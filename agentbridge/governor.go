package agentbridge

import (
	"context"
	"math"
	"sync"
	"time"
)

// Governor is an adaptive concurrency limiter, in the spirit of TCP congestion
// control: additive increase on success, multiplicative decrease on pressure.
//
// WHY THIS EXISTS. Retry logic alone is an amplifier, not a control loop. When a
// provider starts answering 429, retrying every failed call at the same
// concurrency adds load to an already-saturated endpoint, which makes the next
// batch more likely to be throttled as well. What actually absorbs a rate limit
// is narrowing the fan-out while it lasts, then widening it once the provider
// recovers.
//
// This was written after reading two public agent-eval harnesses, which between
// them bracket the failure mode. One classifies 429s correctly and retries them
// correctly, but holds fan-out width at a fixed value for the whole run. The
// other handles it the other way: one 429 flips a `serialized` flag that never
// resets, so a single early blip pins the entire remainder of the run to
// concurrency one. Neither adapts, and both are cheap to fix.
//
// Usage:
//
//	g := agentbridge.NewGovernor(agentbridge.GovernorConfig{Max: 100})
//	sem := g.Limiter("openai") // one per upstream, reused across calls
//
//	if err := sem.AcquireContext(ctx); err != nil { return err }
//	out, err := call()
//	sem.Release()
//	sem.Observe(agentbridge.Outcome{OK: err == nil, Throttled: wasRateLimited})
//
// The same Limiter must guard every call to one upstream. A fresh Limiter per
// call makes each goroutine believe it is the only one, and the aggregate is
// unbounded again -- which is precisely the bug this exists to remove.
type Governor struct {
	cfg   GovernorConfig
	now   func() time.Time
	mu    sync.Mutex
	state map[string]*limitState
}

// GovernorConfig bounds a Governor. Unset fields get defaults suited to
// interactive agent traffic rather than unattended batch jobs.
type GovernorConfig struct {
	// Max is the ceiling on concurrent calls to one upstream, and the width
	// used while the provider is healthy.
	Max int

	// Min is the floor. 1 means fully serial, which is the correct response
	// when a provider is genuinely refusing traffic.
	Min int

	// DecreaseFactor is how far width collapses on throttling. 0.5 is the
	// conventional halving and is deliberately not more aggressive: collapsing
	// straight to Min on a single 429 is what makes a latching governor
	// pathological.
	DecreaseFactor float64

	// IncreaseStep is how much width is added after one full window of
	// successes.
	IncreaseStep float64

	// IncreaseWindow is how many consecutive successes earn one IncreaseStep.
	// Gating the increase on a window stops a run of easy calls from inflating
	// the limit while the provider is still the actual bottleneck.
	IncreaseWindow int

	// Cooldown suppresses increases for this long after a decrease, so a
	// provider that recovers slowly is not probed into flapping.
	Cooldown time.Duration

	// Now is injectable for tests.
	Now func() time.Time
}

func (c GovernorConfig) withDefaults() GovernorConfig {
	if c.Max <= 0 {
		c.Max = 8
	}
	if c.Min <= 0 {
		c.Min = 1
	}
	if c.Min > c.Max {
		c.Min = c.Max
	}
	if c.DecreaseFactor <= 0 || c.DecreaseFactor >= 1 {
		c.DecreaseFactor = 0.5
	}
	if c.IncreaseStep <= 0 {
		c.IncreaseStep = 1
	}
	if c.IncreaseWindow <= 0 {
		c.IncreaseWindow = 16
	}
	if c.Cooldown <= 0 {
		c.Cooldown = 2 * time.Second
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

// limitState is the per-upstream controller state.
//
// changed is closed and replaced on every state transition, which is how
// waiters are woken. Closing the previous channel is a broadcast, so a decrease
// that *lowers* the width still wakes everyone to re-evaluate rather than
// leaving them parked on a stale permit.
type limitState struct {
	cfg           *GovernorConfig
	mu            sync.Mutex
	changed       chan struct{}
	inFlight      int
	width         float64
	consecutiveOK int
	cooldownUntil time.Time
}

func (s *limitState) signal() {
	close(s.changed)
	s.changed = make(chan struct{})
}

func (s *limitState) allowed() int {
	w := int(math.Ceil(s.width))
	if w < 1 {
		w = 1
	}
	return w
}

// NewGovernor returns a Governor safe for concurrent use. It tracks a separate
// limit per key handed to Limiter.
func NewGovernor(cfg GovernorConfig) *Governor {
	c := cfg.withDefaults()
	return &Governor{
		cfg:   c,
		now:   c.Now,
		state: make(map[string]*limitState),
	}
}

// Limiter returns the handle for one upstream, created on first use.
func (g *Governor) Limiter(key string) *Limiter {
	g.mu.Lock()
	defer g.mu.Unlock()
	st, ok := g.state[key]
	if !ok {
		st = &limitState{
			cfg:     &g.cfg,
			changed: make(chan struct{}),
			width:   float64(g.cfg.Max),
		}
		g.state[key] = st
	}
	return &Limiter{st: st}
}

// Width reports the current permitted concurrency for a key. Useful for
// logging, and for asserting in tests that the controller actually moves.
func (g *Governor) Width(key string) float64 {
	g.mu.Lock()
	st, ok := g.state[key]
	g.mu.Unlock()
	if !ok {
		return float64(g.cfg.Max)
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.width
}

// Outcome is the result of one governed call.
type Outcome struct {
	// OK is true when the call succeeded.
	OK bool
	// Throttled is true when the failure was specifically rate limiting
	// (HTTP 429, 529, or an explicit overload signal). Only this narrows the
	// fan-out. A 500 or a timeout means the provider is reachable, and
	// shrinking concurrency would mask a real error while making throughput
	// worse.
	Throttled bool
}

// Limiter is the per-upstream handle. Obtain it from Governor.Limiter and
// reuse it; constructing one per call defeats the whole mechanism.
type Limiter struct {
	st *limitState
}

// AcquireContext takes a slot, blocking while the current width is exhausted or
// ctx is done.
func (l *Limiter) AcquireContext(ctx context.Context) error {
	for {
		l.st.mu.Lock()
		if l.st.inFlight < l.st.allowed() {
			l.st.inFlight++
			l.st.mu.Unlock()
			return nil
		}
		ch := l.st.changed
		l.st.mu.Unlock()

		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Acquire is AcquireContext with a background context, for callers that bound
// the call some other way.
func (l *Limiter) Acquire() { _ = l.AcquireContext(context.Background()) }

// Release returns a slot.
func (l *Limiter) Release() {
	l.st.mu.Lock()
	if l.st.inFlight > 0 {
		l.st.inFlight--
	}
	l.st.signal()
	l.st.mu.Unlock()
}

// Observe reports a call result and adjusts the width for this upstream.
func (l *Limiter) Observe(o Outcome) {
	l.st.mu.Lock()
	defer l.st.mu.Unlock()

	if o.Throttled {
		floor := float64(l.st.cfg.Min)
		if floor < 1 {
			floor = 1
		}
		l.st.width = math.Max(floor, l.st.width*l.st.cfg.DecreaseFactor)
		l.st.consecutiveOK = 0
		l.st.cooldownUntil = l.st.cfg.Now().Add(l.st.cfg.Cooldown)
		l.st.signal()
		return
	}

	if !o.OK {
		// Reachable but failing for another reason. Do not shrink.
		return
	}

	if l.st.cfg.Now().Before(l.st.cooldownUntil) {
		return
	}
	l.st.consecutiveOK++
	if l.st.consecutiveOK < l.st.cfg.IncreaseWindow {
		return
	}
	l.st.consecutiveOK = 0
	ceil := float64(l.st.cfg.Max)
	l.st.width = math.Min(ceil, l.st.width+l.st.cfg.IncreaseStep)
	l.st.signal()
}
