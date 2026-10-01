package agentbridge

import (
	"context"
	"sync"
	"testing"
	"time"
)

// fakeClock must be pointer-based: a closure capturing a time.Time value takes a
// COPY, so advancing the test's variable would never reach the library.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

// TestGovernorHalvesOnThrottle is the core claim: a 429 must narrow fan-out.
func TestGovernorHalvesOnThrottle(t *testing.T) {
	g := NewGovernor(GovernorConfig{Max: 100, Min: 1})
	l := g.Limiter("openai")

	if got := g.Width("openai"); got != 100 {
		t.Fatalf("initial width = %v, want 100", got)
	}

	l.Observe(Outcome{Throttled: true})
	if got := g.Width("openai"); got != 50 {
		t.Errorf("after one 429 width = %v, want 50 (halved)", got)
	}

	l.Observe(Outcome{Throttled: true})
	if got := g.Width("openai"); got != 25 {
		t.Errorf("after two 429s width = %v, want 25 (halved again)", got)
	}
}

// TestGovernorNeverBelowMin is what stops the latching pathology seen in a real
// eval harness, where one early 429 pinned the rest of the run to serial.
func TestGovernorNeverBelowMin(t *testing.T) {
	g := NewGovernor(GovernorConfig{Max: 4, Min: 1})
	l := g.Limiter("anthropic")

	for i := 0; i < 20; i++ {
		l.Observe(Outcome{Throttled: true})
	}
	if got := g.Width("anthropic"); got < 1 {
		t.Errorf("width = %v, must never fall below Min=1", got)
	}
}

// TestGovernorRecovers is the half the latching design never reaches: width
// climbs back once the provider stops complaining.
func TestGovernorRecovers(t *testing.T) {
	clock := &fakeClock{t: time.Unix(1_700_000_000, 0)}
	g := NewGovernor(GovernorConfig{
		Max: 16, Min: 1,
		DecreaseFactor: 0.5, IncreaseStep: 2, IncreaseWindow: 4,
		Cooldown: time.Second, Now: clock.now,
	})
	l := g.Limiter("openai")

	l.Observe(Outcome{Throttled: true})
	if got := g.Width("openai"); got != 8 {
		t.Fatalf("after throttle width = %v, want 8", got)
	}

	// Cooldown is active, so successes must not immediately inflate width.
	for i := 0; i < 10; i++ {
		l.Observe(Outcome{OK: true})
	}
	if got := g.Width("openai"); got != 8 {
		t.Errorf("width = %v during cooldown, want it held at 8", got)
	}

	// Past the cooldown, successes earn width back.
	clock.advance(2 * time.Second)
	for i := 0; i < 8; i++ {
		l.Observe(Outcome{OK: true})
	}
	if got := g.Width("openai"); got <= 8 {
		t.Errorf("width = %v, want it to have recovered above 8", got)
	}
}

// TestGovernorDoesNotShrinkOnNonThrottle is a distinction the latching harness
// also gets wrong by collapsing on any error. A 500 means the provider is
// reachable; narrowing for it makes throughput worse and hides a real fault.
func TestGovernorDoesNotShrinkOnNonThrottle(t *testing.T) {
	g := NewGovernor(GovernorConfig{Max: 32, Min: 1})
	l := g.Limiter("openai")
	before := g.Width("openai")

	for i := 0; i < 5; i++ {
		l.Observe(Outcome{OK: false, Throttled: false})
	}
	if got := g.Width("openai"); got != before {
		t.Errorf("width = %v, want unchanged at %v for non-throttle errors", got, before)
	}
}

// TestGovernorCapsAtMax: recovery must not overshoot the configured ceiling.
func TestGovernorCapsAtMax(t *testing.T) {
	clock := &fakeClock{t: time.Unix(1_700_000_000, 0)}
	g := NewGovernor(GovernorConfig{
		Max: 10, Min: 1, IncreaseStep: 3, IncreaseWindow: 2,
		Cooldown: time.Millisecond, Now: clock.now,
	})
	l := g.Limiter("openai")

	for i := 0; i < 100; i++ {
		clock.advance(time.Second)
		l.Observe(Outcome{OK: true})
	}
	if got := g.Width("openai"); got > 10 {
		t.Errorf("width = %v, must never exceed Max=10", got)
	}
}

// TestGovernorBlocksBeyondWidth is the property that actually matters under
// load: the limiter must stop the N+1'th caller when the current width is full.
func TestGovernorBlocksBeyondWidth(t *testing.T) {
	g := NewGovernor(GovernorConfig{Max: 4, Min: 1, DecreaseFactor: 0.5})
	l := g.Limiter("openai")
	l.Observe(Outcome{Throttled: true}) // width 4 -> 2

	for i := 0; i < 2; i++ {
		if err := l.AcquireContext(context.Background()); err != nil {
			t.Fatalf("acquire %d: %v", i, err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := l.AcquireContext(ctx); err == nil {
		t.Error("third acquire should have blocked past the reduced width of 2")
	}

	l.Release()
	if err := l.AcquireContext(context.Background()); err != nil {
		t.Errorf("after Release, acquire should succeed: %v", err)
	}
}

// TestGovernorIsolatesUpstreams: one throttled provider must not slow an
// unrelated one. This is the property a global semaphore cannot give you.
func TestGovernorIsolatesUpstreams(t *testing.T) {
	g := NewGovernor(GovernorConfig{Max: 8, Min: 1, DecreaseFactor: 0.5})
	openai := g.Limiter("openai")
	_ = g.Limiter("anthropic")

	for i := 0; i < 4; i++ {
		openai.Observe(Outcome{Throttled: true})
	}
	if got := g.Width("openai"); got != 1 {
		t.Errorf("openai width = %v, want 1 after 4 throttles", got)
	}
	if got := g.Width("anthropic"); got != 8 {
		t.Errorf("anthropic width = %v, want it untouched at 8", got)
	}
}

// TestGovernorConcurrentAcquireRelease is a race check: inFlight must never
// drift negative or leak slots under contention.
func TestGovernorConcurrentAcquireRelease(t *testing.T) {
	g := NewGovernor(GovernorConfig{Max: 8, Min: 1})
	l := g.Limiter("openai")

	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := l.AcquireContext(context.Background()); err != nil {
				t.Errorf("acquire: %v", err)
				return
			}
			l.Release()
		}()
	}
	wg.Wait()

	l.st.mu.Lock()
	defer l.st.mu.Unlock()
	if l.st.inFlight != 0 {
		t.Errorf("inFlight = %d after all work completed, want 0 (slot leak)", l.st.inFlight)
	}
}
