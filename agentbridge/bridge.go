// Package agentbridge is a real integration layer for an AI agent that has to
// talk to unreliable third-party services.
//
// It exists because the recurring failure mode of an agent wired straight to
// tools is not a bad model, it is a bad boundary: one tool times out, the whole
// turn is lost, and nobody can say which call failed or how long it took.
//
// What this package does about it, and nothing more:
//
//   - every call is bounded by a context deadline, so one slow tool cannot hang
//     the agent loop;
//   - transient failures are retried with exponential backoff and full jitter,
//     because synchronised retries from many workers are how a recovering
//     service gets knocked over again;
//   - failures are classified, so a 429 is waited out and a 400 is returned
//     immediately rather than burning four attempts on a request that can never
//     succeed;
//   - every call emits structured telemetry: duration, attempt count, outcome,
//     and whether the retry budget was exhausted;
//   - circuit breaking stops a dead dependency from consuming the whole agent's
//     latency budget on every single turn.
//
// There is no model, no prompt, and no orchestration here. This is the layer
// between the agent and its tools, which is where the value is.
package agentbridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// FailureKind is a coarse classification of why a call failed. It exists
// because the correct reaction differs per class, and lumping everything into
// "error" is what causes an agent to retry a malformed request four times.
type FailureKind string

const (
	// KindTransient means the same request may well succeed if tried again.
	KindTransient FailureKind = "transient"
	// KindPermanent means retrying cannot help: bad input, missing auth,
	// malformed schema. Retrying wastes budget and delays a useful error.
	KindPermanent FailureKind = "permanent"
	// KindThrottled means the caller is being rate limited and needs to wait.
	KindThrottled FailureKind = "throttled"
	// KindTimeout means the deadline elapsed before the call returned.
	KindTimeout FailureKind = "timeout"
)

// TerminalError wraps the last underlying failure with enough context to
// diagnose it after the fact, which is the whole point of having telemetry.
type TerminalError struct {
	Kind      FailureKind
	Tool      string
	Attempts  int
	Last      error
	Elapsed   time.Duration
	Exhausted bool // true when retries ran out rather than a permanent refusal
}

func (e *TerminalError) Error() string {
	reason := "permanent failure"
	if e.Exhausted {
		reason = fmt.Sprintf("gave up after %d attempts", e.Attempts)
	}
	return fmt.Sprintf("tool %q: %s after %d attempt(s) in %s: %v",
		e.Tool, reason, e.Attempts, e.Elapsed.Round(time.Millisecond), e.Last)
}

func (e *TerminalError) Unwrap() error { return e.Last }

// IsPermanent reports whether retrying could ever help. A caller can use this
// to fall back to a degraded path instead of failing the turn.
func (e *TerminalError) IsPermanent() bool { return e.Kind == KindPermanent }

// classify maps an underlying error onto a FailureKind. It inspects context
// state first, because a cancelled context can wrap anything and the deadline
// is the more actionable explanation.
func classify(ctx context.Context, err error) FailureKind {
	if err == nil {
		return KindTransient
	}
	if errors.Is(err, context.Canceled) {
		return KindPermanent // the caller went away; nobody is waiting on a retry
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return KindTimeout
	}
	var th *ThrottleError
	if errors.As(err, &th) {
		return KindThrottled
	}
	var pe *ProtocolError
	if errors.As(err, &pe) {
		return KindPermanent
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return KindTimeout
	}
	return KindTransient
}

// ThrottleError is returned by a tool that is rate limiting the caller. It
// carries the server's own retry hint when one is supplied, which is always
// better than a guess.
type ThrottleError struct {
	Status   int
	RetryIn  time.Duration
	Provider string
}

func (e *ThrottleError) Error() string {
	if e.RetryIn > 0 {
		return fmt.Sprintf("%s throttled (status %d), retry in %s",
			e.Provider, e.Status, e.RetryIn.Round(time.Millisecond))
	}
	return fmt.Sprintf("%s throttled (status %d)", e.Provider, e.Status)
}

// ProtocolError is returned when a remote service rejects the shape of a
// request. Retrying an identical bad request cannot help.
type ProtocolError struct {
	Status  int
	Reason  string
	Details string
}

func (e *ProtocolError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("rejected request (status %d): %s: %s", e.Status, e.Reason, e.Details)
	}
	return fmt.Sprintf("rejected request (status %d): %s", e.Status, e.Reason)
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

// CallRecord is one completed call, successful or not. It is a plain struct
// rather than a log line so it can be stored, aggregated, and asserted on in
// tests without parsing text.
type CallRecord struct {
	Tool     string        `json:"tool"`
	Outcome  string        `json:"outcome"` // "ok", "retrying", "failed", "rejected", "circuit_open"
	Kind     FailureKind   `json:"kind,omitempty"`
	Attempts int           `json:"attempts"`
	Elapsed  time.Duration `json:"elapsed"`
	Err      string        `json:"error,omitempty"`
	At       time.Time     `json:"at"`
}

// Observer receives every call record. Implementations must not block: the
// bridge calls this on the hot path, and a slow observer turns a fast agent
// into a slow one. The default observer does nothing.
type Observer interface {
	ObserveCall(CallRecord)
}

// ObserverFunc adapts a function to Observer.
type ObserverFunc func(CallRecord)

func (f ObserverFunc) ObserveCall(r CallRecord) { f(r) }

// NopObserver is the default. Using it explicitly documents that telemetry is
// being discarded on purpose rather than by accident.
var NopObserver Observer = ObserverFunc(func(CallRecord) {})

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

// Tool is a remote operation the agent can invoke.
//
// The signature is deliberately plain: context in, value out, error back. Any
// framework can adapt to it, and it keeps the retry and telemetry logic
// testable without a model in the loop.
type Tool[T any] struct {
	Name string
	Call func(ctx context.Context, in T) (T, error)

	// NonRetryable lets a tool mark specific failures as permanent from the
	// inside, for conditions classify() cannot see from the error alone.
	NonRetryable func(err error) bool
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Config controls retry, timeout, and circuit-breaker behaviour.
//
// The defaults are chosen for interactive agent use, where a turn that takes
// more than a few seconds has already lost the user's attention.
type Config struct {
	MaxAttempts    int
	BaseBackoff    time.Duration
	MaxBackoff     time.Duration
	CallTimeout    time.Duration
	JitterFraction float64

	// Breaker trips after this many consecutive failures and stays open for
	// OpenFor. Zero values fall back to sane defaults.
	BreakerThreshold int
	BreakerOpenFor   time.Duration
}

func (c Config) withDefaults() Config {
	if c.MaxAttempts <= 0 {
		c.MaxAttempts = 4
	}
	if c.BaseBackoff <= 0 {
		c.BaseBackoff = 150 * time.Millisecond
	}
	if c.MaxBackoff <= 0 {
		c.MaxBackoff = 5 * time.Second
	}
	if c.CallTimeout <= 0 {
		c.CallTimeout = 20 * time.Second
	}
	if c.JitterFraction <= 0 {
		c.JitterFraction = 0.3
	}
	if c.BreakerThreshold <= 0 {
		c.BreakerThreshold = 5
	}
	if c.BreakerOpenFor <= 0 {
		c.BreakerOpenFor = 30 * time.Second
	}
	return c
}

// DefaultConfig is the configuration used when a caller passes none.
func DefaultConfig() Config { return Config{}.withDefaults() }

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

// Breaker stops the bridge from spending its whole latency budget talking to a
// dependency that is already known to be down. Without it, every agent turn
// pays the full timeout for every broken tool.
type Breaker struct {
	mu        sync.Mutex
	failures  int
	openedAt  time.Time
	openFor   time.Duration
	threshold int
	now       func() time.Time
}

// NewBreaker returns a closed breaker.
func NewBreaker(threshold int, openFor time.Duration) *Breaker {
	b := &Breaker{threshold: threshold, openFor: openFor}
	b.now = time.Now
	return b
}

// ErrCircuitOpen is returned instead of dialling a dependency known to be down.
var ErrCircuitOpen = errors.New("circuit open: dependency is failing, not dialling it")

// Allow reports whether a call may proceed.
func (b *Breaker) Allow() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failures < b.threshold {
		return nil
	}
	if b.now().Sub(b.openedAt) < b.openFor {
		return ErrCircuitOpen
	}
	// Cool-down elapsed: allow a probe and reset the counter so a single
	// success restores the circuit.
	b.failures = 0
	return nil
}

// Success records a good call.
func (b *Breaker) Success() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures = 0
}

// Failure records a bad call, tripping the breaker at the threshold.
func (b *Breaker) Failure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failures++; b.failures >= b.threshold {
		b.openedAt = b.now()
	}
}

// Trips reports whether the breaker is currently open. Intended for health
// endpoints and tests.
func (b *Breaker) Trips() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.failures >= b.threshold && b.now().Sub(b.openedAt) < b.openFor
}

// ---------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------

// Bridge executes tool calls with retry, deadlines, and telemetry. It is safe
// for concurrent use and is meant to be long-lived: the breaker state is the
// reason to keep one around rather than building per turn.
//
// It is generic over the tool's payload type so that Do can return a typed
// value without a type assertion at every call site. Go has no generic methods
// on non-generic types, hence the type parameter on the struct itself.
type Bridge[T any] struct {
	cfg      Config
	observer Observer
	rand     *rand.Rand
	randMu   sync.Mutex
	breakers map[string]*Breaker
	breakerMu sync.Mutex
}

// NewBridge returns a bridge ready for use. A nil observer discards telemetry.
func NewBridge[T any](cfg Config, observer Observer) *Bridge[T] {
	if observer == nil {
		observer = NopObserver
	}
	return &Bridge[T]{
		cfg:      cfg.withDefaults(),
		observer: observer,
		//nolint:gosec // jitter for retry timing, not a security primitive
		rand:     rand.New(rand.NewSource(time.Now().UnixNano())),
		breakers: make(map[string]*Breaker),
	}
}

// Breaker exposes the breaker for a tool so callers can inspect or reset it.
func (b *Bridge[T]) Breaker(tool string) *Breaker {
	b.breakerMu.Lock()
	defer b.breakerMu.Unlock()
	if br, ok := b.breakers[tool]; ok {
		return br
	}
	br := NewBreaker(b.cfg.BreakerThreshold, b.cfg.BreakerOpenFor)
	b.breakers[tool] = br
	return br
}

// Do executes a tool call, retrying transient failures and giving up fast on
// permanent ones. The returned error is always a *TerminalError, so a caller
// can inspect Kind and decide whether to degrade or surface.
func (b *Bridge[T]) Do(ctx context.Context, tool Tool[T], in T) (T, error) {
	var zero T
	started := time.Now()

	breaker := b.Breaker(tool.Name)
	if err := breaker.Allow(); err != nil {
		b.observer.ObserveCall(CallRecord{
			Tool: tool.Name, Outcome: "circuit_open", Kind: KindTransient,
			Elapsed: time.Since(started), At: time.Now(),
			Err: ErrCircuitOpen.Error(),
		})
		return zero, &TerminalError{
			Kind: KindTransient, Tool: tool.Name, Attempts: 0,
			Elapsed: time.Since(started), Last: ErrCircuitOpen, Exhausted: true,
		}
	}

	var last error
	var lastKind FailureKind

	for attempt := 1; attempt <= b.cfg.MaxAttempts; attempt++ {
		// A fresh per-attempt deadline. Sharing one deadline across retries
		// means a slow first attempt eats the budget and the retries have
		// nothing left, which silently converts retries into a single attempt.
		attemptCtx, cancel := context.WithTimeout(ctx, b.cfg.CallTimeout)
		out, err := tool.Call(attemptCtx, in)
		cancel()

		if err == nil {
			breaker.Success()
			b.observer.ObserveCall(CallRecord{
				Tool: tool.Name, Outcome: "ok", Attempts: attempt,
				Elapsed: time.Since(started), At: time.Now(),
			})
			return out, nil
		}

		last = err
		lastKind = classify(ctx, err)

		permanent := lastKind == KindPermanent || lastKind == KindTimeout ||
			(tool.NonRetryable != nil && tool.NonRetryable(err))
		if permanent {
			// A timeout is terminal for this turn: the caller already waited
			// the full budget, and retrying makes the user wait longer for the
			// same answer. Hand back the error and let the agent degrade.
			breaker.Failure()
			b.observer.ObserveCall(CallRecord{
				Tool: tool.Name, Outcome: "rejected", Kind: lastKind,
				Attempts: attempt, Elapsed: time.Since(started), At: time.Now(),
				Err: err.Error(),
			})
			return zero, &TerminalError{
				Kind: lastKind, Tool: tool.Name, Attempts: attempt,
				Elapsed: time.Since(started), Last: err,
			}
		}

		if attempt == b.cfg.MaxAttempts {
			break
		}

		b.observer.ObserveCall(CallRecord{
			Tool: tool.Name, Outcome: "retrying", Kind: lastKind,
			Attempts: attempt, Elapsed: time.Since(started), At: time.Now(),
			Err: err.Error(),
		})

		wait := b.backoffFor(attempt, last)
		// Respect the caller's own deadline. Sleeping past it wastes the turn.
		if dl, ok := ctx.Deadline(); ok && time.Now().Add(wait).After(dl) {
			break
		}
		select {
		case <-ctx.Done():
			return zero, &TerminalError{
				Kind: KindTransient, Tool: tool.Name, Attempts: attempt,
				Elapsed: time.Since(started), Last: ctx.Err(), Exhausted: true,
			}
		case <-time.After(wait):
		}
	}

	breaker.Failure()
	b.observer.ObserveCall(CallRecord{
		Tool: tool.Name, Outcome: "failed", Kind: lastKind,
		Attempts: b.cfg.MaxAttempts, Elapsed: time.Since(started), At: time.Now(),
		Err: fmt.Sprint(last),
	})
	return zero, &TerminalError{
		Kind: lastKind, Tool: tool.Name, Attempts: b.cfg.MaxAttempts,
		Elapsed: time.Since(started), Last: last, Exhausted: true,
	}
}

// backoffFor computes an exponential delay with full jitter. Full jitter, not
// a fixed exponential, because N workers retrying on the same schedule is how a
// service that just came back up gets immediately overloaded again.
func (b *Bridge[T]) backoffFor(attempt int, last error) time.Duration {
	// Honour the server's own hint when it gave one.
	var th *ThrottleError
	if errors.As(last, &th) && th.RetryIn > 0 {
		if th.RetryIn > b.cfg.MaxBackoff {
			return b.cfg.MaxBackoff
		}
		return th.RetryIn
	}

	exp := float64(b.cfg.BaseBackoff) * math.Pow(2, float64(attempt-1))
	if exp > float64(b.cfg.MaxBackoff) {
		exp = float64(b.cfg.MaxBackoff)
	}
	jitter := b.jitter()
	d := time.Duration(exp * (1 - b.cfg.JitterFraction + 2*b.cfg.JitterFraction*jitter))
	if d < 0 {
		d = 0
	}
	return d
}

func (b *Bridge[T]) jitter() float64 {
	b.randMu.Lock()
	defer b.randMu.Unlock()
	return b.rand.Float64()
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

// Health is a point-in-time view of a bridge, suitable for a status endpoint.
type Health struct {
	Tools  map[string]ToolHealth `json:"tools"`
	Broken int                   `json:"broken"`
}

// ToolHealth is the per-tool view.
type ToolHealth struct {
	CircuitOpen bool `json:"circuit_open"`
}

// Health reports which tools currently have an open breaker. An agent that
// knows a dependency is down can say so instead of appearing to hang.
func (b *Bridge[T]) Health() Health {
	h := Health{Tools: make(map[string]ToolHealth, len(b.breakers))}
	b.breakerMu.Lock()
	breakers := make(map[string]*Breaker, len(b.breakers))
	for k, v := range b.breakers {
		breakers[k] = v
	}
	b.breakerMu.Unlock()
	for name, br := range breakers {
		open := br.Trips()
		h.Tools[name] = ToolHealth{CircuitOpen: open}
		if open {
			h.Broken++
		}
	}
	return h
}

// JSON renders health for a log or an HTTP response.
func (h Health) JSON() string {
	b, err := json.Marshal(h)
	if err != nil {
		return "{}"
	}
	return string(b)
}

