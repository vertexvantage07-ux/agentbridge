package agentbridge

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func echoTool(name string, calls *int) Tool[string] {
	return Tool[string]{
		Name: name,
		Call: func(_ context.Context, in string) (string, error) {
			*calls++
			return in, nil
		},
	}
}

func TestSucceedsFirstTry(t *testing.T) {
	calls := 0
	b := NewBridge[string](DefaultConfig(), nil)
	out, err := b.Do(context.Background(), echoTool("ok", &calls), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "hello" {
		t.Fatalf("got %q, want %q", out, "hello")
	}
	if calls != 1 {
		t.Fatalf("called %d times, want 1", calls)
	}
}

func TestRetriesTransientThenSucceeds(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "flaky",
		Call: func(_ context.Context, in string) (string, error) {
			calls++
			if calls < 3 {
				return "", errors.New("connection reset by peer")
			}
			return in, nil
		},
	}
	cfg := DefaultConfig()
	cfg.BaseBackoff = time.Millisecond
	cfg.MaxBackoff = 2 * time.Millisecond
	b := NewBridge[string](cfg, nil)

	out, err := b.Do(context.Background(), tool, "payload")
	if err != nil {
		t.Fatalf("should have recovered: %v", err)
	}
	if out != "payload" {
		t.Fatalf("got %q", out)
	}
	if calls != 3 {
		t.Fatalf("called %d times, want 3", calls)
	}
}

func TestPermanentFailureIsNotRetried(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "strict",
		Call: func(_ context.Context, _ string) (string, error) {
			calls++
			return "", &ProtocolError{Status: 400, Reason: "malformed request"}
		},
	}
	cfg := DefaultConfig()
	cfg.BaseBackoff = time.Millisecond
	b := NewBridge[string](cfg, nil)

	_, err := b.Do(context.Background(), tool, "x")
	if err == nil {
		t.Fatal("expected an error")
	}
	// The whole point: a 400 must not burn four attempts.
	if calls != 1 {
		t.Fatalf("permanent failure retried %d times, want 1", calls)
	}
	var te *TerminalError
	if !errors.As(err, &te) {
		t.Fatalf("want *TerminalError, got %T", err)
	}
	if te.Kind != KindPermanent {
		t.Fatalf("kind = %q, want %q", te.Kind, KindPermanent)
	}
	if !te.IsPermanent() {
		t.Fatal("IsPermanent() = false, want true")
	}
}

func TestRetriesExhaustedAndReportsAttempts(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "dead",
		Call: func(_ context.Context, _ string) (string, error) {
			calls++
			return "", errors.New("upstream down")
		},
	}
	cfg := DefaultConfig()
	cfg.MaxAttempts = 3
	cfg.BaseBackoff = time.Millisecond
	cfg.MaxBackoff = time.Millisecond
	b := NewBridge[string](cfg, nil)

	_, err := b.Do(context.Background(), tool, "x")
	var te *TerminalError
	if !errors.As(err, &te) {
		t.Fatalf("want *TerminalError, got %v", err)
	}
	if !te.Exhausted {
		t.Fatal("Exhausted = false, want true")
	}
	if te.Attempts != 3 {
		t.Fatalf("attempts = %d, want 3", te.Attempts)
	}
	if calls != 3 {
		t.Fatalf("called %d times, want 3", calls)
	}
}

func TestThrottleIsClassifiedAndHonoursServerHint(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "limited",
		Call: func(_ context.Context, _ string) (string, error) {
			calls++
			if calls == 1 {
				return "", &ThrottleError{Status: 429, RetryIn: time.Millisecond, Provider: "acme"}
			}
			return "ok", nil
		},
	}
	cfg := DefaultConfig()
	cfg.BaseBackoff = time.Millisecond
	b := NewBridge[string](cfg, nil)

	if _, err := b.Do(context.Background(), tool, "x"); err != nil {
		t.Fatalf("should have recovered after throttle: %v", err)
	}
	if calls != 2 {
		t.Fatalf("called %d times, want 2", calls)
	}
}

func TestCallerCancellationStopsImmediately(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "slow",
		Call: func(_ context.Context, _ string) (string, error) {
			calls++
			return "", errors.New("boom")
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before the call

	cfg := DefaultConfig()
	cfg.BaseBackoff = time.Hour // if it slept, the test would hang
	cfg.MaxAttempts = 5
	b := NewBridge[string](cfg, nil)

	start := time.Now()
	_, err := b.Do(ctx, tool, "x")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected an error")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("took %s; it slept despite a cancelled context", elapsed)
	}
	if calls > 2 {
		t.Fatalf("called %d times after cancellation, want <=2", calls)
	}
}

func TestCallTimeoutIsEnforced(t *testing.T) {
	tool := Tool[string]{
		Name: "hangs",
		Call: func(ctx context.Context, _ string) (string, error) {
			<-ctx.Done()
			return "", ctx.Err()
		},
	}
	cfg := DefaultConfig()
	cfg.CallTimeout = 20 * time.Millisecond
	cfg.MaxAttempts = 1
	b := NewBridge[string](cfg, nil)

	start := time.Now()
	_, err := b.Do(context.Background(), tool, "x")
	elapsed := time.Since(start)

	var te *TerminalError
	if !errors.As(err, &te) {
		t.Fatalf("want *TerminalError, got %v", err)
	}
	if te.Kind != KindTimeout {
		t.Fatalf("kind = %q, want %q", te.Kind, KindTimeout)
	}
	if elapsed > time.Second {
		t.Fatalf("timeout not enforced: took %s", elapsed)
	}
}

func TestCircuitBreakerOpensAndBlocks(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "broken",
		Call: func(_ context.Context, _ string) (string, error) {
			calls++
			return "", errors.New("down")
		},
	}
	cfg := DefaultConfig()
	cfg.MaxAttempts = 1
	cfg.BreakerThreshold = 3
	cfg.BreakerOpenFor = time.Minute
	b := NewBridge[string](cfg, nil)

	for i := 0; i < 3; i++ {
		_, _ = b.Do(context.Background(), tool, "x")
	}
	if !b.Breaker("broken").Trips() {
		t.Fatal("breaker should be open after 3 failures")
	}

	// The next call must not reach the dependency at all.
	before := calls
	_, err := b.Do(context.Background(), tool, "x")
	if calls != before {
		t.Fatalf("dialled a known-dead dependency %d extra times", calls-before)
	}
	var te *TerminalError
	if !errors.As(err, &te) {
		t.Fatalf("want *TerminalError, got %v", err)
	}
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatal("error should wrap ErrCircuitOpen")
	}

	h := b.Health()
	if h.Broken != 1 || !h.Tools["broken"].CircuitOpen {
		t.Fatalf("health should report the open circuit: %+v", h)
	}
	if h.JSON() == "{}" {
		t.Fatal("health JSON should render")
	}
}

func TestBreakerRecoversAfterCoolDown(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "healing",
		Call: func(_ context.Context, in string) (string, error) {
			calls++
			if calls <= 2 {
				return "", errors.New("down")
			}
			return in, nil
		},
	}
	cfg := DefaultConfig()
	cfg.MaxAttempts = 1
	cfg.BreakerThreshold = 2
	cfg.BreakerOpenFor = 10 * time.Millisecond
	b := NewBridge[string](cfg, nil)

	_, _ = b.Do(context.Background(), tool, "x")
	_, _ = b.Do(context.Background(), tool, "x")
	if !b.Breaker("healing").Trips() {
		t.Fatal("breaker should be open")
	}
	time.Sleep(20 * time.Millisecond) // let the cool-down lapse

	if _, err := b.Do(context.Background(), tool, "x"); err != nil {
		t.Fatalf("breaker should let a probe through after cool-down: %v", err)
	}
	if b.Breaker("healing").Trips() {
		t.Fatal("a successful probe should close the circuit")
	}
}

func TestTelemetryIsEmittedForEveryCall(t *testing.T) {
	var mu sync.Mutex
	var seen []CallRecord
	obs := ObserverFunc(func(r CallRecord) {
		mu.Lock()
		defer mu.Unlock()
		seen = append(seen, r)
	})

	calls := 0
	tool := Tool[string]{
		Name: "observed",
		Call: func(_ context.Context, in string) (string, error) {
			calls++
			if calls < 2 {
				return "", errors.New("transient")
			}
			return in, nil
		},
	}
	cfg := DefaultConfig()
	cfg.BaseBackoff = time.Millisecond
	b := NewBridge[string](cfg, obs)

	if _, err := b.Do(context.Background(), tool, "x"); err != nil {
		t.Fatalf("unexpected: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 {
		t.Fatalf("got %d records, want 2 (one retry, one ok)", len(seen))
	}
	if seen[0].Outcome != "retrying" {
		t.Fatalf("first record = %q, want retrying", seen[0].Outcome)
	}
	if seen[1].Outcome != "ok" || seen[1].Attempts != 2 {
		t.Fatalf("second record = %+v, want ok with 2 attempts", seen[1])
	}
	if seen[0].Tool != "observed" || seen[0].At.IsZero() {
		t.Fatalf("record missing tool or timestamp: %+v", seen[0])
	}
}

func TestToolMarkedNonRetryableStopsEarly(t *testing.T) {
	calls := 0
	tool := Tool[string]{
		Name: "opaque",
		Call: func(_ context.Context, _ string) (string, error) {
			calls++
			// A plain error the classifier cannot judge, so only the tool knows.
			return "", errors.New("semantic failure: tenant already migrated")
		},
		NonRetryable: func(err error) bool {
			return err != nil && err.Error() == "semantic failure: tenant already migrated"
		},
	}
	cfg := DefaultConfig()
	cfg.MaxAttempts = 5
	cfg.BaseBackoff = time.Millisecond
	b := NewBridge[string](cfg, nil)

	if _, err := b.Do(context.Background(), tool, "x"); err == nil {
		t.Fatal("expected an error")
	}
	if calls != 1 {
		t.Fatalf("called %d times, want 1: NonRetryable was ignored", calls)
	}
}

func TestBackoffIsBoundedAndJittered(t *testing.T) {
	b := NewBridge[string](DefaultConfig(), nil)
	seen := map[time.Duration]bool{}
	for i := 0; i < 40; i++ {
		d := b.backoffFor(3, errors.New("x"))
		if d < 0 || d > b.cfg.MaxBackoff {
			t.Fatalf("backoff %s out of bounds (max %s)", d, b.cfg.MaxBackoff)
		}
		seen[d] = true
	}
	// Full jitter means a spread of values, not one repeated delay.
	if len(seen) < 5 {
		t.Fatalf("only %d distinct delays; jitter looks absent", len(seen))
	}
}

func TestThrottleHintOverridesBackoff(t *testing.T) {
	b := NewBridge[string](DefaultConfig(), nil)
	hint := 42 * time.Millisecond
	got := b.backoffFor(1, &ThrottleError{Status: 429, RetryIn: hint, Provider: "x"})
	if got != hint {
		t.Fatalf("got %s, want the server hint %s", got, hint)
	}
}

func TestZeroConfigStillWorks(t *testing.T) {
	calls := 0
	b := NewBridge[string](Config{}, nil) // every field zero
	if _, err := b.Do(context.Background(), echoTool("d", &calls), "x"); err != nil {
		t.Fatalf("zero config should still work: %v", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d", calls)
	}
}

func TestConcurrentUseIsSafe(t *testing.T) {
	b := NewBridge[string](DefaultConfig(), nil)
	var wg sync.WaitGroup
	var mu sync.Mutex
	failures := 0
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			calls := 0
			tool := Tool[string]{
				Name: fmt.Sprintf("t%d", n%4),
				Call: func(_ context.Context, in string) (string, error) {
					calls++
					if calls < 2 {
						return "", errors.New("flaky")
					}
					return in, nil
				},
			}
			cfg := DefaultConfig()
			cfg.BaseBackoff = time.Millisecond
			cfg.MaxBackoff = time.Millisecond
			if _, err := b.Do(context.Background(), tool, "payload"); err != nil {
				mu.Lock()
				failures++
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	if failures != 0 {
		t.Fatalf("%d concurrent calls failed; the bridge is not safe for shared use", failures)
	}
}

func TestHealthOnFreshBridgeIsEmptyNotNil(t *testing.T) {
	b := NewBridge[string](DefaultConfig(), nil)
	h := b.Health()
	if h.Tools == nil {
		t.Fatal("Tools map should be initialised, not nil")
	}
	if h.Broken != 0 {
		t.Fatalf("Broken = %d on a fresh bridge", h.Broken)
	}
}

