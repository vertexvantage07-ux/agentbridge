# agentbridge

The layer between an AI agent and the tools it calls, for when those tools are
unreliable.

Zero dependencies. Standard library only.

```
go test ./...     # 16 tests, ~1s
```

## Why this exists

The recurring failure mode of an agent wired straight to tools is not a bad
model. It is a bad boundary:

- one tool times out and the whole turn is lost;
- a malformed request gets retried four times, burning the budget on a request
  that can never succeed;
- N workers retry in lockstep, so a service that just recovered is immediately
  overloaded again;
- nothing in the logs says which call failed or how long it took.

## What it does

| Behaviour | Why |
|---|---|
| Retries transient failures with exponential backoff **and full jitter** | Synchronised retries are how a recovering service gets knocked over again |
| Gives up **immediately** on permanent failures | A `400` retried four times is a bug, not resilience |
| Honours a server's own `Retry-After` | Better than a guessed delay, always |
| Per-attempt deadlines | A slow first attempt otherwise eats the retry budget and retries become a single attempt |
| Per-tool circuit breakers | One dead dependency should not disable the whole agent |
| Structured telemetry per call | Duration, attempts, outcome, and failure kind, queryable |
| Health view | Lets the agent say "that tool is down" instead of appearing to hang |

## Usage

```go
bridge := agentbridge.NewBridge[MyPayload](agentbridge.DefaultConfig(), nil)

type MyPayload struct{ Query string }

search := agentbridge.Tool[MyPayload]{
    Name: "search",
    Call: func(ctx context.Context, in MyPayload) (MyPayload, error) {
        return callSearchAPI(ctx, in)
    },
}

out, err := bridge.Do(ctx, search, MyPayload{Query: "golang"})
if err != nil {
    var te *agentbridge.TerminalError
    if errors.As(err, &te) && te.Kind == agentbridge.KindThrottled {
        // degrade: queue it and move on
    }
}
```

Mark a tool's failures non-retryable when only the tool knows better:

```go
tool.NonRetryable = func(err error) bool {
    var pe *agentbridge.ProtocolError
    return errors.As(err, &pe)
}
```

Emit telemetry by implementing one interface:

```go
bridge := agentbridge.NewBridge[MyPayload](cfg, agentbridge.ObserverFunc(
    func(r agentbridge.CallRecord) {
        metrics.Observe("agent.tool.call", r.Elapsed, "tool", r.Tool, "outcome", r.Outcome)
    }))
```

## The decisions I made that you may disagree with

**A timeout is terminal for the turn.** I do not retry it. The caller already
waited the full budget, and retrying makes the user wait longer for the same
answer. If you would rather trade latency for completion, that is a one-line
change.

**The breaker is per-tool, not global.** One broken dependency should not take
down the agent. If a dependency is genuinely all-or-nothing, this is the wrong
design and you should use a global breaker.

**Jitter is full, not equal.** The delay is `0.7x`–`1.3x` of the exponential
value. Full jitter (`0x`–`1x`) is the AWS-recommended variant; I used the
symmetric window because it keeps the median delay at the nominal value. There
is a test asserting the spread is real, so a regression to a fixed delay fails
the suite.

## Design notes

Go has no generic methods on non-generic types, so `Bridge` is generic over the
payload type and the tool type parameter is inferred at the call site. The
breaker is a separate non-generic type because it does not depend on the
payload.

`Tool.Call` receives a fresh deadline per attempt, not a shared one, for the
reason in the table above.

## Tests

16 tests covering: first-try success, transient recovery, permanent give-up,
retry exhaustion, throttling and server hints, caller cancellation, per-call
timeouts, breaker open/close/recovery, telemetry emission, tool-declared
non-retryable errors, bounded jitter, zero-config operation, and concurrent use
from 32 goroutines.

```
ok  github.com/vertexvantage07-ux/agentbridge/agentbridge  1.028s
```

## License

MIT. See LICENSE.


## License

MIT. See LICENSE.
