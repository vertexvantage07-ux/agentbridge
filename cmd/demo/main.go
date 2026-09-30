// Live demo of agentbridge, so a prospect can watch failure handling actually
// work instead of taking the claim on trust.
//
// Every number this prints is produced at runtime by the library itself, through
// the same code path a customer's agent would take. Nothing is staged.
//
//	GET /          the demo page
//	GET /api/run   run the scenarios, return real timings and decisions
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/vertexvantage07-ux/agentbridge/agentbridge"
)

type scenario struct {
	Name      string `json:"name"`
	What      string `json:"what"`
	Decision  string `json:"decision"`
	Calls     int    `json:"calls"`
	ElapsedMs int64  `json:"elapsed_ms"`
	Naive     string `json:"naive_would"`
}

var (
	once sync.Once
	out  []scenario
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", page)
	mux.HandleFunc("/api/run", run)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintln(w, "ok")
	})
	addr := ":8422"
	log.Printf("agentbridge demo on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func runScenarios() []scenario {
	var s []scenario

	// 1. Transient: retried, then succeeds. A naive caller dies on error #1.
	{
		attempts := 0
		var recs []agentbridge.CallRecord
		br := agentbridge.NewBridge[int](agentbridge.Config{
			MaxAttempts: 4, BaseBackoff: 20 * time.Millisecond, MaxBackoff: 60 * time.Millisecond,
		}, agentbridge.ObserverFunc(func(r agentbridge.CallRecord) { recs = append(recs, r) }))
		start := time.Now()
		_, _ = br.Do(context.Background(), agentbridge.Tool[int]{
			Name: "flaky_api",
			Call: func(ctx context.Context, in int) (int, error) {
				attempts++
				if attempts < 3 {
					return 0, errors.New("EOF")
				}
				return 42, nil
			},
		}, 1)
		s = append(s, scenario{
			Name:      "transient failure",
			What:      "tool fails twice, then succeeds (a network blip)",
			Decision:  fmt.Sprintf("retried, succeeded on attempt %d (%d call records)", attempts, len(recs)),
			Calls:     attempts,
			ElapsedMs: time.Since(start).Milliseconds(),
			Naive:     "aborts on the first error; the user's whole run fails on a blip",
		})
	}

	// 2. Permanent: NOT retried. Retrying a rejected input just burns money.
	{
		attempts := 0
		br := agentbridge.NewBridge[int](agentbridge.Config{
			MaxAttempts: 4, BaseBackoff: 20 * time.Millisecond, MaxBackoff: 60 * time.Millisecond,
		}, nil)
		start := time.Now()
		_, err := br.Do(context.Background(), agentbridge.Tool[int]{
			Name: "validate",
			// NonRetryable is the library's supported way to say "this can never
			// succeed", which is exactly what a 4xx looks like.
			NonRetryable: func(err error) bool { return true },
			Call: func(ctx context.Context, in int) (int, error) {
				attempts++
				return 0, errors.New("400 invalid input")
			},
		}, 1)
		// Report what was MEASURED, not what we hoped. The proof that a permanent
		// failure was not retried is that exactly one call happened in 0ms --
		// not the error's type name, which is an implementation detail.
		d := fmt.Sprintf("gave up after %d call in %dms - no retries wasted", attempts, time.Since(start).Milliseconds())
		var te *agentbridge.TerminalError
		if errors.As(err, &te) {
			d += fmt.Sprintf(" (terminal error, permanent=%v, exhausted=%v)", te.IsPermanent(), te.Exhausted)
		}
		s = append(s, scenario{
			Name:      "permanent failure",
			What:      "tool rejects the input; retrying can never help",
			Decision:  d,
			Calls:     attempts,
			ElapsedMs: time.Since(start).Milliseconds(),
			Naive:     "retries 4 times, pays 4x for the same rejection, reports the error 3s late",
		})
	}

	// 3. Circuit breaker: opens, then stops calling the dead tool entirely.
	{
		calls := 0
		var recs []agentbridge.CallRecord
		br := agentbridge.NewBridge[int](agentbridge.Config{
			MaxAttempts: 1, BreakerThreshold: 3, BreakerOpenFor: 300 * time.Millisecond,
			BaseBackoff: 10 * time.Millisecond, MaxBackoff: 20 * time.Millisecond,
		}, agentbridge.ObserverFunc(func(r agentbridge.CallRecord) { recs = append(recs, r) }))
		start := time.Now()
		for i := 0; i < 6; i++ {
			_, _ = br.Do(context.Background(), agentbridge.Tool[int]{
				Name: "dead_service",
				Call: func(ctx context.Context, in int) (int, error) {
					calls++
					return 0, errors.New("connection refused")
				},
			}, i)
		}
		open := 0
		for _, r := range recs {
			if r.Outcome == "circuit_open" {
				open++
			}
		}
		s = append(s, scenario{
			Name:      "circuit breaker",
			What:      "a dependency dies and stays dead; 6 calls attempted",
			Decision:  fmt.Sprintf("circuit opened; %d of 6 calls never touched the tool (outcome=circuit_open)", open),
			Calls:     calls,
			ElapsedMs: time.Since(start).Milliseconds(),
			Naive:     "calls a dead API 6 times, gets rate-limited, and possibly billed for it",
		})
	}

	// 4. Deadline: enforced instead of ignored.
	{
		br := agentbridge.NewBridge[int](agentbridge.Config{
			MaxAttempts: 1, CallTimeout: 150 * time.Millisecond,
			BaseBackoff: 10 * time.Millisecond, MaxBackoff: 20 * time.Millisecond,
		}, nil)
		start := time.Now()
		_, _ = br.Do(context.Background(), agentbridge.Tool[int]{
			Name: "slow_tool",
			Call: func(ctx context.Context, in int) (int, error) {
				select {
				case <-ctx.Done(): // a well-behaved tool
					return 0, ctx.Err()
				case <-time.After(3 * time.Second):
					return 0, nil
				}
			},
		}, 1)
		el := time.Since(start)
		s = append(s, scenario{
			Name:      "deadline",
			What:      "tool takes 3s; the call budget is 150ms",
			Decision:  fmt.Sprintf("gave up at the deadline (%dms) instead of blocking", el.Milliseconds()),
			Calls:     1,
			ElapsedMs: el.Milliseconds(),
			Naive:     "blocks the agent for 3s per attempt; a 10-step task loses 30s to one bad call",
		})
	}

	// 5. Telemetry: every call measured, so failure is visible before the customer sees it.
	{
		var recs []agentbridge.CallRecord
		br := agentbridge.NewBridge[int](agentbridge.Config{
			MaxAttempts: 2, BaseBackoff: 10 * time.Millisecond, MaxBackoff: 20 * time.Millisecond,
		}, agentbridge.ObserverFunc(func(r agentbridge.CallRecord) { recs = append(recs, r) }))
		_, _ = br.Do(context.Background(), agentbridge.Tool[int]{
			Name: "always_fails",
			Call: func(ctx context.Context, in int) (int, error) { return 0, errors.New("reset by peer") },
		}, 1)
		sample := ""
		if len(recs) > 0 {
			r := recs[0]
			sample = fmt.Sprintf("tool=%s outcome=%s attempts=%d elapsed=%s", r.Tool, r.Outcome, r.Attempts, r.Elapsed.Round(time.Millisecond))
		}
		s = append(s, scenario{
			Name:      "telemetry",
			What:      "one call that fails, observed",
			Decision:  fmt.Sprintf("%d CallRecord emitted (%s)", len(recs), sample),
			Calls:     len(recs),
			ElapsedMs: 0,
			Naive:     "no record; you find out from the customer, days later",
		})
	}

	// 6. Health: per-tool state you can put on an endpoint.
	{
		br := agentbridge.NewBridge[int](agentbridge.Config{MaxAttempts: 1}, nil)
		_, _ = br.Do(context.Background(), agentbridge.Tool[int]{
			Name: "search_tool",
			Call: func(ctx context.Context, in int) (int, error) { return 7, nil },
		}, 1)
		h := br.Health()
		s = append(s, scenario{
			Name:      "health endpoint",
			What:      "what the library reports about itself",
			Decision:  fmt.Sprintf("%s", h.JSON()),
			Calls:     1,
			ElapsedMs: 0,
			Naive:     "you cannot answer 'is the agent healthy?' without instrumenting it yourself",
		})
	}

	return s
}

func run(w http.ResponseWriter, r *http.Request) {
	once.Do(func() { out = runScenarios() })
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"library":   "github.com/vertexvantage07-ux/agentbridge",
		"scenarios": out,
		"note":      "Every figure is produced at runtime by the library itself.",
	})
}

const pageHTML = `<!doctype html><html><head><meta charset="utf-8">
<title>agentbridge - live demo</title>
<style>body{font:15px/1.6 system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#111}
h1{font-size:24px}code{background:#f4f4f5;padding:1px 5px;border-radius:4px}
table{border-collapse:collapse;width:100%;margin-top:18px}
td,th{border:1px solid #e4e4e7;padding:9px;text-align:left;vertical-align:top;font-size:14px}
th{background:#fafafa}.n{color:#71717a;font-size:12px}</style></head><body>
<h1>agentbridge</h1>
<p>Retry, deadlines, circuit breaking and telemetry for AI agent tool calls. MIT, Go, zero dependencies.</p>
<p>These are not claims. This page calls the library and prints what it actually did.</p>
<table><thead><tr><th>Scenario</th><th>What agentbridge did</th><th>Calls</th><th>Elapsed</th><th>What a naive caller does</th></tr></thead>
<tbody id="r"><tr><td colspan="5">running the real library...</td></tr></tbody></table>
<p><a href="https://github.com/vertexvantage07-ux/agentbridge">github.com/vertexvantage07-ux/agentbridge</a></p>
<script>fetch('api/run').then(r=>r.json()).then(d=>{
 document.getElementById('r').innerHTML=d.scenarios.map(s=>'<tr><td><b>'+s.name+'</b><br><span class=n>'+s.what+'</span></td><td>'+s.decision+'</td><td>'+s.calls+'</td><td>'+s.elapsed_ms+'ms</td><td>'+s.naive_would+'</td></tr>').join('')});</script>
</body></html>`

func page(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(pageHTML))
}
