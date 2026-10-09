# pi-stall-recovery

A [pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) extension that
recovers from **silent model stalls** — turns that end with no message and no tool call while the
task is still unfinished, leaving you to type "continue" or "why did you stop?" by hand.

It is not a workaround for one model. The condition is detected from the *shape of the response*,
so it applies to any model or backend that produces it.

## The failure it fixes

Some models end a turn by generating reasoning and then emitting EOS, without ever producing content
or a tool call. The provider reports a perfectly ordinary successful completion:

```json
{
  "content": "",
  "tool_calls": null,
  "reasoning_content": "… Continue: write commit message file, commit, ff-merge, live test.",
  "finish_reason": "stop"
}
```

HTTP 200, `finish_reason: "stop"`, nothing wrong at the transport level. pi therefore concludes the
agent is finished and hands control back to you — mid-task, often with the model's own reasoning
having just announced the exact tool call it was about to make.

**pi's built-in retry cannot catch this.** Its predicate requires an error:

```js
isRetryableAssistantError(message){ if(message.stopReason!=="error" || !message.errorMessage) return !1; … }
```

A stall is `stopReason: "stop"` with no error message, so no amount of `retry` configuration will
help. That gap is what this extension fills.

### Measurements

From one real 1,329-turn pi session on Qwen3.8-Flash-Next (Q4_K_XL via llama.cpp):

| | |
|---|---|
| stalls detected | **13** |
| of those, followed by a manual "continue"/"why did you stop" | **12** (the 13th by switching model) |
| false positives across the other 1,247 healthy turns | **0** |
| stalls below 100k context (438 turns) | **0** |
| stalls above 100k context (822 turns) | **13** (1.6%) |

In every one of the 13, output tokens were essentially all reasoning tokens (188/186, 1067/1034,
93/90 …) — the model reasoned and then stopped. A second, distinct mode appeared on a hosted route:
5 hard errors of `tool_calls[1].function.name must be a non-empty string`, which pi's retryable-error
pattern also does not match. Both are handled.

## Install

```bash
pi install git:github.com/lukolszewski/pi-stall-recovery
```

Or clone and point pi at it for a quick trial:

```bash
git clone https://github.com/lukolszewski/pi-stall-recovery
pi -e ./pi-stall-recovery/index.ts
```

For `/reload` support, place it under `~/.pi/agent/extensions/` or `.pi/extensions/`.

## How detection works

The extension listens on **`agent_settled`**, not `agent_end`. That distinction matters: after
`agent_end` pi may still auto-retry, auto-compact, or drain queued follow-up messages.
`agent_settled` fires only when pi will not continue on its own — exactly the moment a stall becomes
the user's problem.

The predicate on the last assistant message is deliberately narrow:

```
stopReason === "stop"
  && no text content (after trimming)
  && no toolCall content
```

There is no legitimate reason for pi to end a turn on an assistant message that says nothing and
does nothing, which is why this needs no heuristics and produced zero false positives in the
measured session.

Three things are deliberately **not** part of the predicate:

- **The model id.** Other models, quants and backends show the same shape.
- **"The previous message was a tool result."** True for 12 of the 13 observed stalls, but one
  stalled directly after a user message. Gating on it would miss that case.
- **`endTurn`.** The provider field that would in principle say whether the model meant to stop is
  `null` on every message in the measured session, so it carries no signal — at least on the
  OpenAI-completions path.

A separate `hasReasoningOnlyFingerprint()` check records whether the stall carried the
reasoning-then-EOS signature. It is **telemetry only** and never gates a recovery, because a
provider that does not report reasoning-token accounting would make it silently false.

## Guards

A detected stall is only acted on when all of these hold:

| Guard | Why |
|---|---|
| not running under `--print` | one-shot mode tears the session down after the turn (see below) |
| user has not aborted | you pressed stop; stay stopped |
| no queued user message | your message wins over an automatic retry |
| pi is idle | another extension may have started a run |
| under the consecutive-retry cap (default 2) | a genuinely wedged model must not loop forever |
| context below the ceiling (default 95%) | a bare `stop` at a full window is exhaustion, not this bug — let compaction handle it |

The cap counts *consecutive* stalls: any turn that produces real output resets it.

### `--print` mode is detect-only

In `pi --print` / `-p`, pi runs one prompt and replaces the session immediately. A turn triggered
from `agent_settled` therefore lands on a session that is already gone, and pi reports
`This extension ctx is stale after session replacement`. Verified against pi 0.84.4: the identical
trigger succeeds in `tui` mode and fails in `print`.

So under `--print` a stall is detected and logged but not retried — better a quiet unfinished run
than an extension error on top of it. Interactive (`tui`), `rpc` and `json` modes recover normally.
If you drive pi non-interactively and need recovery there, retry at the layer that invokes pi.

## How it resumes

It injects a custom message with `triggerTurn`:

```ts
pi.sendMessage({ customType: "stall-recovery/continue", content: cfg.nudge, display: true },
               { triggerTurn: true })
```

This reaches the model (pi's `convertToLlm` maps `role: "custom"` to `role: "user"`) but is stored as
a `custom` session entry, so it stays visible and auditable rather than being forged as something you
typed. `sendUserMessage` would have been indistinguishable from your own input; that is why it is not
used.

You do **not** need to remove the stalled assistant turn — pi's own serializer already drops
contentless, tool-call-less assistant messages from the LLM request
(`if(!(content?.length>0) && !assistantMsg.tool_calls) continue`), so the retry sees clean history.

Every recovery (and every declined detection, with its reason) is written to the session with
`appendEntry`, which is not sent to the LLM. `/stall-recovery` prints the counters.

### Context hygiene

Only the newest resume prompt is kept in LLM context; older ones are stripped via the `context`
event. Without this, repeated recoveries accumulate "continue the task" instructions that cost
tokens and — more importantly — become a pattern the model can imitate. That is the same
self-reinforcement dynamic reported for retained pathological turns in the Qwen3.8-Flash-Next
discussion linked below.

## Configuration

Optional, at `~/.pi/agent/stall-recovery.json`. Defaults are intended to be correct for most users:

```json
{
  "enabled": true,
  "maxAttempts": 2,
  "contextCeiling": 0.95,
  "recoverMalformedToolCalls": true,
  "stripOldNudgesFromContext": true,
  "notify": true,
  "nudge": "Your previous turn ended without any message or tool call …"
}
```

Env overrides, useful for a single run:

| Variable | Effect |
|---|---|
| `PI_STALL_RECOVERY=0` / `off` | disable |
| `PI_STALL_RECOVERY=1` / `on` | enable |
| `PI_STALL_RECOVERY_MAX_ATTEMPTS=n` | override the cap |

`maxAttempts: 0` detects and logs stalls without ever acting — useful for measuring your own rate
before switching the recovery on.

A malformed or missing config file is not an error; the defaults stand.

## What this does not claim

- **It treats a symptom.** The underlying stall is a model-side generation fault. This makes it
  non-fatal; it does not prevent it.
- **The cause is not settled.** In the measured session, stalls above 100k context rose from 0.81%
  to 8.24% across a config change that enabled reasoning replay (`preserve_thinking: true`), which
  inflated prompts by 43% (165k vs 94k tokens for the same turn). That correlation is suggestive but
  **observational** — and a 60-trial replay of one failing request across three prompt
  configurations reproduced nothing, so it neither confirms nor refutes a causal role for context
  length. The stall reproduced once in 61 replays of that request, versus 8.2% in situ, so the
  per-call rate should be treated as unknown.
- **Disabling reasoning replay is not the recommended fix.** On the Qwen3.8 template,
  `preserve_thinking: false` keeps thinking only for turns after the last user message — which at
  the first call of each turn is none of it. Measured on a real request it retained 0% of 285KB of
  reasoning. That is a switch, not a trim, and this extension exists so you do not need it.

## References

- [llama.cpp #30078](https://github.com/ggml-org/llama.cpp/issues/30078) — tool calls returned as
  content with `finish_reason: stop`. A *different* failure; it was ruled out in the measured session
  (zero `<tool_call>` markup ever appeared in content).
- [llama.cpp #27733](https://github.com/ggml-org/llama.cpp/issues/27733) — responses discarded over a
  trailing `<think>` fragment. One benign stray `</think>` leak was observed; it cost no turns.
- [Qwen3.8-Flash-Next discussion #50](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/discussions/50)
  — premature empty turns after tool results, and the context-length dependence.

## Development

```bash
npm install
npm test         # 63 tests: 43 unit + 20 integration
npm run typecheck
```

The live runtime was also checked by driving pi through a pty, which is how the `--print` limitation
above was found: a forced trigger runs a second turn in `tui` mode, the real extension stays silent
on a healthy turn, and `/stall-recovery` reports its counters.

Detection, guards and context hygiene live in `detect.ts` as pure functions with no pi imports, so
they are testable without a running agent (`detect.test.ts`, built from real captured payloads).
`index.ts` is only the wiring, and `index.test.ts` drives it through a fake pi host to check that the
events are connected to the right decisions — a stall on `agent_settled` triggers exactly one turn,
each guard suppresses it, and the retry cap releases after a healthy turn.

Note that the suite calls the real `loadConfig()`, so it reads
`~/.pi/agent/stall-recovery.json` if you have one. The assertions that depend on configuration are
pinned with the `PI_STALL_RECOVERY*` env overrides, but a local config that disables
`recoverMalformedToolCalls` would turn that one test red.

## License

GPL-3.0-or-later. Copyright (C) 2026 Lukasz Olszewski. See [LICENSE](LICENSE).

This is free software: you may redistribute and modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version. It comes with ABSOLUTELY NO WARRANTY.
