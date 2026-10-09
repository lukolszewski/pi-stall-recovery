/**
 * Pure detection logic, deliberately free of any pi imports so it can be unit
 * tested without a running agent. `index.ts` adapts pi's real types onto the
 * structural ones below.
 */

export type StallKind = "empty-turn" | "malformed-tool-call"

/** The subset of pi's AssistantMessage this module needs. */
export interface AssistantLike {
	role: string
	stopReason?: string
	errorMessage?: string
	content?: readonly { type: string; text?: string; thinking?: string }[]
	usage?: { output?: number; reasoning?: number | null }
}

/**
 * Default pattern for the second failure mode: a provider rejecting the model's
 * own malformed tool call. Seen on OpenRouter/Alibaba as
 * `tool_calls[1].function.name must be a non-empty string (got empty string)`.
 * pi's RETRYABLE_PROVIDER_ERROR_PATTERN does not match these, so without us
 * they are terminal.
 */
export const DEFAULT_MALFORMED_TOOL_CALL_PATTERN =
	/function\.name must be a non-empty string|tool_calls\[\d+\]\.function\.name/i

export interface ClassifyOptions {
	/** Set false to ignore provider-side malformed tool calls. */
	recoverMalformedToolCalls?: boolean
	malformedToolCallPattern?: RegExp
}

function joinText(m: AssistantLike): string {
	return (m.content ?? [])
		.filter((c) => c.type === "text")
		.map((c) => c.text ?? "")
		.join("")
		.trim()
}

function countToolCalls(m: AssistantLike): number {
	return (m.content ?? []).filter((c) => c.type === "toolCall").length
}

/**
 * Classify a finalized assistant message, or return null when it looks like a
 * legitimate end of turn.
 *
 * The `empty-turn` predicate is intentionally narrow: a clean `stop` carrying no
 * text and no tool call. pi has no reason to end a turn on such a message — the
 * model said nothing and did nothing — so there is no legitimate case to confuse
 * it with. Measured against a 1,329-turn session on Qwen3.8-Flash-Next it
 * matched 13 turns, every one of them a real stall (12 of which the user had to
 * nudge by hand), and none of the 1,247 healthy turns.
 *
 * Deliberately NOT part of the predicate:
 *  - the model id: other models and quants show the same shape.
 *  - "the previous message was a tool result": true for 12 of the 13, but one
 *    stalled directly after a user message, and gating on it would miss that.
 *  - `endTurn`: null on every message in the measured session, so it carries no
 *    signal on at least the OpenAI-completions path.
 */
export function classifyAssistantMessage(
	m: AssistantLike | undefined,
	opts: ClassifyOptions = {},
): StallKind | null {
	if (!m || m.role !== "assistant") return null

	if (m.stopReason === "stop" && joinText(m) === "" && countToolCalls(m) === 0) {
		return "empty-turn"
	}

	if (opts.recoverMalformedToolCalls !== false && m.stopReason === "error" && m.errorMessage) {
		const pattern = opts.malformedToolCallPattern ?? DEFAULT_MALFORMED_TOOL_CALL_PATTERN
		if (pattern.test(m.errorMessage)) return "malformed-tool-call"
	}

	return null
}

/**
 * True when the stall carries the reasoning-then-EOS fingerprint: the model
 * spent effectively all of its output budget inside the thinking block and
 * emitted nothing after it. Reported for telemetry only — never used to gate a
 * recovery, since a provider that does not report reasoning tokens would make
 * the check silently false.
 */
export function hasReasoningOnlyFingerprint(m: AssistantLike): boolean {
	const out = m.usage?.output ?? 0
	const reasoning = m.usage?.reasoning ?? 0
	if (out <= 0 || !reasoning) return false
	const hasThinking = (m.content ?? []).some(
		(c) => c.type === "thinking" && (c.thinking ?? "").trim().length > 0,
	)
	return hasThinking && reasoning / out >= 0.9
}

/**
 * Indices of every resume prompt except the newest one.
 *
 * The newest is kept so the retry that is about to run still sees the
 * instruction; the older ones are dropped so repeated recoveries do not pile
 * "continue the task" messages into history. Returns an empty array when there
 * is nothing to do, so the caller can skip rebuilding the message list.
 */
export function staleNudgeIndices(
	messages: readonly unknown[],
	isNudge: (m: unknown) => boolean,
): number[] {
	const found: number[] = []
	messages.forEach((m, i) => {
		if (isNudge(m)) found.push(i)
	})
	return found.length <= 1 ? [] : found.slice(0, -1)
}

export interface GuardFacts {
	/** pi is genuinely parked — nothing queued, nothing running. */
	isIdle: boolean
	/** The user already typed something; their message wins over our retry. */
	hasPendingMessages: boolean
	/** The user pressed stop. */
	aborted: boolean
	/** Consecutive recoveries already attempted without a healthy turn between. */
	consecutiveAttempts: number
	maxAttempts: number
	/** Fraction of the context window in use, when pi can report it. */
	contextPercent?: number | null
	/** Above this, treat a bare `stop` as genuine exhaustion and defer to compaction. */
	contextCeiling: number
}

export type GuardVerdict = { ok: true } | { ok: false; reason: string }

/**
 * Decide whether to actually re-trigger. Separated from classification so the
 * "is this a stall" and "may we act" questions can be tested independently.
 */
export function checkGuards(f: GuardFacts): GuardVerdict {
	if (f.aborted) return { ok: false, reason: "aborted by user" }
	if (f.hasPendingMessages) return { ok: false, reason: "user message already queued" }
	if (!f.isIdle) return { ok: false, reason: "another run is already active" }
	if (f.consecutiveAttempts >= f.maxAttempts) {
		return { ok: false, reason: `retry cap reached (${f.maxAttempts})` }
	}
	// A bare `stop` at a full context window is exhaustion, not this bug. pi
	// already treats over-window stops specially; re-triggering would just burn
	// turns against a wall instead of letting compaction run.
	if (f.contextPercent != null && f.contextPercent >= f.contextCeiling) {
		return {
			ok: false,
			reason: `context at ${Math.round(f.contextPercent * 100)}% — deferring to compaction`,
		}
	}
	return { ok: true }
}
