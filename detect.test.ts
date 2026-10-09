import { describe, expect, it } from "vitest"
import { DEFAULT_CONFIG, parseConfig } from "./config.ts"
import {
	checkGuards,
	classifyAssistantMessage,
	hasReasoningOnlyFingerprint,
	staleNudgeIndices,
	type AssistantLike,
	type GuardFacts,
} from "./detect.ts"

/**
 * Verbatim shape of a real stall: session
 * `--home-luk-dev-pi-permission-modes--`, entry 3205, cross-checked against the
 * provider payload (`content: ""`, `tool_calls: null`, `finish_reason: "stop"`,
 * 93 output tokens of which 90 reasoning).
 */
const REAL_STALL: AssistantLike = {
	role: "assistant",
	stopReason: "stop",
	content: [
		{
			type: "thinking",
			thinking:
				"Nothing suspicious in any of it — all tool results this session were my own repo source. " +
				"Continue: write commit message file, commit, ff-merge, live test.",
		},
	],
	usage: { output: 93, reasoning: 90 },
}

const HEALTHY_TEXT: AssistantLike = {
	role: "assistant",
	stopReason: "stop",
	content: [
		{ type: "thinking", thinking: "Task looks done, summarising." },
		{ type: "text", text: "Both items are shipped; suite is green at 852/42." },
	],
	usage: { output: 400, reasoning: 120 },
}

const HEALTHY_TOOL_CALL: AssistantLike = {
	role: "assistant",
	stopReason: "toolUse",
	content: [
		{ type: "thinking", thinking: "Need to read the file first." },
		{ type: "toolCall" },
	],
	usage: { output: 200, reasoning: 150 },
}

describe("classifyAssistantMessage — empty-turn", () => {
	it("flags a reasoning-only clean stop", () => {
		expect(classifyAssistantMessage(REAL_STALL)).toBe("empty-turn")
	})

	it("flags a stop with no content blocks at all", () => {
		expect(classifyAssistantMessage({ role: "assistant", stopReason: "stop", content: [] })).toBe(
			"empty-turn",
		)
	})

	it("treats whitespace-only text as empty", () => {
		expect(
			classifyAssistantMessage({
				role: "assistant",
				stopReason: "stop",
				content: [{ type: "text", text: "  \n\t " }],
			}),
		).toBe("empty-turn")
	})

	it("ignores a stop that produced real text", () => {
		expect(classifyAssistantMessage(HEALTHY_TEXT)).toBeNull()
	})

	it("ignores a turn that produced a tool call", () => {
		expect(classifyAssistantMessage(HEALTHY_TOOL_CALL)).toBeNull()
	})

	it("ignores an empty turn that still carries a tool call", () => {
		expect(
			classifyAssistantMessage({
				role: "assistant",
				stopReason: "stop",
				content: [{ type: "toolCall" }],
			}),
		).toBeNull()
	})
})

describe("classifyAssistantMessage — stop reasons we must not touch", () => {
	// `aborted` is the user pressing stop; `length` is budget exhaustion, which pi
	// handles with compaction; `pending`/`deferred` are mid-flight states.
	it.each(["aborted", "length", "pending", "deferred"])("ignores stopReason %s", (stopReason) => {
		expect(classifyAssistantMessage({ role: "assistant", stopReason, content: [] })).toBeNull()
	})

	it("ignores a non-assistant message", () => {
		expect(classifyAssistantMessage({ role: "toolResult", stopReason: "stop", content: [] })).toBeNull()
	})

	it("ignores undefined", () => {
		expect(classifyAssistantMessage(undefined)).toBeNull()
	})
})

describe("classifyAssistantMessage — malformed tool calls", () => {
	// Observed 5x on the OpenRouter/Alibaba route in the measured session; pi's
	// own retryable-error pattern does not match it, so it is otherwise terminal.
	const OPENROUTER_MALFORMED: AssistantLike = {
		role: "assistant",
		stopReason: "error",
		errorMessage:
			'litellm.BadRequestError: OpenrouterException - {"error":{"message":' +
			'"tool_calls[1].function.name must be a non-empty string (got empty string)","code":400}}',
		content: [],
	}

	it("flags the provider rejection by default", () => {
		expect(classifyAssistantMessage(OPENROUTER_MALFORMED)).toBe("malformed-tool-call")
	})

	it("can be switched off", () => {
		expect(
			classifyAssistantMessage(OPENROUTER_MALFORMED, { recoverMalformedToolCalls: false }),
		).toBeNull()
	})

	it("leaves unrelated provider errors to pi", () => {
		expect(
			classifyAssistantMessage({
				role: "assistant",
				stopReason: "error",
				errorMessage: "litellm.RateLimitError: RateLimitError - temporarily rate-limited upstream",
				content: [],
			}),
		).toBeNull()
	})

	it("ignores an error with no message", () => {
		expect(classifyAssistantMessage({ role: "assistant", stopReason: "error", content: [] })).toBeNull()
	})
})

describe("hasReasoningOnlyFingerprint", () => {
	it("recognises the real stall (90 of 93 tokens were reasoning)", () => {
		expect(hasReasoningOnlyFingerprint(REAL_STALL)).toBe(true)
	})

	it("rejects a turn that spent most tokens on content", () => {
		expect(hasReasoningOnlyFingerprint(HEALTHY_TEXT)).toBe(false)
	})

	it("rejects a turn with no thinking block", () => {
		expect(
			hasReasoningOnlyFingerprint({
				role: "assistant",
				stopReason: "stop",
				content: [],
				usage: { output: 10, reasoning: 10 },
			}),
		).toBe(false)
	})

	it("stays false when the provider reports no reasoning tokens", () => {
		// Must never gate a recovery: providers that omit reasoning accounting
		// would otherwise look healthy.
		expect(
			hasReasoningOnlyFingerprint({
				role: "assistant",
				stopReason: "stop",
				content: [{ type: "thinking", thinking: "..." }],
				usage: { output: 50, reasoning: null },
			}),
		).toBe(false)
	})
})

describe("checkGuards", () => {
	const base: GuardFacts = {
		mode: "tui",
		isIdle: true,
		hasPendingMessages: false,
		aborted: false,
		consecutiveAttempts: 0,
		maxAttempts: 2,
		contextPercent: 0.6,
		contextCeiling: 0.95,
	}

	it("allows a first recovery when parked and idle", () => {
		expect(checkGuards(base)).toEqual({ ok: true })
	})

	it("refuses in print mode, where the session is torn down after the turn", () => {
		const v = checkGuards({ ...base, mode: "print" })
		expect(v.ok).toBe(false)
		expect((v as { reason: string }).reason).toMatch(/print mode/)
	})

	it.each(["tui", "rpc", "json", undefined])("acts in %s mode", (mode) => {
		expect(checkGuards({ ...base, mode })).toEqual({ ok: true })
	})

	it("refuses when the user aborted", () => {
		expect(checkGuards({ ...base, aborted: true })).toMatchObject({ ok: false })
	})

	it("yields to a queued user message", () => {
		expect(checkGuards({ ...base, hasPendingMessages: true })).toMatchObject({ ok: false })
	})

	it("refuses when another run is active", () => {
		expect(checkGuards({ ...base, isIdle: false })).toMatchObject({ ok: false })
	})

	it("enforces the consecutive cap", () => {
		expect(checkGuards({ ...base, consecutiveAttempts: 2 })).toMatchObject({ ok: false })
		expect(checkGuards({ ...base, consecutiveAttempts: 1 })).toEqual({ ok: true })
	})

	it("defers to compaction at a full context window", () => {
		const v = checkGuards({ ...base, contextPercent: 0.97 })
		expect(v.ok).toBe(false)
		expect((v as { reason: string }).reason).toMatch(/compaction/)
	})

	it("proceeds when context usage is unknown", () => {
		expect(checkGuards({ ...base, contextPercent: null })).toEqual({ ok: true })
	})

	it("honours maxAttempts of 0 as 'detect but never act'", () => {
		expect(checkGuards({ ...base, maxAttempts: 0 })).toMatchObject({ ok: false })
	})
})

describe("staleNudgeIndices", () => {
	const nudge = { role: "custom", customType: "stall-recovery/continue" }
	const other = { role: "custom", customType: "gate-ledger" }
	const user = { role: "user" }
	const isNudge = (m: unknown) =>
		(m as { customType?: string })?.customType === "stall-recovery/continue"

	it("does nothing when there are no nudges", () => {
		expect(staleNudgeIndices([user, other, user], isNudge)).toEqual([])
	})

	it("keeps a lone nudge so the pending retry still sees it", () => {
		expect(staleNudgeIndices([user, nudge], isNudge)).toEqual([])
	})

	it("drops every nudge but the newest", () => {
		const msgs = [user, nudge, user, nudge, user, nudge]
		expect(staleNudgeIndices(msgs, isNudge)).toEqual([1, 3])
	})

	it("never selects the final nudge for removal", () => {
		const msgs = [nudge, nudge, nudge]
		const stale = staleNudgeIndices(msgs, isNudge)
		expect(stale).not.toContain(2)
		expect(stale).toEqual([0, 1])
	})

	it("leaves other extensions' custom entries alone", () => {
		expect(staleNudgeIndices([other, other, nudge, nudge], isNudge)).toEqual([2])
	})
})

describe("parseConfig", () => {
	it("falls back to defaults on malformed JSON", () => {
		expect(parseConfig("{nope")).toEqual(DEFAULT_CONFIG)
	})

	it("applies overrides and keeps the rest", () => {
		const cfg = parseConfig('{"maxAttempts":5,"notify":false}')
		expect(cfg.maxAttempts).toBe(5)
		expect(cfg.notify).toBe(false)
		expect(cfg.nudge).toBe(DEFAULT_CONFIG.nudge)
	})

	it("clamps the context ceiling into 0..1", () => {
		expect(parseConfig('{"contextCeiling":42}').contextCeiling).toBe(1)
		expect(parseConfig('{"contextCeiling":-1}').contextCeiling).toBe(0)
	})

	it("rejects a blank nudge rather than sending nothing", () => {
		expect(parseConfig('{"nudge":"   "}').nudge).toBe(DEFAULT_CONFIG.nudge)
	})

	it("ignores wrong-typed fields", () => {
		expect(parseConfig('{"maxAttempts":"lots","enabled":"yes"}')).toEqual(DEFAULT_CONFIG)
	})
})
