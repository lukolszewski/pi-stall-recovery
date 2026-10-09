// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Lukasz Olszewski

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"
import { loadConfig, type StallRecoveryConfig } from "./config.ts"
import {
	checkGuards,
	classifyAssistantMessage,
	hasReasoningOnlyFingerprint,
	staleNudgeIndices,
	type AssistantLike,
	type StallKind,
} from "./detect.ts"

/** Custom message carrying the resume prompt. Reaches the model as a user-role
 *  message (pi's `convertToLlm` maps `role: "custom"` to `role: "user"`) while
 *  staying a distinct, auditable session entry rather than a forged user turn. */
const NUDGE_TYPE = "stall-recovery/continue"
/** Audit entry. `appendEntry` is explicitly not sent to the LLM. */
const AUDIT_TYPE = "stall-recovery/event"

interface Stats {
	recovered: number
	declined: number
	byKind: Record<StallKind, number>
	lastReason?: string
}

function freshStats(): Stats {
	return { recovered: 0, declined: 0, byKind: { "empty-turn": 0, "malformed-tool-call": 0 } }
}

/**
 * pi's message types live in packages that are not direct dependencies of an
 * extension, so this module reads the few fields it needs structurally rather
 * than importing the `AgentMessage` union. `detect.ts` owns the shape.
 */
function asAssistantLike(m: unknown): AssistantLike | undefined {
	const x = m as AssistantLike | undefined
	return x && typeof x.role === "string" ? x : undefined
}

function isNudge(m: unknown): boolean {
	const x = m as { role?: string; customType?: string } | undefined
	return x?.role === "custom" && x?.customType === NUDGE_TYPE
}

/**
 * Fraction of the context window in use, or null when pi cannot say.
 *
 * Computed from `tokens / contextWindow` rather than read from `usage.percent`
 * so the scale is unambiguous here regardless of how pi expresses it.
 */
function contextFraction(usage: { tokens: number | null; contextWindow: number } | undefined) {
	if (!usage || usage.tokens == null || !(usage.contextWindow > 0)) return null
	return usage.tokens / usage.contextWindow
}

export default function (pi: ExtensionAPI) {
	let cfg: StallRecoveryConfig = loadConfig()
	let lastAssistant: AssistantLike | undefined
	let consecutive = 0
	let stats = freshStats()

	pi.on("session_start", async () => {
		cfg = loadConfig()
		lastAssistant = undefined
		consecutive = 0
		stats = freshStats()
	})

	pi.on("message_end", async (event) => {
		const m = asAssistantLike(event.message)
		if (!m || m.role !== "assistant") return
		lastAssistant = m
		// A turn that produced real output clears the consecutive-stall budget, so
		// the cap means "two stalls in a row", not "two stalls per session".
		if (classifyAssistantMessage(m, cfg) === null) consecutive = 0
	})

	/**
	 * `agent_settled` — not `agent_end` — is the correct trigger. pi may still
	 * auto-retry, auto-compact, or drain queued follow-ups after `agent_end`;
	 * `agent_settled` fires only once pi will not continue on its own, which is
	 * exactly the moment the stall becomes visible to the user.
	 */
	pi.on("agent_settled", async (_event, ctx) => {
		if (!cfg.enabled) return

		const kind = classifyAssistantMessage(lastAssistant, cfg)
		if (!kind || !lastAssistant) return

		const verdict = checkGuards({
			mode: ctx.mode,
			isIdle: ctx.isIdle(),
			hasPendingMessages: ctx.hasPendingMessages(),
			aborted: ctx.signal?.aborted === true,
			consecutiveAttempts: consecutive,
			maxAttempts: cfg.maxAttempts,
			contextPercent: contextFraction(ctx.getContextUsage()),
			contextCeiling: cfg.contextCeiling,
		})

		const fingerprint = kind === "empty-turn" && hasReasoningOnlyFingerprint(lastAssistant)

		if (!verdict.ok) {
			stats.declined++
			stats.lastReason = verdict.reason
			pi.appendEntry(AUDIT_TYPE, {
				action: "declined",
				kind,
				reason: verdict.reason,
				reasoningOnly: fingerprint,
				at: Date.now(),
			})
			if (cfg.notify) {
				ctx.ui.notify(
					`stall-recovery: stall detected but not retried — ${verdict.reason}`,
					"warning",
				)
			}
			return
		}

		consecutive++
		stats.recovered++
		stats.byKind[kind]++
		pi.appendEntry(AUDIT_TYPE, {
			action: "recovered",
			kind,
			attempt: consecutive,
			reasoningOnly: fingerprint,
			at: Date.now(),
		})
		if (cfg.notify) {
			ctx.ui.notify(
				`stall-recovery: ${kind} — resuming (attempt ${consecutive}/${cfg.maxAttempts})`,
				"warning",
			)
		}

		// Clear the stale message first: if this retry stalls too, `message_end`
		// will set it again, and we must not re-fire on the previous one.
		lastAssistant = undefined
		pi.sendMessage(
			{ customType: NUDGE_TYPE, content: cfg.nudge, display: true },
			{ triggerTurn: true },
		)
	})

	/**
	 * Keep only the newest resume prompt in LLM context.
	 *
	 * Without this, every recovery leaves a "continue the task" instruction in
	 * history. They accumulate, cost tokens, and — more importantly — turn into a
	 * pattern the model can imitate, which is the self-reinforcement dynamic that
	 * the Qwen3.8-Flash-Next report warns about for retained pathological turns.
	 * The newest one is kept so the retry itself still sees the instruction.
	 */
	pi.on("context", async (event) => {
		if (!cfg.stripOldNudgesFromContext) return
		const stale = staleNudgeIndices(event.messages, isNudge)
		if (stale.length === 0) return
		const drop = new Set(stale)
		return { messages: event.messages.filter((_m, i) => !drop.has(i)) }
	})

	pi.registerCommand("stall-recovery", {
		description: "Show pi-stall-recovery status and counters for this session",
		handler: async (_args, ctx) => {
			const frac = contextFraction(ctx.getContextUsage())
			const lines = [
				`enabled: ${cfg.enabled}`,
				`recovered this session: ${stats.recovered} ` +
					`(empty-turn ${stats.byKind["empty-turn"]}, ` +
					`malformed-tool-call ${stats.byKind["malformed-tool-call"]})`,
				`declined: ${stats.declined}${stats.lastReason ? ` — last: ${stats.lastReason}` : ""}`,
				`consecutive attempts pending reset: ${consecutive}/${cfg.maxAttempts}`,
				`context: ${frac == null ? "unknown" : `${Math.round(frac * 100)}%`} ` +
					`(ceiling ${Math.round(cfg.contextCeiling * 100)}%)`,
				`malformed-tool-call recovery: ${cfg.recoverMalformedToolCalls}`,
			]
			ctx.ui.notify(lines.join("\n"), "info")
		},
	})
}