// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Lukasz Olszewski

/**
 * Integration tests for the wiring in `index.ts`.
 *
 * `detect.test.ts` covers the predicates in isolation; these drive the actual
 * extension through a fake pi host to prove the events are hooked to the right
 * decisions — that a stall on `agent_settled` really does trigger a turn, that
 * the guards really do suppress it, and that the retry cap really does release
 * after a healthy turn.
 */
import { beforeEach, describe, expect, it } from "vitest"
import extension from "./index.ts"

type Handler = (event: any, ctx: any) => Promise<any>

interface SentMessage {
	customType: string
	content: string
	display: boolean
}

class FakePi {
	handlers = new Map<string, Handler[]>()
	commands = new Map<string, { description?: string; handler: Handler }>()
	sent: SentMessage[] = []
	entries: { customType: string; data: any }[] = []
	triggered = 0

	on(event: string, handler: Handler) {
		const list = this.handlers.get(event) ?? []
		list.push(handler)
		this.handlers.set(event, list)
	}

	sendMessage(message: SentMessage, options?: { triggerTurn?: boolean }) {
		this.sent.push(message)
		if (options?.triggerTurn) this.triggered++
	}

	appendEntry(customType: string, data?: any) {
		this.entries.push({ customType, data })
	}

	registerCommand(name: string, options: { description?: string; handler: Handler }) {
		this.commands.set(name, options)
	}

	async emit(event: string, payload: any, ctx: any) {
		const results: any[] = []
		for (const h of this.handlers.get(event) ?? []) results.push(await h(payload, ctx))
		return results
	}
}

function makeCtx(over: Partial<Record<string, any>> = {}) {
	return {
		mode: over.mode ?? "tui",
		isIdle: () => over.isIdle ?? true,
		hasPendingMessages: () => over.hasPendingMessages ?? false,
		signal: over.signal,
		getContextUsage: () => over.contextUsage ?? { tokens: 100_000, contextWindow: 262_144 },
		ui: { notify: (..._a: unknown[]) => {} },
	}
}

const STALL = {
	role: "assistant",
	stopReason: "stop",
	content: [{ type: "thinking", thinking: "Let me run the targeted test file first." }],
	usage: { output: 93, reasoning: 90 },
}

const HEALTHY = {
	role: "assistant",
	stopReason: "toolUse",
	content: [{ type: "toolCall" }],
	usage: { output: 120, reasoning: 80 },
}

/** Env override keeps these tests independent of any real user config file. */
const ENV = { PI_STALL_RECOVERY: "1", PI_STALL_RECOVERY_MAX_ATTEMPTS: "2" }

let pi: FakePi

beforeEach(() => {
	for (const [k, v] of Object.entries(ENV)) process.env[k] = v
	pi = new FakePi()
	extension(pi as any)
})

async function settleWith(message: unknown, ctx = makeCtx()) {
	await pi.emit("message_end", { message }, ctx)
	await pi.emit("agent_settled", {}, ctx)
}

describe("wiring", () => {
	it("registers the events and command it needs", () => {
		expect([...pi.handlers.keys()].sort()).toEqual(
			["agent_settled", "context", "message_end", "session_start"].sort(),
		)
		expect(pi.commands.has("stall-recovery")).toBe(true)
	})
})

describe("recovery on a stall", () => {
	it("triggers exactly one turn with a resume prompt", async () => {
		await settleWith(STALL)
		expect(pi.triggered).toBe(1)
		expect(pi.sent).toHaveLength(1)
		expect(pi.sent[0]!.customType).toBe("stall-recovery/continue")
		expect(pi.sent[0]!.content.length).toBeGreaterThan(20)
	})

	it("records an audit entry flagging the reasoning-only fingerprint", async () => {
		await settleWith(STALL)
		const audit = pi.entries.filter((e) => e.customType === "stall-recovery/event")
		expect(audit).toHaveLength(1)
		expect(audit[0]!.data).toMatchObject({
			action: "recovered",
			kind: "empty-turn",
			reasoningOnly: true,
		})
	})

	it("does nothing on a healthy turn", async () => {
		await settleWith(HEALTHY)
		expect(pi.triggered).toBe(0)
		expect(pi.entries).toHaveLength(0)
	})

	it("does not re-fire when settled twice on one stall", async () => {
		const ctx = makeCtx()
		await pi.emit("message_end", { message: STALL }, ctx)
		await pi.emit("agent_settled", {}, ctx)
		await pi.emit("agent_settled", {}, ctx)
		expect(pi.triggered).toBe(1)
	})
})

describe("guards suppress the retry", () => {
	it.each([
		["user aborted", { signal: { aborted: true } }],
		["queued user message", { hasPendingMessages: true }],
		["another run active", { isIdle: false }],
		["context at the ceiling", { contextUsage: { tokens: 260_000, contextWindow: 262_144 } }],
		["running in print mode", { mode: "print" }],
	])("declines when %s", async (_name, over) => {
		await settleWith(STALL, makeCtx(over))
		expect(pi.triggered).toBe(0)
		expect(pi.entries[0]!.data).toMatchObject({ action: "declined" })
		expect(pi.entries[0]!.data.reason).toBeTruthy()
	})

	it("proceeds when pi cannot report context usage", async () => {
		await settleWith(STALL, makeCtx({ contextUsage: undefined }))
		expect(pi.triggered).toBe(1)
	})
})

describe("consecutive retry cap", () => {
	it("stops after maxAttempts back-to-back stalls", async () => {
		await settleWith(STALL)
		await settleWith(STALL)
		expect(pi.triggered).toBe(2)
		await settleWith(STALL)
		expect(pi.triggered).toBe(2)
		expect(pi.entries.at(-1)!.data).toMatchObject({ action: "declined" })
		expect(pi.entries.at(-1)!.data.reason).toMatch(/cap/)
	})

	it("releases the cap once a turn produces real output", async () => {
		await settleWith(STALL)
		await settleWith(STALL)
		expect(pi.triggered).toBe(2)
		await settleWith(HEALTHY) // resets the consecutive counter
		await settleWith(STALL)
		expect(pi.triggered).toBe(3)
	})

	it("honours PI_STALL_RECOVERY_MAX_ATTEMPTS=0 as detect-only", async () => {
		process.env.PI_STALL_RECOVERY_MAX_ATTEMPTS = "0"
		const p = new FakePi()
		extension(p as any)
		await p.emit("message_end", { message: STALL }, makeCtx())
		await p.emit("agent_settled", {}, makeCtx())
		expect(p.triggered).toBe(0)
		expect(p.entries[0]!.data).toMatchObject({ action: "declined" })
	})
})

describe("kill switch", () => {
	it("does nothing at all when disabled", async () => {
		process.env.PI_STALL_RECOVERY = "0"
		const p = new FakePi()
		extension(p as any)
		await p.emit("message_end", { message: STALL }, makeCtx())
		await p.emit("agent_settled", {}, makeCtx())
		expect(p.triggered).toBe(0)
		expect(p.entries).toHaveLength(0)
	})
})

describe("context hygiene", () => {
	const nudge = { role: "custom", customType: "stall-recovery/continue", content: "resume" }
	const user = { role: "user", content: "do the thing" }

	it("leaves context untouched when there is at most one resume prompt", async () => {
		const [result] = await pi.emit("context", { messages: [user, nudge] }, makeCtx())
		expect(result).toBeUndefined()
	})

	it("drops older resume prompts and keeps the newest", async () => {
		const messages = [user, nudge, user, nudge]
		const [result] = await pi.emit("context", { messages }, makeCtx())
		expect(result.messages).toHaveLength(3)
		expect(result.messages.filter((m: any) => m.customType === "stall-recovery/continue")).toHaveLength(1)
		// the surviving nudge must be the last one, not the first
		expect(result.messages.at(-1)).toBe(nudge)
	})

	it("does not touch other extensions' custom entries", async () => {
		const other = { role: "custom", customType: "gate-ledger" }
		const messages = [other, nudge, other, nudge]
		const [result] = await pi.emit("context", { messages }, makeCtx())
		expect(result.messages.filter((m: any) => m.customType === "gate-ledger")).toHaveLength(2)
	})
})

describe("malformed tool-call recovery", () => {
	it("recovers the provider rejection pi's retry ignores", async () => {
		await settleWith({
			role: "assistant",
			stopReason: "error",
			errorMessage:
				'OpenrouterException - {"error":{"message":"tool_calls[1].function.name must be a non-empty string"}}',
			content: [],
		})
		expect(pi.triggered).toBe(1)
		expect(pi.entries[0]!.data).toMatchObject({ kind: "malformed-tool-call" })
	})

	it("leaves a rate-limit error to pi", async () => {
		await settleWith({
			role: "assistant",
			stopReason: "error",
			errorMessage: "litellm.RateLimitError: temporarily rate-limited upstream",
			content: [],
		})
		expect(pi.triggered).toBe(0)
	})
})