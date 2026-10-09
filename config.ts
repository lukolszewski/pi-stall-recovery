import { readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

export interface StallRecoveryConfig {
	enabled: boolean
	/** Consecutive recoveries allowed before giving the turn back to the user. */
	maxAttempts: number
	/** Context-window fraction above which a bare `stop` is treated as exhaustion. */
	contextCeiling: number
	/** Also recover provider rejections of malformed tool calls. */
	recoverMalformedToolCalls: boolean
	/** Text delivered to the model to resume the task. */
	nudge: string
	/** Keep only the newest recovery prompt in LLM context (see README). */
	stripOldNudgesFromContext: boolean
	/** Surface a TUI notification on each recovery. */
	notify: boolean
}

export const CONFIG_FILENAME = "stall-recovery.json"

export const DEFAULT_CONFIG: StallRecoveryConfig = {
	enabled: true,
	maxAttempts: 2,
	contextCeiling: 0.95,
	recoverMalformedToolCalls: true,
	nudge:
		"Your previous turn ended without any message or tool call, which was not intentional — " +
		"it was a generation fault, not a decision to stop. Resume the task from where you left off. " +
		"Do not apologise, do not summarise what happened, and do not ask whether to continue: " +
		"just take the next action. If the task is genuinely complete, say so in one line.",
	stripOldNudgesFromContext: true,
	notify: true,
}

function coerce(raw: unknown, base: StallRecoveryConfig): StallRecoveryConfig {
	if (!raw || typeof raw !== "object") return base
	const r = raw as Record<string, unknown>
	const num = (v: unknown, fallback: number) =>
		typeof v === "number" && Number.isFinite(v) ? v : fallback
	const bool = (v: unknown, fallback: boolean) => (typeof v === "boolean" ? v : fallback)
	const str = (v: unknown, fallback: string) =>
		typeof v === "string" && v.trim().length > 0 ? v : fallback
	return {
		enabled: bool(r.enabled, base.enabled),
		maxAttempts: Math.max(0, Math.floor(num(r.maxAttempts, base.maxAttempts))),
		contextCeiling: Math.min(1, Math.max(0, num(r.contextCeiling, base.contextCeiling))),
		recoverMalformedToolCalls: bool(r.recoverMalformedToolCalls, base.recoverMalformedToolCalls),
		nudge: str(r.nudge, base.nudge),
		stripOldNudgesFromContext: bool(
			r.stripOldNudgesFromContext,
			base.stripOldNudgesFromContext,
		),
		notify: bool(r.notify, base.notify),
	}
}

/** Exported for tests; `loadConfig` is the real entry point. */
export function parseConfig(json: string, base = DEFAULT_CONFIG): StallRecoveryConfig {
	try {
		return coerce(JSON.parse(json), base)
	} catch {
		return base
	}
}

function applyEnv(cfg: StallRecoveryConfig, env: NodeJS.ProcessEnv): StallRecoveryConfig {
	const out = { ...cfg }
	if (env.PI_STALL_RECOVERY === "0" || env.PI_STALL_RECOVERY === "off") out.enabled = false
	if (env.PI_STALL_RECOVERY === "1" || env.PI_STALL_RECOVERY === "on") out.enabled = true
	const max = Number(env.PI_STALL_RECOVERY_MAX_ATTEMPTS)
	if (Number.isFinite(max) && max >= 0) out.maxAttempts = Math.floor(max)
	return out
}

/**
 * Load `~/.pi/agent/stall-recovery.json` if present, then apply env overrides.
 * A missing or malformed file is not an error — the defaults are the intended
 * configuration for most users.
 */
export function loadConfig(opts: { home?: string; env?: NodeJS.ProcessEnv } = {}): StallRecoveryConfig {
	const home = opts.home ?? homedir()
	const env = opts.env ?? process.env
	let cfg = DEFAULT_CONFIG
	try {
		cfg = parseConfig(readFileSync(join(home, ".pi", "agent", CONFIG_FILENAME), "utf8"))
	} catch {
		// no config file; defaults stand
	}
	return applyEnv(cfg, env)
}
