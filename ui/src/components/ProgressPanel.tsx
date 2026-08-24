import { useEffect, useRef, useState } from 'react'
import type { SSEEvent, SimulateEvent, OptimizeLineupEvent, GppGenerateProgressEvent, GppFieldProgressEvent, GppScoreProgressEvent, GppRescoreFieldProgressEvent, GppRescoreScoreProgressEvent, GppSelectProgressEvent, GppOptimalProgressEvent, GppDetSelectProgressEvent, GppDetRiskStartEvent, GppRefineProgressEvent, SelfPlayPoolStartEvent, SelfPlayPoolDoneEvent, SelfPlayContestProgressEvent, SelfPlayPayoutFallbackEvent, TopnPoolStartEvent, TopnPoolProgressEvent, TopnPoolDoneEvent, TopnContestStartEvent, TopnPickProgressEvent, TopnContestDoneEvent, MrpStartEvent, MrpFrontierStartEvent, MrpFrontierProgressEvent, MrpFrontierDoneEvent, MrpBuildProgressEvent, MrpPickProgressEvent, MrpDoneEvent } from '../types'

interface Props {
  events: SSEEvent[]
  running: boolean
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)
  return `${m}m ${s}s`
}

function formatMsWhole(ms: number): string {
  if (ms < 60000) return `${Math.round(ms / 1000)}s`
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)
  return `${m}m ${s}s`
}

const STAGE_LABELS: Record<string, string> = {
  load_slate: 'Load slate',
  simulate: 'Simulate',
  ppd_applied: 'PPD applied',
  external_ppd_applied: 'PPD applied',
  external_proj_score_floor: 'Ceiling floor',
  external_owncap_cull: 'Ownership cap',
  external_load: 'External pool files',
  external_pool: 'External pool',
  external_pwin: 'P(win) scoring',
  external_pwin_field: 'P(win) opponent field',
  external_pwin_score: 'P(win) scoring',
  compute_target: 'Compute target',
  calibrate_beta: 'Calibrate beta',
  optimize_lineup: 'Optimize lineups',
  gpp_optimal_start: 'Optimal seeding',
  gpp_optimal_done: 'Optimal seeding',
  gpp_sim_optimal_start: 'Sim-optimal seeding',
  gpp_sim_optimal_done: 'Sim-optimal seeding',
  gpp_generate_start: 'Generate candidates',
  gpp_generate_done: 'Generate candidates',
  gpp_score_start: 'Score candidates',
  gpp_score_done: 'Score candidates',
  gpp_refine_start: 'Refine pool',
  gpp_refine_done: 'Refine pool',
  gpp_rescore_start: 'Fresh re-score',
  gpp_rescore_done: 'Fresh re-score',
  gpp_field_inject: 'Field lineups',
  gpp_holdout: 'Holdout evaluation',
  self_play_pool_start: 'Self-play: building pool',
  self_play_pool_done: 'Self-play: pool built',
  self_play_contest_progress: 'Self-play: contest filled',
  self_play_payout_fallback: 'Self-play: approximate payout table',
  topn_sims_autosize: 'Top-N coverage: sim count auto-sized',
  topn_pool_start: 'Top-N coverage: building field pool',
  topn_pool_done: 'Top-N coverage: field pool built',
  topn_pool_augmented: 'Top-N coverage: candidate pool augmented',
  topn_contest_start: 'Top-N coverage: contest started',
  topn_pick_progress: 'Top-N coverage: covering worlds',
  topn_contest_done: 'Top-N coverage: contest filled',
  // mrp_build_progress / mrp_pick_progress are deliberately absent: they are
  // skipped from the log (see the flooding note in the row builder), so a
  // label for them would be dead code.
  mrp_start: 'Marginal reward: allocating',
  mrp_frontier_start: 'Marginal reward: generating frontier',
  mrp_frontier_done: 'Marginal reward: frontier generated',
  mrp_done: 'Marginal reward: allocation complete',
  mrp_payout_fallback: 'Marginal reward: MISSING PAYOUT STRUCTURE',
  complete: 'Complete',
  stopped: 'Stopped',
  error: 'Error',
}

const CONFIG_STAGES = new Set(['simulate', 'compute_target'])
const GPP_PROGRESS_STAGES = new Set(['gpp_generate_progress', 'gpp_score_progress', 'gpp_rescore_score_progress', 'gpp_refine_progress', 'gpp_select_progress', 'gpp_optimal_progress', 'gpp_sim_optimal_progress', 'gpp_det_select_progress', 'gpp_det_risk_start'])

export function ProgressPanel({ events, running }: Props) {
  const [now, setNow] = useState(() => Date.now())
  const tickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastEventTimeRef = useRef<number | null>(null)

  // Clear tick timer when run stops
  useEffect(() => {
    if (!running && tickTimerRef.current) {
      clearTimeout(tickTimerRef.current)
      tickTimerRef.current = null
    }
  }, [running])

  // Cleanup on unmount
  useEffect(() => {
    return () => { if (tickTimerRef.current) clearTimeout(tickTimerRef.current) }
  }, [])

  // Update now on each live progress event; schedule 30s-boundary ticks until the next event
  useEffect(() => {
    const last = events[events.length - 1]
    const isLiveProgressEvent =
      last?.stage === 'optimize_lineup' ||
      last?.stage === 'gpp_optimal_start' ||
      last?.stage === 'gpp_optimal_progress' ||
      last?.stage === 'gpp_optimal_done' ||
      last?.stage === 'gpp_sim_optimal_start' ||
      last?.stage === 'gpp_sim_optimal_progress' ||
      last?.stage === 'gpp_sim_optimal_done' ||
      last?.stage === 'gpp_generate_start' ||
      last?.stage === 'gpp_generate_progress' ||
      last?.stage === 'gpp_generate_done' ||
      last?.stage === 'gpp_score_start' ||
      last?.stage === 'gpp_field_progress' ||
      last?.stage === 'gpp_score_progress' ||
      last?.stage === 'gpp_score_done' ||
      last?.stage === 'gpp_refine_start' ||
      last?.stage === 'gpp_refine_progress' ||
      last?.stage === 'gpp_refine_done' ||
      last?.stage === 'gpp_rescore_start' ||
      last?.stage === 'gpp_rescore_field_progress' ||
      last?.stage === 'gpp_rescore_score_progress' ||
      last?.stage === 'gpp_rescore_done' ||
      last?.stage === 'gpp_field_inject' ||
      last?.stage === 'gpp_select_progress' ||
      last?.stage === 'gpp_mv_select_progress' ||
      last?.stage === 'gpp_hybrid_select_progress' ||
      last?.stage === 'gpp_det_select_progress' ||
      last?.stage === 'gpp_det_risk_start' ||
      last?.stage === 'self_play_pool_start' ||
      last?.stage === 'self_play_pool_done' ||
      last?.stage === 'self_play_contest_progress' ||
      last?.stage === 'self_play_payout_fallback' ||
      last?.stage === 'topn_sims_autosize' ||
      last?.stage === 'topn_pool_start' ||
      last?.stage === 'topn_pool_progress' ||
      last?.stage === 'topn_pool_done' ||
      last?.stage === 'topn_pool_augmented' ||
      last?.stage === 'topn_contest_start' ||
      last?.stage === 'topn_pick_progress' ||
      last?.stage === 'topn_contest_done' ||
      last?.stage === 'mrp_start' ||
      last?.stage === 'mrp_frontier_start' ||
      last?.stage === 'mrp_frontier_progress' ||
      last?.stage === 'mrp_frontier_done' ||
      last?.stage === 'mrp_build_progress' ||
      last?.stage === 'mrp_pick_progress' ||
      last?.stage === 'mrp_done' ||
      (typeof last?.stage === 'string' && last.stage.startsWith('external_'))
    if (!isLiveProgressEvent) return

    const ts = Date.now()
    lastEventTimeRef.current = ts
    setNow(ts)

    if (tickTimerRef.current) clearTimeout(tickTimerRef.current)

    function scheduleTick() {
      const eventTime = lastEventTimeRef.current!
      const sinceEvent = Date.now() - eventTime
      const nextTick = Math.ceil((sinceEvent + 1) / 30000) * 30000
      tickTimerRef.current = setTimeout(() => {
        setNow(Date.now())
        if (lastEventTimeRef.current === eventTime) scheduleTick()
      }, nextTick - sinceEvent)
    }

    scheduleTick()
  }, [events])

  if (events.length === 0 && !running) return null

  const first = events[0]
  const last = events[events.length - 1]
  const elapsed = first && last ? last.timestamp - first.timestamp : null

  const isGpp = events.some(e =>
    e.stage === 'gpp_optimal_start' || e.stage === 'gpp_optimal_done' ||
    e.stage === 'gpp_sim_optimal_start' || e.stage === 'gpp_sim_optimal_done' ||
    e.stage === 'gpp_generate_start' || e.stage === 'gpp_generate_done' ||
    e.stage === 'gpp_score_start' || e.stage === 'gpp_field_inject' ||
    e.stage === 'gpp_rescore_start' || e.stage === 'gpp_rescore_done' ||
    e.stage === 'gpp_select_progress' || e.stage === 'gpp_mv_select_progress' ||
    e.stage === 'gpp_hybrid_select_progress' || e.stage === 'gpp_det_select_progress' ||
    e.stage === 'gpp_det_risk_start'
  )

  // --- Optimal lineup seeding progress ---
  const optimalStartEvent = events.find(e => e.stage === 'gpp_optimal_start') as unknown as { n_optimal: number } | undefined
  const optimalProgressEvents = events.filter(e => e.stage === 'gpp_optimal_progress') as unknown as GppOptimalProgressEvent[]
  const latestOptimalProgress = optimalProgressEvents[optimalProgressEvents.length - 1]
  const optimalDone = events.some(e => e.stage === 'gpp_optimal_done')

  // --- Sim-optimal lineup seeding progress ---
  const simOptimalStartEvent = events.find(e => e.stage === 'gpp_sim_optimal_start') as unknown as { n_sim_optimals: number } | undefined
  const simOptimalProgressEvents = events.filter(e => e.stage === 'gpp_sim_optimal_progress') as unknown as GppOptimalProgressEvent[]
  const latestSimOptimalProgress = simOptimalProgressEvents[simOptimalProgressEvents.length - 1]
  const simOptimalDone = events.some(e => e.stage === 'gpp_sim_optimal_done')

  // --- Non-GPP lineup progress ---
  const latestLineup = [...events]
    .reverse()
    .find(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent | undefined

  const total = latestLineup?.total ?? 0
  const current = latestLineup?.lineup_index ?? 0
  const pct = total > 0 ? Math.round((current / total) * 100) : 0

  const lineupEvents = events.filter(e => e.stage === 'optimize_lineup') as OptimizeLineupEvent[]

  // --- GPP progress ---
  const generateStartEvent = events.find(e => e.stage === 'gpp_generate_start') as unknown as { n_candidates: number; n_from_optimal?: number; n_from_sim_optimal?: number } | undefined
  const generateProgressEvents = events.filter(e => e.stage === 'gpp_generate_progress') as unknown as GppGenerateProgressEvent[]
  const latestGenerateProgress = generateProgressEvents[generateProgressEvents.length - 1]
  const generateDone = events.some(e => e.stage === 'gpp_generate_done')
  const scoreStartEvent = events.find(e => e.stage === 'gpp_score_start') as unknown as { n_field_lineups: number; n_field_samples: number } | undefined
  const fieldProgressEvents = events.filter(e => e.stage === 'gpp_field_progress') as unknown as GppFieldProgressEvent[]
  const latestFieldProgress = fieldProgressEvents[fieldProgressEvents.length - 1]
  const scoreProgressEvents = events.filter(e => e.stage === 'gpp_score_progress') as unknown as GppScoreProgressEvent[]
  const latestScoreProgress = scoreProgressEvents[scoreProgressEvents.length - 1]
  const selectProgressEvents = events.filter(e => e.stage === 'gpp_select_progress') as unknown as GppSelectProgressEvent[]
  const detProgressEvents = events.filter(e => e.stage === 'gpp_det_select_progress') as unknown as GppDetSelectProgressEvent[]
  const detRiskStartEvents = events.filter(e => e.stage === 'gpp_det_risk_start') as unknown as GppDetRiskStartEvent[]
  const scoreDone = events.some(e => e.stage === 'gpp_score_done')
  const fieldInjectEvent = events.find(e => e.stage === 'gpp_field_inject') as unknown as { n_field: number; n_k: number } | undefined
  const refineStartEvent = events.find(e => e.stage === 'gpp_refine_start') as unknown as { rounds: number; top: number; mutants_per_parent: number } | undefined
  const refineProgressEvents = events.filter(e => e.stage === 'gpp_refine_progress') as unknown as GppRefineProgressEvent[]
  const latestRefineProgress = refineProgressEvents[refineProgressEvents.length - 1]
  const refineDone = events.some(e => e.stage === 'gpp_refine_done')

  // --- Fresh re-score progress (Phase 2c) — distinct event names from the
  // first-stage scoring above so this phase gets its own live readout
  // instead of showing a stale "Scoring batch"/"Generating field" label.
  const rescoreStartEvent = events.find(e => e.stage === 'gpp_rescore_start') as unknown as { n_candidates: number; n_field_samples: number } | undefined
  const rescoreFieldProgressEvents = events.filter(e => e.stage === 'gpp_rescore_field_progress') as unknown as GppRescoreFieldProgressEvent[]
  const latestRescoreFieldProgress = rescoreFieldProgressEvents[rescoreFieldProgressEvents.length - 1]
  const rescoreScoreProgressEvents = events.filter(e => e.stage === 'gpp_rescore_score_progress') as unknown as GppRescoreScoreProgressEvent[]
  const latestRescoreScoreProgress = rescoreScoreProgressEvents[rescoreScoreProgressEvents.length - 1]
  const rescoreDone = events.some(e => e.stage === 'gpp_rescore_done')

  // Det-EV hoisted variables (used in both gppLabel, detSegments, and ETA)
  const isDetEv = detRiskStartEvents.length > 0 || detProgressEvents.length > 0
  const latestDetStep = detProgressEvents[detProgressEvents.length - 1]
  const latestDetRiskStart = detRiskStartEvents[detRiskStartEvents.length - 1]
  const totalRisksCount = latestDetRiskStart?.total_risks ?? latestDetStep?.total_risks ?? 5
  const currentActiveRiskIdx = latestDetStep?.risk_index ?? latestDetRiskStart?.risk_index ?? 0
  const portfolioSizeDet = latestDetStep?.portfolio_size ?? 0
  const currentDetStepNum = latestDetStep?.step ?? 0
  const detSegments = isDetEv ? Array.from({ length: totalRisksCount }, (_, i) => {
    const riskIdx = i + 1
    const isComplete = riskIdx < currentActiveRiskIdx
    const isActive = riskIdx === currentActiveRiskIdx
    const pct = isComplete ? 100 : isActive && portfolioSizeDet > 0
      ? Math.round((currentDetStepNum / portfolioSizeDet) * 100)
      : 0
    return { pct, isComplete, isActive }
  }) : []

  // --- Self-play: single sequential best-response allocation, no risk
  // sweep (never emits gpp_* stages, so it's a sibling to isGpp/isDetEv
  // rather than nested inside them) ---
  const selfPlayPoolStartEvent = events.find(e => e.stage === 'self_play_pool_start') as unknown as SelfPlayPoolStartEvent | undefined
  const selfPlayPoolDoneEvent = events.find(e => e.stage === 'self_play_pool_done') as unknown as SelfPlayPoolDoneEvent | undefined
  const selfPlayContestEvents = events.filter(e => e.stage === 'self_play_contest_progress') as unknown as SelfPlayContestProgressEvent[]
  const latestSelfPlayContest = selfPlayContestEvents[selfPlayContestEvents.length - 1]
  const selfPlayFallbackEvents = events.filter(e => e.stage === 'self_play_payout_fallback') as unknown as SelfPlayPayoutFallbackEvent[]
  const isSelfPlay = selfPlayPoolStartEvent !== undefined || selfPlayContestEvents.length > 0

  let selfPlayPct = 0
  let selfPlayLabel = ''
  if (isSelfPlay) {
    if (latestSelfPlayContest) {
      selfPlayPct = Math.round((latestSelfPlayContest.contests_done / latestSelfPlayContest.contests_total) * 100)
      selfPlayLabel = `Contest ${latestSelfPlayContest.contests_done} / ${latestSelfPlayContest.contests_total}: ${latestSelfPlayContest.contest_id} filled (${latestSelfPlayContest.n_rounds} rounds, ${latestSelfPlayContest.n_swaps} refinement swap${latestSelfPlayContest.n_swaps === 1 ? '' : 's'})`
    } else if (selfPlayPoolDoneEvent) {
      selfPlayPct = 0
      selfPlayLabel = `Pool built (${selfPlayPoolDoneEvent.pool_size.toLocaleString()} lineups) — filling contests…`
    } else if (selfPlayPoolStartEvent) {
      selfPlayPct = 0
      selfPlayLabel = `Building opponent pool for ${selfPlayPoolStartEvent.n_contests} contest${selfPlayPoolStartEvent.n_contests === 1 ? '' : 's'}…`
    }
  }

  // --- Top-N coverage: single greedy set-cover allocation, no risk sweep
  // (sibling to isSelfPlay -- never emits gpp_* stages). ETA follows the
  // same "rate from a recent window of progress events" pattern as every
  // other stage: per-pick rate within the current contest for the
  // in-contest estimate, per-contest rate across completed contests for
  // the overall slate estimate. ---
  const topnPoolStartEvent = events.find(e => e.stage === 'topn_pool_start') as unknown as TopnPoolStartEvent | undefined
  const topnPoolProgressEvents = events.filter(e => e.stage === 'topn_pool_progress') as unknown as TopnPoolProgressEvent[]
  const latestTopnPoolProgress = topnPoolProgressEvents[topnPoolProgressEvents.length - 1]
  const topnPoolDoneEvent = events.find(e => e.stage === 'topn_pool_done') as unknown as TopnPoolDoneEvent | undefined
  const topnContestStartEvents = events.filter(e => e.stage === 'topn_contest_start') as unknown as TopnContestStartEvent[]
  const topnPickEvents = events.filter(e => e.stage === 'topn_pick_progress') as unknown as TopnPickProgressEvent[]
  const topnContestDoneEvents = events.filter(e => e.stage === 'topn_contest_done') as unknown as TopnContestDoneEvent[]
  const latestTopnContestStart = topnContestStartEvents[topnContestStartEvents.length - 1]
  const latestTopnPick = topnPickEvents[topnPickEvents.length - 1]
  const latestTopnContestDone = topnContestDoneEvents[topnContestDoneEvents.length - 1]
  const isTopnCoverage = topnPoolStartEvent !== undefined || topnContestStartEvents.length > 0

  // --- Marginal reward (MRP): three sequential phases, one bar ------------
  // Frontier generation -> per-contest state build -> entry picks. The bar
  // tracks the CURRENT phase rather than a weighted whole, because the phase
  // costs are wildly unequal and slate-dependent (frontier ~2.5 min, picks
  // ~2 min, build seconds) so a blended percentage would be a worse lie than
  // an honest per-phase one. Same convention as the topn bar above.
  const mrpStartEvent = events.find(e => e.stage === 'mrp_start') as unknown as MrpStartEvent | undefined
  const mrpFrontierStartEvent = events.find(e => e.stage === 'mrp_frontier_start') as unknown as MrpFrontierStartEvent | undefined
  const mrpFrontierProgressEvents = events.filter(e => e.stage === 'mrp_frontier_progress') as unknown as MrpFrontierProgressEvent[]
  const latestMrpFrontier = mrpFrontierProgressEvents[mrpFrontierProgressEvents.length - 1]
  const mrpFrontierDoneEvent = events.find(e => e.stage === 'mrp_frontier_done') as unknown as MrpFrontierDoneEvent | undefined
  const mrpBuildEvents = events.filter(e => e.stage === 'mrp_build_progress') as unknown as MrpBuildProgressEvent[]
  const latestMrpBuild = mrpBuildEvents[mrpBuildEvents.length - 1]
  const mrpPickEvents = events.filter(e => e.stage === 'mrp_pick_progress') as unknown as MrpPickProgressEvent[]
  const latestMrpPick = mrpPickEvents[mrpPickEvents.length - 1]
  const mrpDoneEvent = events.find(e => e.stage === 'mrp_done') as unknown as MrpDoneEvent | undefined
  const isMrp = mrpStartEvent !== undefined

  let mrpPct = 0
  let mrpLabel = ''
  let mrpEtaMs: number | null = null
  if (isMrp) {
    // Rate over the last few events of the CURRENT phase. Deliberately recent
    // rather than cumulative: frontier solve cost climbs steeply with lambda,
    // so an average over the whole phase reads far too optimistic near the end.
    const rate = (evs: Array<{ timestamp: number; done: number }>, remaining: number) => {
      if (evs.length < 2 || remaining <= 0) return null
      const recent = evs.slice(-4)
      const dt = recent[recent.length - 1].timestamp - recent[0].timestamp
      const dDone = recent[recent.length - 1].done - recent[0].done
      return dDone > 0 && dt > 0 ? (dt / dDone) * remaining : null
    }

    if (latestMrpPick && latestMrpPick.done < latestMrpPick.total) {
      mrpPct = latestMrpPick.total > 0 ? Math.round((latestMrpPick.done / latestMrpPick.total) * 100) : 0
      mrpLabel = `Filling entries: ${latestMrpPick.done.toLocaleString()} / ${latestMrpPick.total.toLocaleString()}`
      mrpEtaMs = rate(mrpPickEvents, latestMrpPick.total - latestMrpPick.done)
    } else if (latestMrpPick) {
      mrpPct = 100
      mrpLabel = `Filling entries: ${latestMrpPick.done.toLocaleString()} / ${latestMrpPick.total.toLocaleString()}`
    } else if (latestMrpBuild) {
      mrpPct = latestMrpBuild.total > 0 ? Math.round((latestMrpBuild.done / latestMrpBuild.total) * 100) : 0
      mrpLabel = `Building contest states: ${latestMrpBuild.done} / ${latestMrpBuild.total}`
      mrpEtaMs = rate(mrpBuildEvents, latestMrpBuild.total - latestMrpBuild.done)
    } else if (mrpFrontierStartEvent && !mrpFrontierDoneEvent) {
      // Two phases with no shared denominator. Until line 4 reports back there
      // is no operating-point count to divide by -- and the SEARCH grid size is
      // emphatically not it (16 searched, ~5 generated at), so falling back to
      // it made the bar jump from "0 / 16" to "1 / 5" mid-run.
      if (!latestMrpFrontier) {
        mrpPct = 0
        mrpLabel = `Frontier: sampling ${mrpFrontierStartEvent.n_sample.toLocaleString()} candidates, `
          + `searching ${mrpFrontierStartEvent.n_lambda_search} λ for each contest's λ*…`
      } else {
        const { done, total } = latestMrpFrontier
        mrpPct = total > 0 ? Math.round((done / total) * 100) : 0
        mrpLabel = `Frontier: λ* ${done} / ${total}`
          + (done > 0 ? ` · ${latestMrpFrontier.n_lineups.toLocaleString()} lineups generated`
                      : ' · generating at the first operating point…')
        mrpEtaMs = rate(mrpFrontierProgressEvents, total - done)
      }
    } else {
      mrpLabel = 'Preparing allocation…'
    }
  }

  let topnPct = 0
  let topnLabel = ''
  let topnEtaMs: number | null = null
  if (isTopnCoverage) {
    const contestsDone = latestTopnContestDone?.contests_done ?? 0
    const contestsTotal = latestTopnContestDone?.contests_total ?? latestTopnContestStart?.contests_total ?? 0
    if (latestTopnPick && latestTopnContestStart && latestTopnPick.contest_id === latestTopnContestStart.contest_id) {
      const covered = latestTopnPick.uncovered_total - latestTopnPick.uncovered_remaining
      const coveredPct = latestTopnPick.uncovered_total > 0
        ? Math.round((covered / latestTopnPick.uncovered_total) * 100) : 0
      topnPct = contestsTotal > 0 ? Math.round((contestsDone / contestsTotal) * 100) : 0
      topnLabel = `Contest ${contestsDone + 1} / ${contestsTotal}: ${latestTopnPick.contest_id} — `
        + `pick ${latestTopnPick.pick_num} / ${latestTopnContestStart.k}, `
        + `${coveredPct}% of ${latestTopnPick.uncovered_total.toLocaleString()} simulated worlds covered`
        + (latestTopnPick.relaxations_so_far > 0 ? ` (${latestTopnPick.relaxations_so_far} relaxations)` : '')
      // Within-contest ETA: rate of picks over the last few pick events.
      const recentPicks = topnPickEvents.filter(e => e.contest_id === latestTopnPick.contest_id).slice(-5)
      if (recentPicks.length >= 2) {
        const dtMs = recentPicks[recentPicks.length - 1].timestamp - recentPicks[0].timestamp
        const dPicks = recentPicks[recentPicks.length - 1].pick_num - recentPicks[0].pick_num
        if (dPicks > 0 && dtMs > 0) {
          const msPerPick = dtMs / dPicks
          const picksRemainingHere = latestTopnContestStart.k - latestTopnPick.pick_num
          const contestsRemaining = Math.max(contestsTotal - contestsDone - 1, 0)
          // Cross-contest ETA: average completed-contest duration, when we have any.
          let msPerContest = msPerPick * latestTopnContestStart.k
          if (topnContestDoneEvents.length > 0) {
            msPerContest = topnContestDoneEvents.reduce((s, e) => s + e.elapsed_s * 1000, 0) / topnContestDoneEvents.length
          }
          topnEtaMs = msPerPick * picksRemainingHere + msPerContest * contestsRemaining
        }
      }
    } else if (latestTopnContestStart) {
      topnPct = contestsTotal > 0 ? Math.round((contestsDone / contestsTotal) * 100) : 0
      topnLabel = `Contest ${contestsDone + 1} / ${contestsTotal}: ${latestTopnContestStart.contest_id} `
        + `(field ${latestTopnContestStart.field_size_g.toLocaleString()}, top-${latestTopnContestStart.effective_rank}, `
        + `${latestTopnContestStart.n_sims_g.toLocaleString()} sim worlds)…`
    } else if (latestTopnPoolProgress) {
      // Single updating bar (not one row per chunk -- see buildDisplayEvents,
      // which skips topn_pool_progress from the raw event list entirely).
      topnPct = latestTopnPoolProgress.n_total > 0
        ? Math.round((latestTopnPoolProgress.n_done / latestTopnPoolProgress.n_total) * 100) : 0
      topnLabel = `Building opponent field pool: ${latestTopnPoolProgress.n_done.toLocaleString()} `
        + `/ ${latestTopnPoolProgress.n_total.toLocaleString()} lineups generated…`
      const recent = topnPoolProgressEvents.slice(-5)
      if (recent.length >= 2) {
        const dtMs = recent[recent.length - 1].timestamp - recent[0].timestamp
        const dDone = recent[recent.length - 1].n_done - recent[0].n_done
        if (dDone > 0 && dtMs > 0) {
          const remaining = latestTopnPoolProgress.n_total - latestTopnPoolProgress.n_done
          topnEtaMs = (dtMs / dDone) * remaining
        }
      }
    } else if (topnPoolDoneEvent) {
      topnPct = 0
      topnLabel = `Field pool built (${topnPoolDoneEvent.field_pool_size.toLocaleString()} lineups) — filling contests…`
    } else if (topnPoolStartEvent) {
      topnPct = 0
      topnLabel = `Building opponent field pool (${topnPoolStartEvent.field_pool_size.toLocaleString()} lineups)…`
    }
  }

  // --- External pool: p_win field generation + scoring (runs before the
  // Det-EV risk sweep, in this fixed order: field-gen A, field-gen B,
  // score A, score B). `isGpp` only flips true once an actual gpp_* stage
  // event fires (the risk sweep, later) — until then this is the one part
  // of an external-pool run with genuine chunked progress data to build an
  // ETA from, so it gets its own substep tracker mirroring the Det-EV
  // current-step + avg-per-completed-step pattern above.
  type PwinProgEvent = { phase: 'A' | 'B'; n_done: number; n_total: number; timestamp: number }
  const pwinSubsteps = ([
    ['external_pwin_field', 'A'], ['external_pwin_field', 'B'],
    ['external_pwin_score', 'A'], ['external_pwin_score', 'B'],
  ] as const).map(([stage, phase]) => ({
    stage, phase,
    evs: events.filter(e => e.stage === stage && (e as unknown as { phase?: string }).phase === phase) as unknown as PwinProgEvent[],
  }))
  const isExternalPwin = events.some(e => e.stage === 'external_pwin' || e.stage === 'external_pwin_field' || e.stage === 'external_pwin_score')
  const pwinNonEmptyIdxs = pwinSubsteps.map((s, i) => (s.evs.length > 0 ? i : -1)).filter(i => i >= 0)
  const pwinCurrentIdx = pwinNonEmptyIdxs.length ? pwinNonEmptyIdxs[pwinNonEmptyIdxs.length - 1] : -1

  let gppPct = 0
  let gppLabel = ''
  if (isGpp) {
    if (isDetEv) {
      const currentRisk = latestDetStep?.risk ?? latestDetRiskStart?.risk ?? null
      const riskOfN = latestDetRiskStart
        ? `Risk ${latestDetRiskStart.risk_index}/${latestDetRiskStart.total_risks}`
        : currentRisk != null ? `Risk ${currentRisk}` : 'Starting…'
      gppLabel = `Portfolio (Det-EV): ${riskOfN}${latestDetStep ? ` · step ${latestDetStep.step}/${portfolioSizeDet}` : ''}`
    } else if (selectProgressEvents.length > 0) {
      const lastSel = selectProgressEvents[selectProgressEvents.length - 1]
      gppPct = 100
      gppLabel = `Portfolio selection: round ${lastSel.round + 1} — ${lastSel.pct_covered.toFixed(1)}% covered`
    } else if (rescoreDone) {
      gppPct = 100
      gppLabel = 'Fresh re-score complete'
    } else if (latestRescoreScoreProgress) {
      gppPct = Math.round((latestRescoreScoreProgress.batches_done / latestRescoreScoreProgress.batches_total) * 100)
      gppLabel = `Fresh re-score: scoring batch ${latestRescoreScoreProgress.batches_done} / ${latestRescoreScoreProgress.batches_total}`
    } else if (latestRescoreFieldProgress) {
      gppPct = Math.round((latestRescoreFieldProgress.n_done / latestRescoreFieldProgress.n_total) * 100)
      gppLabel = `Fresh re-score: generating field ${latestRescoreFieldProgress.n_done.toLocaleString()} / ${latestRescoreFieldProgress.n_total.toLocaleString()} lineups`
    } else if (rescoreStartEvent) {
      gppPct = 0
      gppLabel = `Fresh re-score: ${rescoreStartEvent.n_candidates.toLocaleString()} candidates × K=${rescoreStartEvent.n_field_samples} fresh fields…`
    } else if (refineStartEvent && !refineDone) {
      gppPct = latestRefineProgress
        ? Math.round((latestRefineProgress.round / latestRefineProgress.rounds) * 100)
        : 0
      gppLabel = latestRefineProgress
        ? `Refining pool: round ${latestRefineProgress.round} / ${latestRefineProgress.rounds} · +${latestRefineProgress.n_mutants} mutants · top-${latestRefineProgress.top_k} EV $${latestRefineProgress.topk_ev_before.toFixed(2)} → $${latestRefineProgress.topk_ev_after.toFixed(2)}${latestRefineProgress.topk_ev_holdout_after != null ? ` (holdout $${latestRefineProgress.topk_ev_holdout_after.toFixed(2)})` : ''}`
        : `Refining pool: round 1 / ${refineStartEvent.rounds}…`
    } else if (latestScoreProgress) {
      gppPct = Math.round((latestScoreProgress.batches_done / latestScoreProgress.batches_total) * 100)
      gppLabel = `Scoring batch ${latestScoreProgress.batches_done} / ${latestScoreProgress.batches_total}`
    } else if (fieldInjectEvent) {
      gppPct = 100
      gppLabel = `Field lineups loaded from cache`
    } else if (latestFieldProgress && !scoreDone) {
      gppPct = Math.round((latestFieldProgress.n_done / latestFieldProgress.n_total) * 100)
      gppLabel = `Generating field: ${latestFieldProgress.n_done.toLocaleString()} / ${latestFieldProgress.n_total.toLocaleString()} lineups`
    } else if (scoreStartEvent && fieldProgressEvents.length === 0) {
      const nTotal = scoreStartEvent.n_field_lineups * scoreStartEvent.n_field_samples
      gppPct = 0
      gppLabel = `Generating field: 0 / ${nTotal.toLocaleString()} lineups`
    } else if (generateDone) {
      gppPct = 100
      gppLabel = 'Candidates generated'
    } else if (latestGenerateProgress && generateStartEvent) {
      const nToGenerate = generateStartEvent.n_candidates - (generateStartEvent.n_from_optimal ?? 0) - (generateStartEvent.n_from_sim_optimal ?? 0)
      gppPct = Math.round((latestGenerateProgress.n / nToGenerate) * 100)
      gppLabel = `Generating candidates: ${latestGenerateProgress.n.toLocaleString()} / ${nToGenerate.toLocaleString()}`
    } else if (generateStartEvent) {
      gppPct = 0
      gppLabel = 'Generating candidates…'
    } else if (simOptimalDone) {
      gppPct = 100
      gppLabel = 'Sim-optimal seeding complete'
    } else if (latestSimOptimalProgress && simOptimalStartEvent) {
      gppPct = Math.round((latestSimOptimalProgress.n / latestSimOptimalProgress.total) * 100)
      gppLabel = `Seeding sim-optimal: ${latestSimOptimalProgress.n} / ${latestSimOptimalProgress.total}`
    } else if (simOptimalStartEvent) {
      gppPct = 0
      gppLabel = `Seeding sim-optimal: 0 / ${simOptimalStartEvent.n_sim_optimals}`
    } else if (optimalDone) {
      gppPct = 100
      gppLabel = 'Optimal seeding complete'
    } else if (latestOptimalProgress && optimalStartEvent) {
      gppPct = Math.round((latestOptimalProgress.n / latestOptimalProgress.total) * 100)
      gppLabel = `Seeding optimal: ${latestOptimalProgress.n} / ${latestOptimalProgress.total}`
    } else if (optimalStartEvent) {
      gppPct = 0
      gppLabel = `Seeding optimal: 0 / ${optimalStartEvent.n_optimal}`
    }
  }

  // --- ETA ---
  let etaMs: number | null = null
  if (running && isExternalPwin && !isGpp && pwinCurrentIdx >= 0) {
    // Current substep: remaining time from its own recent (done, total) rate.
    let currentRemainingMs: number | null = null
    const curEvs = pwinSubsteps[pwinCurrentIdx].evs
    const curLatest = curEvs[curEvs.length - 1]
    if (curLatest.n_done < curLatest.n_total) {
      const recent = curEvs.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const recentDone = recent[recent.length - 1].n_done - recent[0].n_done
        if (recentDone > 0) {
          currentRemainingMs = (recentElapsed / recentDone) * (curLatest.n_total - curLatest.n_done)
        }
      }
    } else {
      currentRemainingMs = 0 // this substep finished; the next just hasn't emitted yet
    }

    // Average wall time per FULLY completed substep (every non-empty
    // substep before the current one is complete by construction, since
    // they run strictly in this fixed order) — same idea as the Det-EV
    // avg-per-risk estimate above.
    const completedDurations = pwinNonEmptyIdxs.slice(0, -1).map(i => {
      const evs = pwinSubsteps[i].evs
      return evs[evs.length - 1].timestamp - evs[0].timestamp
    }).filter(d => d > 0)
    let avgMsPerSubstep: number | null = completedDurations.length > 0
      ? completedDurations.reduce((a, b) => a + b, 0) / completedDurations.length
      : null
    // No completed substep yet (still in the first one, field-gen A):
    // estimate this substep's own total time from its rate and assume the
    // rest take similarly — a rough first guess that firms up once a
    // substep actually finishes.
    if (avgMsPerSubstep === null && curLatest.n_done > 0) {
      const recent = curEvs.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const recentDone = recent[recent.length - 1].n_done - recent[0].n_done
        if (recentDone > 0) avgMsPerSubstep = (recentElapsed / recentDone) * curLatest.n_total
      }
    }

    const remainingSubsteps = pwinSubsteps.length - 1 - pwinCurrentIdx
    if (currentRemainingMs !== null) {
      // Without a completed-substep average yet, this understates the true
      // total (silent on the still-unknown remaining substeps) rather than
      // guessing — it firms up once the first substep finishes.
      etaMs = avgMsPerSubstep !== null
        ? currentRemainingMs + avgMsPerSubstep * remainingSubsteps
        : currentRemainingMs
    }
  } else if (isGpp) {
    if (running && latestOptimalProgress && !optimalDone) {
      const elapsed = latestOptimalProgress.timestamp - optimalProgressEvents[0].timestamp
      const n = latestOptimalProgress.n
      const remaining = latestOptimalProgress.total - n
      if (n > 0 && remaining > 0 && elapsed > 0) {
        etaMs = (elapsed / n) * remaining
      }
    } else if (running && latestGenerateProgress && !generateDone && generateStartEvent) {
      const recent = generateProgressEvents.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerChunk = recentElapsed / (recent.length - 1)
        const nToGenerate = generateStartEvent.n_candidates - (generateStartEvent.n_from_optimal ?? 0) - (generateStartEvent.n_from_sim_optimal ?? 0)
        const remainingChunks = (nToGenerate - latestGenerateProgress.n) / 500
        if (remainingChunks > 0) etaMs = avgPerChunk * remainingChunks
      }
    } else if (running && latestFieldProgress && !scoreDone && latestScoreProgress === undefined) {
      const remaining = latestFieldProgress.n_total - latestFieldProgress.n_done
      if (remaining > 0 && fieldProgressEvents.length >= 2) {
        const recent = fieldProgressEvents.slice(-4)
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const recentLineups = recent[recent.length - 1].n_done - recent[0].n_done
        if (recentLineups > 0) etaMs = (recentElapsed / recentLineups) * remaining
      }
    } else if (running && latestScoreProgress && !scoreDone) {
      const recent = scoreProgressEvents.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerBatch = recentElapsed / (recent.length - 1)
        const remaining = latestScoreProgress.batches_total - latestScoreProgress.batches_done
        if (remaining > 0) etaMs = avgPerBatch * remaining
      }
    } else if (running && latestRescoreFieldProgress && !rescoreDone && latestRescoreScoreProgress === undefined) {
      const remaining = latestRescoreFieldProgress.n_total - latestRescoreFieldProgress.n_done
      if (remaining > 0 && rescoreFieldProgressEvents.length >= 2) {
        const recent = rescoreFieldProgressEvents.slice(-4)
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const recentLineups = recent[recent.length - 1].n_done - recent[0].n_done
        if (recentLineups > 0) etaMs = (recentElapsed / recentLineups) * remaining
      }
    } else if (running && latestRescoreScoreProgress && !rescoreDone) {
      const recent = rescoreScoreProgressEvents.slice(-4)
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerBatch = recentElapsed / (recent.length - 1)
        const remaining = latestRescoreScoreProgress.batches_total - latestRescoreScoreProgress.batches_done
        if (remaining > 0) etaMs = avgPerBatch * remaining
      }
    } else if (running && isDetEv) {
      // ETA for Det-EV: remaining in current risk + (avgMsPerRisk * remaining risks)
      const currentRiskStartTs = latestDetRiskStart?.timestamp ?? null
      const elapsedInCurrentRisk = currentRiskStartTs ? now - currentRiskStartTs : 0

      // Remaining time in current risk from step rate
      let currentRiskRemainingMs: number | null = null
      if (currentDetStepNum > 0 && portfolioSizeDet > 0 && elapsedInCurrentRisk > 0) {
        currentRiskRemainingMs = (elapsedInCurrentRisk / currentDetStepNum) * (portfolioSizeDet - currentDetStepNum)
      }

      // Average time per completed risk (from consecutive risk_start timestamps)
      let avgMsPerRisk: number | null = null
      if (detRiskStartEvents.length >= 2) {
        const durations: number[] = []
        for (let ri = 1; ri < detRiskStartEvents.length; ri++) {
          durations.push(detRiskStartEvents[ri].timestamp - detRiskStartEvents[ri - 1].timestamp)
        }
        avgMsPerRisk = durations.reduce((a, b) => a + b, 0) / durations.length
      } else if (portfolioSizeDet > 0 && currentDetStepNum > 0 && elapsedInCurrentRisk > 0) {
        // Estimate per-risk duration from current rate within first risk
        avgMsPerRisk = (elapsedInCurrentRisk / currentDetStepNum) * portfolioSizeDet
      }

      const risksAfterCurrent = totalRisksCount - currentActiveRiskIdx
      if (currentRiskRemainingMs !== null && avgMsPerRisk !== null) {
        etaMs = currentRiskRemainingMs + avgMsPerRisk * risksAfterCurrent
      } else if (avgMsPerRisk !== null && risksAfterCurrent >= 0) {
        etaMs = avgMsPerRisk * (risksAfterCurrent + 1)
      }
    }
  } else if (running && isSelfPlay && latestSelfPlayContest) {
    // Rolling average wall time per COMPLETED contest, same chunked
    // n_done/n_total pattern used elsewhere in this panel (e.g. the
    // gpp_field_progress branch above). Per-contest cost is genuinely
    // heterogeneous here (a 72-round mini-max-sized contest can cost ~100x
    // a 1-round small one, see self_play.py), so unlike the other chunked
    // ETAs in this panel this one can swing hard moment-to-moment,
    // especially early in a slate before a large contest has been seen --
    // a rough progress signal, not a precise one.
    const remaining = latestSelfPlayContest.contests_total - latestSelfPlayContest.contests_done
    if (remaining > 0 && selfPlayContestEvents.length >= 2) {
      const recent = selfPlayContestEvents.slice(-4)
      const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
      const recentDone = recent[recent.length - 1].contests_done - recent[0].contests_done
      if (recentDone > 0) etaMs = (recentElapsed / recentDone) * remaining
    }
  } else if (running && isTopnCoverage) {
    etaMs = topnEtaMs
  } else if (running && isMrp) {
    etaMs = mrpEtaMs
  } else {
    if (running && current > 0 && total > current) {
      const recent = lineupEvents.slice(-4) // up to 4 events → 3 intervals
      if (recent.length >= 2) {
        const recentElapsed = recent[recent.length - 1].timestamp - recent[0].timestamp
        const avgPerLineup = recentElapsed / (recent.length - 1)
        etaMs = avgPerLineup * (total - current)
      }
    }
  }

  // isExternalRun covers the whole external-pool lifecycle (file discovery,
  // sim, p_win field/scoring, the risk sweep) so elapsed ticks from the
  // start of the run rather than only once isGpp flips true partway
  // through (which for external mode doesn't happen until the risk sweep).
  const isExternalRun = events.some(e => typeof e.stage === 'string' && e.stage.startsWith('external_'))
  const liveElapsedMs = running && first && (current > 0 || isGpp || isExternalRun || isSelfPlay || isTopnCoverage || isMrp) ? now - first.timestamp : null

  return (
    <div className="progress-panel">
      <h3>
        Run Progress
        {liveElapsedMs !== null && (
          <span className="muted" style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.9em' }}>
            {formatMsWhole(liveElapsedMs)} elapsed
            {etaMs !== null && etaMs > 0 && (
              <span style={{ marginLeft: 12 }}>
                ~{formatMsWhole(etaMs)} remaining
              </span>
            )}
          </span>
        )}
        {elapsed !== null && !running && (
          <span className="muted" style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.9em' }}>
            ({formatMs(elapsed)} total)
          </span>
        )}
      </h3>

      {/* Non-GPP progress bar */}
      {!isGpp && (running || latestLineup) && total > 0 && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${pct}%` }} />
          <span className="progress-label">
            Lineup {current} / {total}
          </span>
        </div>
      )}

      {/* GPP progress bar — Det-EV: 5-segment bar */}
      {isGpp && running && isDetEv && (
        <div>
          <div className="progress-bar-segmented">
            {detSegments.map((seg, i) => (
              <div key={i} className="progress-segment">
                <div
                  className={`progress-bar${seg.isComplete ? ' progress-bar-complete' : ''}`}
                  style={{ width: `${seg.pct}%` }}
                />
              </div>
            ))}
          </div>
          <div className="progress-segment-labels">
            {detSegments.map((seg, i) => (
              <span
                key={i}
                className={`progress-segment-label${seg.isComplete ? ' complete' : seg.isActive ? ' active' : ''}`}
              >
                R{i + 1}
              </span>
            ))}
          </div>
          {gppLabel && <div className="progress-det-label">{gppLabel}</div>}
        </div>
      )}

      {/* GPP progress bar — all other methods */}
      {isGpp && running && gppLabel && !isDetEv && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${gppPct}%` }} />
          <span className="progress-label">{gppLabel}</span>
        </div>
      )}

      {/* Self-play progress bar */}
      {isSelfPlay && running && selfPlayLabel && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${selfPlayPct}%` }} />
          <span className="progress-label">{selfPlayLabel}</span>
        </div>
      )}

      {/* Top-N coverage progress bar */}
      {isTopnCoverage && running && topnLabel && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${topnPct}%` }} />
          <span className="progress-label">{topnLabel}</span>
        </div>
      )}

      {/* Marginal reward progress bar */}
      {isMrp && running && mrpLabel && (
        <div className="progress-bar-wrap">
          <div className="progress-bar" style={{ width: `${mrpPct}%` }} />
          <span className="progress-label">{mrpLabel}</span>
        </div>
      )}

      {/* How much of the portfolio the generated frontier actually won.
          Gated on the frontier having RUN, not on it having won anything:
          "0 of 103" is a real result worth seeing when generation was on,
          but pure noise when it was off. */}
      {mrpFrontierDoneEvent != null && mrpDoneEvent != null
        && mrpDoneEvent.n_entries != null && mrpDoneEvent.n_entries > 0
        && mrpDoneEvent.n_generated_picked != null && (
        <div className="progress-det-label">
          {mrpDoneEvent.n_generated_picked.toLocaleString()} of{' '}
          {mrpDoneEvent.n_entries.toLocaleString()} entries
          {' '}({Math.round((mrpDoneEvent.n_generated_picked / mrpDoneEvent.n_entries) * 100)}%)
          {' '}came from the generated frontier
          {mrpFrontierDoneEvent?.n_kept != null && (
            <span className="muted"> · {mrpFrontierDoneEvent.n_kept.toLocaleString()} generated lineups offered</span>
          )}
        </div>
      )}

      {selfPlayFallbackEvents.length > 0 && (
        <div className="progress-warning-list">
          <div className="progress-warning-header">
            Approximate payout table used (contest name not one of DK's known types) —
            add a real one to <code>src/optimization/payout.py</code>'s{' '}
            <code>CONTEST_STRUCTURES</code> / <code>data/payout_structures/</code> for
            more accurate results:
          </div>
          {selfPlayFallbackEvents.map((ev, i) => (
            <div key={i} className="progress-warning-row">
              "{ev.contest_name}" (implied ~{Math.round(ev.implied_field_size).toLocaleString()} entries)
              → matched a {ev.matched_total_entries.toLocaleString()}-entry table
            </div>
          ))}
        </div>
      )}

      <div className="event-list">
        {buildDisplayEvents(events).map((item, i) => (
          <div key={i} className={`event-row event-${item.stage}`}>
            <span className="event-stage">{item.label}</span>
            <span className="event-detail">{item.detail}</span>
          </div>
        ))}
        {running && !latestLineup && !isGpp && (
          <div className="event-row">
            <span className="event-stage muted">…</span>
          </div>
        )}
      </div>

      {/* Non-GPP lineup grid */}
      {events.some(e => e.stage === 'optimize_lineup') && (
        <div className="event-list event-list-four-col">
          {events.filter(e => e.stage === 'optimize_lineup').map((e, i) => {
            const ev = e as OptimizeLineupEvent
            return (
              <div key={i} className="event-row event-optimize_lineup">
                <span className="event-stage event-stage-lineup">{ev.lineup_index}/{ev.total}</span>
                <span className="event-detail">{renderDetail(e)}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* GPP selection grid (legacy EVPortfolioSelector) */}
      {selectProgressEvents.length > 0 && (
        <div className="event-list event-list-four-col">
          {selectProgressEvents.map((ev, i) => (
            <div key={i} className="event-row event-gpp_select_progress">
              <span className="event-stage event-stage-lineup">{ev.round + 1}</span>
              <span className="event-detail">
                EV ${ev.lineup_ev.toFixed(2)} · {ev.n_covered.toLocaleString()} sims ({ev.pct_covered.toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Self-play: one row per completed contest */}
      {selfPlayContestEvents.length > 0 && (
        <div className="event-list">
          {selfPlayContestEvents.map((ev, i) => (
            <div key={i} className="event-row event-self_play_contest_progress">
              <span className="event-stage event-stage-lineup">{ev.contest_id}</span>
              <span className="event-detail">
                k={ev.k} · field {ev.n_field.toLocaleString()} · {ev.n_rounds} round{ev.n_rounds === 1 ? '' : 's'}
                {' '}(round {ev.round_elapsed_s.toFixed(1)}s{ev.refine_elapsed_s > 0 ? ` + refine ${ev.refine_elapsed_s.toFixed(1)}s` : ''})
                {' '}· {ev.n_swaps} refinement swap{ev.n_swaps === 1 ? '' : 's'}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Pool refinement: one row per round showing what mutation changed */}
      {refineProgressEvents.length > 0 && (
        <div className="event-list">
          {refineProgressEvents.map((ev, i) => (
            <div key={i} className="event-row event-gpp_refine_progress">
              <span className="event-stage event-stage-lineup">Refine {ev.round}/{ev.rounds}</span>
              <span className="event-detail">
                +{ev.n_mutants} mutants from {ev.n_parents} parents · {ev.n_beat_parent} beat parent · {ev.n_in_topk} now in top-{ev.top_k} · top-{ev.top_k} EV ${ev.topk_ev_before.toFixed(2)} → ${ev.topk_ev_after.toFixed(2)}
                {ev.topk_ev_holdout_before != null && ev.topk_ev_holdout_after != null && (
                  <> (holdout ${ev.topk_ev_holdout_before.toFixed(2)} → ${ev.topk_ev_holdout_after.toFixed(2)})</>
                )}
                {ev.best_swap_out.length > 0 && (
                  <> · best swap: {formatSwap(ev.best_swap_out, ev.best_swap_in)} ({ev.best_swap_ev_delta >= 0 ? '+' : ''}${ev.best_swap_ev_delta.toFixed(2)}
                    {ev.best_swap_ev_delta_holdout != null && (
                      <>, holdout {ev.best_swap_ev_delta_holdout >= 0 ? '+' : ''}${ev.best_swap_ev_delta_holdout.toFixed(2)}</>
                    )})
                  </>
                )}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Det-EV: one row per risk value showing completion status */}
      {(detRiskStartEvents.length > 0 || detProgressEvents.length > 0) && (() => {
        const totalRisks = detRiskStartEvents[0]?.total_risks ?? detProgressEvents[0]?.total_risks ?? 5
        const lastStep = detProgressEvents[detProgressEvents.length - 1]
        const currentRiskIdx = lastStep?.risk_index ?? detRiskStartEvents[detRiskStartEvents.length - 1]?.risk_index ?? 0
        const rows = Array.from({ length: totalRisks }, (_, i) => i + 1)
        return (
          <div className="event-list">
            {rows.map(riskIdx => {
              const risk = riskIdx  // risk value = risk_index for [1..5]
              const isComplete = riskIdx < currentRiskIdx
              const isActive = riskIdx === currentRiskIdx
              const stepForRisk = isActive ? lastStep : null
              const label = isComplete ? '✓' : isActive ? '→' : '·'
              return (
                <div key={riskIdx} className="event-row event-gpp_det_select_progress">
                  <span className="event-stage event-stage-lineup">{label} R{risk}</span>
                  <span className="event-detail">
                    {isComplete && 'complete'}
                    {isActive && stepForRisk && `step ${stepForRisk.step}/${stepForRisk.portfolio_size} · EV $${stepForRisk.lineup_ev.toFixed(3)} · dist ${(stepForRisk.distance * 100).toFixed(1)}%`}
                    {isActive && !stepForRisk && 'starting…'}
                    {!isComplete && !isActive && ''}
                  </span>
                </div>
              )
            })}
          </div>
        )
      })()}

    </div>
  )
}

function formatSwap(out: string[], inn: string[]): string {
  // Same-position 1-2 player swaps; labels are "POS Name" sorted, so
  // index-wise pairing lines the positions up.
  if (out.length === inn.length) {
    return out.map((o, i) => `${o} → ${inn[i]}`).join(' · ')
  }
  return `${out.join(', ')} → ${inn.join(', ')}`
}

function buildConfigDetail(events: SSEEvent[]): string {
  const sim = events.find(e => e.stage === 'simulate') as SimulateEvent | undefined
  const parts: string[] = []
  if (sim) parts.push(`${sim.n_sims.toLocaleString()} simulations`)
  return parts.join(', ')
}

function buildDisplayEvents(events: SSEEvent[]): Array<{ stage: string; label: string; detail: string }> {
  const result: Array<{ stage: string; label: string; detail: string }> = []
  let configInserted = false

  const hasEvent = (stage: string) => events.some(e => e.stage === stage)

  for (const e of events) {
    if (e.stage === 'optimize_lineup' || e.stage === 'upload_files') continue
    if (GPP_PROGRESS_STAGES.has(e.stage) || e.stage === 'gpp_field_progress' || e.stage === 'gpp_rescore_field_progress') continue
    // self_play_contest_progress/self_play_payout_fallback each get their
    // own dedicated list (per-contest row list / warning banner) below.
    if (e.stage === 'self_play_contest_progress' || e.stage === 'self_play_payout_fallback') continue
    // topn_pick_progress/topn_contest_start/topn_pool_progress each get
    // their own live progress bar (see isTopnCoverage) rather than a row
    // per event -- topn_pool_progress in particular can fire dozens of
    // times during one field-pool build, so without this skip it spammed
    // the event log with near-duplicate rows instead of showing one
    // updating count.
    if (e.stage === 'topn_pick_progress' || e.stage === 'topn_contest_start'
        || e.stage === 'topn_pool_progress') continue
    if (e.stage === 'topn_pool_start' && hasEvent('topn_pool_done')) continue
    // Per-chunk p_win progress (many events per stage, A and B) — no live
    // counter surface exists for it yet, same treatment as the gpp_*
    // _progress events above; the one-shot 'external_pwin' summary row
    // below is what's shown.
    if (e.stage === 'external_pwin_field' || e.stage === 'external_pwin_score') continue
    // MRP fires one event per pick (114 on a typical slate) and one per
    // contest state built — same flooding problem as topn_pick_progress
    // above. They still drive the live elapsed timer; only the log rows go.
    if (e.stage === 'mrp_pick_progress' || e.stage === 'mrp_build_progress') continue
    // Per-lambda frontier progress drives the live bar above; a row each
    // would push the surrounding stages off screen for no added information.
    if (e.stage === 'mrp_frontier_progress') continue
    if (e.stage === 'mrp_frontier_start' && hasEvent('mrp_frontier_done')) continue
    if (e.stage === 'mrp_start' && hasEvent('mrp_done')) continue
    // Skip start event once done event is present (collapse into one row)
    if (e.stage === 'gpp_optimal_start' && hasEvent('gpp_optimal_done')) continue
    if (e.stage === 'gpp_sim_optimal_start' && hasEvent('gpp_sim_optimal_done')) continue
    if (e.stage === 'gpp_generate_start' && hasEvent('gpp_generate_done')) continue
    if (e.stage === 'gpp_score_start' && hasEvent('gpp_score_done')) continue
    if (e.stage === 'gpp_refine_start' && hasEvent('gpp_refine_done')) continue
    if (e.stage === 'gpp_rescore_start' && hasEvent('gpp_rescore_done')) continue
    // field_inject is a one-shot cache notification; skip once score is done
    if (e.stage === 'gpp_field_inject' && hasEvent('gpp_score_done')) continue
    if (CONFIG_STAGES.has(e.stage)) {
      if (!configInserted) {
        result.push({ stage: 'config', label: 'Configuration', detail: buildConfigDetail(events) })
        configInserted = true
      }
      continue
    }
    result.push({ stage: e.stage, label: STAGE_LABELS[e.stage] ?? e.stage, detail: renderDetail(e) })
  }

  return result
}

function renderDetail(e: SSEEvent): string {
  switch (e.stage) {
    case 'load_slate': {
      const ev = e as unknown as {
        n_teams: number; n_batters: number; n_pitchers: number;
        multi_pitcher_teams: Record<string, number>;
        n_teams_excluded: number; n_batters_ind_excluded: number; n_pitchers_ind_excluded: number;
        n_pitchers_value_excluded: number; n_batters_value_excluded: number;
      }
      const multiPitcher = ev.multi_pitcher_teams && Object.keys(ev.multi_pitcher_teams).length > 0
        ? ` (${Object.entries(ev.multi_pitcher_teams).map(([t, n]) => `${n} ${t}`).join(', ')})`
        : ''
      const loaded = `${ev.n_teams} teams, ${ev.n_batters} batters, ${ev.n_pitchers} pitchers${multiPitcher} loaded`
      const exclParts: string[] = []
      if (ev.n_teams_excluded > 0) exclParts.push(`${ev.n_teams_excluded} team${ev.n_teams_excluded !== 1 ? 's' : ''}`)
      if (ev.n_batters_ind_excluded > 0) exclParts.push(`${ev.n_batters_ind_excluded} batter${ev.n_batters_ind_excluded !== 1 ? 's' : ''}`)
      if (ev.n_pitchers_ind_excluded > 0) exclParts.push(`${ev.n_pitchers_ind_excluded} pitcher${ev.n_pitchers_ind_excluded !== 1 ? 's' : ''}`)
      const valueParts: string[] = []
      if (ev.n_pitchers_value_excluded > 0) valueParts.push(`${ev.n_pitchers_value_excluded} pitcher${ev.n_pitchers_value_excluded !== 1 ? 's' : ''}`)
      if (ev.n_batters_value_excluded > 0) valueParts.push(`${ev.n_batters_value_excluded} batter${ev.n_batters_value_excluded !== 1 ? 's' : ''}`)
      let detail = exclParts.length > 0 ? `${loaded}. ${exclParts.join(', ')} excluded` : loaded
      if (valueParts.length > 0) detail += `. ${valueParts.join(', ')} below value cutoff`
      return detail
    }
    case 'simulate': {
      const ev = e as unknown as { n_sims: number }
      return `${ev.n_sims.toLocaleString()} simulations`
    }
    case 'external_load': {
      const ev = e as unknown as { lineups_files: string[]; projections_file: string; paired_by_token: boolean }
      const filesLabel = ev.lineups_files.length > 1
        ? `${ev.lineups_files.length} lineup files (${ev.lineups_files.join(', ')})`
        : ev.lineups_files[0]
      return `${filesLabel} + ${ev.projections_file}${ev.paired_by_token ? '' : ' (unpaired companion)'}`
    }
    case 'external_pool': {
      const ev = e as unknown as {
        n_lineups: number; n_files: number; n_contests_covered: number;
        n_dropped_unknown: number; n_dropped_duplicates: number; n_dropped_near_duplicates: number;
      }
      const fileNote = ev.n_files > 1 ? ` across ${ev.n_files} files` : ''
      const dupNote = ev.n_dropped_duplicates > 0
        ? `, ${ev.n_dropped_duplicates.toLocaleString()} duplicate${ev.n_dropped_duplicates !== 1 ? 's' : ''} removed`
        : ''
      const nearDupNote = ev.n_dropped_near_duplicates > 0
        ? `, ${ev.n_dropped_near_duplicates.toLocaleString()} near-duplicate${ev.n_dropped_near_duplicates !== 1 ? 's' : ''} (9/10 overlap) removed`
        : ''
      const unkNote = ev.n_dropped_unknown > 0
        ? `, ${ev.n_dropped_unknown.toLocaleString()} unknown-player row${ev.n_dropped_unknown !== 1 ? 's' : ''} dropped`
        : ''
      return `${ev.n_lineups.toLocaleString()} lineups imported${fileNote}${dupNote}${nearDupNote}${unkNote} · ${ev.n_contests_covered} contest${ev.n_contests_covered !== 1 ? 's' : ''} covered`
    }
    case 'external_proj_score_floor': {
      const ev = e as unknown as { cutoff: number; n_culled: number; percentile: number; pool_size: number }
      return `Bottom ${ev.percentile}% by ceiling culled — ${ev.n_culled.toLocaleString()} of ${ev.pool_size.toLocaleString()} lineups below ${ev.cutoff.toFixed(1)} 99th-percentile points`
    }
    case 'external_owncap_cull': {
      const ev = e as unknown as {
        contest: string; field_size: number; cap_pct: number; cap_cutoff: number;
        n_before: number; n_after: number; n_culled: number;
      }
      return `${ev.contest} (${Math.round(ev.field_size).toLocaleString()} field): ` +
        `${ev.n_culled.toLocaleString()} of ${ev.n_before.toLocaleString()} lineups above the ` +
        `${ev.cap_pct.toFixed(0)}th ownership percentile (${ev.cap_cutoff.toFixed(1)} total own pts) ` +
        `culled — ${ev.n_after.toLocaleString()} remain`
    }
    case 'external_pwin': {
      const ev = e as unknown as {
        n_sims_per_stage: number; field_size: number; n_contests: number;
        admit_n: number; sharpness: number;
      }
      const admitNote = ev.admit_n > 0 ? `, stage-A admit ${ev.admit_n.toLocaleString()}` : ''
      return `${ev.n_sims_per_stage.toLocaleString()} sims/stage vs a ${ev.field_size.toLocaleString()}-lineup simulated field ` +
        `· ${ev.n_contests} contest${ev.n_contests !== 1 ? 's' : ''} · sharpness ${ev.sharpness}${admitNote}`
    }
    case 'ppd_applied':
    case 'external_ppd_applied': {
      const ev = e as unknown as { games: { game: string; ppd_pct: number; n_sims_zeroed: number }[]; n_sims_total: number }
      const parts = ev.games.map(g => `${g.game} ${g.ppd_pct}% (${g.n_sims_zeroed.toLocaleString()} sims)`)
      return `${parts.join(', ')} — zeroed independently`
    }
    case 'compute_target': {
      const ev = e as unknown as { target: number; percentile: number | null }
      return ev.percentile
        ? `Target: ${ev.target.toFixed(1)} pts (p${ev.percentile})`
        : `Target: ${ev.target.toFixed(1)} pts (manual)`
    }
    case 'optimize_lineup': {
      const ev = e as OptimizeLineupEvent
      const pctFmt = (v: number | null | undefined) => v != null ? `${v.toFixed(1)}%` : '—'
      const ptLabel = ev.target_percentile != null ? `p${ev.target_percentile}` : 'target'
      return `p90: ${pctFmt(ev.pct_above_p90)} · ${ptLabel}: ${pctFmt(ev.pct_above_target)} · p99: ${pctFmt(ev.pct_above_p99)}`
    }
    case 'gpp_optimal_start': {
      const ev = e as unknown as { n_optimal: number }
      return `Generating ${ev.n_optimal} optimal candidates…`
    }
    case 'gpp_optimal_done': {
      const ev = e as unknown as { n_generated: number }
      return `${ev.n_generated} optimal candidates seeded`
    }
    case 'gpp_sim_optimal_start': {
      const ev = e as unknown as { n_sim_optimals: number }
      return `Solving ${ev.n_sim_optimals} per-sim optimal lineups…`
    }
    case 'gpp_sim_optimal_done': {
      const ev = e as unknown as { n_generated: number }
      return `${ev.n_generated} sim-optimal candidates seeded`
    }
    case 'gpp_generate_start': {
      const ev = e as unknown as { n_candidates: number; n_from_cache?: number; n_from_optimal?: number; n_from_sim_optimal?: number }
      if (ev.n_from_cache != null && ev.n_from_cache > 0) {
        return `Generating ${ev.n_candidates.toLocaleString()} + ${ev.n_from_cache.toLocaleString()} from cache…`
      }
      const nSeeded = (ev.n_from_optimal ?? 0) + (ev.n_from_sim_optimal ?? 0)
      if (nSeeded > 0) {
        const parts = []
        if (ev.n_from_optimal) parts.push(`${ev.n_from_optimal} optimal`)
        if (ev.n_from_sim_optimal) parts.push(`${ev.n_from_sim_optimal} sim-optimal`)
        return `Generating ${(ev.n_candidates - nSeeded).toLocaleString()} random + ${parts.join(' + ')} candidates…`
      }
      return `Generating ${ev.n_candidates.toLocaleString()} candidates…`
    }
    case 'gpp_generate_done': {
      const ev = e as unknown as { n_generated: number; from_cache?: boolean }
      const suffix = ev.from_cache ? ' (from cache)' : ''
      return `${ev.n_generated.toLocaleString()} candidates${suffix}`
    }
    case 'gpp_field_inject': {
      const ev = e as unknown as { n_field: number; n_k: number }
      return `${(ev.n_field * ev.n_k).toLocaleString()} lineups loaded from cache`
    }
    case 'gpp_score_start': {
      const ev = e as unknown as { n_candidates: number; n_field_samples: number }
      return `${ev.n_candidates.toLocaleString()} candidates × ${ev.n_field_samples} field samples`
    }
    case 'gpp_score_done':
      return 'Scoring complete'
    case 'gpp_refine_start': {
      const ev = e as unknown as { rounds: number; top: number; mutants_per_parent: number }
      return `${ev.rounds} round${ev.rounds !== 1 ? 's' : ''} × ${ev.top} parents × ${ev.mutants_per_parent} mutants…`
    }
    case 'gpp_refine_done': {
      const ev = e as unknown as { pool_size: number; n_added: number; best_ev: number; best_ev_before: number }
      const uplift = ev.best_ev - ev.best_ev_before
      return `+${ev.n_added.toLocaleString()} mutants (pool ${ev.pool_size.toLocaleString()}) · best EV $${ev.best_ev_before.toFixed(2)} → $${ev.best_ev.toFixed(2)}${uplift > 0 ? ` (+$${uplift.toFixed(2)})` : ''}`
    }
    case 'gpp_rescore_start': {
      const ev = e as unknown as { n_candidates: number; n_field_samples: number }
      return `Re-scoring top ${ev.n_candidates.toLocaleString()} candidates × ${ev.n_field_samples} fresh field samples…`
    }
    case 'gpp_rescore_done': {
      const ev = e as unknown as { pool_size: number; n_field_samples: number; topk: number; topk_ev_mined: number; topk_ev_fresh: number }
      const curse = ev.topk_ev_mined - ev.topk_ev_fresh
      return `${ev.pool_size.toLocaleString()} candidates × ${ev.n_field_samples} fresh fields · top-${ev.topk} EV $${ev.topk_ev_mined.toFixed(2)} → $${ev.topk_ev_fresh.toFixed(2)} (curse $${curse.toFixed(2)})`
    }
    case 'gpp_holdout': {
      const ev = e as unknown as { holdout_mean_payout: number }
      return `Holdout mean payout: ${ev.holdout_mean_payout.toFixed(4)}`
    }
    case 'self_play_pool_start': {
      const ev = e as unknown as { n_contests: number }
      return `${ev.n_contests} contest${ev.n_contests === 1 ? '' : 's'}…`
    }
    case 'self_play_pool_done': {
      const ev = e as unknown as { pool_size: number; precise_n_sims: number | null; n_promoted: number }
      const refine = ev.precise_n_sims
        ? `refinement at ${ev.precise_n_sims.toLocaleString()} sims, ${ev.n_promoted.toLocaleString()} promoted candidates`
        : 'refinement disabled'
      return `${ev.pool_size.toLocaleString()} lineups · ${refine}`
    }
    case 'topn_sims_autosize': {
      const ev = e as unknown as { configured_n_sims: number; total_demand: number; effective_n_sims: number }
      return `${ev.configured_n_sims.toLocaleString()} configured → ${ev.effective_n_sims.toLocaleString()} `
        + `(every contest needs a disjoint set totaling ${ev.total_demand.toLocaleString()} sim worlds)`
    }
    case 'topn_pool_start': {
      const ev = e as unknown as { field_pool_size: number }
      return `${ev.field_pool_size.toLocaleString()} lineups…`
    }
    case 'topn_pool_done': {
      const ev = e as unknown as { field_pool_size: number; n_sims: number }
      return `${ev.field_pool_size.toLocaleString()} lineups × ${ev.n_sims.toLocaleString()} sim worlds`
    }
    case 'topn_pool_augmented': {
      const ev = e as unknown as { n_requested: number; n_added: number; pool_size: number }
      return `+${ev.n_added.toLocaleString()} generated candidates added (of ${ev.n_requested.toLocaleString()} `
        + `requested) — candidate pool now ${ev.pool_size.toLocaleString()}`
    }
    case 'mrp_start': {
      const ev = e as unknown as {
        n_contests: number; n_pool: number; n_entries: number
        gamma_in: number; gamma_out: number
      }
      return `${ev.n_entries} entries across ${ev.n_contests} contests `
        + `from ${ev.n_pool.toLocaleString()} lineups (γ_in ${ev.gamma_in}, γ_out ${ev.gamma_out})`
    }
    case 'mrp_frontier_start': {
      const ev = e as unknown as MrpFrontierStartEvent
      return `Sampling ${ev.n_sample.toLocaleString()} candidates · searching `
        + `${ev.n_lambda_search} λ for each contest's λ* · keeping `
        + `${ev.per_team} per team · ${ev.n_pairs.toLocaleString()} covariance pairs`
    }
    case 'mrp_frontier_done': {
      const ev = e as unknown as MrpFrontierDoneEvent
      if (ev.skipped) return `Skipped — ${ev.skipped}`
      const dropped = ev.n_dropped_duplicate
        ? `, ${ev.n_dropped_duplicate.toLocaleString()} exact duplicate${ev.n_dropped_duplicate === 1 ? '' : 's'} dropped`
        : ''
      const universe = ev.n_players_kept != null && ev.n_players_before != null
        ? ` · solved over ${ev.n_players_kept} of ${ev.n_players_before} players`
          + (ev.n_pitchers_kept != null && ev.n_teams != null
            ? ` (${ev.n_pitchers_kept} starters / ${ev.n_teams} teams)` : '')
        : ''
      const lam = ev.lambda_min != null && ev.lambda_max != null
        ? ` · λ* ${ev.lambda_min.toPrecision(3)}–${ev.lambda_max.toPrecision(3)}`
          + (ev.n_lambda_star != null ? ` over ${ev.n_lambda_star} operating point${ev.n_lambda_star === 1 ? '' : 's'}` : '')
          + (ev.lambda_star_at_edge ? ' ⚠ at search edge' : '')
        : ''
      const blend = ev.sigma_dG_contests != null
        ? ` · σ_dG blended over ${ev.sigma_dG_contests} contest${ev.sigma_dG_contests === 1 ? '' : 's'}`
          + (ev.sigma_dG_min_corr != null ? ` (min corr ${ev.sigma_dG_min_corr.toFixed(3)})` : '')
        : ''
      return `${(ev.n_kept ?? 0).toLocaleString()} lineups added to the pool${dropped}${lam}${blend}${universe}`
    }
    case 'mrp_payout_fallback': {
      const ev = e as unknown as { n_missing: number; n_contests: number }
      return `${ev.n_missing} of ${ev.n_contests} contests have no registered payout `
        + `table -- run paused for confirmation`
    }
    case 'mrp_done': {
      const ev = e as unknown as {
        total_reward: number; n_unfilled: number
        per_contest: { contest_name: string; k: number; reward: number }[]
      }
      const n = ev.per_contest?.length ?? 0
      return `R(S) = $${ev.total_reward.toFixed(2)} across ${n} contest${n === 1 ? '' : 's'}`
        + `${ev.n_unfilled > 0 ? `, ${ev.n_unfilled} entries UNFILLED (constraints exhausted the pool)` : ''}`
    }
    case 'topn_contest_done': {
      const ev = e as unknown as {
        contest_id: string; k: number; n_filled: number; n_relaxations: number; n_wave_resets: number
        elapsed_s: number; sim_lap: number; sim_lap_used_pct: number; sim_total_taken: number
        n_sims_total: number; n_sims_g: number; worlds_claimed: number; worlds_claimed_pct: number
        n_generated_picks: number
      }
      return `${ev.contest_id}: ${ev.n_filled} / ${ev.k} filled${ev.n_relaxations > 0 ? `, ${ev.n_relaxations} relaxations` : ''}`
        + `${ev.n_wave_resets > 0 ? `, ${ev.n_wave_resets} wave reset${ev.n_wave_resets === 1 ? '' : 's'}` : ''}`
        + `${ev.n_generated_picks > 0 ? `, ${ev.n_generated_picks} generated pick${ev.n_generated_picks === 1 ? '' : 's'}` : ''} `
        + `(${ev.elapsed_s.toFixed(1)}s) — used ${ev.n_sims_g.toLocaleString()} of ${ev.n_sims_total.toLocaleString()} `
        + `sims, claimed ${ev.worlds_claimed.toLocaleString()} / ${ev.n_sims_g.toLocaleString()} `
        + `(${ev.worlds_claimed_pct.toFixed(0)}%)`
    }
    case 'complete': {
      const ev = e as unknown as { n_lineups: number }
      return `${ev.n_lineups} lineups built`
    }
    case 'error': {
      const ev = e as unknown as { message: string }
      return ev.message
    }
    default:
      return ''
  }
}
