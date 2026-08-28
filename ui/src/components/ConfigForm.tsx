import { useMemo, useState, useEffect, useRef } from 'react'
import type { AppConfig, GppConfig, PlatformType } from '../types'
import { fetchConfig, saveConfig } from '../api'

interface Props {
  config: AppConfig
  onSaved: (cfg: AppConfig) => void
  disabled?: boolean
}

function FieldRow({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="field-row">
      <label>{label}</label>
      {children}
    </div>
  )
}

export function ConfigForm({ config, onSaved, disabled }: Props) {
  const [draft, setDraft] = useState<AppConfig>(config)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  const configJson = useMemo(() => JSON.stringify(config), [config])
  // Track the config snapshot the draft was last synced from, so isDirty
  // measures user edits relative to where we started — not the incoming value.
  const lastSyncedJson = useRef(configJson)
  const isDirty = JSON.stringify(draft) !== lastSyncedJson.current

  // `draft` is cloned from `config` once (on mount / next clean sync) and this
  // form only ever renders inputs for a subset of AppConfig/GppConfig fields.
  // Every other field just rides along in `draft` at whatever value it had
  // at that snapshot. If we POSTed `draft` wholesale, saving a change to any
  // rendered field would silently overwrite every un-rendered field (e.g.
  // gpp.external_pool_ceiling_weight, which has no UI control here) back to
  // that stale snapshot value — clobbering any out-of-band edit (direct
  // config.yaml edit, another tab) made after this page loaded. So we track
  // exactly which "section.key" paths the user actually touched, and at
  // submit time apply only those onto a freshly-fetched server config.
  const touchedPaths = useRef<Set<string>>(new Set())

  // Sync draft when the external config changes (e.g. after a reselect save).
  // Only resets if the user has no unsaved edits.
  useEffect(() => {
    if (!isDirty) {
      setDraft(config)
      lastSyncedJson.current = configJson
    }
  }, [configJson])

  const setGpp = (key: keyof GppConfig, value: unknown) => {
    touchedPaths.current.add(`gpp.${key}`)
    setDraft(d => ({ ...d, gpp: { ...d.gpp, [key]: value } }))
    setSaved(false)
  }

  const set = (section: keyof AppConfig, key: string, value: unknown) => {
    touchedPaths.current.add(`${section}.${key}`)
    setDraft(d => ({
      ...d,
      [section]: { ...(d[section] as object), [key]: value },
    }))
    setSaved(false)
  }

  const handlePlatformChange = (p: PlatformType) => {
    touchedPaths.current.add('platform')
    setDraft(d => {
      const floor = d.optimizer.salary_floor
      // Auto-adjust salary floor when the current value is invalid or clearly wrong for the target platform
      let newFloor = floor
      if (p === 'fanduel' && floor != null && floor > 35000) {
        newFloor = 30000
      } else if (p === 'draftkings' && floor != null && floor <= 35000) {
        newFloor = 48500
      }
      if (newFloor !== floor) touchedPaths.current.add('optimizer.salary_floor')
      return { ...d, platform: p, optimizer: { ...d.optimizer, salary_floor: newFloor } }
    })
    setSaved(false)
  }

  const str = (v: unknown) => (v == null ? '' : String(v))
  const num = (v: unknown) => (v == null || v === '' ? null : Number(v))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError(null)
    try {
      // Base the payload on the server's current config, not the possibly
      // stale `draft` clone, then layer in only the fields the user actually
      // edited through this form. See the touchedPaths comment above.
      const fresh = await fetchConfig()
      const payload: AppConfig = JSON.parse(JSON.stringify(fresh))
      for (const path of touchedPaths.current) {
        const [section, key] = path.split('.') as [keyof AppConfig, string | undefined]
        if (key === undefined) {
          ;(payload as unknown as Record<string, unknown>)[section] =
            (draft as unknown as Record<string, unknown>)[section]
        } else {
          const draftSection = draft[section] as unknown as Record<string, unknown>
          const payloadSection = payload[section] as unknown as Record<string, unknown>
          payloadSection[key] = draftSection[key]
        }
      }
      const saved = await saveConfig(payload)
      touchedPaths.current.clear()
      lastSyncedJson.current = JSON.stringify(saved)
      setDraft(saved)
      onSaved(saved)
      setSaved(true)
    } catch (err) {
      setError(String(err))
    } finally {
      setSaving(false)
    }
  }

  // The line-2 frontier generator is shared by the two per-contest shapes:
  // MARGINAL REWARD merges its lineups into one global greedy, PER-CONTEST
  // merges them into the pool each contest picks its own slice from. Both
  // read the same `marginal_reward:` keys, so the controls are defined once
  // and rendered under either mode rather than duplicated and left to drift.
  const frontierControls = (
    <>
            <FieldRow label="Generate along the mean-variance frontier">
              <input type="checkbox"
                checked={draft.marginal_reward?.frontier_enabled ?? false}
                onChange={e => set('marginal_reward', 'frontier_enabled', e.target.checked)}
                disabled={disabled} />
            </FieldRow>
            <p className="field-hint">
              Haugh &amp; Singal's line 2 — sweep <code>w'μ + λ(w'Σw − 2w'σ_dG)</code> and
              add the results to the candidate pool. The λ-term is the variance of your
              margin over the payout cutoff, so the sweep runs from the plain projection
              lineup out to the highest-variance one. Marginal reward cannot select a
              lineup that is not in the pool, and the SaberSim pool measured as not
              spanning this region on 9 of 9 slates. Costs CP-SAT solve time and changes
              the pool, so it is off by default.
            </p>
            {draft.marginal_reward?.frontier_enabled && (
              <>
                <FieldRow label="λ search grid size">
                  <input type="number" step={1} min={2} max={40}
                    value={draft.marginal_reward?.frontier_n_lambdas ?? 6}
                    onChange={e => set('marginal_reward', 'frontier_n_lambdas', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Target generated lineups (total)">
                  <input type="number" step={500} min={500} max={20000}
                    value={draft.marginal_reward?.frontier_target_lineups ?? 8000}
                    onChange={e => set('marginal_reward', 'frontier_target_lineups', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Min lineups per team">
                  <input type="number" step={1} min={1} max={50}
                    value={draft.marginal_reward?.frontier_min_per_team ?? 4}
                    onChange={e => set('marginal_reward', 'frontier_min_per_team', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Candidates sampled">
                  <input type="number" step={5000} min={5000} max={200000}
                    value={draft.marginal_reward?.frontier_sample_n ?? 30000}
                    onChange={e => set('marginal_reward', 'frontier_sample_n', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Exact solver anchors (0 = none)">
                  <input type="number" step={1} min={0} max={12}
                    value={draft.marginal_reward?.frontier_n_anchors ?? 2}
                    onChange={e => set('marginal_reward', 'frontier_n_anchors', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Mutation generations">
                  <input type="number" step={1} min={0} max={12}
                    value={draft.marginal_reward?.frontier_n_generations ?? 5}
                    onChange={e => set('marginal_reward', 'frontier_n_generations', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Mutants per parent">
                  <input type="number" step={1} min={1} max={40}
                    value={draft.marginal_reward?.frontier_mutants_per_parent ?? 6}
                    onChange={e => set('marginal_reward', 'frontier_mutants_per_parent', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Generated lineup salary floor">
                  <input type="number" step={100} min={0} max={50000}
                    value={draft.marginal_reward?.frontier_salary_floor ?? 47500}
                    onChange={e => set('marginal_reward', 'frontier_salary_floor', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Set this to the salary floor SaberSim was given, so generated lineups
                  sit in the same salary regime as the external pool they're merged into
                  and compete with. It is deliberately separate from the optimizer's own
                  salary floor, which is a holdover and higher. 0 disables it — but with
                  no floor the sampler builds lineups leaving thousands unspent, a shape
                  no other stage of the funnel produces.
                </p>
                <FieldRow label="Solver timeout per lineup (s)">
                  <input type="number" step="any" min={0.5}
                    value={draft.marginal_reward?.frontier_solver_timeout_s ?? 8}
                    onChange={e => set('marginal_reward', 'frontier_solver_timeout_s', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Candidates are sampled from the team-round-robin generator and ranked
                  by the exact objective. λ is then chosen <strong>per contest</strong> by
                  the paper's line 4 — the λ maximising expected top-heavy payout against
                  that contest's own simulated field and payout table. Small fields come
                  out with higher-projection, less contrarian builds and large fields with
                  more extreme ones, because that is what the payout maths says rather
                  than anything hand-tuned. The grid size below is only the search range
                  line 4 picks from. <strong>Lineups per team</strong> bounds any one
                  team's share: near its optimum the objective barely separates stacks, so
                  without a cap the pool collapses onto one.
                  Frontier lineups skip the ceiling floor and the 9/10 near-duplicate
                  cull (mutants differ from their parent by one player, which is what
                  that cull targets), so they still have to earn a slot on marginal
                  dollars alone.
                </p>
              </>
            )}
    </>
  )

  return (
    <form className="config-form" onSubmit={handleSubmit}>
      <div className="config-form-footer">
        <button
          type="submit"
          className={isDirty ? 'btn-dirty' : 'btn-clean'}
          disabled={disabled || saving || (!isDirty && !saving)}
        >
          {saving ? 'Saving…' : 'Save Config'}
        </button>
        {error && <span className="error">{error}</span>}
        {saved && !isDirty && <span className="success">Saved.</span>}
      </div>
      <div className="config-form-grid">
        <div>
          <section>
            <h3>Platform</h3>
            <FieldRow label="Platform">
              <select value={draft.platform}
                onChange={e => handlePlatformChange(e.target.value as PlatformType)} disabled={disabled}>
                <option value="draftkings">DraftKings</option>
                <option value="fanduel">FanDuel</option>
              </select>
            </FieldRow>
            <FieldRow label={draft.platform === 'fanduel' ? 'FD Slate CSV' : 'DK Slate CSV'}>
              {draft.platform === 'fanduel' ? (
                <input type="text" value={draft.paths.fd_slate ?? ''}
                  onChange={e => set('paths', 'fd_slate', e.target.value)} disabled={disabled}
                  placeholder="data/raw/FanDuel-MLB-….csv" />
              ) : (
                <input type="text" value={draft.paths.dk_slate ?? ''}
                  onChange={e => set('paths', 'dk_slate', e.target.value)} disabled={disabled}
                  placeholder="data/raw/DKSalaries.csv" />
              )}
            </FieldRow>
          </section>

          <section>
            <h3>Projections</h3>
            <FieldRow label="Source">
              <select value={draft.paths.projections_source}
                onChange={e => set('paths', 'projections_source', e.target.value)} disabled={disabled}>
                <option value="rotowire">RotoWire</option>
                <option value="dailyfantasyfuel">Daily Fantasy Fuel</option>
                <option value="market_odds">Market Odds</option>
                <option value="sabersim">SaberSim</option>
              </select>
            </FieldRow>
          </section>

          <section>
            <h3>Optimizer</h3>
            <FieldRow label="Salary floor ($)">
              <input type="number" step={500} value={str(draft.optimizer.salary_floor)}
                onChange={e => set('optimizer', 'salary_floor', num(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="RNG seed (blank = random)">
              <input type="number" value={str(draft.optimizer.rng_seed)}
                onChange={e => set('optimizer', 'rng_seed', num(e.target.value))} disabled={disabled} />
            </FieldRow>
          </section>
        </div>

        <div>
          <section>
            <h3>Simulation</h3>
            <FieldRow label="Simulations (n_sims)">
              <input type="number" min={1000} step={1000} value={draft.simulation.n_sims}
                onChange={e => set('simulation', 'n_sims', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
          </section>

          <section>
            <h3>GPP</h3>
            <FieldRow label="Base EVw (risk 1)">
              <input type="number" step={0.01} min={0} max={1} value={draft.gpp.evw_base ?? 0.10}
                onChange={e => setGpp('evw_base', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Max EVw (risk 5)">
              <input type="number" step={0.01} min={0} max={1} value={draft.gpp.evw_max ?? 0.40}
                onChange={e => setGpp('evw_max', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            {(draft.gpp.field_source ?? 'simulated') === 'historical' && (
              <FieldRow label="Historical slates (N)">
                <input type="number" step={1} min={1} max={50}
                  value={draft.gpp.historical_n_slates ?? 10}
                  onChange={e => setGpp('historical_n_slates', Number(e.target.value))}
                  disabled={disabled} />
              </FieldRow>
            )}
            {(() => {
              // selector_mode is a comma list of arms. Rendering it as
              // checkboxes rather than a fixed set of combo options keeps every
              // combination reachable without enumerating 2^n of them. The four
              // playable arms are always shown; "det" appears only when it is
              // already selected, so a config that has it can still turn it off
              // instead of having it silently dropped by the first toggle.
              const ARMS: { id: string; name: string; hint: string }[] = [
                { id: 'kelly', name: 'Kelly', hint: 'log bankroll growth · no upstream cost' },
                { id: 'dr', name: 'dR', hint: 'exact demotion · holds the sorted field (~0.6–1.2 GB)' },
                { id: 'emax', name: 'E[max]', hint: 'best entry hits · no upstream cost' },
                { id: 'coverage', name: 'Coverage', hint: 'world coverage · ~69 MB packed bits · contest-blind' },
              ]
              const raw = draft.gpp.selector_mode ?? 'kelly,dr'
              const sel = new Set(
                raw.toLowerCase() === 'all'
                  ? ['det', 'kelly', 'coverage', 'emax', 'dr']
                  : raw.split(',').map(x => x.trim()).filter(Boolean)
              )
              const shown = sel.has('det')
                ? [...ARMS, { id: 'det', name: 'Determinant (legacy)', hint: 'correlation proxy · 9–13 pts behind the others' }]
                : ARMS
              const toggle = (id: string, on: boolean) => {
                const next = new Set(sel)
                if (on) next.add(id); else next.delete(id)
                // Never write an empty list: the pipeline would fall back to
                // Determinant, which is the one arm this UI exists to retire.
                setGpp('selector_mode', next.size ? [...next].join(',') : 'kelly')
              }
              return (
                <>
                  <div className="field-row"><label>Selection arms</label></div>
                  {shown.map(a => (
                    <FieldRow key={a.id} label={a.name}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input type="checkbox" checked={sel.has(a.id)}
                          onChange={e => toggle(a.id, e.target.checked)}
                          disabled={disabled} />
                        <span style={{ fontSize: '0.85em', color: 'var(--color-text-muted)' }}>
                          {a.hint}
                        </span>
                      </span>
                    </FieldRow>
                  ))}
                  <p className="field-hint">
                    Each arm prices self-competition between your own entries a different
                    way, and each builds a full portfolio you can compare in the Portfolio
                    tab before making one active. Selection itself is seconds next to
                    generation and scoring — the cost that varies is upstream, noted per
                    arm above. Kelly and dR measured strongest across four contests and
                    are the default; coverage ties them on large fields but does not react
                    to contest size at all, so treat it with care on small ones.
                  </p>
                </>
              )
            })()}
            <FieldRow label="Per-contest shortlist">
              <input type="number" step={100} min={0}
                value={draft.gpp.per_contest_shortlist ?? 8000}
                onChange={e => setGpp('per_contest_shortlist', Number(e.target.value))}
                disabled={disabled} />
            </FieldRow>
            <FieldRow label="Per-contest candidates per entry (0 = no cap)">
              <input type="number" step={50} min={0}
                value={draft.gpp.per_contest_cand_per_entry ?? 400}
                onChange={e => setGpp('per_contest_cand_per_entry', Number(e.target.value))}
                disabled={disabled} />
            </FieldRow>
            <p className="field-hint">
              Caps how many candidates each contest chooses from, as a rate per
              purchased entry (floored at 2,000). A 2-entry contest otherwise pays
              the same per-candidate cost as a 60-entry one — measured at 22% of the
              selection stage to pick six lineups. Safe there because dR's first pick
              is exactly the highest-EV lineup and each later one subtracts a demotion
              term the earlier picks create, so at two entries there is almost nothing
              to price. At 400 this is a no-op above roughly 20 entries, which is
              deliberate: with many slots to fill that demotion term is doing real work
              on the contrarian tail an EV ranking cuts first. Unlike the other speed
              knobs, this changes which lineups get picked. It is forced to 0
              automatically unless dR or Determinant is among your selection arms —
              they are the only two whose cost grows faster than linearly in the
              candidate count, so for Kelly, coverage or E[max] the cap costs picks
              and saves nothing.
            </p>
            <FieldRow label="Per-contest field samples">
              <input type="number" step={1} min={1}
                value={draft.gpp.per_contest_field_samples ?? 1}
                onChange={e => setGpp('per_contest_field_samples', Number(e.target.value))}
                disabled={disabled} />
            </FieldRow>
            <FieldRow label="Disjoint contest slices">
              <input type="checkbox"
                checked={draft.gpp.per_contest_disjoint ?? true}
                onChange={e => setGpp('per_contest_disjoint', e.target.checked)}
                disabled={disabled} />
            </FieldRow>
            <p className="field-hint">
              When the slate has entry files, each contest in them selects its own
              slice against its own ladder and field size, largest top prize first,
              instead of one portfolio being spread across all of them. The contest
              above still sets the reference ladder the funnel (EV floor, fresh
              re-score) is denominated in. Shortlist is how many survivors the
              per-contest selectors see — everything downstream is shortlist × sims.
              Field samples multiply both the field build and a per-contest sort, so
              raise it only to damp field noise when validating. Disjoint slices stop
              the same lineup being entered in two contests, which concentrates risk
              without diversifying it.
            </p>
            <details className="config-collapse">
              <summary>
                External candidate pool (SaberSim)
                <span className="config-collapse-sub">
                  {' '}— only applies when the run dialog's external-pool box is ticked;
                  currently <code>{draft.gpp.external_pool_ev_type ?? 'roi'}</code>
                </span>
              </summary>
            <FieldRow label="External pool EV type">
              <select value={draft.gpp.external_pool_ev_type ?? 'roi'}
                onChange={e => setGpp('external_pool_ev_type', e.target.value)} disabled={disabled}>
                <option value="roi">ROI</option>
                <option value="prj_own">PRJ/OWN</option>
                <option value="p_win">P(WIN)</option>
                <option value="proj_top">PROJ TOP</option>
                <option value="self_play">SELF-PLAY</option>
                <option value="topn_coverage">TOP-N COVERAGE</option>
                <option value="marginal_reward">MARGINAL REWARD</option>
                <option value="per_contest">PER-CONTEST (kelly/dR/E[max])</option>
              </select>
            </FieldRow>
            {draft.gpp.external_pool_ev_type === 'marginal_reward' && (
              <>
                <p className="field-hint">
                  Fills every purchased slot by one global greedy over (lineup, contest)
                  pairs, ranked by the marginal expected dollars each would add — with our
                  OWN entries inside the ranking, so a near-duplicate of something already
                  picked is penalised because it cannot also take first place. Contest
                  routing falls out of the objective instead of a fixed fee/prize-pool sort.
                  Produces a single portfolio: risk is an EVw dial belonging to the Det
                  selector, and this objective has no such knob.
                </p>
                <p className="field-hint">
                  Not yet validated against production — the archive read it at production's
                  top-1% level and it was the only negative dollar arm, on a single seed.
                  Treat it as the A/B challenger it is, not a replacement.
                </p>
                <FieldRow label="Max overlap within a contest (γ_in)">
                  <input type="number" step={1} min={1} max={10}
                    value={draft.marginal_reward?.gamma_in ?? 7}
                    onChange={e => set('marginal_reward', 'gamma_in', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Max overlap across contests (γ_out)">
                  <input type="number" step={1} min={1} max={10}
                    value={draft.marginal_reward?.gamma_out ?? 8}
                    onChange={e => set('marginal_reward', 'gamma_out', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  γ_in is the EV rule — entries in the same contest are the only ones that
                  compete. γ_out is bankroll-variance control only, and 8 is a no-op against
                  a pool already through the 9/10 near-duplicate cull.
                </p>
                <FieldRow label="Allow the same lineup in two contests">
                  <input type="checkbox"
                    checked={draft.marginal_reward?.allow_cross_contest_duplicates ?? false}
                    onChange={e => set('marginal_reward', 'allow_cross_contest_duplicates', e.target.checked)}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Smoothing width (0 = exact estimator)">
                  <input type="number" step="any" min={0}
                    value={draft.marginal_reward?.smooth_tau_scale ?? 0}
                    onChange={e => set('marginal_reward', 'smooth_tau_scale', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Opponent field pool size">
                  <input type="number" step={1000} min={1000}
                    value={draft.marginal_reward?.field_pool_size ?? 25000}
                    onChange={e => set('marginal_reward', 'field_pool_size', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Max sim worlds per contest">
                  <input type="number" step={500} min={500}
                    value={draft.marginal_reward?.max_sims_per_contest ?? 12500}
                    onChange={e => set('marginal_reward', 'max_sims_per_contest', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Every contest's state is held in memory at once (the greedy compares across
                  them), so this caps the world axis rather than letting n_sims multiply by
                  the contest count.
                </p>
                {frontierControls}
              </>
            )}
            {draft.gpp.external_pool_ev_type === 'per_contest' && (
              <>
              <p className="field-hint">
                Runs the imported pool through the same per-contest selection the
                generated pool uses: each contest on the entries files picks its own
                slice against its own payout ladder and field size, largest top prize
                first, with no lineup entered twice. The objectives come from
                <strong> Selection arms</strong> above (Kelly / dR / E[max] / coverage),
                and each arm builds a full portfolio you can compare before making one
                active. It differs from MARGINAL REWARD in shape rather than objective —
                that one runs a single global greedy over (lineup, contest) pairs, this
                fills one contest at a time. Shortlist size and field draws per contest
                are the <code>per_contest_*</code> keys in config.yaml.
              </p>
              <p className="field-hint">
                An imported pool is a menu somebody else wrote, and the arms can only
                pick what is on it — a real run filled 110 slots out of 4,219 SaberSim
                lineups, a 2.6% cut where the generated path cuts 0.4%. Turning the
                frontier on below adds generated candidates to that pool rather than
                replacing any of it: your own lineups are all still there and still
                eligible, they just stop being the only thing on offer.
              </p>
              {frontierControls}
              </>
            )}
            {draft.gpp.external_pool_ev_type === 'proj_top' && (
              <>
                <p className="field-hint">
                  Ranks purely on projected mean — best backtested currency for catching a
                  slate's own top-10-real-score lineups, but produces a concentrated portfolio
                  (few distinct teams, high single-player exposure). Use risk 1-5 to blend
                  diversity back in.
                </p>
                <FieldRow label="Ceiling tiers (rank medium/large fields on sim p95/p99)">
                  <input type="checkbox"
                    checked={draft.gpp.external_pool_proj_top_ceiling_tiers ?? false}
                    onChange={e => setGpp('external_pool_proj_top_ceiling_tiers', e.target.checked)}
                    disabled={disabled} />
                </FieldRow>
                {draft.gpp.external_pool_proj_top_ceiling_tiers && (
                  <FieldRow label="Medium/large field boundary (implied entries)">
                    <input type="number" step={1000} min={5000}
                      value={draft.gpp.external_pool_proj_top_medium_large_boundary ?? 15000}
                      onChange={e => setGpp('external_pool_proj_top_medium_large_boundary', Number(e.target.value))}
                      disabled={disabled} />
                  </FieldRow>
                )}
                <p className="field-hint">
                  Below 5,000 implied entries, proj_top always ranks on mean projected score.
                  With ceiling tiers on, fields from 5,000 up to the boundary rank on each
                  lineup's simulated 95th-percentile score, and fields at/above the boundary on
                  the 99th percentile — validated on 10 archived slates (positive drop_max).
                </p>
              </>
            )}
            {draft.gpp.external_pool_ev_type === 'prj_own' && (
              <FieldRow label="Ownership scale (entries)">
                <input type="number" step={1000} min={1000}
                  value={draft.gpp.external_pool_own_scale ?? 30000}
                  onChange={e => setGpp('external_pool_own_scale', Number(e.target.value))}
                  disabled={disabled} />
              </FieldRow>
            )}
            {draft.gpp.external_pool_ev_type === 'p_win' && (
              <>
                <FieldRow label="P(win) sharpness (× entry count)">
                  <input type="number" step={0.05} min={0}
                    value={draft.gpp.external_pool_pwin_sharpness ?? 1.0}
                    onChange={e => setGpp('external_pool_pwin_sharpness', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="P(win) flat exponent ref (0 = per-contest)">
                  <input type="number" step={1000} min={0}
                    value={draft.gpp.external_pool_pwin_flat_reference ?? 10000}
                    onChange={e => setGpp('external_pool_pwin_flat_reference', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="P(win) stage-A admit N">
                  <input type="number" step={100} min={0}
                    value={draft.gpp.external_pool_pwin_admit_n ?? 2000}
                    onChange={e => setGpp('external_pool_pwin_admit_n', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="P(win) admit scaling (× entries)">
                  <input type="number" step={1} min={0}
                    value={draft.gpp.external_pool_pwin_admit_multiplier ?? 12.0}
                    onChange={e => setGpp('external_pool_pwin_admit_multiplier', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
              </>
            )}
            {draft.gpp.external_pool_ev_type === 'self_play' && (
              <>
                <p className="field-hint">
                  Fills each contest by iterative best-response against opponents plus its own
                  prior picks, using the contest's real DK payout table every round — diversity
                  is a byproduct of the round loop, not a separate mechanism, so this produces a
                  single portfolio (no risk sweep). Materially slower than the other EV types:
                  offline timing on archived slates ran ~5-9 minutes per slate, not seconds.
                  Contests whose name isn't one of DK's ~14 known recurring types use an
                  approximate closest-size payout table — watch the progress panel for a warning
                  naming any affected contest.
                </p>
                <FieldRow label="Round-loop sims (per pick)">
                  <input type="number" step={500} min={500}
                    value={draft.gpp.external_pool_self_play_round_n_sims ?? 2000}
                    onChange={e => setGpp('external_pool_self_play_round_n_sims', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Precision-refinement sims (0 = disable)">
                  <input type="number" step={1000} min={0}
                    value={draft.gpp.external_pool_self_play_precise_n_sims ?? 20000}
                    onChange={e => setGpp('external_pool_self_play_precise_n_sims', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Shortlist size (candidates re-scored per round)">
                  <input type="number" step={100} min={100}
                    value={draft.gpp.external_pool_self_play_shortlist_size ?? 1000}
                    onChange={e => setGpp('external_pool_self_play_shortlist_size', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
              </>
            )}
            {draft.gpp.external_pool_ev_type === 'topn_coverage' && (
              <>
                <p className="field-hint">
                  Fills each contest by greedily picking whichever remaining candidate would have
                  finished top-N most often against a sub-sampled opponent field, then removes the
                  simulated worlds a pick "claimed" so later picks have to prove themselves
                  elsewhere — diversity is a byproduct of the coverage race itself, so this
                  produces a single portfolio (no risk sweep).
                </p>
                <FieldRow label="Top-N rank (floor)">
                  <input type="number" step={1} min={1}
                    value={draft.gpp.external_pool_topn_rank ?? 10}
                    onChange={e => setGpp('external_pool_topn_rank', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Percentile floor for large fields (0 = off)">
                  <input type="number" step="any" min={0} max={1}
                    value={draft.gpp.external_pool_topn_percentile_floor ?? 0.001}
                    onChange={e => setGpp('external_pool_topn_percentile_floor', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Effective rank per contest = max(Top-N rank, ceil(percentile floor × field size)),
                  clipped to field size. 0.001 ("top 0.1%") makes a 17,000-entry field effectively
                  top-17 instead of a literal top-10, while fields under ~10,000 entries stay at the
                  flat Top-N rank — keeps the bar's real difficulty comparable across contest sizes.
                </p>
                <FieldRow label="Field pool size">
                  <input type="number" step={1000} min={1000}
                    value={draft.gpp.external_pool_topn_field_pool_size ?? 25000}
                    onChange={e => setGpp('external_pool_topn_field_pool_size', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Field draws per contest (K)">
                  <input type="number" step={1} min={1}
                    value={draft.gpp.external_pool_topn_field_samples ?? 5}
                    onChange={e => setGpp('external_pool_topn_field_samples', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Generated candidates to add (0 = off)">
                  <input type="number" step={500} min={0}
                    value={draft.gpp.external_pool_topn_generated_pool_size ?? 0}
                    onChange={e => setGpp('external_pool_topn_generated_pool_size', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Adds this many extra candidates from the same stacked-lineup generator the
                  opponent field uses, merged into the real external pool after 9/10-overlap dedup
                  (a real lineup always wins a conflict). Lets the selector pick a high-performing
                  lineup that's visible in the simulated field but wasn't in the real export.
                  Unvalidated — off by default.
                </p>
                <FieldRow label="Leverage weight for generated candidates (0 = ownership-only, 1 = pure leverage)">
                  <input type="number" step={0.05} min={0} max={1}
                    value={draft.gpp.external_pool_topn_generated_leverage_weight ?? 0}
                    onChange={e => setGpp('external_pool_topn_generated_leverage_weight', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Blends the generated candidates' sampling ownership toward game-theoretic
                  "optimal ownership" (players whose true edge exceeds their projected ownership)
                  instead of plain projected ownership, biasing extra candidates toward genuine
                  leverage rather than mimicking the field. 0 = today's behavior (unaffected).
                  Unvalidated as a generation-time bias — off by default; only has any effect when
                  "Generated candidates to add" above is greater than 0.
                </p>
                <FieldRow label="Sims-needed floor (at reference field size)">
                  <input type="number" step={1} min={0}
                    value={draft.gpp.external_pool_topn_sims_min ?? 4607}
                    onChange={e => setGpp('external_pool_topn_sims_min', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Reference field size (entries)">
                  <input type="number" step={1} min={0}
                    value={draft.gpp.external_pool_topn_sims_reference_field_size ?? 392}
                    onChange={e => setGpp('external_pool_topn_sims_reference_field_size', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <FieldRow label="Sims-vs-field-size power">
                  <input type="number" step="any" min={0}
                    value={draft.gpp.external_pool_topn_sims_power ?? 0.222}
                    onChange={e => setGpp('external_pool_topn_sims_power', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
                <p className="field-hint">
                  Field-size-aware sim budget per contest: n_sims_g = sims-needed-floor ×
                  (field_size / reference_field_size) ^ power, clipped to at least the floor.
                  Calibrated 2026-08-09 against one archived slate — 392-1,189-entry fields needed
                  ~5,000 sims, 5,945-17,835-entry fields needed ~10,000 (see
                  scripts/calibrate_topn_sims_per_contest.py). Set any of the three to 0 to fall
                  back to a flat fraction of total sim worlds instead.
                </p>
                <FieldRow label="Fallback: sim-worlds per contest (fraction of n_sims)">
                  <input type="number" step="any" min={0.01} max={1}
                    value={draft.gpp.external_pool_topn_sims_per_contest_fraction ?? 0.5}
                    onChange={e => setGpp('external_pool_topn_sims_per_contest_fraction', Number(e.target.value))}
                    disabled={disabled} />
                </FieldRow>
              </>
            )}
            </details>
          </section>
        </div>
      </div>

    </form>
  )
}
