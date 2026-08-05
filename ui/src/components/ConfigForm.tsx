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
            <FieldRow label="External pool EV type">
              <select value={draft.gpp.external_pool_ev_type ?? 'roi'}
                onChange={e => setGpp('external_pool_ev_type', e.target.value)} disabled={disabled}>
                <option value="roi">ROI</option>
                <option value="prj_own">PRJ/OWN</option>
                <option value="p_win">P(WIN)</option>
                <option value="proj_top">PROJ TOP</option>
              </select>
            </FieldRow>
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
          </section>
        </div>
      </div>

    </form>
  )
}
