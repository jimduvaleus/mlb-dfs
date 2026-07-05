import { useMemo, useState, useEffect, useRef } from 'react'
import type { AppConfig, GppConfig, PlatformType } from '../types'
import { saveConfig } from '../api'

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

  // Sync draft when the external config changes (e.g. after a reselect save).
  // Only resets if the user has no unsaved edits.
  useEffect(() => {
    if (!isDirty) {
      setDraft(config)
      lastSyncedJson.current = configJson
    }
  }, [configJson])

  const setGpp = (key: keyof GppConfig, value: unknown) => {
    setDraft(d => ({ ...d, gpp: { ...d.gpp, [key]: value } }))
    setSaved(false)
  }

  const set = (section: keyof AppConfig, key: string, value: unknown) => {
    setDraft(d => ({
      ...d,
      [section]: { ...(d[section] as object), [key]: value },
    }))
    setSaved(false)
  }

  const handlePlatformChange = (p: PlatformType) => {
    setDraft(d => {
      const floor = d.optimizer.salary_floor
      // Auto-adjust salary floor when the current value is invalid or clearly wrong for the target platform
      let newFloor = floor
      if (p === 'fanduel' && floor != null && floor > 35000) {
        newFloor = 30000
      } else if (p === 'draftkings' && floor != null && floor <= 35000) {
        newFloor = 48500
      }
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
      const saved = await saveConfig(draft)
      lastSyncedJson.current = JSON.stringify(saved)
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
              </select>
            </FieldRow>
          </section>

          <section>
            <h3>Optimizer</h3>
            <FieldRow label="Salary floor ($)">
              <input type="number" step={500} value={str(draft.optimizer.salary_floor)}
                onChange={e => set('optimizer', 'salary_floor', num(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Min pitcher value (blank = off)">
              <input type="number" step={0.1} min={0} value={str(draft.optimizer.min_pitcher_value)}
                onChange={e => set('optimizer', 'min_pitcher_value', num(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Min batter value (blank = off)">
              <input type="number" step={0.1} min={0} value={str(draft.optimizer.min_batter_value)}
                onChange={e => set('optimizer', 'min_batter_value', num(e.target.value))} disabled={disabled} />
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
            <FieldRow label="Candidates">
              <input type="number" min={100} value={draft.gpp.n_candidates}
                onChange={e => setGpp('n_candidates', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Field lineups">
              <input type="number" min={100} value={draft.gpp.n_field_lineups}
                onChange={e => setGpp('n_field_lineups', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Field samples">
              <input type="number" min={1} max={10} value={draft.gpp.n_field_samples}
                onChange={e => setGpp('n_field_samples', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Candidate floor relief ($)">
              <input type="number" step={500} min={0} max={10000} value={draft.gpp.candidate_floor_relief}
                onChange={e => setGpp('candidate_floor_relief', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Refine rounds (0 = off)">
              <input type="number" step={1} min={0} max={10} value={draft.gpp.refine_rounds ?? 2}
                onChange={e => setGpp('refine_rounds', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Refine parents per round">
              <input type="number" step={10} min={10} value={draft.gpp.refine_top ?? 150}
                onChange={e => setGpp('refine_top', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Mutants per parent">
              <input type="number" step={1} min={1} max={50} value={draft.gpp.refine_mutants ?? 8}
                onChange={e => setGpp('refine_mutants', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Final rescore samples (0 = off)">
              <input type="number" step={1} min={0} max={20} value={draft.gpp.final_n_field_samples ?? 5}
                onChange={e => setGpp('final_n_field_samples', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Rescore pool (top M by EV)">
              <input type="number" step={100} min={100} value={draft.gpp.final_rescore_top ?? 2000}
                onChange={e => setGpp('final_rescore_top', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Dupe penalty">
              <select value={(draft.gpp.dupe_penalty ?? false) ? 'on' : 'off'}
                onChange={e => setGpp('dupe_penalty', e.target.value === 'on')} disabled={disabled}>
                <option value="off">Off</option>
                <option value="on">On</option>
              </select>
            </FieldRow>
            {(draft.gpp.dupe_penalty ?? false) && (
              <>
                <FieldRow label="Dupe intercept">
                  <input type="number" step={0.1} value={draft.gpp.dupe_intercept ?? 3.698}
                    onChange={e => setGpp('dupe_intercept', Number(e.target.value))} disabled={disabled} />
                </FieldRow>
                <FieldRow label="Dupe Σlog(own) coef">
                  <input type="number" step={0.01} min={0} value={draft.gpp.dupe_log_own_coef ?? 0.212}
                    onChange={e => setGpp('dupe_log_own_coef', Number(e.target.value))} disabled={disabled} />
                </FieldRow>
                <FieldRow label="Dupe salary coef (per $100 unused)">
                  <input type="number" step={0.01} min={0} value={draft.gpp.dupe_salary_coef ?? 0.089}
                    onChange={e => setGpp('dupe_salary_coef', Number(e.target.value))} disabled={disabled} />
                </FieldRow>
                <FieldRow label="Dupe stack coef">
                  <input type="number" step={0.05} value={draft.gpp.dupe_stack_coef ?? 0.024}
                    onChange={e => setGpp('dupe_stack_coef', Number(e.target.value))} disabled={disabled} />
                </FieldRow>
                <FieldRow label="Dupe min gross payout ($)">
                  <input type="number" step={1} min={0} value={draft.gpp.dupe_min_gross_payout ?? 15}
                    onChange={e => setGpp('dupe_min_gross_payout', Number(e.target.value))} disabled={disabled} />
                </FieldRow>
              </>
            )}
            <FieldRow label="Base EVw (risk 1)">
              <input type="number" step={0.01} min={0} max={1} value={draft.gpp.evw_base ?? 0.10}
                onChange={e => setGpp('evw_base', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Max EVw (risk 5)">
              <input type="number" step={0.01} min={0} max={1} value={draft.gpp.evw_max ?? 0.40}
                onChange={e => setGpp('evw_max', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="EV floor ($)">
              <input type="number" step={0.01} value={draft.gpp.ev_floor ?? 0.20}
                onChange={e => setGpp('ev_floor', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Field source">
              <select value={draft.gpp.field_source ?? 'simulated'}
                onChange={e => setGpp('field_source', e.target.value)} disabled={disabled}>
                <option value="simulated">Simulated</option>
                <option value="historical">Historical</option>
              </select>
            </FieldRow>
            {(draft.gpp.field_source ?? 'simulated') === 'historical' && (
              <FieldRow label="Historical slates (N)">
                <input type="number" step={1} min={1} max={50}
                  value={draft.gpp.historical_n_slates ?? 10}
                  onChange={e => setGpp('historical_n_slates', Number(e.target.value))}
                  disabled={disabled} />
              </FieldRow>
            )}
          </section>
        </div>
      </div>

    </form>
  )
}
