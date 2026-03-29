import { useMemo, useState } from 'react'
import type { AppConfig } from '../types'
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
  const isDirty = JSON.stringify(draft) !== configJson

  const set = (section: keyof AppConfig, key: string, value: unknown) => {
    setDraft(d => ({
      ...d,
      [section]: { ...(d[section] as object), [key]: value },
    }))
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
      <div className="config-form-grid">
        <div>
          <section>
            <h3>Simulation</h3>
            <FieldRow label="Simulations (n_sims)">
              <input type="number" min={1000} step={1000} value={draft.simulation.n_sims}
                onChange={e => set('simulation', 'n_sims', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
          </section>

          <section>
            <h3>Optimizer</h3>
            <FieldRow label="Objective">
              <select value={draft.optimizer.objective}
                onChange={e => set('optimizer', 'objective', e.target.value)} disabled={disabled}>
                <option value="expected_surplus">Expected Surplus</option>
                <option value="p_hit">Probability Hit</option>
                <option value="marginal_payout">Marginal Payout</option>
              </select>
            </FieldRow>
            {draft.optimizer.objective === 'marginal_payout' && (
              <FieldRow label="Payout beta (blank = auto)">
                <input type="number" step={0.1} min={1} max={5} value={str(draft.optimizer.payout_beta)}
                  onChange={e => set('optimizer', 'payout_beta', num(e.target.value))} disabled={disabled} />
              </FieldRow>
            )}
            <FieldRow label="Chains (n_chains)">
              <input type="number" min={1} value={draft.optimizer.n_chains}
                onChange={e => set('optimizer', 'n_chains', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Temperature">
              <input type="number" step={0.0001} value={draft.optimizer.temperature}
                onChange={e => set('optimizer', 'temperature', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Steps per chain">
              <input type="number" min={1} value={draft.optimizer.n_steps}
                onChange={e => set('optimizer', 'n_steps', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Niter success">
              <input type="number" min={1} value={draft.optimizer.niter_success}
                onChange={e => set('optimizer', 'niter_success', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Workers">
              <input type="number" min={1} value={draft.optimizer.n_workers}
                onChange={e => set('optimizer', 'n_workers', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
          </section>
        </div>

        <div>
          <section>
            <h3>Optimizer (cont.)</h3>
            <FieldRow label="Early stop window">
              <input type="number" min={1} value={draft.optimizer.early_stopping_window}
                onChange={e => set('optimizer', 'early_stopping_window', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Early stop threshold">
              <input type="number" step={0.0001} value={draft.optimizer.early_stopping_threshold}
                onChange={e => set('optimizer', 'early_stopping_threshold', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Salary floor ($)">
              <input type="number" step={500} value={str(draft.optimizer.salary_floor)}
                onChange={e => set('optimizer', 'salary_floor', num(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="RNG seed (blank = random)">
              <input type="number" value={str(draft.optimizer.rng_seed)}
                onChange={e => set('optimizer', 'rng_seed', num(e.target.value))} disabled={disabled} />
            </FieldRow>
          </section>

          <section>
            <h3>Portfolio</h3>
            <FieldRow label="Portfolio size">
              <input type="number" min={1} value={draft.portfolio.size}
                onChange={e => set('portfolio', 'size', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Target percentile">
              <input type="number" min={1} max={99} value={draft.portfolio.target_percentile}
                onChange={e => set('portfolio', 'target_percentile', Number(e.target.value))} disabled={disabled} />
            </FieldRow>
            <FieldRow label="Target score (blank = auto)">
              <input type="number" step={0.5} value={str(draft.portfolio.target_score)}
                onChange={e => set('portfolio', 'target_score', num(e.target.value))} disabled={disabled} />
            </FieldRow>
          </section>
        </div>
      </div>

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
    </form>
  )
}
