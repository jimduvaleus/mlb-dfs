import { useState } from 'react'
import type { CacheStatus } from '../types'

interface Props {
  cacheStatus: CacheStatus
  onStart: (useCandidates: boolean, useField: boolean, seedOptimal: boolean, seedSimOptimal: boolean, useExternalPool: boolean) => void
  onDismiss: () => void
  fieldSource?: string
}

export function RunOptionsDialog({ cacheStatus, onStart, onDismiss, fieldSource }: Props) {
  const isHistorical = fieldSource === 'historical'
  const [useCandidates, setUseCandidates] = useState(cacheStatus.candidates !== null)
  const [useField, setUseField] = useState(!isHistorical && cacheStatus.field_k !== null)
  const [seedOptimal, setSeedOptimal] = useState(false)
  const [seedSimOptimal, setSeedSimOptimal] = useState(false)
  const [useExternal, setUseExternal] = useState(false)

  const candAvailable = cacheStatus.candidates !== null
  const fieldAvailable = cacheStatus.field_k !== null
  const ext = cacheStatus.external_pool
  const extAvailable = !!ext?.available
  const internalDisabled = useExternal && extAvailable

  const candLabel = candAvailable
    ? `${cacheStatus.candidates!.toLocaleString()} cached`
    : 'none cached'
  const fieldLabel = fieldAvailable
    ? `${cacheStatus.field_k} sample${cacheStatus.field_k !== 1 ? 's' : ''} cached`
    : 'none cached'

  // Seed optimal is only meaningful when generating fresh candidates
  // (and never in external pool mode, which bypasses generation entirely).
  const seedOptimalDisabled = (useCandidates && candAvailable) || internalDisabled

  return (
    <div className="dialog-overlay" onClick={onDismiss}>
      <div className="dialog" onClick={e => e.stopPropagation()}>
        <p className="dialog-title">Run options</p>
        <p className="dialog-message" style={{ marginBottom: 16 }}>
          Cached lineups reflect a previous run's ownership configuration.
          Ownership changes since then are not included.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 20 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: extAvailable ? 'pointer' : 'default', opacity: extAvailable ? 1 : 0.45 }}>
            <input
              type="checkbox"
              checked={useExternal && extAvailable}
              disabled={!extAvailable}
              onChange={e => setUseExternal(e.target.checked)}
            />
            <span>
              <strong>External candidate pool (SaberSim)</strong>
              {extAvailable ? (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                  {' '}— {ext!.lineups_file} + {ext!.projections_file} ({ext!.n_lineups?.toLocaleString()} lineups · {ext!.n_contests} contests{ext!.paired_by_token ? '' : ' · unpaired companion'})
                </span>
              ) : (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                  {' '}— {ext?.error || 'no lineups_*.csv found in data/raw'}
                </span>
              )}
            </span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: candAvailable && !internalDisabled ? 'pointer' : 'default', opacity: candAvailable && !internalDisabled ? 1 : 0.45 }}>
            <input
              type="checkbox"
              checked={useCandidates && candAvailable && !internalDisabled}
              disabled={!candAvailable || internalDisabled}
              onChange={e => setUseCandidates(e.target.checked)}
            />
            <span>
              <strong>Candidates</strong> — {candLabel}
              {internalDisabled && (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}> (bypassed in external mode)</span>
              )}
              {!internalDisabled && candAvailable && cacheStatus.candidates! < cacheStatus.n_configured_candidates && (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                  {' '}(will generate {(cacheStatus.n_configured_candidates - cacheStatus.candidates!).toLocaleString()} more)
                </span>
              )}
            </span>
          </label>
          {!isHistorical && (
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: fieldAvailable && !internalDisabled ? 'pointer' : 'default', opacity: fieldAvailable && !internalDisabled ? 1 : 0.45 }}>
              <input
                type="checkbox"
                checked={useField && fieldAvailable && !internalDisabled}
                disabled={!fieldAvailable || internalDisabled}
                onChange={e => setUseField(e.target.checked)}
              />
              <span>
                <strong>Field lineups</strong> — {fieldLabel}
                {internalDisabled && (
                  <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}> (bypassed in external mode)</span>
                )}
              </span>
            </label>
          )}
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: seedOptimalDisabled ? 'default' : 'pointer', opacity: seedOptimalDisabled ? 0.45 : 1 }}>
            <input
              type="checkbox"
              checked={seedOptimal && !seedOptimalDisabled}
              disabled={seedOptimalDisabled}
              onChange={e => setSeedOptimal(e.target.checked)}
            />
            <span>
              <strong>Seed with optimal lineups</strong> — {cacheStatus.n_batter_teams * 60} ILP-optimal candidates ({cacheStatus.n_batter_teams} teams × 35 five-stack + 25 four-stack)
              {seedOptimalDisabled && (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                  {' '}(unavailable when using cached candidates)
                </span>
              )}
            </span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: seedOptimalDisabled ? 'default' : 'pointer', opacity: seedOptimalDisabled ? 0.45 : 1 }}>
            <input
              type="checkbox"
              checked={seedSimOptimal && !seedOptimalDisabled}
              disabled={seedOptimalDisabled}
              onChange={e => setSeedSimOptimal(e.target.checked)}
            />
            <span>
              <strong>Seed with sim-optimal lineups</strong> — per-sim ILP winners across sampled simulated worlds (ceiling candidates by construction)
              {seedOptimalDisabled && (
                <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                  {' '}(unavailable when using cached candidates)
                </span>
              )}
            </span>
          </label>
        </div>
        <div className="dialog-actions">
          <button className="btn-run" onClick={() => onStart(
            useCandidates && candAvailable && !internalDisabled,
            useField && fieldAvailable && !internalDisabled,
            seedOptimal && !seedOptimalDisabled,
            seedSimOptimal && !seedOptimalDisabled,
            useExternal && extAvailable,
          )}>
            Start Run
          </button>
          <button className="btn-secondary" onClick={onDismiss}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
