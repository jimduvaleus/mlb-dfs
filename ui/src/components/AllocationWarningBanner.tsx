interface Props {
  warnings: string[]
  onDismiss?: () => void
}

/**
 * Allocation warnings shown above the portfolio.
 *
 * These are money problems, not diagnostics: an unfilled entry is a purchased
 * slot that went unused, and a relaxed overlap cap means the portfolio is more
 * concentrated than the configuration asked for. Both were previously visible
 * only as one clause on a progress line that scrolls away, so they are surfaced
 * here — and persisted on the sweep payload, so a page refresh does not lose
 * them.
 */
export function AllocationWarningBanner({ warnings, onDismiss }: Props) {
  if (!warnings.length) return null
  return (
    <div className="allocation-warning" role="alert">
      <div className="allocation-warning-head">
        <strong>
          Allocation warning{warnings.length !== 1 ? 's' : ''}
        </strong>
        {onDismiss && (
          <button className="btn-secondary" onClick={onDismiss} aria-label="Dismiss">
            Dismiss
          </button>
        )}
      </div>
      <ul className="allocation-warning-list">
        {warnings.map((w, i) => <li key={i}>{w}</li>)}
      </ul>
    </div>
  )
}
