import type { PayoutFallbackContest } from '../types'

interface Props {
  contests: PayoutFallbackContest[]
  nContests: number
  onProceed: () => void
  onCancel: () => void
}

/**
 * Mid-run gate. Marginal reward could not find a registered payout table for
 * one or more contests and would silently borrow another contest's structure.
 *
 * This blocks the run rather than warning, because dR is denominated in
 * dollars and the greedy compares marginal dollars ACROSS contests — a wrong
 * payout curve does not just misrank candidates inside one contest, it
 * misallocates entries between them, slate-wide.
 */
export function PayoutFallbackDialog({ contests, nContests, onProceed, onCancel }: Props) {
  return (
    <div className="dialog-overlay">
      <div className="dialog" style={{ maxWidth: 720 }}>
        <p className="dialog-title">Missing payout structure</p>
        <p className="dialog-message">
          {contests.length} of {nContests} contest{nContests !== 1 ? 's' : ''} have no
          registered payout table. Marginal reward will fall back to another contest's
          structure:
        </p>

        <div style={{ maxHeight: 280, overflowY: 'auto', margin: '12px 0' }}>
          {contests.map(c => (
            <div key={c.contest_id}
                 style={{ padding: '8px 0', borderTop: '1px solid var(--color-border)' }}>
              <div style={{ fontWeight: 600 }}>{c.contest_name}</div>
              <div style={{ fontSize: '0.85em', color: 'var(--color-text-muted)' }}>
                {c.k} {c.k === 1 ? 'entry' : 'entries'} · ~{c.implied_field_size.toLocaleString()} field
                · ${c.entry_fee.toFixed(2)} entry
              </div>
              <div style={{ fontSize: '0.9em', marginTop: 4 }}>
                → will use <strong>{c.table_name}</strong>{' '}
                <span style={{ color: 'var(--color-text-muted)' }}>
                  ({c.table_entries.toLocaleString()} entries · ${c.table_entry_fee.toFixed(2)} entry
                  · ${c.table_prize_pool.toLocaleString()} prize pool)
                </span>
              </div>
            </div>
          ))}
        </div>

        <p className="dialog-message" style={{ fontSize: '0.85em' }}>
          This objective is denominated in dollars and compares contests against each
          other, so a borrowed payout curve misallocates entries <em>between</em> contests,
          not just within one. The proper fix is to add the real table to{' '}
          <code>data/payout_structures/</code> and register it in{' '}
          <code>payout.CONTEST_STRUCTURES</code>.
        </p>

        <div className="dialog-actions">
          <button className="btn-secondary" onClick={onCancel}>Stop run</button>
          <button className="btn-run" onClick={onProceed}>Proceed anyway</button>
        </div>
      </div>
    </div>
  )
}
