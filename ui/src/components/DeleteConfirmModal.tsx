interface Props {
  lineupIndex: number
  onConfirm: () => void
  onCancel: () => void
}

export function DeleteConfirmModal({ lineupIndex, onConfirm, onCancel }: Props) {
  return (
    <div className="dialog-overlay">
      <div className="dialog">
        <p className="dialog-message">
          Delete Lineup #{lineupIndex}? A replacement lineup will be generated from
          the remaining simulations and added to the end of the portfolio.
        </p>
        <div className="dialog-actions">
          <button className="btn-secondary" onClick={onCancel}>Cancel</button>
          <button className="btn-danger" onClick={onConfirm}>Delete &amp; Replace</button>
        </div>
      </div>
    </div>
  )
}
