interface Props {
  lineupCount: number
  onConfirm: () => void
  onDismiss: () => void
}

export function StopUploadDialog({ lineupCount, onConfirm, onDismiss }: Props) {
  return (
    <div className="dialog-overlay">
      <div className="dialog">
        <p className="dialog-message">
          Run stopped after {lineupCount} lineup{lineupCount !== 1 ? 's' : ''} generated.
          Write partial portfolio to upload files?
        </p>
        <div className="dialog-actions">
          <button className="btn-run" onClick={onConfirm}>
            Write to Files
          </button>
          <button className="btn-secondary" onClick={onDismiss}>
            Skip
          </button>
        </div>
      </div>
    </div>
  )
}
