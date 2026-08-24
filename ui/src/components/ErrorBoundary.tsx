import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

interface Props {
  /** Names the area in the fallback, so a contained crash says what broke. */
  label: string
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Contains a render crash to one panel instead of the whole app.
 *
 * React unmounts the entire tree when a render throws and nothing catches it,
 * so a single bad field in a progress event took the app to a blank page
 * mid-run -- with the pipeline still happily running and the finished
 * portfolio unreachable behind the blankness. Nothing in the read-only
 * display layer is worth that: a panel that cannot render should say so and
 * leave the rest of the page usable.
 *
 * Deliberately NOT wrapped around the whole app -- that would just relocate
 * the blank page. Wrap the individual panels that render server-supplied
 * payloads.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The stack is the only breadcrumb pointing at which field was missing.
    console.error(`[${this.props.label}] render failed:`, error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (error === null) return this.props.children
    return (
      <div className="error-boundary">
        <div className="error-boundary-header">
          {this.props.label} stopped rendering
        </div>
        <div className="error-boundary-message">{error.message}</div>
        <p className="error-boundary-hint">
          The run is unaffected and is still going — this is a display fault
          only. Details are in the browser console.
        </p>
        {/* Props change on every event, so a later render can genuinely
            succeed even though the current one cannot. */}
        <button
          className="error-boundary-retry"
          onClick={() => this.setState({ error: null })}
        >
          Try again
        </button>
      </div>
    )
  }
}
