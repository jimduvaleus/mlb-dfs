import { useState, useEffect, useRef } from 'react'
import type { SSEEvent } from '../types'

export type SSEHookStatus = 'idle' | 'connecting' | 'streaming' | 'done' | 'error'

interface UseSSEResult {
  events: SSEEvent[]
  status: SSEHookStatus
  start: () => void
  reset: () => void
}

export function useSSE(url: string): UseSSEResult {
  const [events, setEvents] = useState<SSEEvent[]>([])
  const [status, setStatus] = useState<SSEHookStatus>('idle')
  const esRef = useRef<EventSource | null>(null)

  const start = () => {
    if (esRef.current) {
      esRef.current.close()
    }
    setEvents([])
    setStatus('connecting')

    const es = new EventSource(url)
    esRef.current = es

    es.onopen = () => setStatus('streaming')

    es.onmessage = (e: MessageEvent) => {
      try {
        const event: SSEEvent = JSON.parse(e.data)
        setEvents(prev => [...prev, event])
        if (event.stage === 'complete' || event.stage === 'error') {
          setStatus('done')
          es.close()
          esRef.current = null
        }
      } catch {
        // ignore malformed events
      }
    }

    es.onerror = () => {
      setStatus('done')
      es.close()
      esRef.current = null
    }
  }

  const reset = () => {
    esRef.current?.close()
    esRef.current = null
    setEvents([])
    setStatus('idle')
  }

  useEffect(() => {
    return () => {
      esRef.current?.close()
    }
  }, [])

  return { events, status, start, reset }
}
