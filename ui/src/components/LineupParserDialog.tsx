import { useState } from 'react'
import type { ParsedSlot, PlayerMatch, TwitterLineupParseResponse, TwitterLineupSaveRequest, TwitterLineupSlot } from '../types'

interface Props {
  parseResult: TwitterLineupParseResponse
  onConfirm: (req: TwitterLineupSaveRequest) => Promise<void>
  onCancel: () => void
}

function getSelectedPlayer(slot: ParsedSlot, selectedId: number | null): PlayerMatch | null {
  if (selectedId === null) return null
  return slot.matches.find(m => m.player_id === selectedId) ?? null
}

export function LineupParserDialog({ parseResult, onConfirm, onCancel }: Props) {
  const [selections, setSelections] = useState<(number | null)[]>(
    () => parseResult.slots.map(s => s.matches[0]?.player_id ?? null)
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canConfirm = selections.every(s => s !== null)

  const handleConfirm = async () => {
    const slots: TwitterLineupSlot[] = parseResult.slots.map((s, i) => ({
      slot: s.slot,
      player_id: selections[i]!,
      name: s.raw_name,
    }))
    const req: TwitterLineupSaveRequest = {
      team: parseResult.team!,
      notification_id: parseResult.notification_id,
      slots,
    }
    setSaving(true)
    setError(null)
    try {
      await onConfirm(req)
    } catch (e) {
      setError(String(e))
      setSaving(false)
    }
  }

  return (
    <div className="dialog-overlay" onClick={e => { if (e.target === e.currentTarget) onCancel() }}>
      <div className="dialog dialog--wide">
        <div className="dialog-title">
          Confirm Lineup — {parseResult.team ?? 'Unknown Team'}
        </div>

        {parseResult.warning && (
          <div className={`dialog-warning ${parseResult.team_in_slate ? 'dialog-warning--info' : 'dialog-warning--caution'}`}>
            {parseResult.warning}
          </div>
        )}

        {!parseResult.team_in_slate && parseResult.team && (
          <div className="dialog-warning dialog-warning--caution">
            {parseResult.team} is not on the current slate. The lineup will be saved but won't affect this run.
          </div>
        )}

        <div className="lineup-parser-slots">
          {parseResult.slots.map((slot, i) => {
            const selected = getSelectedPlayer(slot, selections[i])
            const isExact = slot.matches.length === 1 && slot.matches[0].match_confidence === 'exact'
            const noMatch = slot.matches.length === 0

            return (
              <div key={slot.slot} className="lp-slot-row">
                <span className="lp-slot-num">{slot.slot}</span>
                <span className="lp-raw-name">{slot.raw_name} <span className="lp-pos-chip">{slot.position}</span></span>
                <span className="lp-player-cell">
                  {noMatch ? (
                    <span className="lp-no-match">No player found in pool</span>
                  ) : isExact ? (
                    <span className="lp-match-exact">{slot.matches[0].name}</span>
                  ) : (
                    <select
                      className="lp-match-select"
                      value={selections[i] ?? ''}
                      onChange={e => {
                        const val = e.target.value === '' ? null : Number(e.target.value)
                        setSelections(prev => {
                          const next = [...prev]
                          next[i] = val
                          return next
                        })
                      }}
                    >
                      {slot.matches.map(m => (
                        <option key={m.player_id} value={m.player_id}>
                          {m.name} ({m.position} · ${(m.salary / 1000).toFixed(1)}k)
                          {m.match_confidence === 'fuzzy' ? ' ~' : ''}
                        </option>
                      ))}
                    </select>
                  )}
                </span>
                <span className="lp-salary">
                  {selected ? `$${(selected.salary / 1000).toFixed(1)}k` : ''}
                </span>
              </div>
            )
          })}
        </div>

        {error && <div className="dialog-warning dialog-warning--caution">{error}</div>}

        <div className="dialog-actions">
          <button className="btn-secondary" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={handleConfirm}
            disabled={!canConfirm || saving}
          >
            {saving ? 'Saving…' : 'Confirm Lineup'}
          </button>
        </div>
      </div>
    </div>
  )
}
