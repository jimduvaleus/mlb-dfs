interface TeamBadgeProps {
  team: string
  className?: string
}

export default function TeamBadge({ team, className }: TeamBadgeProps) {
  return (
    <span className={`team-badge${className ? ' ' + className : ''}`}>
      <img
        src={`/team-logos/${team.toUpperCase()}.png`}
        alt=""
        className="team-badge-logo"
        onError={(e) => { e.currentTarget.style.display = 'none' }}
      />
      {team}
    </span>
  )
}
