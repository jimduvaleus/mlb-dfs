"""
Factory that returns platform-specific entry file handlers.

Usage
-----
    handlers = get_entry_handlers(Platform.FANDUEL)
    paths    = handlers["scan"](raw_dir)
    entries  = [(p, handlers["parse"](p)) for p in paths]
    assigns  = handlers["assign"](entries, portfolio)
    written  = handlers["write"](entries, assigns, slate_df, output_dir)
"""
from src.platforms.base import Platform
from src.api.dk_entries import (
    scan_entry_files,
    parse_entry_file,
    assign_lineups_to_entries,
    write_upload_files,
)
from src.api.fd_entries import (
    scan_fd_entry_files,
    parse_fd_entry_file,
    assign_fd_lineups_to_entries,
    write_fd_upload_files,
)

_DK_HANDLERS = {
    "scan":   scan_entry_files,
    "parse":  parse_entry_file,
    "assign": assign_lineups_to_entries,
    "write":  write_upload_files,
}

_FD_HANDLERS = {
    "scan":   scan_fd_entry_files,
    "parse":  parse_fd_entry_file,
    "assign": assign_fd_lineups_to_entries,
    "write":  write_fd_upload_files,
}


def get_entry_handlers(platform: Platform) -> dict:
    """
    Return a dict of {scan, parse, assign, write} callables for *platform*.

    Raises
    ------
    ValueError for unsupported platforms.
    """
    if platform == Platform.DRAFTKINGS:
        return _DK_HANDLERS
    if platform == Platform.FANDUEL:
        return _FD_HANDLERS
    raise ValueError(f"No entry handlers registered for platform: {platform!r}")
