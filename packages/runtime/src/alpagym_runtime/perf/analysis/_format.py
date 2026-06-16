# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared terminal-rendering helpers for the perf analysis CLI.

Color, section framing, table layout, and adaptive duration formatting, so the summary
tables and the AlpaSim telemetry section render consistently. Color is a light-touch
hierarchy (bold cyan section titles, cyan sub-headers, dim column headers) emitted only
on an interactive terminal, so piped or pasted output stays plain ASCII.
"""

from __future__ import annotations

import os
import sys

_RESET = "\033[0m"
_BOLD = "\033[1m"
_UNDERLINE = "\033[4m"
_BRIGHT_CYAN = "\033[96m"
_BRIGHT_YELLOW = "\033[93m"
_ORANGE = "\033[38;5;208m"

# The hierarchy is high-contrast and colorblind-safe: every header tier is BOLD (luminance
# contrast works for any color vision), and each tier also carries a non-color cue -- the
# `---` rule for sections, the underline for column headers -- so the structure reads even
# if the colors are indistinguishable. Bright cyan and bright yellow are the blue/yellow
# axis, the pair most reliably told apart with red-green color vision deficiency.


def color_enabled() -> bool:
    """Return whether to emit ANSI styling (interactive stdout, `NO_COLOR` unset)."""
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def bold(text: str, color: bool) -> str:
    """Bold (run title, TOTAL row): high-contrast emphasis, independent of color vision."""
    return f"{_BOLD}{text}{_RESET}" if color else text


def subheader(text: str, color: bool) -> str:
    """Sub-headers (`Device`, `Simulator internals`): bold bright yellow, one tier below a
    section title (which is bold bright cyan).
    """
    return f"{_BOLD}{_BRIGHT_YELLOW}{text}{_RESET}" if color else text


def warning(text: str, color: bool) -> str:
    """Warning text: bold orange, distinct from the yellow sub-header color."""
    return f"{_BOLD}{_ORANGE}{text}{_RESET}" if color else text


def colheader(text: str, color: bool) -> str:
    """Table column headers: bold + underline. The underline marks the header band without
    relying on color, so it reads as a header under any color vision.
    """
    return f"{_BOLD}{_UNDERLINE}{text}{_RESET}" if color else text


def print_section(title: str, color: bool) -> None:
    """Start a top-level section with a separator rule and a bold, colored title.

    The rule and title print in both modes, so sections stay visible when the output is
    piped or pasted (e.g. into an MR description). With color the title is bold bright
    cyan; the `---` rule above it marks the section without relying on color. The blank
    line above comes from the previous block.
    """
    print("-" * 60)
    if color:
        print(f"{_BOLD}{_BRIGHT_CYAN}{title}{_RESET}")
    else:
        print(title)


def duration_parts(value_ms: float) -> tuple[str, str]:
    """Split a duration into its `(number, unit)` parts with an adaptive unit.

    The unit steps up with magnitude (`us`, `ms`, `s`, `min`, then `h`), promoting to
    minutes at >= 60 s and to hours at >= 60 min, so long runs read in sensible units.
    Returning the parts separately lets the caller line up numbers and units in their own
    sub-columns.
    """
    if value_ms < 1.0:
        return f"{value_ms * 1000.0:.0f}", "us"
    if value_ms < 1000.0:
        return f"{value_ms:.2f}", "ms"
    seconds = value_ms / 1000.0
    if seconds < 60.0:
        return f"{seconds:.2f}", "s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.2f}", "min"
    return f"{minutes / 60.0:.2f}", "h"


def duration_str(value_ms: float) -> str:
    """Format a single duration as `number unit` (a space separates value and unit)."""
    number, unit = duration_parts(value_ms)
    return f"{number} {unit}"


def duration_column(values_ms: list[float]) -> list[str]:
    """Format a column of durations as aligned `number unit` cells.

    Every number is right-justified and every unit left-justified to the widest at that
    position across the column, so values and units each line up vertically down the
    column (a space always separates the value from its unit).
    """
    parts = [duration_parts(value) for value in values_ms]
    num_width = max((len(number) for number, _ in parts), default=0)
    unit_width = max((len(unit) for _, unit in parts), default=0)
    return [f"{number:>{num_width}} {unit:<{unit_width}}" for number, unit in parts]


def render_table(headers: list[str], rows: list[list[str]], color: bool) -> None:
    """Print a table whose columns are sized to their widest value.

    Each column is padded to the longest cell (header included) so values line up
    vertically. Columns whose every value is an integer are right-justified; the rest are
    left-justified. A two-space gutter separates columns, trailing padding is stripped, and
    the header row is dimmed so it reads as a quiet label band above the data.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    right = [all(row[i].isdigit() for row in rows) for i in range(len(headers))]

    def line_for(cells: list[str]) -> str:
        return "  ".join(
            f"{cell:{'>' if right[i] else '<'}{widths[i]}}" for i, cell in enumerate(cells)
        ).rstrip()

    print(colheader(line_for(headers), color))
    for row in rows:
        print(line_for(row))
