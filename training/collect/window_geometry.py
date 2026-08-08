"""Shared fixed geometry for historical and live chat windows."""

import math


WINDOW_BEFORE_SECONDS = 5
WINDOW_AFTER_SECONDS = 30
WINDOW_SECONDS = WINDOW_BEFORE_SECONDS + WINDOW_AFTER_SECONDS
TARGET_LAG_SECONDS = WINDOW_AFTER_SECONDS
WINDOW_GEOMETRY_VERSION = 2
WINDOW_GEOMETRY_NAME = "clip_start_minus_5_plus_30"
GEOMETRY_TOLERANCE_SECONDS = 1e-6


def window_bounds(target_offset):
    """Return a fixed 35-second window around a clip-start/negative anchor."""
    target_offset = float(target_offset)
    # A negative start near the beginning of a VOD is intentional: it preserves
    # the fixed 35-second feature denominator, with no messages in the
    # pre-broadcast portion.
    start = target_offset - WINDOW_BEFORE_SECONDS
    return start, start + WINDOW_SECONDS


def has_current_geometry(record, expected_target=None):
    """Return whether a raw record uses the current fixed window contract."""
    try:
        target = float(record["target_offset"])
        start = float(record["window_start"])
        end = float(record["window_end"])
        version = int(record["window_geometry_version"])
    except (KeyError, TypeError, ValueError):
        return False

    if expected_target is not None and not math.isclose(
        target,
        float(expected_target),
        abs_tol=GEOMETRY_TOLERANCE_SECONDS,
    ):
        return False

    expected_start, expected_end = window_bounds(target)
    return (
        version == WINDOW_GEOMETRY_VERSION
        and record.get("window_geometry") == WINDOW_GEOMETRY_NAME
        and math.isclose(
            start,
            expected_start,
            abs_tol=GEOMETRY_TOLERANCE_SECONDS,
        )
        and math.isclose(
            end,
            expected_end,
            abs_tol=GEOMETRY_TOLERANCE_SECONDS,
        )
    )
