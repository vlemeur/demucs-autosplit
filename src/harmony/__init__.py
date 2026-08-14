"""
Harmony, voicing, and notation helpers for music-workbench.
"""

from harmony.chord_library import (
    detect_chord_from_notes,
    detect_chords_from_notes,
    get_chord_notes,
)
from harmony.trainer import (
    VISIBLE_MAJOR_KEYS,
    VISIBLE_MINOR_KEYS,
    build_visible_exercises,
    compare_note_sets,
    display_chord_symbol,
    export_live_chord_progression_svg,
    render_detected_voicing_gallery,
    render_played_chord_gallery,
    render_progression_svg,
)

__all__ = [
    "detect_chord_from_notes",
    "detect_chords_from_notes",
    "get_chord_notes",
    "VISIBLE_MAJOR_KEYS",
    "VISIBLE_MINOR_KEYS",
    "build_visible_exercises",
    "compare_note_sets",
    "display_chord_symbol",
    "export_live_chord_progression_svg",
    "render_detected_voicing_gallery",
    "render_played_chord_gallery",
    "render_progression_svg",
]
