"""
Mapping of chord labels to corresponding notes (from bass to treble).
Intended for visualization or playback via MIDI or synthesis.
"""

CHORD_MAP = {
    # C family
    "C:maj": ["C3", "E3", "G3"],
    "C:min": ["C3", "D#3", "G3"],
    "C:7": ["C3", "E3", "G3", "A#3"],
    "C:maj7": ["C3", "E3", "G3", "B3"],
    "C:min7": ["C3", "D#3", "G3", "A#3"],
    # C#/Db
    "C#:maj": ["C#3", "F3", "G#3"],
    "C#:min": ["C#3", "E3", "G#3"],
    "C#:7": ["C#3", "F3", "G#3", "B3"],
    "C#:maj7": ["C#3", "F3", "G#3", "C4"],
    "C#:min7": ["C#3", "E3", "G#3", "B3"],
    # D
    "D:maj": ["D3", "F#3", "A3"],
    "D:min": ["D3", "F3", "A3"],
    "D:7": ["D3", "F#3", "A3", "C4"],
    "D:maj7": ["D3", "F#3", "A3", "C#4"],
    "D:min7": ["D3", "F3", "A3", "C4"],
    # D#/Eb
    "D#:maj": ["D#3", "G3", "A#3"],
    "D#:min": ["D#3", "F#3", "A#3"],
    "D#:7": ["D#3", "G3", "A#3", "C#4"],
    "D#:maj7": ["D#3", "G3", "A#3", "D4"],
    "D#:min7": ["D#3", "F#3", "A#3", "C#4"],
    # E
    "E:maj": ["E3", "G#3", "B3"],
    "E:min": ["E3", "G3", "B3"],
    "E:7": ["E3", "G#3", "B3", "D4"],
    "E:maj7": ["E3", "G#3", "B3", "D#4"],
    "E:min7": ["E3", "G3", "B3", "D4"],
    # F
    "F:maj": ["F3", "A3", "C4"],
    "F:min": ["F3", "G#3", "C4"],
    "F:7": ["F3", "A3", "C4", "D#4"],
    "F:maj7": ["F3", "A3", "C4", "E4"],
    "F:min7": ["F3", "G#3", "C4", "D#4"],
    # F#/Gb
    "F#:maj": ["F#3", "A#3", "C#4"],
    "F#:min": ["F#3", "A3", "C#4"],
    "F#:7": ["F#3", "A#3", "C#4", "E4"],
    "F#:maj7": ["F#3", "A#3", "C#4", "F4"],
    "F#:min7": ["F#3", "A3", "C#4", "E4"],
    # G
    "G:maj": ["G3", "B3", "D4"],
    "G:min": ["G3", "A#3", "D4"],
    "G:7": ["G3", "B3", "D4", "F4"],
    "G:maj7": ["G3", "B3", "D4", "F#4"],
    "G:min7": ["G3", "A#3", "D4", "F4"],
    # G#/Ab
    "G#:maj": ["G#3", "C4", "D#4"],
    "G#:min": ["G#3", "B3", "D#4"],
    "G#:7": ["G#3", "C4", "D#4", "F#4"],
    "G#:maj7": ["G#3", "C4", "D#4", "G4"],
    "G#:min7": ["G#3", "B3", "D#4", "F#4"],
    # A
    "A:maj": ["A3", "C#4", "E4"],
    "A:min": ["A3", "C4", "E4"],
    "A:7": ["A3", "C#4", "E4", "G4"],
    "A:maj7": ["A3", "C#4", "E4", "G#4"],
    "A:min7": ["A3", "C4", "E4", "G4"],
    # A#/Bb
    "A#:maj": ["A#3", "D4", "F4"],
    "A#:min": ["A#3", "C#4", "F4"],
    "A#:7": ["A#3", "D4", "F4", "G#4"],
    "A#:maj7": ["A#3", "D4", "F4", "A4"],
    "A#:min7": ["A#3", "C#4", "F4", "G#4"],
    # B
    "B:maj": ["B3", "D#4", "F#4"],
    "B:min": ["B3", "D4", "F#4"],
    "B:7": ["B3", "D#4", "F#4", "A4"],
    "B:maj7": ["B3", "D#4", "F#4", "A#4"],
    "B:min7": ["B3", "D4", "F#4", "A4"],
}


def get_chord_notes(chord_label: str) -> list[str]:
    """
    Return the ordered notes for a given chord label.
    """
    return CHORD_MAP.get(chord_label, [])


def all_chords() -> list[str]:
    """
    Return all known chord labels.
    """
    return sorted(CHORD_MAP.keys())
