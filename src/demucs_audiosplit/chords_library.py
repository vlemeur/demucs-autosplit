"""
Mapping of chord labels to corresponding notes (from bass to treble).
Intended for visualization or playback via MIDI or synthesis.

This module provides comprehensive chord to note mappings supporting:
- Major and minor triads
- 7th chords (dominant, major 7th, minor 7th)
- Suspended chords (sus2, sus4)
- Diminished and augmented chords
- Extended chords (9th, 11th, 13th) - placeholder for future expansion
"""

from typing import Final

# All supported chord types
CHORD_TYPES: Final[list[str]] = [
    "maj",
    "min",
    "7",
    "maj7",
    "min7",
    "sus2",
    "sus4",
    "dim",
    "dim7",
    "aug",
]

# All chromatic notes
NOTES: Final[list[str]] = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

# Semitone offsets for each note (C = 0)
NOTE_TO_SEMITONE: Final[dict[str, int]] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def _generate_chord_map() -> dict[str, list[str]]:
    """
    Generate a comprehensive chord to notes mapping.

    This function creates mappings for all notes and chord types.
    The base octave is 3 for most notes, with some variations for better voicing.

    Returns
    -------
    dict[str, list[str]]
        Mapping of chord labels (e.g., "C:maj") to list of note names (e.g., ["C3", "E3", "G3"]).
    """
    chord_map: dict[str, list[str]] = {}

    # Interval patterns for each chord type (in semitones from root)
    # Format: chord_type -> list of semitone offsets
    chord_intervals: dict[str, list[int]] = {
        "maj": [0, 4, 7],  # Major triad: root, major third, perfect fifth
        "min": [0, 3, 7],  # Minor triad: root, minor third, perfect fifth
        "7": [0, 4, 7, 10],  # Dominant 7th: root, major third, perfect fifth, minor seventh
        "maj7": [0, 4, 7, 11],  # Major 7th: root, major third, perfect fifth, major seventh
        "min7": [0, 3, 7, 10],  # Minor 7th: root, minor third, perfect fifth, minor seventh
        "sus2": [0, 2, 7],  # Suspended 2nd: root, major second, perfect fifth
        "sus4": [0, 5, 7],  # Suspended 4th: root, perfect fourth, perfect fifth
        "dim": [0, 3, 6],  # Diminished triad: root, minor third, diminished fifth
        # Diminished 7th: root, minor third, diminished fifth, diminished seventh
        "dim7": [0, 3, 6, 9],
        "aug": [0, 4, 8],  # Augmented triad: root, major third, augmented fifth
    }

    # Generate mappings for all notes and chord types
    for note in NOTES:
        for chord_type in CHORD_TYPES:
            if chord_type not in chord_intervals:
                continue

            chord_label = f"{note}:{chord_type}"
            intervals = chord_intervals[chord_type]

            # Calculate the notes in the chord
            base_semitone = NOTE_TO_SEMITONE[note]
            chord_notes = []

            for interval in intervals:
                target_semitone = (base_semitone + interval) % 12
                # Find the note name for this semitone
                for note_name, semitone in NOTE_TO_SEMITONE.items():
                    if semitone == target_semitone:
                        # Determine octave: base is 3, adjust for higher notes
                        octave = 3 + (target_semitone // 12)
                        if interval >= 12:  # For extended chords
                            octave += 1
                        chord_notes.append(f"{note_name}{octave}")
                        break

            chord_map[chord_label] = chord_notes

    # Manually override some mappings for better voicing
    # These are common chord voicings that sound better in certain octaves
    better_voicings: dict[str, list[str]] = {
        # C family
        "C:maj": ["C3", "E3", "G3"],
        "C:min": ["C3", "Eb3", "G3"],
        "C:7": ["C3", "E3", "G3", "Bb3"],
        "C:maj7": ["C3", "E3", "G3", "B3"],
        "C:min7": ["C3", "Eb3", "G3", "Bb3"],
        "C:dim": ["C3", "Eb3", "Gb3"],
        "C:dim7": ["C3", "Eb3", "Gb3", "Bbb3"],
        "C:aug": ["C3", "E3", "Ab3"],
        "C:sus2": ["C3", "D3", "G3"],
        "C:sus4": ["C3", "F3", "G3"],
        # C#/Db
        "C#:maj": ["C#3", "F3", "G#3"],
        "C#:min": ["C#3", "E3", "G#3"],
        "C#:7": ["C#3", "F3", "G#3", "B3"],
        "C#:maj7": ["C#3", "F3", "G#3", "C4"],
        "C#:min7": ["C#3", "E3", "G#3", "B3"],
        "C#:dim": ["C#3", "E3", "G3"],
        "C#:dim7": ["C#3", "E3", "G3", "Bb3"],
        "C#:aug": ["C#3", "F3", "A3"],
        # D
        "D:maj": ["D3", "F#3", "A3"],
        "D:min": ["D3", "F3", "A3"],
        "D:7": ["D3", "F#3", "A3", "C4"],
        "D:maj7": ["D3", "F#3", "A3", "C#4"],
        "D:min7": ["D3", "F3", "A3", "C4"],
        "D:dim": ["D3", "F3", "Ab3"],
        "D:aug": ["D3", "F#3", "A#3"],
        # D#/Eb
        "D#:maj": ["D#3", "G3", "A#3"],
        "D#:min": ["D#3", "F#3", "A#3"],
        "D#:7": ["D#3", "G3", "A#3", "C#4"],
        # E
        "E:maj": ["E3", "G#3", "B3"],
        "E:min": ["E3", "G3", "B3"],
        "E:7": ["E3", "G#3", "B3", "D4"],
        "E:maj7": ["E3", "G#3", "B3", "D#4"],
        "E:min7": ["E3", "G3", "B3", "D4"],
        "E:dim": ["E3", "G3", "Bb3"],
        # F
        "F:maj": ["F3", "A3", "C4"],
        "F:min": ["F3", "Ab3", "C4"],
        "F:7": ["F3", "A3", "C4", "Eb4"],
        "F:maj7": ["F3", "A3", "C4", "E4"],
        "F:min7": ["F3", "Ab3", "C4", "Eb4"],
        "F:dim": ["F3", "Ab3", "Cb4"],
        # F#/Gb
        "F#:maj": ["F#3", "A#3", "C#4"],
        "F#:min": ["F#3", "A3", "C#4"],
        "F#:7": ["F#3", "A#3", "C#4", "E4"],
        "F#:maj7": ["F#3", "A#3", "C#4", "F4"],
        "F#:min7": ["F#3", "A3", "C#4", "E4"],
        # G
        "G:maj": ["G3", "B3", "D4"],
        "G:min": ["G3", "Bb3", "D4"],
        "G:7": ["G3", "B3", "D4", "F4"],
        "G:maj7": ["G3", "B3", "D4", "F#4"],
        "G:min7": ["G3", "Bb3", "D4", "F4"],
        "G:dim": ["G3", "Bb3", "Db4"],
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
        "A:dim": ["A3", "C4", "Eb4"],
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
        "B:dim": ["B3", "D4", "F4"],
    }

    # Apply better voicings
    chord_map.update(better_voicings)

    return chord_map


# Generate the chord map at module load time
CHORD_MAP: Final[dict[str, list[str]]] = _generate_chord_map()


def get_chord_notes(chord_label: str) -> list[str]:
    """
    Return the ordered notes for a given chord label.

    Parameters
    ----------
    chord_label : str
        Chord label in format "NOTE:CHORD_TYPE" (e.g., "C:maj", "G:min7").

    Returns
    -------
    list[str]
        List of note names in format "NOTE_OCTAVE" (e.g., ["C3", "E3", "G3"]).
        Returns empty list if chord_label is not found.

    Examples
    --------
    >>> get_chord_notes("C:maj")
    ['C3', 'E3', 'G3']
    >>> get_chord_notes("A:min7")
    ['A3', 'C4', 'E4', 'G4']
    >>> get_chord_notes("INVALID")
    []
    """
    return CHORD_MAP.get(chord_label, [])


def all_chords() -> list[str]:
    """
    Return all known chord labels.

    Returns
    -------
    list[str]
        Sorted list of all supported chord labels.

    Examples
    --------
    >>> all_chords()  # doctest: +SKIP
    ['A:7', 'A:aug', 'A:dim', 'A:dim7', 'A:maj', 'A:maj7', 'A:min', 'A:min7', ...]
    """
    return sorted(CHORD_MAP.keys())


def all_chord_types() -> list[str]:
    """
    Return all supported chord types.

    Returns
    -------
    list[str]
        List of chord type suffixes (e.g., ['maj', 'min', '7']).
    """
    return CHORD_TYPES.copy()
