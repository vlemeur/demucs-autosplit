"""Chord templates and lightweight chord detection helpers."""

from typing import Final

QUALITY_COMPONENTS: Final[dict[str, tuple[tuple[int, int], ...]]] = {
    "maj": ((1, 0), (3, 4), (5, 7)),
    "min": ((1, 0), (3, 3), (5, 7)),
    "dim": ((1, 0), (3, 3), (5, 6)),
    "aug": ((1, 0), (3, 4), (5, 8)),
    "sus2": ((1, 0), (2, 2), (5, 7)),
    "sus4": ((1, 0), (4, 5), (5, 7)),
    "6": ((1, 0), (3, 4), (5, 7), (6, 9)),
    "min6": ((1, 0), (3, 3), (5, 7), (6, 9)),
    "7": ((1, 0), (3, 4), (5, 7), (7, 10)),
    "maj7": ((1, 0), (3, 4), (5, 7), (7, 11)),
    "min7": ((1, 0), (3, 3), (5, 7), (7, 10)),
    "dim7": ((1, 0), (3, 3), (5, 6), (7, 9)),
    "m7b5": ((1, 0), (3, 3), (5, 6), (7, 10)),
    "add9": ((1, 0), (3, 4), (5, 7), (2, 2)),
    "minadd9": ((1, 0), (3, 3), (5, 7), (2, 2)),
    "9": ((1, 0), (3, 4), (5, 7), (7, 10), (2, 2)),
    "maj9": ((1, 0), (3, 4), (5, 7), (7, 11), (2, 2)),
    "min9": ((1, 0), (3, 3), (5, 7), (7, 10), (2, 2)),
    "11": ((1, 0), (3, 4), (5, 7), (7, 10), (2, 2), (4, 5)),
    "min11": ((1, 0), (3, 3), (5, 7), (7, 10), (2, 2), (4, 5)),
    "13": ((1, 0), (3, 4), (5, 7), (7, 10), (2, 2), (4, 5), (6, 9)),
    "min13": ((1, 0), (3, 3), (5, 7), (7, 10), (2, 2), (4, 5), (6, 9)),
}

CHORD_INTERVALS: Final[dict[str, tuple[int, ...]]] = {
    quality: tuple(offset for _degree, offset in components)
    for quality, components in QUALITY_COMPONENTS.items()
}

# All supported chord types
CHORD_TYPES: Final[list[str]] = list(CHORD_INTERVALS.keys())
ROOT_LABELS: Final[list[str]] = [
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
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
    "Cb": 11,
    "B#": 0,
}
NATURAL_NOTE_TO_SEMITONE: Final[dict[str, int]] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
NOTE_LETTERS: Final[list[str]] = ["C", "D", "E", "F", "G", "A", "B"]


def _advance_note_letter(root_letter: str, degree: int) -> str:
    start_index = NOTE_LETTERS.index(root_letter)
    return NOTE_LETTERS[(start_index + degree - 1) % len(NOTE_LETTERS)]


def _accidental_suffix(delta: int) -> str | None:
    normalized = ((delta + 6) % 12) - 6
    mapping = {
        -2: "bb",
        -1: "b",
        0: "",
        1: "#",
        2: "##",
    }
    return mapping.get(normalized)


def _spell_chord_tones(root: str, quality: str) -> list[str]:
    root_pitch_class = NOTE_TO_SEMITONE[root]
    components = QUALITY_COMPONENTS.get(quality)
    if components is None:
        return []

    root_letter = root[0]
    spelled: list[str] = []
    for degree, semitone_offset in components:
        target_letter = _advance_note_letter(root_letter, degree)
        target_pitch_class = (root_pitch_class + semitone_offset) % 12
        natural_pitch_class = NATURAL_NOTE_TO_SEMITONE[target_letter]
        accidental = _accidental_suffix(target_pitch_class - natural_pitch_class)
        if accidental is None:
            return []
        spelled.append(f"{target_letter}{accidental}")
    return spelled


def _label_spelling_complexity(chord_label: str) -> int:
    tones = _spell_chord_tones_for_label(chord_label)
    if not tones:
        return 999
    score = 0
    for tone in tones:
        score += tone.count("#") + tone.count("b")
        if "##" in tone or "bb" in tone:
            score += 4
    return score


def _spell_chord_tones_for_label(chord_label: str) -> list[str]:
    normalized_label = chord_label.split("/", 1)[0]
    if ":" in normalized_label:
        root, quality = normalized_label.split(":", 1)
    else:
        root, quality = normalized_label, "maj"
    return _spell_chord_tones(root, quality)


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

    # Generate mappings for the common chord spellings first.
    for note in ROOT_LABELS:
        for chord_type in CHORD_TYPES:
            chord_label = f"{note}:{chord_type}"
            chord_tones = _spell_chord_tones(note, chord_type)
            if not chord_tones:
                continue
            chord_map[chord_label] = [f"{tone}3" for tone in chord_tones]

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
        "C#:m7b5": ["C#3", "E3", "G3", "B3"],
        # D
        "D:maj": ["D3", "F#3", "A3"],
        "D:min": ["D3", "F3", "A3"],
        "D:7": ["D3", "F#3", "A3", "C4"],
        "D:maj7": ["D3", "F#3", "A3", "C#4"],
        "D:min7": ["D3", "F3", "A3", "C4"],
        "D:dim": ["D3", "F3", "Ab3"],
        "D:aug": ["D3", "F#3", "A#3"],
        "D:m7b5": ["D3", "F3", "Ab3", "C4"],
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
        "F:m7b5": ["F3", "Ab3", "B3", "Eb4"],
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
        "G:m7b5": ["G3", "Bb3", "Db4", "F4"],
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
        "A:m7b5": ["A3", "C4", "Eb4", "G4"],
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
        "B:m7b5": ["B3", "D4", "F4", "A4"],
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
    normalized_label = chord_label.split("/", 1)[0]
    if normalized_label in CHORD_MAP:
        return CHORD_MAP[normalized_label]

    spelled = _spell_chord_tones_for_label(normalized_label)
    if not spelled:
        return []
    return [f"{note_name}3" for note_name in spelled]


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
    return sorted(f"{root}:{quality}" for root in ROOT_LABELS for quality in CHORD_TYPES)


def all_chord_types() -> list[str]:
    """
    Return all supported chord types.

    Returns
    -------
    list[str]
        List of chord type suffixes (e.g., ['maj', 'min', '7']).
    """
    return CHORD_TYPES.copy()


def _get_note_set(chord_label: str) -> set[str]:
    """
    Get the set of base notes (without octave) for a chord label.

    Parameters
    ----------
    chord_label : str
        Chord label in format "NOTE:CHORD_TYPE" (e.g., "C:maj", "G:min7").

    Returns
    -------
    set[str]
        Set of base note names (e.g., {"C", "E", "G"} for "C:maj").
    """
    notes = get_chord_notes(chord_label)
    return {note[:-1] for note in notes if note}  # Remove octave number


def _note_name_without_octave(note: str) -> str:
    if len(note) > 1 and note[-1].isdigit():
        return note[:-1]
    return note


def _normalize_input_notes(notes: list[str]) -> tuple[list[str], set[str]]:
    ordered = [_note_name_without_octave(note) for note in notes]
    return ordered, set(ordered)


def _note_names_to_pitch_classes(notes: set[str]) -> set[int]:
    pitch_classes: set[int] = set()
    for note in notes:
        pitch_class = NOTE_TO_SEMITONE.get(note)
        if pitch_class is not None:
            pitch_classes.add(pitch_class)
    return pitch_classes


def _candidate_score(
    *,
    chord_pitch_classes: set[int],
    input_pitch_classes: set[int],
) -> tuple[int, int, int, int]:
    match_count = len(chord_pitch_classes & input_pitch_classes)
    missing = len(chord_pitch_classes - input_pitch_classes)
    extras = len(input_pitch_classes - chord_pitch_classes)
    exact_bonus = 1 if missing == 0 and extras == 0 else 0
    return (exact_bonus, match_count, -extras, -missing)


def _quality_priority(chord_label: str) -> int:
    normalized = chord_label.split("/", 1)[0]
    quality = normalized.split(":", 1)[1] if ":" in normalized else "maj"
    priorities = {
        "min9": 18,
        "9": 17,
        "maj9": 16,
        "min7": 15,
        "7": 14,
        "maj7": 13,
        "m7b5": 12,
        "min11": 11,
        "11": 10,
        "min6": 9,
        "6": 8,
        "min": 7,
        "maj": 6,
        "sus4": 5,
        "sus2": 4,
        "dim7": 3,
        "dim": 2,
        "aug": 1,
    }
    return priorities.get(quality, 0)


def _spelling_priority(chord_label: str) -> int:
    root = chord_label.split("/", 1)[0].split(":", 1)[0]
    if "b" in root:
        return 2
    if "#" in root:
        return 0
    return 1


def _with_slash_bass(chord_label: str, ordered_input_notes: list[str]) -> str:
    if not ordered_input_notes:
        return chord_label

    bass = ordered_input_notes[0]
    root = chord_label.split(":", 1)[0]
    chord_pitch_classes = _get_note_set(chord_label)
    if bass == root or bass not in chord_pitch_classes:
        return chord_label
    return f"{chord_label}/{bass}"


def _detect_candidate_chords(notes: list[str], min_match_ratio: float) -> list[str]:
    if not notes:
        return []

    ordered_input_notes, input_note_set = _normalize_input_notes(notes)
    input_pitch_classes = _note_names_to_pitch_classes(input_note_set)
    results: list[tuple[tuple[int, int, int, int], int, int, str]] = []

    for chord_label in all_chords():
        chord_notes = _get_note_set(chord_label)
        if not chord_notes:
            continue
        chord_pitch_classes = _note_names_to_pitch_classes(chord_notes)
        if not chord_pitch_classes:
            continue

        match_ratio = len(chord_pitch_classes & input_pitch_classes) / len(chord_pitch_classes)
        if match_ratio < min_match_ratio:
            continue

        if len(input_pitch_classes - chord_pitch_classes) > 2:
            continue

        results.append(
            (
                _candidate_score(
                    chord_pitch_classes=chord_pitch_classes,
                    input_pitch_classes=input_pitch_classes,
                ),
                _quality_priority(chord_label),
                _spelling_priority(chord_label),
                _with_slash_bass(chord_label, ordered_input_notes),
            )
        )

    results.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[0][3],
            -item[1],
            -item[2],
            _label_spelling_complexity(item[3]),
            item[3],
        )
    )
    return [label for *_prefix, label in results]


def detect_chord_from_notes(notes: list[str], min_match_ratio: float = 0.7) -> str | None:
    """
    Detect the most likely chord from a list of played notes.

    This function finds the chord that best matches the input notes.
    A chord matches if all its notes are present in the input (ignoring octaves).
    The chord with the most matching notes wins.

    Parameters
    ----------
    notes : list[str]
        List of note names with octaves (e.g., ["C4", "E4", "G4"]).
    min_match_ratio : float, default=0.7
        Minimum ratio of chord notes that must be present in input to consider a match.
        For example, 0.7 means at least 70% of the chord's notes must be in the input.

    Returns
    -------
    str or None
        The detected chord label (e.g., "C:maj"), or None if no match found.

    Examples
    --------
    >>> detect_chord_from_notes(["C4", "E4", "G4"])
    'C:maj'
    >>> detect_chord_from_notes(["C4", "E4", "G4", "B4"])
    'C:maj7'
    >>> detect_chord_from_notes(["C4", "Eb4", "G4"])
    'C:min'
    """
    candidates = _detect_candidate_chords(notes, min_match_ratio)
    return candidates[0] if candidates else None


def detect_chords_from_notes(notes: list[str], min_match_ratio: float = 0.7) -> list[str]:
    """
    Detect all possible chords from a list of played notes, sorted by likelihood.

    Parameters
    ----------
    notes : list[str]
        List of note names with octaves (e.g., ["C4", "E4", "G4"]).
    min_match_ratio : float, default=0.7
        Minimum ratio of chord notes that must be present in input.

    Returns
    -------
    list[str]
        List of matching chord labels, sorted by match quality (best first).
    """
    return _detect_candidate_chords(notes, min_match_ratio)
