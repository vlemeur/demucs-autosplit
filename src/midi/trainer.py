from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path

import verovio
from music21 import chord, clef, key, layout, meter, musicxml, note, stream

NOTATION_BACKEND_AVAILABLE = True
NOTATION_BACKEND_ERROR = ""


NOTE_NAME_TO_PITCH_CLASS: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
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
}

FLAT_NOTE_NAMES: list[str] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SHARP_NOTE_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_FLAT_KEYS: set[str] = {"F", "Bb", "Eb", "Ab", "Db", "Gb"}
MINOR_FLAT_KEYS: set[str] = {"D", "G", "C", "F", "Bb", "Eb", "Ab"}

VISIBLE_MAJOR_KEYS: list[str] = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "B", "E", "A", "D", "G"]
VISIBLE_MINOR_KEYS: list[str] = ["A", "D", "G", "C", "F", "Bb", "Eb", "Ab", "C#", "F#", "B", "E"]

MAJOR_EXERCISE_VOICINGS: dict[str, dict[str, list[tuple[str, tuple[str, ...]]]]] = {
    "C": {
        "Type A": [
            ("Dm7", ("D3", "F4", "C5")),
            ("G7", ("G2", "F4", "B4")),
            ("Cmaj7", ("C3", "E4", "B4")),
        ],
        "Type B": [
            ("Dm7", ("D3", "C4", "F4")),
            ("G7", ("G2", "B3", "F4")),
            ("Cmaj7", ("C3", "B3", "E4")),
        ],
    },
    "F": {
        "Type A": [
            ("Gm7", ("G2", "Bb3", "F4")),
            ("C7", ("C3", "Bb3", "E4")),
            ("Fmaj7", ("F2", "A3", "E4")),
        ],
        "Type B": [
            ("Gm7", ("G2", "F4", "Bb4")),
            ("C7", ("C3", "E4", "Bb4")),
            ("Fmaj7", ("F2", "E4", "A4")),
        ],
    },
    "Bb": {
        "Type A": [
            ("Cm7", ("C3", "Eb4", "Bb4")),
            ("F7", ("F2", "Eb4", "A4")),
            ("Bbmaj7", ("Bb2", "D4", "A4")),
        ],
        "Type B": [
            ("Cm7", ("C3", "Bb3", "Eb4")),
            ("F7", ("F2", "A3", "Eb4")),
            ("Bbmaj7", ("Bb2", "A3", "D4")),
        ],
    },
    "Eb": {
        "Type A": [
            ("Fm7", ("F2", "Ab4", "Eb5")),
            ("Bb7", ("Bb2", "Ab4", "D5")),
            ("Ebmaj7", ("Eb2", "G4", "D5")),
        ],
        "Type B": [
            ("Fm7", ("F3", "Eb4", "Ab4")),
            ("Bb7", ("Bb2", "D4", "Ab4")),
            ("Ebmaj7", ("Eb2", "D4", "G4")),
        ],
    },
    "Ab": {
        "Type A": [
            ("Bbm7", ("Bb2", "Db4", "Ab4")),
            ("Eb7", ("Eb2", "Db4", "G4")),
            ("Abmaj7", ("Ab2", "C4", "G4")),
        ],
        "Type B": [
            ("Bbm7", ("Bb2", "Ab3", "Db4")),
            ("Eb7", ("Eb2", "G3", "Db4")),
            ("Abmaj7", ("Ab2", "G3", "C4")),
        ],
    },
    "Db": {
        "Type A": [
            ("Ebm7", ("Eb3", "Gb4", "Db5")),
            ("Ab7", ("Ab2", "Gb4", "C5")),
            ("Dbmaj7", ("Db3", "F4", "C5")),
        ],
        "Type B": [
            ("Ebm7", ("Eb3", "Db4", "Gb4")),
            ("Ab7", ("Ab2", "C4", "Gb4")),
            ("Dbmaj7", ("Db3", "C4", "F4")),
        ],
    },
    "Gb": {
        "Type A": [
            ("Abm7", ("Ab2", "B3", "Gb4")),
            ("Db7", ("Db3", "B3", "F4")),
            ("Gbmaj7", ("Gb2", "Bb3", "F4")),
        ],
        "Type B": [
            ("Abm7", ("Ab2", "Gb4", "B4")),
            ("Db7", ("Db3", "F4", "B4")),
            ("Gbmaj7", ("Gb2", "F4", "Bb4")),
        ],
    },
    "B": {
        "Type A": [
            ("C#m7", ("C#3", "E4", "B4")),
            ("F#7", ("F#3", "E4", "A#4")),
            ("Bmaj7", ("B2", "D#4", "A#4")),
        ],
        "Type B": [
            ("C#m7", ("C#3", "B4", "E5")),
            ("F#7", ("F#3", "A#4", "E5")),
            ("Bmaj7", ("B2", "A#4", "D#5")),
        ],
    },
    "E": {
        "Type A": [
            ("F#m7", ("F#3", "A3", "E4")),
            ("B7", ("B2", "A3", "D#4")),
            ("Emaj7", ("E2", "G#3", "D#4")),
        ],
        "Type B": [
            ("F#m7", ("F#2", "E4", "A4")),
            ("B7", ("B2", "D#4", "A4")),
            ("Emaj7", ("E3", "D#4", "G#4")),
        ],
    },
    "A": {
        "Type A": [
            ("Bm7", ("B2", "D4", "A4")),
            ("E7", ("E3", "D4", "G#4")),
            ("Amaj7", ("A2", "C#4", "G#4")),
        ],
        "Type B": [
            ("Bm7", ("B2", "A4", "D5")),
            ("E7", ("E3", "G#4", "D5")),
            ("Amaj7", ("A2", "G#4", "C#5")),
        ],
    },
    "D": {
        "Type A": [
            ("Em7", ("E3", "G4", "D5")),
            ("A7", ("A3", "G4", "C#5")),
            ("Dmaj7", ("D3", "F#4", "C#5")),
        ],
        "Type B": [
            ("Em7", ("E3", "D5", "G5")),
            ("A7", ("A3", "C#5", "G5")),
            ("Dmaj7", ("D3", "C#5", "F#5")),
        ],
    },
    "G": {
        "Type A": [
            ("Am7", ("A2", "C4", "G4")),
            ("D7", ("D3", "C4", "F#4")),
            ("Gmaj7", ("G2", "B3", "F#4")),
        ],
        "Type B": [
            ("Am7", ("A2", "G4", "C5")),
            ("D7", ("D3", "F#4", "C5")),
            ("Gmaj7", ("G2", "F#4", "B4")),
        ],
    },
}

MINOR_EXERCISE_VOICINGS: dict[str, dict[str, list[tuple[str, tuple[str, ...]]]]] = {
    "A": {
        "Type A": [
            ("Bm7(b5)", ("D3", "F3", "A3", "B3")),
            ("E7(b9#5)", ("D3", "F3", "G#3", "C4")),
            ("Am7", ("C3", "E3", "G3", "B3")),
        ],
        "Type B": [
            ("Bm7(b5)", ("A3", "B3", "D4", "F4")),
            ("E7(b9#5)", ("G#3", "C4", "D4", "F4")),
            ("Am7", ("G3", "B3", "C4", "E4")),
        ],
    },
    "D": {
        "Type A": [
            ("Em7(b5)", ("G3", "Bb3", "D4", "E4")),
            ("A7(b9#5)", ("G3", "Bb3", "C#4", "F4")),
            ("Dm7", ("F3", "A3", "C4", "E4")),
        ],
        "Type B": [
            ("Em7(b5)", ("D4", "E4", "G4", "Bb4")),
            ("A7(b9#5)", ("C#4", "F4", "G4", "Bb4")),
            ("Dm7", ("C4", "E4", "F4", "A4")),
        ],
    },
    "G": {
        "Type A": [
            ("Am7(b5)", ("C4", "Eb4", "G4", "A4")),
            ("D7(b9#5)", ("C4", "Eb4", "F#4", "Bb4")),
            ("Gm7", ("Bb3", "D4", "F4", "A4")),
        ],
        "Type B": [
            ("Am7(b5)", ("G3", "A3", "C4", "Eb4")),
            ("D7(b9#5)", ("F#3", "Bb3", "C4", "Eb4")),
            ("Gm7", ("F3", "A3", "Bb3", "D4")),
        ],
    },
    "C": {
        "Type A": [
            ("Dm7(b5)", ("F3", "Ab3", "C4", "D4")),
            ("G7(b9#5)", ("F3", "Ab3", "B3", "Eb4")),
            ("Cm7", ("Eb3", "G3", "Bb3", "D4")),
        ],
        "Type B": [
            ("Dm7(b5)", ("C4", "D4", "F4", "Ab4")),
            ("G7(b9#5)", ("B3", "Eb4", "F4", "Ab4")),
            ("Cm7", ("Bb3", "D4", "Eb4", "G4")),
        ],
    },
    "F": {
        "Type A": [
            ("Gm7(b5)", ("Bb3", "Db4", "F4", "G4")),
            ("C7(b9#5)", ("Bb3", "Db4", "E4", "Ab4")),
            ("Fm7", ("Ab3", "C4", "Eb4", "G4")),
        ],
        "Type B": [
            ("Gm7(b5)", ("F3", "G3", "Bb3", "Db4")),
            ("C7(b9#5)", ("E3", "Ab3", "Bb3", "Db4")),
            ("Fm7", ("Eb3", "G3", "Ab3", "C4")),
        ],
    },
    "Bb": {
        "Type A": [
            ("Cm7(b5)", ("Eb3", "Gb3", "Bb3", "C4")),
            ("F7(b9#5)", ("Eb3", "Gb3", "A3", "Db4")),
            ("Bbm7", ("Db3", "F3", "Ab3", "C4")),
        ],
        "Type B": [
            ("Cm7(b5)", ("Bb3", "C4", "Eb4", "Gb4")),
            ("F7(b9#5)", ("A3", "Db4", "Eb4", "Gb4")),
            ("Bbm7", ("Ab3", "C4", "Db4", "F4")),
        ],
    },
    "Eb": {
        "Type A": [
            ("Fm7(b5)", ("Ab3", "B3", "Eb4", "F4")),
            ("Bb7(b9#5)", ("Ab3", "B3", "D4", "Gb4")),
            ("Ebm7", ("Gb3", "Bb3", "Db4", "F4")),
        ],
        "Type B": [
            ("Fm7(b5)", ("Eb4", "F4", "Ab4", "B4")),
            ("Bb7(b9#5)", ("D4", "Gb4", "Ab4", "B4")),
            ("Ebm7", ("Db4", "F4", "Gb4", "Bb4")),
        ],
    },
    "Ab": {
        "Type A": [
            ("Bbm7(b5)", ("Db4", "E4", "Ab4", "Bb4")),
            ("Eb7(b9#5)", ("Db4", "E4", "G4", "B4")),
            ("Abm7", ("Cb4", "Eb4", "Gb4", "Bb4")),
        ],
        "Type B": [
            ("Bbm7(b5)", ("Ab3", "Bb3", "Db4", "E4")),
            ("Eb7(b9#5)", ("G3", "B3", "Db4", "E4")),
            ("Abm7", ("Gb3", "Bb3", "Cb4", "Eb4")),
        ],
    },
    "C#": {
        "Type A": [
            ("D#m7(b5)", ("F#3", "A3", "C#4", "D#4")),
            ("G#7(b9#5)", ("F#3", "A3", "C4", "E4")),
            ("C#m7", ("E3", "G#3", "B3", "D#4")),
        ],
        "Type B": [
            ("D#m7(b5)", ("C#4", "D#4", "F#4", "A4")),
            ("G#7(b9#5)", ("C4", "E4", "F#4", "A4")),
            ("C#m7", ("B3", "D#4", "E4", "G#4")),
        ],
    },
    "F#": {
        "Type A": [
            ("G#m7(b5)", ("B3", "D4", "F#4", "G#4")),
            ("C#7(b9#5)", ("B3", "D4", "F4", "A4")),
            ("F#m7", ("A3", "C#4", "E4", "G#4")),
        ],
        "Type B": [
            ("G#m7(b5)", ("F#3", "G#3", "B3", "D4")),
            ("C#7(b9#5)", ("F3", "A3", "B3", "D4")),
            ("F#m7", ("E3", "G#3", "A3", "C#4")),
        ],
    },
    "B": {
        "Type A": [
            ("C#m7(b5)", ("E3", "G3", "B3", "C#4")),
            ("F#7(b9#5)", ("E3", "G3", "A#3", "D4")),
            ("Bm7", ("D3", "F#3", "A3", "C#4")),
        ],
        "Type B": [
            ("C#m7(b5)", ("B3", "C#4", "E4", "G4")),
            ("F#7(b9#5)", ("A#3", "D4", "E4", "G4")),
            ("Bm7", ("A3", "C#4", "D4", "F#4")),
        ],
    },
    "E": {
        "Type A": [
            ("F#m7(b5)", ("A3", "C4", "E4", "F#4")),
            ("B7(b9#5)", ("A3", "C4", "D#4", "G4")),
            ("Em7", ("G3", "B3", "D4", "F#4")),
        ],
        "Type B": [
            ("F#m7(b5)", ("E3", "F#3", "A3", "C4")),
            ("B7(b9#5)", ("D#3", "G3", "A3", "C4")),
            ("Em7", ("D3", "F#3", "G3", "B3")),
        ],
    },
}


@dataclass(frozen=True)
class TrainerStep:
    symbol: str
    expected_notes: tuple[str, ...]
    display_notes: tuple[str, ...]
    lower_notes: tuple[str, ...]
    upper_notes: tuple[str, ...]
    staff: str


@dataclass(frozen=True)
class TrainerExercise:
    mode: str
    key: str
    variant: str
    steps: tuple[TrainerStep, ...]

    @property
    def title(self) -> str:
        return f"{self.mode.title()} 251 in {self.key} ({self.variant})"

    @property
    def exercise_id(self) -> str:
        return f"{self.mode}:{self.key}:{self.variant}"


@dataclass(frozen=True)
class NoteMatchResult:
    exact: bool
    pitch_class_only: bool
    expected_midis: tuple[int, ...]
    played_midis: tuple[int, ...]


def note_name_to_midi(note_name: str) -> int:
    accidental = ""
    octave_text = note_name[-1]
    if not octave_text.isdigit():
        msg = f"Unsupported note name: {note_name}"
        raise ValueError(msg)

    if len(note_name) >= 3 and note_name[1] in {"#", "b"}:
        accidental = note_name[1]

    note_key = note_name[0] + accidental
    octave = int(octave_text)
    pitch_class = NOTE_NAME_TO_PITCH_CLASS[note_key]
    return 12 * (octave + 1) + pitch_class


def midi_to_note_name(midi_note: int, *, prefer_flats: bool) -> str:
    note_names = FLAT_NOTE_NAMES if prefer_flats else SHARP_NOTE_NAMES
    pitch_class = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{note_names[pitch_class]}{octave}"


def compare_note_sets(expected: list[str] | tuple[str, ...], played: list[str]) -> NoteMatchResult:
    expected_midis = tuple(sorted(note_name_to_midi(note_name) for note_name in expected))
    played_midis = tuple(sorted(note_name_to_midi(note_name) for note_name in played))
    exact = expected_midis == played_midis
    expected_pitch_classes = sorted(midi_note % 12 for midi_note in expected_midis)
    played_pitch_classes = sorted(midi_note % 12 for midi_note in played_midis)
    return NoteMatchResult(
        exact=exact,
        pitch_class_only=expected_pitch_classes == played_pitch_classes,
        expected_midis=expected_midis,
        played_midis=played_midis,
    )


def build_visible_exercises() -> dict[str, TrainerExercise]:
    exercises: list[TrainerExercise] = []

    for key_name in VISIBLE_MAJOR_KEYS:
        for variant in ("Type A", "Type B"):
            exercises.append(build_major_exercise(key=key_name, variant=variant))

    for key_name in VISIBLE_MINOR_KEYS:
        for variant in ("Type A", "Type B"):
            exercises.append(build_minor_exercise(key=key_name, variant=variant))

    return {exercise.exercise_id: exercise for exercise in exercises}


def build_major_exercise(*, key: str, variant: str) -> TrainerExercise:
    voicings = MAJOR_EXERCISE_VOICINGS[key][variant]
    steps = tuple(
        TrainerStep(
            symbol=symbol,
            expected_notes=notes,
            display_notes=notes,
            lower_notes=(notes[0],),
            upper_notes=notes[1:],
            staff="grand",
        )
        for symbol, notes in voicings
    )
    return TrainerExercise(mode="major", key=key, variant=variant, steps=steps)


def build_minor_exercise(*, key: str, variant: str) -> TrainerExercise:
    voicings = MINOR_EXERCISE_VOICINGS[key][variant]
    steps = tuple(
        TrainerStep(
            symbol=symbol,
            expected_notes=notes,
            display_notes=notes,
            lower_notes=notes,
            upper_notes=(),
            staff="bass",
        )
        for symbol, notes in voicings
    )
    return TrainerExercise(mode="minor", key=key, variant=variant, steps=steps)


def render_progression_svg(
    exercise: TrainerExercise,
    *,
    active_step: int,
    played_notes: list[str],
) -> str:
    if not NOTATION_BACKEND_AVAILABLE:
        return _render_notation_error()

    legend = _render_step_legend(exercise=exercise, active_step=active_step)
    progression_svg = _render_score_svg(
        score_xml=_exercise_musicxml(exercise),
        container_class="trainer-score",
    )

    played_svg = ""
    if played_notes:
        prefer_flats = _exercise_prefers_flats(exercise)
        respelled_played_notes = tuple(
            midi_to_note_name(note_name_to_midi(note_name), prefer_flats=prefer_flats)
            for note_name in played_notes
        )
        played_upper, played_lower = _split_played_notes_for_staff(
            played_notes=list(respelled_played_notes),
            staff=exercise.steps[active_step].staff,
        )
        played_step = TrainerStep(
            symbol="Played",
            expected_notes=respelled_played_notes,
            display_notes=respelled_played_notes,
            lower_notes=played_lower,
            upper_notes=played_upper,
            staff=exercise.steps[active_step].staff,
        )
        played_svg = (
            '<div class="trainer-played">'
            '<div class="trainer-subtitle">Played shape</div>'
            f"{
                _render_score_svg(
                    score_xml=_single_step_musicxml(
                        played_step,
                        key_signature_sharps=_exercise_key_signature_sharps(exercise),
                    ),
                    container_class='trainer-score trainer-score--played',
                )
            }"
            "</div>"
        )

    score_layout = (
        "<div class='trainer-score-grid'>"
        "<div class='trainer-expected'>"
        "<div class='trainer-subtitle'>Expected</div>"
        f"{progression_svg}"
        "</div>"
        f"{played_svg}"
        "</div>"
    )

    return (
        "<div class='trainer-wrap'>"
        f"{_trainer_style_block()}"
        f"<div class='trainer-title'>{escape(exercise.title)}</div>"
        f"{legend}"
        f"{score_layout}"
        "</div>"
    )


def _render_notation_error() -> str:
    return (
        "<div class='trainer-wrap'>"
        f"{_trainer_style_block()}"
        "<div class='trainer-error'>"
        "Notation backend unavailable. Install `music21` and `verovio` to render the staves."
        f"<br/><small>{escape(NOTATION_BACKEND_ERROR)}</small>"
        "</div></div>"
    )


def _render_step_legend(*, exercise: TrainerExercise, active_step: int) -> str:
    items: list[str] = []
    for index, step in enumerate(exercise.steps):
        class_name = "trainer-step trainer-step--active" if index == active_step else "trainer-step"
        items.append(f"<div class='{class_name}'>{escape(step.symbol)}</div>")
    return "<div class='trainer-steps'>" + "".join(items) + "</div>"


@lru_cache(maxsize=128)
def _exercise_musicxml(exercise: TrainerExercise) -> str:
    score = _build_music21_score(exercise)
    return musicxml.m21ToXml.GeneralObjectExporter(score).parse().decode("utf-8")


@lru_cache(maxsize=256)
def _single_step_musicxml(step: TrainerStep, key_signature_sharps: int = 0) -> str:
    score = _build_single_step_score(step, key_signature_sharps=key_signature_sharps)
    return musicxml.m21ToXml.GeneralObjectExporter(score).parse().decode("utf-8")


def _build_music21_score(exercise: TrainerExercise):
    if not NOTATION_BACKEND_AVAILABLE:
        msg = "Notation backend unavailable"
        raise RuntimeError(msg)

    if exercise.mode == "major":
        return _build_major_score(exercise)
    return _build_minor_score(exercise)


def _build_major_score(exercise: TrainerExercise):
    score = stream.Score(id=exercise.exercise_id)
    upper_part = stream.Part(id=f"{exercise.exercise_id}-upper")
    lower_part = stream.Part(id=f"{exercise.exercise_id}-lower")

    upper_part.append(clef.TrebleClef())
    lower_part.append(clef.BassClef())

    ks = key.Key(exercise.key).sharps
    upper_part.append(key.KeySignature(ks))
    lower_part.append(key.KeySignature(ks))

    for step in exercise.steps:
        upper_part.append(_build_measure(label=None, pitches=step.upper_notes))
        lower_part.append(_build_measure(label=None, pitches=step.lower_notes))

    score.insert(0, upper_part)
    score.insert(0, lower_part)
    score.insert(0, layout.StaffGroup([upper_part, lower_part], symbol="brace", barTogether=True))
    return score


def _build_minor_score(exercise: TrainerExercise):
    score = stream.Score(id=exercise.exercise_id)
    part = stream.Part(id=f"{exercise.exercise_id}-bass")
    part.append(clef.BassClef())
    ks = key.Key(exercise.key, "minor").sharps
    part.append(key.KeySignature(ks))

    for step in exercise.steps:
        part.append(_build_measure(label=None, pitches=step.lower_notes))

    score.insert(0, part)
    return score


def _build_single_step_score(step: TrainerStep, *, key_signature_sharps: int):
    score = stream.Score(id=f"single-{step.symbol}")
    if step.staff == "grand":
        upper_part = stream.Part(id="played-upper")
        lower_part = stream.Part(id="played-lower")
        upper_part.append(clef.TrebleClef())
        lower_part.append(clef.BassClef())
        upper_part.append(key.KeySignature(key_signature_sharps))
        lower_part.append(key.KeySignature(key_signature_sharps))
        upper_part.append(_build_measure(label=None, pitches=_played_upper(step)))
        lower_part.append(_build_measure(label=None, pitches=_played_lower(step)))
        score.insert(0, upper_part)
        score.insert(0, lower_part)
        score.insert(
            0, layout.StaffGroup([upper_part, lower_part], symbol="brace", barTogether=True)
        )
        return score

    part = stream.Part(id="played-bass")
    part.append(clef.BassClef())
    part.append(key.KeySignature(key_signature_sharps))
    part.append(_build_measure(label=None, pitches=step.display_notes))
    score.insert(0, part)
    return score


def _exercise_prefers_flats(exercise: TrainerExercise) -> bool:
    if exercise.mode == "major":
        return exercise.key in MAJOR_FLAT_KEYS
    return exercise.key in MINOR_FLAT_KEYS


def _exercise_key_signature_sharps(exercise: TrainerExercise) -> int:
    sharps = (
        key.Key(exercise.key).sharps
        if exercise.mode == "major"
        else key.Key(exercise.key, "minor").sharps
    )
    return 0 if sharps is None else sharps


def _played_upper(step: TrainerStep) -> tuple[str, ...]:
    if step.upper_notes:
        return step.upper_notes
    if len(step.display_notes) <= 2:
        return step.display_notes
    return step.display_notes[1:]


def _played_lower(step: TrainerStep) -> tuple[str, ...]:
    if step.lower_notes:
        return step.lower_notes
    if step.display_notes:
        return (step.display_notes[0],)
    return ()


def _split_played_notes_for_staff(
    *, played_notes: list[str], staff: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if staff == "bass":
        return (), tuple(played_notes)

    upper = tuple(note_name for note_name in played_notes if note_name_to_midi(note_name) >= 60)
    lower = tuple(note_name for note_name in played_notes if note_name_to_midi(note_name) < 60)
    if not lower and played_notes:
        lower = (played_notes[0],)
        upper = tuple(played_notes[1:])
    return upper, lower


def _build_measure(*, label: str | None, pitches: tuple[str, ...]):
    measure = stream.Measure()
    time_signature = meter.TimeSignature("4/4")
    time_signature.style.hideObjectOnPrint = True
    measure.insert(0, time_signature)

    if not pitches:
        measure.append(note.Rest(quarterLength=4))
        return measure

    if len(pitches) == 1:
        measure.append(note.Note(pitches[0], quarterLength=4))
    else:
        measure.append(chord.Chord(pitches, quarterLength=4))
    return measure


def _render_score_svg(*, score_xml: str, container_class: str) -> str:
    toolkit = verovio.toolkit(False)
    resource_path = Path(verovio.__file__).resolve().parent / "data"
    toolkit.setResourcePath(str(resource_path))
    svg = toolkit.renderData(
        score_xml,
        {
            "inputFrom": "xml",
            "pageWidth": 1800,
            "scale": 38,
            "svgViewBox": True,
            "adjustPageHeight": True,
            "header": "none",
            "footer": "none",
            "breaks": "none",
        },
    )
    if not svg:
        return (
            f"<div class='{container_class}'>"
            "<div class='trainer-error'>Verovio could not render this score.</div>"
            "</div>"
        )
    return f"<div class='{container_class}'>{svg}</div>"


def _voice_two_notes(
    first_pitch_class: int,
    second_pitch_class: int,
    *,
    prefer_flats: bool,
    low_target: int,
    high_target: int,
) -> tuple[str, str]:
    lower_midi = _nearest_midi_for_pitch_class(first_pitch_class, target=low_target)
    upper_midi = _nearest_midi_for_pitch_class(second_pitch_class, target=high_target)
    while upper_midi <= lower_midi:
        upper_midi += 12
    return (
        midi_to_note_name(lower_midi, prefer_flats=prefer_flats),
        midi_to_note_name(upper_midi, prefer_flats=prefer_flats),
    )


def _voice_three_notes(
    first_pitch_class: int,
    second_pitch_class: int,
    third_pitch_class: int,
    *,
    prefer_flats: bool,
    low_target: int,
    middle_target: int,
    high_target: int,
) -> tuple[str, str, str]:
    first_midi = _nearest_midi_for_pitch_class(first_pitch_class, target=low_target)
    second_midi = _nearest_midi_for_pitch_class(second_pitch_class, target=middle_target)
    while second_midi <= first_midi:
        second_midi += 12
    third_midi = _nearest_midi_for_pitch_class(third_pitch_class, target=high_target)
    while third_midi <= second_midi:
        third_midi += 12
    return (
        midi_to_note_name(first_midi, prefer_flats=prefer_flats),
        midi_to_note_name(second_midi, prefer_flats=prefer_flats),
        midi_to_note_name(third_midi, prefer_flats=prefer_flats),
    )


def _root_note(pitch_class: int, *, prefer_flats: bool, target: int) -> str:
    midi_note = _nearest_midi_for_pitch_class(pitch_class, target=target)
    return midi_to_note_name(midi_note, prefer_flats=prefer_flats)


def _nearest_midi_for_pitch_class(pitch_class: int, *, target: int) -> int:
    target_pitch_class = pitch_class % 12
    base = target - (target % 12) + target_pitch_class
    candidates = [base - 12, base, base + 12]
    return min(candidates, key=lambda candidate: abs(candidate - target))


def _note_name(pitch_class: int, *, prefer_flats: bool) -> str:
    note_names = FLAT_NOTE_NAMES if prefer_flats else SHARP_NOTE_NAMES
    return note_names[pitch_class % 12]


def _trainer_style_block() -> str:
    return """
    <style>
    .trainer-wrap {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 16px;
        padding: 1rem 1rem 0.6rem 1rem;
        background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(241,245,249,0.92));
    }
    .trainer-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }
    .trainer-steps {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
    }
    .trainer-step {
        border: 1px solid rgba(148, 163, 184, 0.45);
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.88rem;
        color: #334155;
        background: rgba(255, 255, 255, 0.75);
    }
    .trainer-step--active {
        border-color: #2563eb;
        background: rgba(37, 99, 235, 0.12);
        color: #1d4ed8;
        font-weight: 700;
    }
    .trainer-score {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        overflow-x: auto;
    }
    .trainer-score-grid {
        display: grid;
        grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);
        gap: 0.9rem;
        align-items: start;
    }
    .trainer-expected,
    .trainer-played {
        min-width: 0;
    }
    .trainer-score svg {
        width: 100%;
        height: auto;
        display: block;
    }
    .trainer-subtitle {
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        margin: 0 0 0.35rem 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .trainer-error {
        color: #991b1b;
        background: rgba(254, 242, 242, 0.9);
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    @media (max-width: 900px) {
        .trainer-score-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
