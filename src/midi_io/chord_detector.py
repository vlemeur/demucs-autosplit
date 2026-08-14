"""
Real-time chord detection from MIDI notes.

This module provides chord detection functionality that:
- Takes a list of currently pressed MIDI notes
- Identifies the most likely chord being played
- Handles inversions and chord voicings
- Provides multiple candidate chords with confidence scores
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harmony.chord_library import detect_chord_from_notes, detect_chords_from_notes

if TYPE_CHECKING:
    from midi_io.handler import MIDIHandler

logger = logging.getLogger(__name__)

# Type alias for chord callback
ChordCallback = Callable[[str | None], None]
MultiChordCallback = Callable[[list[str]], None]


@dataclass
class ChordDetectionResult:
    """Result of a chord detection operation."""

    primary_chord: str | None
    alternative_chords: list[str] = field(default_factory=list)
    input_notes: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __str__(self) -> str:
        if self.primary_chord:
            return f"{self.primary_chord} (confidence: {self.confidence:.2f})"
        return "No chord detected"


class ChordDetector:
    """
    Real-time chord detector for MIDI notes.

    This class takes a stream of MIDI notes and detects the chord being played.
    It can work in two modes:
    - Single chord mode: Returns only the most likely chord
    - Multi chord mode: Returns all possible matching chords

    Parameters
    ----------
    min_notes : int, default=2
        Minimum number of notes required to attempt chord detection.
    min_match_ratio : float, default=0.7
        Minimum ratio of chord notes that must match input notes.
    allow_inversions : bool, default=True
        Whether to detect inverted chords (same notes, different bass note).

    Examples
    --------
    >>> detector = ChordDetector()
    >>> result = detector.detect(["C4", "E4", "G4"])
    >>> print(result.primary_chord)
    C:maj
    >>> result = detector.detect(["E4", "G4", "C5"])
    >>> print(result.primary_chord)
    C:maj
    """

    def __init__(
        self,
        min_notes: int = 2,
        min_match_ratio: float = 0.7,
        allow_inversions: bool = True,
    ) -> None:
        self.min_notes = min_notes
        self.min_match_ratio = min_match_ratio
        self.allow_inversions = allow_inversions

    def detect(self, notes: list[str]) -> ChordDetectionResult:
        """
        Detect the chord from a list of note names.

        Parameters
        ----------
        notes : list[str]
            List of note names with octaves (e.g., ["C4", "E4", "G4"]).

        Returns
        -------
        ChordDetectionResult
            Object containing the detected chord(s) and metadata.
        """
        if not notes or len(notes) < self.min_notes:
            return ChordDetectionResult(
                primary_chord=None,
                alternative_chords=[],
                input_notes=notes,
                confidence=0.0,
            )

        # Detect primary chord
        primary_chord = detect_chord_from_notes(notes, self.min_match_ratio)

        # Detect all matching chords
        all_chords = detect_chords_from_notes(notes, self.min_match_ratio)

        # Remove the primary chord from alternatives if present
        alternatives = [c for c in all_chords if c != primary_chord]

        # Calculate confidence (ratio of chord notes found in input)
        confidence = 0.0
        if primary_chord:
            from harmony.chord_library import get_chord_notes

            chord_notes = get_chord_notes(primary_chord)
            if chord_notes:
                chord_base_notes = {n[:-1] for n in chord_notes if len(n) > 1}
                input_base_notes = {n[:-1] if len(n) > 1 and n[-1].isdigit() else n for n in notes}
                confidence = len(chord_base_notes & input_base_notes) / len(chord_base_notes)

        return ChordDetectionResult(
            primary_chord=primary_chord,
            alternative_chords=alternatives[:5],  # Limit to top 5 alternatives
            input_notes=notes,
            confidence=confidence,
        )

    def detect_simple(self, notes: list[str]) -> str | None:
        """
        Simple chord detection that returns only the primary chord.

        Parameters
        ----------
        notes : list[str]
            List of note names with octaves.

        Returns
        -------
        str or None
            The detected chord label, or None if no chord detected.
        """
        return self.detect(notes).primary_chord

    def detect_all(self, notes: list[str]) -> list[str]:
        """
        Get all possible matching chords for the input notes.

        Parameters
        ----------
        notes : list[str]
            List of note names with octaves.

        Returns
        -------
        list[str]
            List of all matching chord labels, sorted by likelihood.
        """
        result = self.detect(notes)
        if result.primary_chord:
            return result.alternative_chords + [result.primary_chord]
        return result.alternative_chords


class LiveChordDetector:
    """
    Live chord detector that integrates with MIDIHandler.

    This class combines MIDI input handling with chord detection,
    providing a simple interface for real-time chord detection.

    Parameters
    ----------
    device_name : str
        Name of the MIDI input device.
    chord_callback : ChordCallback, optional
        Function to call when chord changes. Receives chord label or None.
    detector_kwargs : dict, optional
        Additional keyword arguments to pass to ChordDetector.

    Examples
    --------
    >>> def on_chord_changed(chord):
    ...     print(f"Detected chord: {chord}")
    >>> live_detector = LiveChordDetector("Nord Stage 3", on_chord_changed)
    >>> live_detector.start()
    >>> # ... play some chords ...
    >>> live_detector.stop()
    """

    def __init__(
        self,
        device_name: str,
        chord_callback: ChordCallback | None = None,
        **detector_kwargs,
    ) -> None:
        self.device_name = device_name
        self.chord_callback = chord_callback
        self.detector = ChordDetector(**detector_kwargs)
        self._handler: MIDIHandler | None = None
        self._current_chord: str | None = None

    def start(self) -> bool:
        """
        Start listening for MIDI notes and detecting chords.

        Returns
        -------
        bool
            True if successfully started, False otherwise.
        """

        def note_callback(notes: list[str]) -> None:
            new_chord = self.detector.detect_simple(notes)
            if new_chord != self._current_chord:
                self._current_chord = new_chord
                if self.chord_callback:
                    try:
                        self.chord_callback(new_chord)
                    except Exception as e:
                        logger.error(f"Error in chord callback: {e}")

        self._handler = MIDIHandler(self.device_name, note_callback)
        return self._handler.start()

    def stop(self) -> None:
        """Stop listening and detecting."""
        if self._handler:
            self._handler.stop()
            self._handler = None
        self._current_chord = None

    @property
    def current_chord(self) -> str | None:
        """Get the currently detected chord."""
        return self._current_chord

    @property
    def is_running(self) -> bool:
        """Check if the detector is running."""
        return self._handler is not None and self._handler.is_running

    def get_current_notes(self) -> list[str]:
        """Get the list of currently pressed notes."""
        if self._handler:
            return self._handler.get_active_notes()
        return []
