"""
MIDI I/O module for music-workbench.

Provides MIDI device discovery, input handling, and live chord detection hooks.
"""

from midi_io.chord_detector import ChordDetectionResult, ChordDetector, LiveChordDetector
from midi_io.handler import (
    MIDIDeviceQueryResult,
    MIDIHandler,
    list_midi_devices,
    query_midi_devices,
)

__all__ = [
    "MIDIHandler",
    "ChordDetector",
    "LiveChordDetector",
    "ChordDetectionResult",
    "MIDIDeviceQueryResult",
    "list_midi_devices",
    "query_midi_devices",
]
