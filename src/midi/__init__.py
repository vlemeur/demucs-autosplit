"""
MIDI module for music-workbench.

Provides MIDI input handling and chord detection for live harmony workflows.
"""

from midi.chord_detector import ChordDetectionResult, ChordDetector, LiveChordDetector
from midi.handler import MIDIDeviceQueryResult, MIDIHandler, list_midi_devices, query_midi_devices

__all__ = [
    "MIDIHandler",
    "ChordDetector",
    "LiveChordDetector",
    "ChordDetectionResult",
    "MIDIDeviceQueryResult",
    "list_midi_devices",
    "query_midi_devices",
]
