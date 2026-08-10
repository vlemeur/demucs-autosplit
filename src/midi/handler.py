"""
MIDI input handler for real-time note and chord detection.

This module provides functionality to:
- List available MIDI input devices
- Connect to a MIDI device and listen for note events
- Track currently pressed notes
- Call callbacks when notes change
"""

import logging
from collections.abc import Callable

import mido

logger = logging.getLogger(__name__)

# Type alias for note callback
NoteCallback = Callable[[list[str]], None]

# MIDI note number to note name mapping (0-127)
MIDI_NOTE_TO_NAME: list[str] = [
    "C-1",
    "C#-1",
    "D-1",
    "D#-1",
    "E-1",
    "F-1",
    "F#-1",
    "G-1",
    "G#-1",
    "A-1",
    "A#-1",
    "B-1",
    "C0",
    "C#0",
    "D0",
    "D#0",
    "E0",
    "F0",
    "F#0",
    "G0",
    "G#0",
    "A0",
    "A#0",
    "B0",
    "C1",
    "C#1",
    "D1",
    "D#1",
    "E1",
    "F1",
    "F#1",
    "G1",
    "G#1",
    "A1",
    "A#1",
    "B1",
    "C2",
    "C#2",
    "D2",
    "D#2",
    "E2",
    "F2",
    "F#2",
    "G2",
    "G#2",
    "A2",
    "A#2",
    "B2",
    "C3",
    "C#3",
    "D3",
    "D#3",
    "E3",
    "F3",
    "F#3",
    "G3",
    "G#3",
    "A3",
    "A#3",
    "B3",
    "C4",
    "C#4",
    "D4",
    "D#4",
    "E4",
    "F4",
    "F#4",
    "G4",
    "G#4",
    "A4",
    "A#4",
    "B4",
    "C5",
    "C#5",
    "D5",
    "D#5",
    "E5",
    "F5",
    "F#5",
    "G5",
    "G#5",
    "A5",
    "A#5",
    "B5",
    "C6",
    "C#6",
    "D6",
    "D#6",
    "E6",
    "F6",
    "F#6",
    "G6",
    "G#6",
    "A6",
    "A#6",
    "B6",
    "C7",
    "C#7",
    "D7",
    "D#7",
    "E7",
    "F7",
    "F#7",
    "G7",
    "G#7",
    "A7",
    "A#7",
    "B7",
    "C8",
    "C#8",
    "D8",
    "D#8",
    "E8",
    "F8",
    "F#8",
    "G8",
]


def list_midi_devices() -> list[str]:
    """
    List all available MIDI input device names.

    Returns
    -------
    list[str]
        List of MIDI input device names. Returns empty list if no devices found.
    """
    try:
        # mido types are not recognized by ty, using type: ignore
        devices = mido.get_input_names()  # type: ignore
        logger.info(f"Found {len(devices)} MIDI input devices: {devices}")
        return devices
    except Exception as e:
        logger.error(f"Failed to list MIDI devices: {e}")
        return []


class MIDIHandler:
    """
    Handler for MIDI input with real-time note tracking.

    This class manages a MIDI input connection and tracks which notes
    are currently pressed. It calls a callback whenever the set of
    pressed notes changes.

    Parameters
    ----------
    device_name : str
        Name of the MIDI input device to connect to.
    note_callback : NoteCallback, optional
        Function to call when notes change. Receives a list of note names
        (e.g., ["C4", "E4", "G4"]).
    velocity_threshold : int, default=1
        Minimum velocity to consider a note as "on". Notes with velocity below this are ignored.

    Examples
    --------
    >>> def on_notes_changed(notes):
    ...     print(f"Current notes: {notes}")
    >>> handler = MIDIHandler("Nord Stage 3", on_notes_changed)
    >>> handler.start()
    >>> # ... play some notes ...
    >>> handler.stop()
    """

    def __init__(
        self,
        device_name: str,
        note_callback: NoteCallback | None = None,
        velocity_threshold: int = 1,
    ) -> None:
        self.device_name = device_name
        self.note_callback = note_callback
        self.velocity_threshold = velocity_threshold
        self._port: object | None = None  # mido.Port | None
        self._active_notes: dict[int, int] = {}  # {midi_note_number: velocity}
        self._running: bool = False

    def start(self) -> bool:
        """
        Start listening to the MIDI device.

        Returns
        -------
        bool
            True if successfully started, False otherwise.
        """
        if self._running:
            logger.warning("MIDI handler is already running")
            return False

        try:
            # mido types are not recognized by ty, using type: ignore
            self._port = mido.open_input(self.device_name)  # type: ignore
            self._running = True
            logger.info(f"Started listening to MIDI device: {self.device_name}")
            self._listen()
            return True
        except Exception as e:
            logger.error(f"Failed to start MIDI handler for {self.device_name}: {e}")
            self._running = False
            return False

    def stop(self) -> None:
        """Stop listening to the MIDI device."""
        self._running = False
        if self._port:
            try:
                self._port.close()  # type: ignore
                logger.info(f"Stopped MIDI device: {self.device_name}")
            except Exception as e:
                logger.error(f"Error stopping MIDI port: {e}")
            finally:
                self._port = None

    def _listen(self) -> None:
        """Internal method to process MIDI messages."""
        if not self._port or not self._running:
            return

        try:
            for msg in self._port:  # type: ignore
                if not self._running:
                    break

                self._process_message(msg)

        except Exception as e:
            logger.error(f"MIDI listener error: {e}")
            self.stop()

    def _process_message(self, msg: object) -> None:
        """Process a MIDI message and update note state."""
        # msg is a mido.Message object, but ty doesn't recognize its attributes
        if msg.type == "note_on":  # type: ignore
            self._handle_note_on(msg.note, msg.velocity)  # type: ignore
        elif msg.type == "note_off":  # type: ignore
            self._handle_note_off(msg.note, msg.velocity)  # type: ignore

    def _handle_note_on(self, note_num: int, velocity: int) -> None:
        """Handle a note-on MIDI message."""
        if velocity < self.velocity_threshold:
            # Note-on with velocity 0 is equivalent to note-off
            self._handle_note_off(note_num, velocity)
            return

        if note_num not in self._active_notes:
            self._active_notes[note_num] = velocity
            self._notify_callback()

    def _handle_note_off(self, note_num: int, velocity: int) -> None:
        """Handle a note-off MIDI message."""
        if note_num in self._active_notes:
            del self._active_notes[note_num]
            self._notify_callback()

    def _notify_callback(self) -> None:
        """Call the note callback with current notes."""
        if self.note_callback:
            try:
                notes = self.get_active_notes()
                self.note_callback(notes)
            except Exception as e:
                logger.error(f"Error in note callback: {e}")

    def get_active_notes(self) -> list[str]:
        """
        Get the list of currently active note names.

        Returns
        -------
        list[str]
            List of note names (e.g., ["C4", "E4", "G4"]).
        """
        note_numbers = sorted(self._active_notes.keys())
        return [MIDI_NOTE_TO_NAME[n] for n in note_numbers if 0 <= n < len(MIDI_NOTE_TO_NAME)]

    def get_active_notes_raw(self) -> dict[int, int]:
        """
        Get the raw dictionary of active MIDI notes and velocities.

        Returns
        -------
        dict[int, int]
            Dictionary mapping MIDI note numbers to velocities.
        """
        return self._active_notes.copy()

    @property
    def is_running(self) -> bool:
        """Check if the handler is currently running."""
        return self._running

    @property
    def device(self) -> str:
        """Get the device name."""
        return self.device_name
