from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


SCRIPT_PATH = Path("scripts") / "midi_monitor.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List MIDI inputs or monitor a selected input for incoming messages."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available MIDI input ports and exit.",
    )
    parser.add_argument(
        "--port",
        help="Exact MIDI input port name to open.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.01,
        help="Polling sleep in seconds while waiting for messages.",
    )
    return parser


def _get_mido_functions(
    mido_module: Any,
) -> tuple[Callable[[], Iterable[str]], Callable[[str], Any]]:
    get_input_names = getattr(mido_module, "get_input_names", None)
    open_input = getattr(mido_module, "open_input", None)

    if not callable(get_input_names):
        raise RuntimeError("The installed `mido` package does not expose `get_input_names()`.")
    if not callable(open_input):
        raise RuntimeError("The installed `mido` package does not expose `open_input()`.")

    return cast("Callable[[], Iterable[str]]", get_input_names), cast(
        "Callable[[str], Any]", open_input
    )


def _override_command(port_name: str) -> str:
    quoted_port = shlex.quote(port_name)
    return f"python {SCRIPT_PATH.as_posix()} --port {quoted_port}"


def _select_port(ports: list[str], requested_port: str | None) -> str | None:
    if not ports:
        return None
    if requested_port is not None:
        return requested_port
    return ports[0]


def _print_port_selection(selected_port: str, ports: list[str], requested_port: str | None) -> None:
    if requested_port is not None:
        print(f"Selected MIDI input: {selected_port} (requested with --port)")
        return

    if len(ports) == 1:
        print(f"Selected MIDI input: {selected_port} (only available input)")
        return

    print(f"Selected MIDI input: {selected_port} (default from {len(ports)} available inputs)")
    print("Use one of these commands to monitor a different input:")
    for port_name in ports:
        if port_name == selected_port:
            continue
        print(f"- {_override_command(port_name)}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        import mido
    except ModuleNotFoundError:
        print("`mido` is not installed in the current Python environment.", file=sys.stderr)
        return 1

    try:
        get_input_names, open_input = _get_mido_functions(mido)
        ports = list(get_input_names())
    except Exception as exc:
        print(f"Unable to query MIDI inputs: {exc}", file=sys.stderr)
        return 1

    if args.list:
        if not ports:
            print("No MIDI input ports found.")
            return 0
        print("Available MIDI input ports:")
        for index, port_name in enumerate(ports, start=1):
            print(f"{index}. {port_name}")
        return 0

    if args.port is not None and args.port not in ports:
        print(f"Unknown MIDI input port: {args.port}", file=sys.stderr)
        print("Known ports:", file=sys.stderr)
        for port_name in ports:
            print(f"- {port_name}", file=sys.stderr)
        return 1

    selected_port = _select_port(ports, args.port)
    if selected_port is None:
        print("No MIDI input ports found.")
        return 0

    _print_port_selection(selected_port, ports, args.port)
    print(f"Opening MIDI input: {selected_port}")
    print("Press Ctrl+C to stop.")

    message_count = 0
    try:
        with open_input(selected_port) as input_port:
            while True:
                seen_message = False
                for message in input_port.iter_pending():
                    seen_message = True
                    message_count += 1
                    print(f"[{message_count}] {message}")
                if not seen_message:
                    time.sleep(args.timeout)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception as exc:
        print(f"Failed while monitoring MIDI input: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
