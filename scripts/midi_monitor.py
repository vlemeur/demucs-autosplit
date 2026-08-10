from __future__ import annotations

import argparse
import sys
import time


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


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        import mido
    except ModuleNotFoundError:
        print("`mido` is not installed in the current Python environment.", file=sys.stderr)
        return 1

    try:
        ports = list(mido.get_input_names())
    except Exception as exc:
        print(f"Unable to query MIDI inputs: {exc}", file=sys.stderr)
        return 1

    if args.list or not args.port:
        if not ports:
            print("No MIDI input ports found.")
            return 0
        print("Available MIDI input ports:")
        for index, port_name in enumerate(ports, start=1):
            print(f"{index}. {port_name}")
        if args.list:
            return 0
        print('\nRe-run with --port "Exact Port Name" to monitor one input.')
        return 0

    if args.port not in ports:
        print(f"Unknown MIDI input port: {args.port}", file=sys.stderr)
        print("Known ports:", file=sys.stderr)
        for port_name in ports:
            print(f"- {port_name}", file=sys.stderr)
        return 1

    print(f"Opening MIDI input: {args.port}")
    print("Press Ctrl+C to stop.")

    message_count = 0
    try:
        with mido.open_input(args.port) as input_port:
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
