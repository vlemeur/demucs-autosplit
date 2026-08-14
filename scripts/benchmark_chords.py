import argparse
from pathlib import Path

from audio_analysis.chord_detection import list_chord_detection_backends
from audio_analysis.inspection import benchmark_chord_detection


def _build_parser() -> argparse.ArgumentParser:
    backend_ids = [backend.backend_id for backend in list_chord_detection_backends()]
    parser = argparse.ArgumentParser(
        description="Benchmark chord-detection backends on one audio file"
    )
    parser.add_argument("input_wav", type=Path, help="Input WAV file to analyze")
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=backend_ids,
        help="Backend to run. Repeat the option to benchmark several backends.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "chord_benchmark",
        help="Directory where backend .lab files will be written",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    backend_ids = args.backends or [
        backend.backend_id for backend in list_chord_detection_backends()
    ]
    results = benchmark_chord_detection(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        backend_ids=backend_ids,
    )

    for result in results:
        status = "ok" if result.success else "failed"
        output_lab = str(result.output_lab) if result.output_lab else "-"
        print(
            f"{result.label}: {status} | runtime={result.runtime_s:.2f}s | "
            f"segments={result.segment_count} | unique_labels={result.unique_label_count} | "
            f"output={output_lab}"
        )
        if result.error:
            print(f"  error: {result.error}")


if __name__ == "__main__":
    main()
