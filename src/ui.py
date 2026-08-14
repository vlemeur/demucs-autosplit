from __future__ import annotations

import base64
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from app_files import (
    clear_workspace,
    read_text_file,
    safe_filename,
    save_bytes_to_file,
    validate_extension,
)
from audio_analysis.chord_detection import (
    get_chord_detection_backend,
    list_chord_detection_backends,
)
from audio_analysis.inspection import (
    benchmark_chord_detection,
    compute_chord_label_agreement,
    extract_wav_clip_bytes,
    load_waveform_for_plot,
    predict_chords_for_stem,
    read_chords_lab,
)
from audio_analysis.separation import DEFAULT_MODEL, DEMUCS_MODELS
from audio_analysis.workspace import list_stems_wav, read_stems, run_split, zip_stems
from harmony.trainer import (
    VISIBLE_MAJOR_KEYS,
    VISIBLE_MINOR_KEYS,
    build_visible_exercises,
    compare_note_sets,
    render_detected_voicing_gallery,
    render_progression_svg,
)
from midi_io import query_midi_devices
from midi_io.chord_detector import ChordDetector
from midi_io.handler import MIDI_NOTE_TO_NAME

APP_TITLE: str = "🎵 Music Workbench"

WORK_DIR: Path = Path(".streamlit_workdir")
UPLOAD_DIR: Path = WORK_DIR / "uploads"
OUTPUT_DIR: Path = WORK_DIR / "outputs"

SUPPORTED_EXT: set[str] = {".wav", ".mp3"}

# Model-specific stems for Demucs v4
MODEL_STEMS: dict[str, list[str]] = {
    "htdemucs": ["drums", "bass", "other", "vocals"],
    "htdemucs_ft": ["drums", "bass", "other", "vocals"],
    "htdemucs_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
}

# Session state keys
SESSION_KEY_STEMS_DIR: str = "stems_dir"
SESSION_KEY_ZOOM_START_S: str = "zoom_start_s"
SESSION_KEY_ZOOM_END_S: str = "zoom_end_s"
SESSION_KEY_SELECTED_MODEL: str = "selected_model"
SESSION_KEY_ACTIVE_WORKSPACE: str = "active_workspace"
SESSION_KEY_CHORD_BACKEND: str = "chord_backend"
SESSION_KEY_CHORD_BENCHMARK: str = "chord_benchmark"
SESSION_KEY_CHORD_SINGLE_HAND: str = "chord_single_hand"

# Live Harmony session state keys
SESSION_KEY_LIVE_NOTES: str = "live_notes"
SESSION_KEY_LIVE_CHORD: str = "live_chord"
SESSION_KEY_LIVE_CHORD_ALTS: str = "live_chord_alts"
SESSION_KEY_LIVE_CONFIDENCE: str = "live_confidence"
SESSION_KEY_LIVE_RUNNING: str = "live_running"
SESSION_KEY_LIVE_THREAD: str = "live_thread"
SESSION_KEY_LIVE_STOP_EVENT: str = "live_stop_event"
SESSION_KEY_LIVE_SNAPSHOT: str = "live_snapshot"
SESSION_KEY_LIVE_ERROR: str = "live_error"
SESSION_KEY_CHORD_SEQ: str = "chord_sequence"
SESSION_KEY_251_MODE: str = "trainer_251_mode"
SESSION_KEY_251_KEY: str = "trainer_251_key"
SESSION_KEY_251_VARIANT: str = "trainer_251_variant"
SESSION_KEY_251_STEP: str = "trainer_251_step"
SESSION_KEY_251_AUTO_ADVANCE: str = "trainer_251_auto_advance"
SESSION_KEY_251_LAST_MATCHED_STEP: str = "trainer_251_last_matched_step"
SESSION_KEY_251_CHAIN_MODE: str = "trainer_251_chain_mode"
SESSION_KEY_251_CHAIN_KEYS: str = "trainer_251_chain_keys"
SESSION_KEY_251_CHAIN_INDEX: str = "trainer_251_chain_index"
SESSION_KEY_MIDI_SELECTED_DEVICE: str = "live_selected_device"
LIVE_REFRESH_INTERVAL_S: float = 0.2
VISIBLE_251_EXERCISES = build_visible_exercises()


@dataclass(frozen=True)
class ChordsPlotConfig:
    """
    Configuration for chord waveform plotting.

    Attributes
    ----------
    start_s : float
        Start time (seconds).
    end_s : float
        End time (seconds).
    """

    start_s: float
    end_s: float


def _collapse_chord_segments(segments: list) -> list:
    """Filter out non-chords and keep unique chord labels in first-seen order."""
    collapsed: list = []
    seen_labels: set[str] = set()

    for segment in segments:
        normalized_label = segment.label.strip()
        if normalized_label in {"", "N", "N.C.", "NC", "X"}:
            continue
        if normalized_label in seen_labels:
            continue
        seen_labels.add(normalized_label)
        collapsed.append(segment)

    return collapsed


def _segments_in_time_range(segments: list, start_s: float, end_s: float) -> list:
    """Return only the chord segments that overlap the requested time range."""
    return [
        segment for segment in segments if not (segment.end_s < start_s or segment.start_s > end_s)
    ]


def _render_detected_chord_chart(segments: list) -> None:
    """Render detected chord voicings without time-based text clutter."""
    condensed = _collapse_chord_segments(segments)
    if not condensed:
        return

    st.subheader("Chord chart")
    single_hand = st.toggle(
        "Single-hand voicings",
        key=SESSION_KEY_CHORD_SINGLE_HAND,
        help="Switch between one-hand and split two-hand keyboard voicings.",
    )

    notation_labels = [segment.label for segment in condensed[:24]]
    st.markdown(
        render_detected_voicing_gallery(notation_labels, single_hand=bool(single_hand)),
        unsafe_allow_html=True,
    )
    if len(condensed) > 24:
        st.caption(
            "The voicing gallery shows the first 24 harmonic events to keep the layout readable."
        )


@dataclass
class LiveHarmonySnapshot:
    """Thread-safe snapshot for the Live Harmony listener state."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    notes: list[str] = field(default_factory=list)
    chord: str | None = None
    alternatives: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None

    def update(
        self,
        *,
        notes: list[str],
        chord: str | None,
        alternatives: list[str],
        confidence: float,
    ) -> None:
        with self.lock:
            self.notes = notes
            self.chord = chord
            self.alternatives = alternatives
            self.confidence = confidence
            self.error = None

    def set_error(self, message: str) -> None:
        with self.lock:
            self.error = message

    def reset(self) -> None:
        with self.lock:
            self.notes = []
            self.chord = None
            self.alternatives = []
            self.confidence = 0.0
            self.error = None

    def read(self) -> tuple[list[str], str | None, list[str], float, str | None]:
        with self.lock:
            return (
                self.notes.copy(),
                self.chord,
                self.alternatives.copy(),
                self.confidence,
                self.error,
            )


def _init_session_state() -> None:
    """
    Initialize Streamlit session state keys used by the app.

    Returns
    -------
    None
    """
    if SESSION_KEY_STEMS_DIR not in st.session_state:
        st.session_state[SESSION_KEY_STEMS_DIR] = None
    if SESSION_KEY_ZOOM_START_S not in st.session_state:
        st.session_state[SESSION_KEY_ZOOM_START_S] = 0.0
    if SESSION_KEY_ZOOM_END_S not in st.session_state:
        st.session_state[SESSION_KEY_ZOOM_END_S] = 0.0
    if SESSION_KEY_SELECTED_MODEL not in st.session_state:
        st.session_state[SESSION_KEY_SELECTED_MODEL] = DEFAULT_MODEL
    if SESSION_KEY_ACTIVE_WORKSPACE not in st.session_state:
        st.session_state[SESSION_KEY_ACTIVE_WORKSPACE] = "Audio Analysis"
    elif st.session_state[SESSION_KEY_ACTIVE_WORKSPACE] == "Analyse audio":
        st.session_state[SESSION_KEY_ACTIVE_WORKSPACE] = "Audio Analysis"
    if SESSION_KEY_CHORD_BACKEND not in st.session_state:
        st.session_state[SESSION_KEY_CHORD_BACKEND] = "chordmini_chordnet"
    if SESSION_KEY_CHORD_BENCHMARK not in st.session_state:
        st.session_state[SESSION_KEY_CHORD_BENCHMARK] = None


def _render_sidebar() -> tuple[str, str, str]:
    """
    Render the sidebar controls.

    Returns
    -------
    tuple[str, str, str]
        The selected model, active workspace, and chord backend.
    """
    with st.sidebar:
        st.header("Navigation")
        active_workspace = st.radio(
            "Workspace",
            options=("Audio Analysis", "Live Harmony"),
            key=SESSION_KEY_ACTIVE_WORKSPACE,
            label_visibility="collapsed",
        )

        if active_workspace == "Audio Analysis":
            st.caption("Upload a track, separate its stems, then analyze the chord progression.")
        else:
            st.caption("Play your MIDI keyboard to see notes and chords update in real time.")

        st.markdown("---")

        selected_model = st.session_state.get(SESSION_KEY_SELECTED_MODEL, DEFAULT_MODEL)
        selected_backend = st.session_state.get(SESSION_KEY_CHORD_BACKEND, "chordmini_chordnet")

        if selected_model not in DEMUCS_MODELS:
            selected_model = DEFAULT_MODEL
            st.session_state[SESSION_KEY_SELECTED_MODEL] = selected_model

        backend_ids = [backend.backend_id for backend in list_chord_detection_backends()]
        if selected_backend not in backend_ids:
            selected_backend = "chordmini_chordnet"
            st.session_state[SESSION_KEY_CHORD_BACKEND] = selected_backend

        if active_workspace == "Audio Analysis":
            st.header("Settings")
            model_tooltip = {
                "htdemucs": "Default Hybrid Transformer model - 9.0 dB SDR, best balance",
                "htdemucs_ft": "Fine-tuned Hybrid Transformer - 9.2 dB SDR, 4x slower",
                "htdemucs_6s": "6-source model: drums, bass, vocals, other, guitar, piano",
            }

            selected_model = st.selectbox(
                "Demucs Model",
                options=DEMUCS_MODELS,
                key=SESSION_KEY_SELECTED_MODEL,
                help="\n".join(
                    f"**{m}**: {model_tooltip.get(m, 'No description')}" for m in DEMUCS_MODELS
                ),
            )

            selected_backend = st.selectbox(
                "Chord Backend",
                options=backend_ids,
                key=SESSION_KEY_CHORD_BACKEND,
                format_func=lambda backend_id: get_chord_detection_backend(backend_id).label,
                help="Select the chord-recognition backend used in the Chord Detection tab.",
            )
            st.caption(get_chord_detection_backend(selected_backend).description)

            st.markdown("---")

        if st.button("🧹 Clear workspace"):
            clear_workspace(WORK_DIR)
            st.session_state[SESSION_KEY_STEMS_DIR] = None
            st.success("Workspace cleared.")

    return selected_model, active_workspace, selected_backend


def _simplify_chord_label(label: str) -> str:
    """
    Simplify chord labels for display purposes.

    This removes common inversion/voicing notation such as slash bass notes.
    Example: "G:maj/B" -> "G:maj".

    Parameters
    ----------
    label : str
        Original chord label from the .lab file.

    Returns
    -------
    str
        Simplified chord label.
    """
    simplified = label.strip()
    if "/" in simplified:
        simplified = simplified.split("/", maxsplit=1)[0].strip()
    return simplified


def _build_chords_waveform_figure(times_s, mono, segments, config: ChordsPlotConfig) -> go.Figure:
    """
    Build a Plotly figure with waveform and chord regions.

    The plot stays minimal (no chord text annotations). Chords are readable via
    a color legend (simplified labels) and full details are shown on hover
    (original label + timestamps).

    Parameters
    ----------
    times_s : numpy.ndarray
        Time axis in seconds.
    mono : numpy.ndarray
        Mono waveform samples (downsampled).
    segments : Sequence[ChordSegment]
        Chord segments parsed from a .lab file.
    config : ChordsPlotConfig
        Plot configuration.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure ready to be displayed in Streamlit.
    """
    fig = go.Figure()

    # Waveform (GL trace for performance)
    fig.add_trace(
        go.Scattergl(
            x=times_s,
            y=mono,
            mode="lines",
            name="Waveform",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    palette = go.layout.Template().layout.colorway or [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    visible_segments = _segments_in_time_range(segments, config.start_s, config.end_s)
    simplified_labels = sorted({seg.label for seg in visible_segments})
    label_to_color: dict[str, str] = {
        chord_label: palette[i % len(palette)] for i, chord_label in enumerate(simplified_labels)
    }

    # Dummy traces to build a clean legend.
    for chord_label in simplified_labels:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": label_to_color[chord_label]},
                name=chord_label,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Regions + invisible markers for hover details.
    for seg in visible_segments:
        color = label_to_color.get(seg.label, "#999999")

        fig.add_vrect(
            x0=seg.start_s,
            x1=seg.end_s,
            fillcolor=color,
            opacity=0.22,
            line_width=0,
            layer="below",
        )

        mid = 0.5 * (seg.start_s + seg.end_s)
        fig.add_trace(
            go.Scattergl(
                x=[mid],
                y=[0.0],
                mode="markers",
                marker={"size": 6, "opacity": 0.0},
                hovertemplate=(
                    f"Chord: <b>{seg.label}</b><br>"
                    f"{seg.start_s:.2f}s → {seg.end_s:.2f}s"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="x",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    fig.update_xaxes(range=[config.start_s, config.end_s])

    return fig


def _render_split_tab(model: str) -> None:
    """
    Render the stem separation tab.

    Parameters
    ----------
    model : str
        The Demucs-family model to use for separation.

    Returns
    -------
    None
    """
    stems = MODEL_STEMS.get(model, ["drums", "bass", "other", "vocals"])
    num_stems = len(stems)
    st.caption(f"Upload a track and get {num_stems} stems: {', '.join(stems)}.")

    stored_stems_dir = _get_stems_dir()
    if stored_stems_dir is not None:
        stored_stems = list_stems_wav(stems_dir=stored_stems_dir, stems=stems)
        if stored_stems:
            st.subheader("Latest split")
            st.write("**Output folder:**", str(stored_stems_dir))

            stems_bytes = read_stems(stems_dir=stored_stems_dir, stems=list(stored_stems.keys()))
            zip_bytes = zip_stems(stems_dir=stored_stems_dir, stems=list(stored_stems.keys()))
            track_name = stored_stems_dir.name

            st.download_button(
                "⬇️ Download all (ZIP)",
                data=zip_bytes,
                file_name=f"{track_name}_stems_{model}.zip",
                mime="application/zip",
            )

            num_cols = min(4, len(stored_stems))
            cols = st.columns(num_cols)
            for idx, stem in enumerate(stored_stems.keys()):
                with cols[idx % num_cols]:
                    st.download_button(
                        f"⬇️ {stem}.wav",
                        data=stems_bytes[stem],
                        file_name=f"{track_name}_{stem}.wav",
                        mime="audio/wav",
                        key=f"stored_{track_name}_{stem}",
                    )

            for stem in stored_stems.keys():
                st.markdown(f"**{stem}**")
                st.audio(stems_bytes[stem], format="audio/wav")

            st.markdown("---")

    uploaded = st.file_uploader(
        "Upload audio",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXT],
    )
    if uploaded is None:
        st.info("Choose a .wav or .mp3 file to start.")
        return

    # Save upload
    filename = safe_filename(uploaded.name)
    audio_path = save_bytes_to_file(uploaded.getbuffer(), UPLOAD_DIR / filename)

    if not validate_extension(audio_path, SUPPORTED_EXT):
        st.error(f"Unsupported file type: {audio_path.suffix}")
        return

    track_name = audio_path.stem
    st.write("**File:**", audio_path.name)
    st.write("**Model:**", model)

    if st.button("🚀 Split track", type="primary"):
        with st.status("Running stem separation…", expanded=True) as status:
            st.write(
                f"Splitting with model {model}… this can take a while depending on your CPU/GPU."
            )
            try:
                stems_dir = run_split(
                    audio_path=audio_path,
                    output_dir=OUTPUT_DIR,
                    model=model,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError) as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.exception(exc)
                return

            if stems_dir is None:
                status.update(label="Failed", state="error", expanded=True)
                st.error(
                    f"Stem separation finished, but stems were not found for model {model}. "
                    "Inspect .streamlit_workdir/outputs to verify the output layout."
                )
                return

            status.update(label="Done ✔", state="complete", expanded=False)

        st.success("✅ Split complete!")
        st.write("**Output folder:**", str(stems_dir))

        # Store stems_dir for the Chord detection tab.
        st.session_state[SESSION_KEY_STEMS_DIR] = str(stems_dir)

        stems_bytes = read_stems(stems_dir=stems_dir, stems=stems)
        zip_bytes = zip_stems(stems_dir=stems_dir, stems=stems)

        st.subheader("Download")
        st.download_button(
            "⬇️ Download all (ZIP)",
            data=zip_bytes,
            file_name=f"{track_name}_stems_{model}.zip",
            mime="application/zip",
        )

        # Create columns based on number of stems
        num_cols = min(4, num_stems)
        cols = st.columns(num_cols)
        for idx, stem in enumerate(stems):
            with cols[idx % num_cols]:
                st.download_button(
                    f"⬇️ {stem}.wav",
                    data=stems_bytes[stem],
                    file_name=f"{track_name}_{stem}.wav",
                    mime="audio/wav",
                )

        st.subheader("Preview")
        for stem in stems:
            st.markdown(f"**{stem}**")
            st.audio(stems_bytes[stem], format="audio/wav")


def _get_stems_dir() -> Path | None:
    """
    Get the stems directory stored in the session state.

    Returns
    -------
    Path or None
        Stems directory if available and existing, otherwise None.
    """
    from urllib.parse import unquote

    stems_dir_str = st.session_state.get(SESSION_KEY_STEMS_DIR)
    if stems_dir_str is None:
        st.info("Run a stem separation first, then come back here to detect chords.")
        return None

    # Streamlit may encode spaces as %20 in session state, decode them
    stems_dir_str = unquote(stems_dir_str)
    stems_dir = Path(stems_dir_str)

    if not stems_dir.exists():
        st.warning("Stored stems folder does not exist anymore. Please run a split again.")
        st.session_state[SESSION_KEY_STEMS_DIR] = None
        return None

    return stems_dir


def _get_stems_paths(stems_dir: Path, model: str) -> dict[str, Path] | None:
    """
    Collect existing stem wav paths for chord prediction.

    Parameters
    ----------
    stems_dir : Path
        Directory containing stem wav files.
    model : str
        The Demucs model used for separation.

    Returns
    -------
    dict of str to Path or None
        Mapping {stem_name: wav_path}, or None if no stems are found.
    """
    stems = MODEL_STEMS.get(model, ["drums", "bass", "other", "vocals"])
    stems_paths: dict[str, Path] = list_stems_wav(stems_dir=stems_dir, stems=stems)
    if not stems_paths:
        st.warning("No stems were found in the stored stems folder.")
        return None
    return stems_paths


def _render_chords_controls(stems_paths: dict[str, Path]) -> tuple[str, bool]:
    """
    Render stem selection and action button.

    Parameters
    ----------
    stems_paths : dict of str to Path
        Available stems.

    Returns
    -------
    selected_stem : str
        Selected stem name.
    run_button : bool
        True if the user clicked the "Predict chords" button.
    """
    cols = st.columns([2, 1])
    with cols[0]:
        stem_names = list(stems_paths.keys())
        default_index = stem_names.index("other") if "other" in stem_names else 0
        selected_stem = st.selectbox(
            "Stem", stem_names, index=default_index, label_visibility="visible"
        )
    with cols[1]:
        st.write("")
        st.write("")
        run_button = st.button("🎹 Predict chords", type="primary", width="stretch")

    return selected_stem, run_button


def _maybe_run_chords_prediction(
    input_wav: Path,
    output_lab: Path,
    backend_id: str,
    run_button: bool,
) -> None:
    """
    Run chord prediction if requested.

    Parameters
    ----------
    input_wav : Path
        Selected stem wav path.
    output_lab : Path
        Output chords lab file path.
    backend_id : str
        Chord-detection backend id.
    run_button : bool
        Whether the button was clicked.

    Returns
    -------
    None
    """
    if not run_button:
        return

    try:
        with st.spinner("Predicting chords..."):
            predict_chords_for_stem(
                input_wav=input_wav, output_lab=output_lab, backend_id=backend_id
            )
        st.success("Chord prediction completed.")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))


def _get_plot_config(duration_s: float) -> ChordsPlotConfig:
    """
    Render zoom UI and compute plot configuration.

    Parameters
    ----------
    duration_s : float
        Audio duration in seconds.

    Returns
    -------
    ChordsPlotConfig
        Plot configuration.
    """
    zoom = st.toggle("Zoom", value=False)
    if zoom:
        start_s, end_s = st.slider(
            "Time range (seconds)",
            min_value=0.0,
            max_value=float(duration_s),
            value=(0.0, float(min(duration_s, 20.0))),
            step=0.1,
        )
        st.session_state[SESSION_KEY_ZOOM_START_S] = float(start_s)
        st.session_state[SESSION_KEY_ZOOM_END_S] = float(end_s)
        return ChordsPlotConfig(start_s=float(start_s), end_s=float(end_s))

    st.session_state[SESSION_KEY_ZOOM_START_S] = 0.0
    st.session_state[SESSION_KEY_ZOOM_END_S] = float(duration_s)
    return ChordsPlotConfig(start_s=0.0, end_s=float(duration_s))


def _render_chords_plot(output_lab: Path, input_wav: Path) -> tuple[float | None, list]:
    """
    Render the Plotly visualization if chords are available.

    Parameters
    ----------
    output_lab : Path
        Chords lab file path.
    input_wav : Path
        Selected stem wav path.

    Returns
    -------
    float or None
        Audio duration in seconds if the plot was rendered, otherwise None.
    """
    if not output_lab.exists():
        st.info("No chords detected yet. Click “Predict chords” to generate chords.lab.")
        return None, []

    try:
        segments = read_chords_lab(output_lab)
        times_s, mono, duration_s = load_waveform_for_plot(input_wav)
    except FileNotFoundError as exc:
        st.error(f"File not found: {exc}")
        return None, []
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Error loading data: {exc}")
        return None, []

    # Check if any chords were detected
    if not segments:
        st.warning(
            "No chords were detected in this stem. "
            "This can happen if the stem is silent or contains only drums/bass. "
            "Try selecting a different stem (e.g., 'other' for harmonic instruments)."
        )
        return float(duration_s), segments

    config = _get_plot_config(duration_s=float(duration_s))
    visible_segments = _segments_in_time_range(segments, config.start_s, config.end_s)
    fig = _build_chords_waveform_figure(
        times_s=times_s, mono=mono, segments=segments, config=config
    )
    st.plotly_chart(fig, width="stretch")

    return float(duration_s), visible_segments


def _render_playback_controls(input_wav: Path, duration_s: float) -> None:
    """
    Render playback controls driven by the current visible time range.

    Parameters
    ----------
    input_wav : Path
        Selected stem WAV path.
    duration_s : float
        Total audio duration in seconds.

    Returns
    -------
    None
    """
    zoom_start = float(st.session_state.get(SESSION_KEY_ZOOM_START_S, 0.0))
    zoom_end = float(st.session_state.get(SESSION_KEY_ZOOM_END_S, float(duration_s)))
    visible_range_duration = max(0.0, zoom_end - zoom_start)
    is_full_track = zoom_start <= 0.0 and abs(zoom_end - float(duration_s)) < 0.1

    st.caption(
        "The visible time range drives both the chart and playback, "
        "so there is only one selection to manage."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if is_full_track:
            st.caption(f"Current selection: full track ({duration_s:.1f}s)")
        else:
            st.caption(
                f"Current selection: {zoom_start:.1f}s → {zoom_end:.1f}s "
                f"({visible_range_duration:.1f}s)"
            )
    with col2:
        st.caption("The player updates automatically when the visible range changes.")

    if visible_range_duration <= 0.0:
        st.warning("The current selection is empty.")
        return

    try:
        clip_bytes, clip_duration = extract_wav_clip_bytes(
            wav_path=input_wav,
            start_s=float(zoom_start),
            duration_s=float(visible_range_duration),
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        st.error(str(exc))
        return

    if not clip_bytes or clip_duration <= 0.0:
        st.warning("No playable audio was found in the current selection.")
        return

    st.caption(f"Selection clip: {zoom_start:.1f}s → {zoom_start + clip_duration:.1f}s")
    audio_base64 = base64.b64encode(clip_bytes).decode("ascii")
    player_id = (
        f"audio-player-{int(round(zoom_start * 10))}-"
        f"{int(round((zoom_start + clip_duration) * 10))}"
    )
    player_html = f"""
    <div style="width: 100%;">
      <audio id="{player_id}" controls preload="metadata" style="width: 100%;">
        Your browser does not support the audio element.
      </audio>
    </div>
    <script>
      (function() {{
        const audio = document.getElementById("{player_id}");
        const base64 = "{audio_base64}";
        const binary = window.atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {{
          bytes[i] = binary.charCodeAt(i);
        }}
        const blob = new Blob([bytes], {{ type: "audio/wav" }});
        const url = URL.createObjectURL(blob);
        audio.src = url;
      }})();
    </script>
    """
    components.html(player_html, height=80)


def _render_chords_results_expander(output_lab: Path) -> None:
    """
    Render the results expander containing timestamps and download button.

    Parameters
    ----------
    output_lab : Path
        Chords lab file path.

    Returns
    -------
    None
    """
    if not output_lab.exists():
        return

    with st.expander("Results (timestamps) / Download", expanded=False):
        try:
            lab_content = read_text_file(output_lab)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        st.download_button(
            label="⬇️ Download chords.lab",
            data=lab_content.encode("utf-8"),
            file_name="chords.lab",
            mime="text/plain",
        )
        st.code(lab_content, language="text")


def _render_chord_benchmark_section(input_wav: Path, benchmark_dir: Path) -> None:
    """Render a lightweight benchmark UI for available chord backends."""
    available_backends = list_chord_detection_backends()
    backend_ids = [backend.backend_id for backend in available_backends]

    with st.expander("Benchmark backends", expanded=False):
        st.caption(
            "This compares runtime and output consistency across backends. "
            "Without reference annotations, it is not a ground-truth accuracy benchmark."
        )

        selected_backends = st.multiselect(
            "Backends to benchmark",
            backend_ids,
            default=backend_ids,
            format_func=lambda backend_id: get_chord_detection_backend(backend_id).label,
        )

        run_benchmark = st.button(
            "Run benchmark",
            type="secondary",
            width="stretch",
            disabled=not selected_backends,
        )

        if run_benchmark:
            try:
                with st.spinner("Running benchmark..."):
                    results = benchmark_chord_detection(
                        input_wav=input_wav,
                        output_dir=benchmark_dir,
                        backend_ids=selected_backends,
                    )
                st.session_state[SESSION_KEY_CHORD_BENCHMARK] = {
                    "input_wav": str(input_wav),
                    "results": [
                        {
                            "backend_id": result.backend_id,
                            "label": result.label,
                            "output_lab": str(result.output_lab) if result.output_lab else None,
                            "runtime_s": result.runtime_s,
                            "segment_count": result.segment_count,
                            "unique_label_count": result.unique_label_count,
                            "success": result.success,
                            "error": result.error,
                        }
                        for result in results
                    ],
                }
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                st.error(str(exc))

        benchmark_report = st.session_state.get(SESSION_KEY_CHORD_BENCHMARK)
        if not benchmark_report or benchmark_report.get("input_wav") != str(input_wav):
            return

        benchmark_rows = benchmark_report["results"]
        if not benchmark_rows:
            return

        st.table(
            [
                {
                    "Backend": row["label"],
                    "Status": "ok" if row["success"] else "failed",
                    "Runtime (s)": f"{row['runtime_s']:.2f}",
                    "Segments": row["segment_count"],
                    "Unique labels": row["unique_label_count"],
                    "Output": row["output_lab"] or "-",
                }
                for row in benchmark_rows
            ]
        )

        failures = [row for row in benchmark_rows if not row["success"]]
        for row in failures:
            st.warning(f"{row['label']}: {row['error']}")

        successful_rows = [row for row in benchmark_rows if row["success"] and row["output_lab"]]
        if len(successful_rows) < 2:
            return

        st.caption("Cross-backend exact-label agreement on a 100 ms grid.")
        for index, first_row in enumerate(successful_rows):
            first_output = Path(first_row["output_lab"])
            first_segments = read_chords_lab(first_output)
            for second_row in successful_rows[index + 1 :]:
                second_output = Path(second_row["output_lab"])
                second_segments = read_chords_lab(second_output)
                agreement = compute_chord_label_agreement(first_segments, second_segments)
                if agreement is None:
                    st.write(f"{first_row['label']} vs {second_row['label']}: n/a")
                else:
                    st.write(
                        f"{first_row['label']} vs {second_row['label']}: {agreement:.1%} agreement"
                    )


def _render_chords_tab() -> None:
    """
    Render the chord detection tab.

    Returns
    -------
    None
    """
    stems_dir = _get_stems_dir()
    if stems_dir is None:
        return

    # Get the model from session state
    model = st.session_state.get(SESSION_KEY_SELECTED_MODEL, DEFAULT_MODEL)
    stems_paths = _get_stems_paths(stems_dir=stems_dir, model=model)
    if stems_paths is None:
        return

    selected_stem, run_button = _render_chords_controls(stems_paths=stems_paths)
    selected_backend = st.session_state.get(SESSION_KEY_CHORD_BACKEND, "chordmini_chordnet")

    input_wav = stems_paths[selected_stem]
    output_lab = stems_dir / f"chords_{selected_backend}.lab"
    benchmark_dir = stems_dir / "chord_benchmark"

    _maybe_run_chords_prediction(
        input_wav=input_wav,
        output_lab=output_lab,
        backend_id=selected_backend,
        run_button=run_button,
    )

    _render_chord_benchmark_section(input_wav=input_wav, benchmark_dir=benchmark_dir)

    with st.container(border=True):
        st.subheader("Audio inspector")

        audio_duration, segments = _render_chords_plot(output_lab=output_lab, input_wav=input_wav)
        if audio_duration is None:
            return

        _render_playback_controls(input_wav=input_wav, duration_s=audio_duration)
        _render_detected_chord_chart(segments)
        _render_chords_results_expander(output_lab=output_lab)


# =============================================================================
# Live Harmony Tab Functions
# =============================================================================


def _init_live_harmony_state() -> None:
    """
    Initialize session state for Live Harmony tab.

    Returns
    -------
    None
    """
    if SESSION_KEY_LIVE_NOTES not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_NOTES] = []
    if SESSION_KEY_LIVE_CHORD not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_CHORD] = None
    if SESSION_KEY_LIVE_CHORD_ALTS not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_CHORD_ALTS] = []
    if SESSION_KEY_LIVE_CONFIDENCE not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_CONFIDENCE] = 0.0
    if SESSION_KEY_LIVE_RUNNING not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_RUNNING] = False
    if SESSION_KEY_LIVE_THREAD not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_THREAD] = None
    if SESSION_KEY_LIVE_STOP_EVENT not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_STOP_EVENT] = None
    if SESSION_KEY_LIVE_SNAPSHOT not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_SNAPSHOT] = LiveHarmonySnapshot()
    if SESSION_KEY_LIVE_ERROR not in st.session_state:
        st.session_state[SESSION_KEY_LIVE_ERROR] = None
    if SESSION_KEY_CHORD_SEQ not in st.session_state:
        st.session_state[SESSION_KEY_CHORD_SEQ] = []
    if SESSION_KEY_251_MODE not in st.session_state:
        st.session_state[SESSION_KEY_251_MODE] = "major"
    if SESSION_KEY_251_KEY not in st.session_state:
        st.session_state[SESSION_KEY_251_KEY] = VISIBLE_MAJOR_KEYS[0]
    if SESSION_KEY_251_VARIANT not in st.session_state:
        st.session_state[SESSION_KEY_251_VARIANT] = "Type A"
    if SESSION_KEY_251_STEP not in st.session_state:
        st.session_state[SESSION_KEY_251_STEP] = 0
    if SESSION_KEY_251_AUTO_ADVANCE not in st.session_state:
        st.session_state[SESSION_KEY_251_AUTO_ADVANCE] = True
    if SESSION_KEY_251_LAST_MATCHED_STEP not in st.session_state:
        st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = None
    if SESSION_KEY_251_CHAIN_MODE not in st.session_state:
        st.session_state[SESSION_KEY_251_CHAIN_MODE] = False
    if SESSION_KEY_251_CHAIN_KEYS not in st.session_state:
        st.session_state[SESSION_KEY_251_CHAIN_KEYS] = [VISIBLE_MAJOR_KEYS[0]]
    if SESSION_KEY_251_CHAIN_INDEX not in st.session_state:
        st.session_state[SESSION_KEY_251_CHAIN_INDEX] = 0
    if SESSION_KEY_MIDI_SELECTED_DEVICE not in st.session_state:
        st.session_state[SESSION_KEY_MIDI_SELECTED_DEVICE] = None


def _reset_live_harmony_session_values() -> None:
    """Clear live harmony values stored in the Streamlit session."""
    st.session_state[SESSION_KEY_LIVE_NOTES] = []
    st.session_state[SESSION_KEY_LIVE_CHORD] = None
    st.session_state[SESSION_KEY_LIVE_CHORD_ALTS] = []
    st.session_state[SESSION_KEY_LIVE_CONFIDENCE] = 0.0
    st.session_state[SESSION_KEY_LIVE_ERROR] = None


def _sync_live_harmony_snapshot() -> None:
    """Copy the listener snapshot into Streamlit session state."""
    snapshot = st.session_state.get(SESSION_KEY_LIVE_SNAPSHOT)
    if snapshot is None:
        return

    notes, chord, alternatives, confidence, error = snapshot.read()
    st.session_state[SESSION_KEY_LIVE_NOTES] = notes
    st.session_state[SESSION_KEY_LIVE_CHORD] = chord
    st.session_state[SESSION_KEY_LIVE_CHORD_ALTS] = alternatives
    st.session_state[SESSION_KEY_LIVE_CONFIDENCE] = confidence
    st.session_state[SESSION_KEY_LIVE_ERROR] = error


def _stop_midi_listener() -> None:
    """Stop the MIDI listener thread if running."""
    stop_event = st.session_state.get(SESSION_KEY_LIVE_STOP_EVENT)
    if stop_event is not None:
        stop_event.set()

    thread = st.session_state.get(SESSION_KEY_LIVE_THREAD)
    if thread and thread.is_alive():
        thread.join(timeout=1.0)

    snapshot = st.session_state.get(SESSION_KEY_LIVE_SNAPSHOT)
    if snapshot is not None:
        snapshot.reset()

    st.session_state[SESSION_KEY_LIVE_RUNNING] = False
    st.session_state[SESSION_KEY_LIVE_THREAD] = None
    st.session_state[SESSION_KEY_LIVE_STOP_EVENT] = None
    _reset_live_harmony_session_values()


def _midi_listener_thread(
    device_name: str,
    detector: ChordDetector,
    stop_event: threading.Event,
    snapshot: LiveHarmonySnapshot,
) -> None:
    """
    Background thread that listens to MIDI input and updates session state.

    Parameters
    ----------
    device_name : str
        Name of the MIDI device to listen to.
    detector : ChordDetector
        Chord detector instance.

    Returns
    -------
    None
    """
    import mido

    active_notes: dict[int, int] = {}

    try:
        # mido types are not recognized by ty, using type: ignore
        with mido.open_input(device_name) as port:  # type: ignore
            while not stop_event.is_set():
                try:
                    state_changed = False
                    for msg in port.iter_pending():
                        if msg.type == "note_on" and msg.velocity > 0:
                            active_notes[msg.note] = msg.velocity
                            state_changed = True
                        elif msg.type == "note_off" or (
                            msg.type == "note_on" and msg.velocity == 0
                        ):
                            active_notes.pop(msg.note, None)
                            state_changed = True

                    if state_changed:
                        note_numbers = sorted(active_notes.keys())
                        note_names = [
                            MIDI_NOTE_TO_NAME[n]
                            for n in note_numbers
                            if 0 <= n < len(MIDI_NOTE_TO_NAME)
                        ]
                        result = detector.detect(note_names)
                        snapshot.update(
                            notes=note_names,
                            chord=result.primary_chord,
                            alternatives=result.alternative_chords,
                            confidence=result.confidence,
                        )

                    time.sleep(0.01)

                except Exception as exc:
                    snapshot.set_error(f"MIDI error: {exc}")
                    stop_event.set()
                    break

    except Exception as exc:
        snapshot.set_error(f"Failed to open MIDI device: {exc}")
        stop_event.set()


def _start_midi_listener(device_name: str) -> None:
    """
    Start the MIDI listener in a background thread.

    Parameters
    ----------
    device_name : str
        Name of the MIDI device to connect to.

    Returns
    -------
    None
    """
    # Stop any existing listener first
    _stop_midi_listener()

    detector = ChordDetector(min_notes=2, min_match_ratio=0.6)
    stop_event = threading.Event()
    snapshot = st.session_state[SESSION_KEY_LIVE_SNAPSHOT]
    snapshot.reset()

    # Start new thread
    thread = threading.Thread(
        target=_midi_listener_thread,
        args=(device_name, detector, stop_event, snapshot),
        daemon=True,
    )
    st.session_state[SESSION_KEY_LIVE_THREAD] = thread
    st.session_state[SESSION_KEY_LIVE_STOP_EVENT] = stop_event
    st.session_state[SESSION_KEY_LIVE_RUNNING] = True
    thread.start()


def _render_live_harmony_tab() -> None:
    """
    Render the Live Harmony tab for real-time MIDI chord detection.

    Returns
    -------
    None
    """
    _init_live_harmony_state()
    _sync_live_harmony_snapshot()

    live_thread = st.session_state.get(SESSION_KEY_LIVE_THREAD)
    if st.session_state.get(SESSION_KEY_LIVE_RUNNING, False) and (
        live_thread is None or not live_thread.is_alive()
    ):
        st.session_state[SESSION_KEY_LIVE_RUNNING] = False

    st.header("🎹 Live Harmony")
    st.markdown(
        """
        Connect your MIDI keyboard and play chords. The app will detect the notes
        you're playing and identify the chord in real-time.
        """
    )

    settings_tab, detection_tab, trainer_tab = st.tabs(
        ["MIDI Settings", "Chord Detection", "Jazz Trainer"]
    )

    midi_query = query_midi_devices()
    devices = midi_query.devices
    selected_device_name = st.session_state.get(SESSION_KEY_MIDI_SELECTED_DEVICE)
    if selected_device_name not in devices:
        if devices:
            selected_device_name = devices[0]
            st.session_state[SESSION_KEY_MIDI_SELECTED_DEVICE] = selected_device_name
        else:
            selected_device_name = None

    with settings_tab:
        st.subheader("🎛️ MIDI Settings")
        refresh_col, status_col = st.columns([1, 2])

        with refresh_col:
            if st.button("🔄 Refresh devices", width="stretch"):
                st.rerun()

        with status_col:
            if devices:
                st.caption(f"{len(devices)} MIDI input device(s) detected.")
            else:
                st.caption("No MIDI input device detected for now.")

        if not devices:
            if midi_query.error:
                st.error("Unable to query MIDI input devices from the current app environment.")
                st.code(
                    "\n".join(
                        [
                            f"Python: {midi_query.python_executable}",
                            f"Mido backend: {midi_query.backend}",
                            f"Error: {midi_query.error}",
                        ]
                    ),
                    language="text",
                )
                st.info(
                    "The debug script and Streamlit are probably not running in the same "
                    "Python environment. Launch the app with `uv run streamlit run src/ui.py` "
                    "if your MIDI debug script works with `uv run python ...`."
                )
            else:
                st.warning(
                    "No MIDI input devices found. "
                    "Make sure your keyboard is connected and the driver is installed."
                )
            st.markdown(
                """
                **Troubleshooting:**
                - On macOS: Check Audio MIDI Setup app to verify your device is connected
                - Make sure your keyboard is set to send MIDI (not just audio)
                - Try closing and reopening other MIDI applications
                - Launch Streamlit from the same environment as the MIDI debug script
                """
            )
        else:
            resolved_selected_device = selected_device_name or devices[0]
            with st.expander("MIDI diagnostics", expanded=False):
                st.code(
                    "\n".join(
                        [
                            f"Python: {midi_query.python_executable}",
                            f"Mido backend: {midi_query.backend}",
                            f"Detected devices: {len(devices)}",
                        ]
                    ),
                    language="text",
                )

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_device = st.selectbox(
                    "Select MIDI Input Device",
                    devices,
                    index=devices.index(resolved_selected_device),
                    key=SESSION_KEY_MIDI_SELECTED_DEVICE,
                    help="Choose your MIDI keyboard or controller",
                )

            with col2:
                if st.session_state.get(SESSION_KEY_LIVE_RUNNING, False):
                    if st.button("⏹️ Stop", type="secondary", width="stretch"):
                        _stop_midi_listener()
                        st.success("MIDI listener stopped")
                else:
                    if st.button("▶️ Start", type="primary", width="stretch"):
                        _start_midi_listener(selected_device)
                        st.success(f"Listening to {selected_device}...")

    live_error = st.session_state.get(SESSION_KEY_LIVE_ERROR)
    current_notes = st.session_state.get(SESSION_KEY_LIVE_NOTES, [])
    current_chord = st.session_state.get(SESSION_KEY_LIVE_CHORD, None)
    current_alts = st.session_state.get(SESSION_KEY_LIVE_CHORD_ALTS, [])
    confidence = st.session_state.get(SESSION_KEY_LIVE_CONFIDENCE, 0.0)

    with detection_tab:
        st.subheader("🎵 Chord Detection")
        show_alts = st.toggle("Show alternatives", value=False)

        if not devices:
            st.info("Connect and refresh a MIDI input device from the MIDI Settings tab.")
        else:
            if live_error:
                st.error(live_error)

            if st.session_state.get(SESSION_KEY_LIVE_RUNNING, False):
                st.caption(
                    f"Listening to `{st.session_state[SESSION_KEY_MIDI_SELECTED_DEVICE]}`..."
                )

            # Display in columns
            col1, col2 = st.columns([2, 1])

            with col1:
                if current_notes:
                    st.markdown("**Current Notes:**")
                    # Group notes by octave for better readability
                    notes_by_octave: dict[str, list[str]] = {}
                    for note in current_notes:
                        if len(note) > 1 and note[-1].isdigit():
                            octave = note[-1]
                            base = note[:-1]
                        else:
                            octave = "?"
                            base = note
                        if octave not in notes_by_octave:
                            notes_by_octave[octave] = []
                        notes_by_octave[octave].append(base)

                    for octave in sorted(notes_by_octave.keys()):
                        notes = sorted(notes_by_octave[octave])
                        st.markdown(f"- **Octave {octave}:** {', '.join(notes)}")
                else:
                    st.info("No notes being played. Play something on your keyboard!")

            with col2:
                if current_chord:
                    st.success(f"**Detected Chord:** {current_chord}")
                    st.caption(f"Confidence: {confidence:.1%}")
                else:
                    if current_notes:
                        st.warning("Notes detected but no chord recognized")

            if show_alts and current_alts:
                st.markdown("**Alternative Chords:**")
                for alt in current_alts[:5]:
                    st.markdown(f"- {alt}")

            st.markdown("---")
            _render_chord_sequence_editor()

    with trainer_tab:
        st.subheader("🎼 Jazz Trainer")
        if not devices:
            st.info("Connect and refresh a MIDI input device from the MIDI Settings tab.")
        else:
            if live_error:
                st.error(live_error)
            _render_251_trainer(current_notes=current_notes)

    if st.session_state.get(SESSION_KEY_LIVE_RUNNING, False):
        time.sleep(LIVE_REFRESH_INTERVAL_S)
        st.rerun()


def _render_251_trainer(current_notes: list[str]) -> None:
    """Render guided ii-V-I practice with target voicings and staff notation."""
    st.subheader("🎼 251 Trainer")
    st.markdown(
        """
        Practice guided `ii-V-I` voicings with a target progression on staff.
        The trainer validates the current step and can auto-advance when the voicing matches.
        """
    )

    mode = st.radio(
        "Mode",
        options=["major", "minor"],
        horizontal=True,
        key=SESSION_KEY_251_MODE,
    )
    key_options = VISIBLE_MAJOR_KEYS if mode == "major" else VISIBLE_MINOR_KEYS
    current_key = st.session_state.get(SESSION_KEY_251_KEY, key_options[0])
    if current_key not in key_options:
        st.session_state[SESSION_KEY_251_KEY] = key_options[0]
        st.session_state[SESSION_KEY_251_STEP] = 0
        current_key = key_options[0]
    _sync_251_chain_state(key_options=key_options, fallback_key=current_key)

    col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 1])
    with col1:
        selected_key = st.selectbox(
            "Key",
            options=key_options,
            key=SESSION_KEY_251_KEY,
            disabled=st.session_state.get(SESSION_KEY_251_CHAIN_MODE, False),
        )
    with col2:
        st.selectbox(
            "Voicing",
            options=["Type A", "Type B"],
            key=SESSION_KEY_251_VARIANT,
        )
    with col3:
        st.toggle(
            "Auto-advance",
            key=SESSION_KEY_251_AUTO_ADVANCE,
            help="Advance to the next chord when the current target voicing is matched exactly.",
        )
    with col4:
        if st.button("↺ Restart", width="stretch"):
            st.session_state[SESSION_KEY_251_STEP] = 0
            st.session_state[SESSION_KEY_251_CHAIN_INDEX] = 0
            st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = None
            st.rerun()
    with col5:
        st.toggle(
            "Chain keys",
            key=SESSION_KEY_251_CHAIN_MODE,
            help="Practice several keys in sequence "
            "and move on only after validating the current chord.",
        )

    chain_mode = st.session_state.get(SESSION_KEY_251_CHAIN_MODE, False)
    if chain_mode:
        selected_chain_keys = st.multiselect(
            "Keys sequence",
            options=key_options,
            key=SESSION_KEY_251_CHAIN_KEYS,
            help="The trainer will walk through these keys in order.",
        )
        if not selected_chain_keys:
            st.session_state[SESSION_KEY_251_CHAIN_KEYS] = [selected_key]
            selected_chain_keys = st.session_state[SESSION_KEY_251_CHAIN_KEYS]
        chain_index = min(
            int(st.session_state.get(SESSION_KEY_251_CHAIN_INDEX, 0)),
            len(selected_chain_keys) - 1,
        )
        st.session_state[SESSION_KEY_251_CHAIN_INDEX] = chain_index
        selected_key = selected_chain_keys[chain_index]
        st.caption(
            f"Sequence {chain_index + 1}/{len(selected_chain_keys)}: current key `{selected_key}`"
        )

    exercise_id = (
        f"{st.session_state[SESSION_KEY_251_MODE]}:"
        f"{selected_key}:"
        f"{st.session_state[SESSION_KEY_251_VARIANT]}"
    )
    exercise = VISIBLE_251_EXERCISES[exercise_id]

    current_step = int(st.session_state.get(SESSION_KEY_251_STEP, 0))
    current_step = max(0, min(current_step, len(exercise.steps) - 1))
    step = exercise.steps[current_step]
    match_result = compare_note_sets(step.expected_notes, current_notes)
    _maybe_auto_advance_251(
        match_result=match_result,
        step_index=current_step,
        num_steps=len(exercise.steps),
    )

    nav1, nav2, nav3 = st.columns([1, 1, 2])
    with nav1:
        if st.button("← Previous", width="stretch", disabled=current_step == 0):
            st.session_state[SESSION_KEY_251_STEP] = current_step - 1
            st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = None
            st.rerun()
    with nav2:
        if st.button(
            "Next →",
            width="stretch",
            disabled=chain_mode or current_step >= len(exercise.steps) - 1,
        ):
            st.session_state[SESSION_KEY_251_STEP] = current_step + 1
            st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = None
            st.rerun()
    with nav3:
        suffix = f" in `{selected_key}`" if chain_mode else ""
        st.caption(f"Step {current_step + 1}/{len(exercise.steps)}: target `{step.symbol}`{suffix}")

    status_col1, status_col2, status_col3 = st.columns([1.2, 1.2, 2.2])
    with status_col1:
        if match_result.exact:
            st.success("Exact voicing match")
        elif match_result.pitch_class_only:
            st.info("Right chord tones, different octave/spacing")
        elif current_notes:
            st.warning("Different voicing or chord")
        else:
            st.info("Play the target voicing to validate this step.")
    with status_col2:
        st.caption("Expected")
        st.code(", ".join(step.expected_notes))
    with status_col3:
        st.caption("Played")
        st.code(", ".join(current_notes) if current_notes else "No notes")

    st.markdown(
        render_progression_svg(exercise, active_step=current_step, played_notes=current_notes),
        unsafe_allow_html=True,
    )

    if exercise.mode == "minor":
        st.caption(
            "Minor voicings use compact four-note shapes with the altered dominant color "
            "(`7b9`) on staff for step-by-step practice."
        )


def _maybe_auto_advance_251(*, match_result, step_index: int, num_steps: int) -> None:
    """Advance the trainer when auto-advance is enabled and the voicing is correct."""
    if not st.session_state.get(SESSION_KEY_251_AUTO_ADVANCE, True):
        return

    last_matched_step = st.session_state.get(SESSION_KEY_251_LAST_MATCHED_STEP)
    if match_result.exact and last_matched_step != step_index:
        st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = step_index
        if step_index < num_steps - 1:
            st.session_state[SESSION_KEY_251_STEP] = step_index + 1
            st.rerun()
        if st.session_state.get(SESSION_KEY_251_CHAIN_MODE, False):
            chain_keys = st.session_state.get(SESSION_KEY_251_CHAIN_KEYS, [])
            chain_index = int(st.session_state.get(SESSION_KEY_251_CHAIN_INDEX, 0))
            if chain_index < len(chain_keys) - 1:
                st.session_state[SESSION_KEY_251_CHAIN_INDEX] = chain_index + 1
                st.session_state[SESSION_KEY_251_STEP] = 0
                st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = None
                st.rerun()
        return

    if not match_result.exact:
        st.session_state[SESSION_KEY_251_LAST_MATCHED_STEP] = None


def _sync_251_chain_state(*, key_options: list[str], fallback_key: str) -> None:
    """Keep chained-key practice state aligned with the currently selected mode."""
    chain_keys = st.session_state.get(SESSION_KEY_251_CHAIN_KEYS, [fallback_key])
    filtered_keys = [key_name for key_name in chain_keys if key_name in key_options]
    if not filtered_keys:
        filtered_keys = [fallback_key]
    if filtered_keys != chain_keys:
        st.session_state[SESSION_KEY_251_CHAIN_KEYS] = filtered_keys

    chain_index = int(st.session_state.get(SESSION_KEY_251_CHAIN_INDEX, 0))
    max_index = len(filtered_keys) - 1
    if chain_index > max_index:
        st.session_state[SESSION_KEY_251_CHAIN_INDEX] = max_index


def _render_chord_sequence_editor() -> None:
    """
    Render the chord sequence editor section.

    Returns
    -------
    None
    """
    st.subheader("📝 Chord Sequence Editor")
    st.markdown("Create and save chord progressions for practice or reference.")

    # Get or initialize chord sequence
    if SESSION_KEY_CHORD_SEQ not in st.session_state:
        st.session_state[SESSION_KEY_CHORD_SEQ] = []

    sequence = st.session_state[SESSION_KEY_CHORD_SEQ]

    # Controls
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # Chord input
        all_possible_chords = [
            f"{note}:{chord_type}"
            for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            for chord_type in ["maj", "min", "7", "maj7", "min7", "sus2", "sus4", "dim", "aug"]
        ]
        new_chord = st.selectbox(
            "Add Chord",
            ["""""" + chord for chord in all_possible_chords],
            index=0,
            help="Select a chord to add to the sequence",
        )

    with col2:
        if st.button("➕ Add", width="stretch"):
            if new_chord:
                sequence.append(new_chord)
                st.session_state[SESSION_KEY_CHORD_SEQ] = sequence

    with col3:
        if st.button("🗑️ Clear", width="stretch"):
            st.session_state[SESSION_KEY_CHORD_SEQ] = []

    # Display sequence
    if sequence:
        st.markdown("**Current Sequence:**")
        cols = st.columns(min(len(sequence), 6))
        for idx, chord in enumerate(sequence):
            with cols[idx % 6]:
                if st.button(f"{chord} ❌", key=f"chord_{idx}", width="stretch"):
                    # Remove this chord
                    sequence.pop(idx)
                    st.session_state[SESSION_KEY_CHORD_SEQ] = sequence
                    st.rerun()
    else:
        st.info("No chords in sequence yet. Add some using the controls above!")


def main() -> None:
    """
    Streamlit UI entrypoint.

    Returns
    -------
    None
    """
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)

    WORK_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _init_session_state()
    selected_model, active_workspace, _selected_backend = _render_sidebar()

    if active_workspace == "Audio Analysis":
        st.caption("Load a track, isolate its stems, and explore its chord progression.")
        tab_split, tab_chords = st.tabs(["Stem separation", "Chord detection"])
        with tab_split:
            _render_split_tab(model=selected_model)
        with tab_chords:
            _render_chords_tab()
    else:
        st.caption("Connect a MIDI keyboard to view live notes, chords, and exercises.")
        _render_live_harmony_tab()


if __name__ == "__main__":
    main()
