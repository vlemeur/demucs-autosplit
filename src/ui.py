from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import plotly.graph_objects as go
import streamlit as st  # pylint: disable=import-error
from service import (
    clear_workspace,
    find_stems_dir,
    list_stems_wav,
    load_waveform_for_plot,
    predict_chords_for_stem,
    read_chords_lab,
    read_stems,
    read_text_file,
    run_split,
    safe_filename,
    save_bytes_to_file,
    validate_extension,
    zip_stems,
)

APP_TITLE: str = "🥷🏻 Demucs Audio Splitter"

WORK_DIR: Path = Path(".streamlit_workdir")
UPLOAD_DIR: Path = WORK_DIR / "uploads"
OUTPUT_DIR: Path = WORK_DIR / "outputs"

TRY_FILTERS_OTHERS_DEFAULT: bool = False
SUPPORTED_EXT: Set[str] = {".wav", ".mp3"}
STEMS: List[str] = ["drums", "bass", "other", "vocals"]

SESSION_KEY_STEMS_DIR: str = "stems_dir"


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


def _init_session_state() -> None:
    """
    Initialize Streamlit session state keys used by the app.

    Returns
    -------
    None
    """
    if SESSION_KEY_STEMS_DIR not in st.session_state:
        st.session_state[SESSION_KEY_STEMS_DIR] = None


def _render_sidebar() -> bool:
    """
    Render the sidebar controls.

    Returns
    -------
    bool
        Whether the "Try filter others" toggle is enabled.
    """
    with st.sidebar:
        st.header("Settings")
        try_filters = st.toggle("Try filter others", value=TRY_FILTERS_OTHERS_DEFAULT)
        st.markdown("---")
        if st.button("🧹 Clear workspace"):
            clear_workspace(WORK_DIR)
            st.session_state[SESSION_KEY_STEMS_DIR] = None
            st.success("Workspace cleared.")
    return try_filters


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

    simplified_labels = sorted({_simplify_chord_label(seg.label) for seg in segments})
    label_to_color: Dict[str, str] = {
        chord_label: palette[i % len(palette)] for i, chord_label in enumerate(simplified_labels)
    }

    # Dummy traces to build a clean legend (simplified labels only).
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
    for seg in segments:
        if seg.end_s < config.start_s or seg.start_s > config.end_s:
            continue

        simplified = _simplify_chord_label(seg.label)
        color = label_to_color.get(simplified, "#999999")

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
                    f"Simplified: <b>{simplified}</b><br>"
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


def _render_split_tab(try_filters: bool) -> None:
    """
    Render the stem separation tab.

    Parameters
    ----------
    try_filters : bool
        Whether to enable "Try filter others" in the Demucs pipeline.

    Returns
    -------
    None
    """
    st.caption("Upload a track and get 4 stems: drums, bass, other, vocals.")

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

    if st.button("🚀 Split track", type="primary"):
        with st.status("Running Demucs…", expanded=True) as status:
            st.write("Splitting… this can take a while depending on your CPU/GPU.")
            try:
                run_split(
                    audio_path=audio_path,
                    output_dir=OUTPUT_DIR,
                    try_filter_others=try_filters,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.exception(exc)
                return

            stems_dir = find_stems_dir(
                output_root=OUTPUT_DIR,
                track_name=track_name,
                stems=STEMS,
            )
            if stems_dir is None:
                status.update(label="Stems not found", state="error", expanded=True)
                st.error(
                    "Demucs finished, but the 4 stems were not found. "
                    "Inspect .streamlit_workdir/outputs to verify the output layout."
                )
                return

            status.update(label="Done ✔", state="complete", expanded=False)

        st.success("✅ Split complete!")
        st.write("**Output folder:**", str(stems_dir))

        # Store stems_dir for the Chord detection tab.
        st.session_state[SESSION_KEY_STEMS_DIR] = str(stems_dir)

        stems_bytes = read_stems(stems_dir=stems_dir, stems=STEMS)
        zip_bytes = zip_stems(stems_dir=stems_dir, stems=STEMS)

        st.subheader("Download")
        st.download_button(
            "⬇️ Download all (ZIP)",
            data=zip_bytes,
            file_name=f"{track_name}_stems.zip",
            mime="application/zip",
        )

        cols = st.columns(4)
        for idx, stem in enumerate(STEMS):
            with cols[idx]:
                st.download_button(
                    f"⬇️ {stem}.wav",
                    data=stems_bytes[stem],
                    file_name=f"{track_name}_{stem}.wav",
                    mime="audio/wav",
                )

        st.subheader("Preview")
        for stem in STEMS:
            st.markdown(f"**{stem}**")
            st.audio(stems_bytes[stem], format="audio/wav")


def _get_stems_dir() -> Optional[Path]:
    """
    Get the stems directory stored in the session state.

    Returns
    -------
    Path or None
        Stems directory if available and existing, otherwise None.
    """
    stems_dir_str = st.session_state.get(SESSION_KEY_STEMS_DIR)
    if stems_dir_str is None:
        st.info("Run a stem separation first, then come back here to detect chords.")
        return None

    stems_dir = Path(stems_dir_str)
    if not stems_dir.exists():
        st.warning("Stored stems folder does not exist anymore. Please run a split again.")
        st.session_state[SESSION_KEY_STEMS_DIR] = None
        return None

    return stems_dir


def _get_stems_paths(stems_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Collect existing stem wav paths for chord prediction.

    Parameters
    ----------
    stems_dir : Path
        Directory containing stem wav files.

    Returns
    -------
    dict of str to Path or None
        Mapping {stem_name: wav_path}, or None if no stems are found.
    """
    stems_paths: Dict[str, Path] = list_stems_wav(stems_dir=stems_dir, stems=STEMS)
    if not stems_paths:
        st.warning("No stems were found in the stored stems folder.")
        return None
    return stems_paths


def _render_chords_controls(stems_paths: Dict[str, Path]) -> Tuple[str, bool]:
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
        run_button = st.button("🎹 Predict chords", type="primary", use_container_width=True)

    return selected_stem, run_button


def _maybe_run_chords_prediction(input_wav: Path, output_lab: Path, run_button: bool) -> None:
    """
    Run chord prediction if requested.

    Parameters
    ----------
    input_wav : Path
        Selected stem wav path.
    output_lab : Path
        Output chords lab file path.
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
            predict_chords_for_stem(input_wav=input_wav, output_lab=output_lab)
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
        return ChordsPlotConfig(start_s=start_s, end_s=end_s)

    return ChordsPlotConfig(start_s=0.0, end_s=float(duration_s))


def _render_chords_plot(output_lab: Path, input_wav: Path) -> bool:
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
    bool
        True if the plot was rendered, otherwise False.
    """
    if not output_lab.exists():
        st.info("No chords detected yet. Click “Predict chords” to generate chords.lab.")
        return False

    try:
        segments = read_chords_lab(output_lab)
        times_s, mono, duration_s = load_waveform_for_plot(input_wav)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return False

    config = _get_plot_config(duration_s=duration_s)
    fig = _build_chords_waveform_figure(
        times_s=times_s, mono=mono, segments=segments, config=config
    )
    st.plotly_chart(fig, use_container_width=True)
    return True


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

    stems_paths = _get_stems_paths(stems_dir=stems_dir)
    if stems_paths is None:
        return

    selected_stem, run_button = _render_chords_controls(stems_paths=stems_paths)

    input_wav = stems_paths[selected_stem]
    output_lab = stems_dir / "chords.lab"

    _maybe_run_chords_prediction(input_wav=input_wav, output_lab=output_lab, run_button=run_button)

    plot_rendered = _render_chords_plot(output_lab=output_lab, input_wav=input_wav)
    if plot_rendered:
        _render_chords_results_expander(output_lab=output_lab)


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
    try_filters = _render_sidebar()

    tab_split, tab_chords = st.tabs(["Stem separation", "Chord detection"])
    with tab_split:
        _render_split_tab(try_filters=try_filters)
    with tab_chords:
        _render_chords_tab()


if __name__ == "__main__":
    main()
