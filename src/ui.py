from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Set

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


def _build_chords_waveform_figure(
    times_s,
    mono,
    segments,
    start_s: float,
    end_s: float,
) -> go.Figure:
    """
    Build a Plotly figure with waveform and chord regions.

    Parameters
    ----------
    times_s : numpy.ndarray
        Time axis in seconds.
    mono : numpy.ndarray
        Mono waveform samples (downsampled).
    segments : Sequence[ChordSegment]
        Chord segments parsed from a .lab file.
    start_s : float
        Start time for x-axis (seconds).
    end_s : float
        End time for x-axis (seconds).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure ready to be displayed in Streamlit.
    """
    fig = go.Figure()

    # Waveform (GL trace for better performance on long files)
    fig.add_trace(
        go.Scattergl(
            x=times_s,
            y=mono,
            mode="lines",
            name="Waveform",
            hoverinfo="skip",
        )
    )

    # Use Plotly's default colorway if available, otherwise fallback to a small palette.
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

    labels = sorted({seg.label for seg in segments})
    label_to_color: Dict[str, str] = {
        label: palette[i % len(palette)] for i, label in enumerate(labels)
    }

    # Chord regions + annotations
    for seg in segments:
        if seg.end_s < start_s or seg.start_s > end_s:
            continue

        color = label_to_color.get(seg.label, "#999999")

        fig.add_vrect(
            x0=seg.start_s,
            x1=seg.end_s,
            fillcolor=color,
            opacity=0.20,
            line_width=0,
            layer="below",
        )

        # Add labels only for regions long enough to avoid clutter.
        if (seg.end_s - seg.start_s) >= 0.5:
            fig.add_annotation(
                x=0.5 * (seg.start_s + seg.end_s),
                y=0.95,
                xref="x",
                yref="paper",
                text=seg.label,
                showarrow=False,
                font={"size": 10},
            )

    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        hovermode="x",
    )
    fig.update_xaxes(range=[start_s, end_s])

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


def _render_chords_tab() -> None:  # pylint: disable=too-many-locals,too-many-return-statements,too-many-statements
    """
    Render the chord detection tab.

    Returns
    -------
    None
    """
    stems_dir_str = st.session_state.get(SESSION_KEY_STEMS_DIR)
    if stems_dir_str is None:
        st.info("Run a stem separation first, then come back here to detect chords.")
        return

    stems_dir = Path(stems_dir_str)
    if not stems_dir.exists():
        st.warning("Stored stems folder does not exist anymore. Please run a split again.")
        st.session_state[SESSION_KEY_STEMS_DIR] = None
        return

    st.write("**Stems folder:**", str(stems_dir))

    stems_paths: Dict[str, Path] = list_stems_wav(stems_dir=stems_dir, stems=STEMS)
    if not stems_paths:
        st.warning("No stems were found in the stored stems folder.")
        return

    stem_names = list(stems_paths.keys())
    default_index = stem_names.index("other") if "other" in stem_names else 0
    selected_stem = st.selectbox("Select a stem", stem_names, index=default_index)

    input_wav = stems_paths[selected_stem]
    output_lab = stems_dir / "chords.lab"

    col1, col2 = st.columns(2)
    with col1:
        run_button = st.button("🎹 Predict chords", type="primary")
    with col2:
        st.write(f"Output: `{output_lab.name}`")

    if run_button:
        try:
            with st.spinner("Predicting chords..."):
                predict_chords_for_stem(input_wav=input_wav, output_lab=output_lab)
            st.success("Chord prediction completed.")
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
            return

    if not output_lab.exists():
        st.info("No chords file found yet. Run chord prediction to generate chords.lab.")
        return

    try:
        lab_content = read_text_file(output_lab)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.subheader("Result")
    st.code(lab_content, language="text")

    st.download_button(
        label="⬇️ Download chords.lab",
        data=lab_content.encode("utf-8"),
        file_name="chords.lab",
        mime="text/plain",
    )

    st.subheader("Visualization")

    show_viz = st.toggle("Show waveform + chord regions (Plotly)", value=True)
    if not show_viz:
        return

    try:
        segments = read_chords_lab(output_lab)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    try:
        times_s, mono, duration_s = load_waveform_for_plot(input_wav)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return

    zoom = st.checkbox("Zoom", value=False)
    if zoom:
        start_s, end_s = st.slider(
            "Time range (seconds)",
            min_value=0.0,
            max_value=float(duration_s),
            value=(0.0, float(min(duration_s, 30.0))),
            step=0.1,
        )
    else:
        start_s, end_s = 0.0, float(duration_s)

    fig = _build_chords_waveform_figure(
        times_s=times_s,
        mono=mono,
        segments=segments,
        start_s=start_s,
        end_s=end_s,
    )
    st.plotly_chart(fig, use_container_width=True)


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
