from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Set

import streamlit as st  # pylint: disable=import-error
from service import (
    clear_workspace,
    find_stems_dir,
    list_stems_wav,
    predict_chords_for_stem,
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


def _render_chords_tab() -> None:
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

    if output_lab.exists():
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
