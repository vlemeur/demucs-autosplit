from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Set

import streamlit as st  # pylint: disable=import-error
from service import (
    clear_workspace,
    find_stems_dir,
    read_stems,
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


def main() -> None:
    """
    Streamlit UI entrypoint.

    Returns
    -------
    None
    """
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.caption("Upload a track and get 4 stems: drums, bass, other, vocals.")

    WORK_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Settings")
        try_filters = st.toggle("Try filter others", value=TRY_FILTERS_OTHERS_DEFAULT)
        st.markdown("---")
        if st.button("🧹 Clear workspace"):
            clear_workspace(WORK_DIR)
            st.success("Workspace cleared.")

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
                    audio_path=audio_path, output_dir=OUTPUT_DIR, try_filter_others=try_filters
                )
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
                status.update(label="Failed", state="error", expanded=True)
                st.exception(exc)
                return

            stems_dir = find_stems_dir(output_root=OUTPUT_DIR, track_name=track_name, stems=STEMS)
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


if __name__ == "__main__":
    main()
