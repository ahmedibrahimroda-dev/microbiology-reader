from pathlib import Path
import tempfile
import os
import streamlit as st

from microbio_notes_tool import (
    ToolConfig,
    MediaProcessor,
    build_transcription_provider,
    build_ocr_provider,
    build_llm_provider,
    LectureNotesBuilder,
    DummyTranscriptionProvider,
    RuleBasedLLMProvider,
    export_to_docx,
    export_to_json,
)

def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

# Make API key available to the backend provider
_api_key = _get_secret("OPENAI_API_KEY", "")
_model_name = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
if _api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = _api_key
if _model_name and not os.getenv("OPENAI_MODEL"):
    os.environ["OPENAI_MODEL"] = _model_name

st.set_page_config(page_title="Microbio Notes Builder", page_icon="🧫", layout="wide")
st.title("🧫 Microbiology Lecture Notes Builder")
st.caption("Upload a lecture video or audio file, then export structured English study notes to Word.")

with st.sidebar:
    st.header("Settings")
    lecture_title = st.text_input("Lecture title", "")
    output_language = st.selectbox("Output language", ["English"], index=0)
    frame_interval = st.slider("Frame interval (sec)", min_value=5, max_value=60, value=20, step=5)
    max_chars = st.slider("Chunk size", min_value=1500, max_value=7000, value=4200, step=100)
    use_ocr = st.checkbox("Enable OCR", value=True)
    writer_mode = st.selectbox("Notes writer", ["auto", "rule"], index=0)
    transcriber_mode = st.selectbox("Transcriber", ["auto", "dummy"], index=0)
    if _api_key:
        st.success("OpenAI API key detected.")
    else:
        st.info("No OpenAI API key detected. The app will use the rule-based writer.")

uploaded = st.file_uploader("Upload lecture video/audio", type=["mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a"])

if uploaded:
    st.success(f"Loaded: {uploaded.name}")
    if st.button("Build notes", type="primary"):
        with st.spinner("Processing lecture..."):
            with tempfile.TemporaryDirectory(prefix="streamlit_microbio_") as tmp:
                input_path = Path(tmp) / uploaded.name
                input_path.write_bytes(uploaded.read())

                config = ToolConfig(
                    output_language=output_language,
                    frame_interval_sec=frame_interval,
                    max_chunk_chars=max_chars,
                    enable_ocr=use_ocr,
                    llm_model_name=_model_name or "gpt-4o-mini",
                )

                if transcriber_mode == "auto":
                    try:
                        transcription_provider = build_transcription_provider(config, "auto")
                    except Exception:
                        transcription_provider = DummyTranscriptionProvider()
                        st.warning("faster-whisper unavailable. Using dummy transcription.")
                else:
                    transcription_provider = DummyTranscriptionProvider()

                try:
                    ocr_provider = build_ocr_provider(config)
                except Exception:
                    ocr_provider = None
                    st.warning("OCR unavailable. Continuing without OCR.")

                if writer_mode == "auto":
                    try:
                        llm_provider = build_llm_provider(config, "auto")
                        if not _api_key:
                            raise RuntimeError("Missing OPENAI_API_KEY")
                    except Exception:
                        llm_provider = RuleBasedLLMProvider()
                        st.warning("OpenAI writer unavailable. Using rule-based writer.")
                else:
                    llm_provider = RuleBasedLLMProvider()

                builder = LectureNotesBuilder(
                    media_processor=MediaProcessor(),
                    transcription_provider=transcription_provider,
                    ocr_provider=ocr_provider,
                    llm_provider=llm_provider,
                    config=config,
                )

                notes = builder.process(str(input_path), lecture_title=lecture_title or None)

                docx_path = Path(tmp) / "study_notes.docx"
                json_path = Path(tmp) / "study_notes.json"
                export_to_docx(notes, str(docx_path))
                export_to_json(notes, str(json_path))

                st.subheader("High-Yield Summary")
                for item in notes.summary:
                    st.markdown(f"- {item}")

                for section in notes.sections:
                    with st.expander(section.title, expanded=False):
                        st.markdown(f"**Learning goal:** {section.learning_goal}")
                        st.markdown("**Key Points**")
                        for point in section.key_points:
                            st.markdown(f"- {point}")
                        if section.comparison_table:
                            st.markdown("**Comparison Table**")
                            st.dataframe(section.comparison_table, use_container_width=True)
                        if section.common_mistakes:
                            st.markdown("**Common Mistakes**")
                            for item in section.common_mistakes:
                                st.markdown(f"- {item}")
                        if section.quick_recall:
                            st.markdown("**Quick Recall**")
                            for item in section.quick_recall:
                                st.markdown(f"- **{item}**")
                        if section.mini_mind_map:
                            st.markdown("**Mini Mind Map**")
                            for item in section.mini_mind_map:
                                st.markdown(f"- {item}")

                st.download_button(
                    "Download DOCX",
                    data=docx_path.read_bytes(),
                    file_name="study_notes.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                st.download_button(
                    "Download JSON",
                    data=json_path.read_bytes(),
                    file_name="study_notes.json",
                    mime="application/json",
                )