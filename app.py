import streamlit as st
import tempfile
import json
from qc_engine.pipeline import run_pipeline

st.title("QC Automation – Speaking Review")

audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
claimed_level = st.selectbox(
    "Claimed ILR Level",
    ["ILR_0","ILR_0_plus","ILR_1","ILR_1_plus","ILR_2","ILR_2_plus","ILR_3"]
)

if st.button("Run QC") and audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.read())
        audio_path = tmp.name

    with st.spinner("Running QC pipeline…"):
        report = run_pipeline(audio_path, claimed_level)

    st.subheader("Human-Readable Summary")
    st.write(f"Score Decision: **{report['score_qc']['decision'].upper()}**")
    st.write(report["score_qc"]["justification"])

    if report["procedural_qc"]["violations"]:
        st.warning("Procedural Violations Detected")
        for v in report["procedural_qc"]["violations"]:
            st.write(f"- {v}")
    else:
        st.success("No procedural violations detected")

    st.subheader("Download Reports")

    st.download_button(
        "Download JSON Report",
        json.dumps(report, indent=2),
        file_name="qc_report.json",
        mime="application/json"
    )
