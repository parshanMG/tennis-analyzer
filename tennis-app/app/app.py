# app/app.py
import streamlit as st
from pathlib import Path
from src.io_utils import frames
# from src.pose import PoseEstimator
# from src.metrics import compute_metrics_for_segment
# from src.phases import segment_serves

st.title("AI Tennis Serve Analyzer (MVP)")
uploaded = st.file_uploader("Upload a serve video (mp4/mov/avi)", type=["mp4","mov","avi"])

if uploaded:
    tmp = Path("data/interim") / uploaded.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(uploaded.read())
    st.video(str(tmp))

    if st.button("Analyze"):
        # 1) Load frames
        # 2) Pose estimation per frame
        # 3) Segment serves
        # 4) Compute per-serve metrics
        # 5) Aggregate summary
        # For now, stub a fake response:
        per_serve = [
            {"serve_id": 1, "knee_bend_deg": 32.4, "contact_height_rel": 0.22, "x_factor_deg": 28.0, "score": 73.2},
            {"serve_id": 2, "knee_bend_deg": 29.7, "contact_height_rel": 0.20, "x_factor_deg": 26.5, "score": 70.5},
        ]
        st.success("Analysis complete (demo).")
        st.write("Per-serve metrics:")
        st.dataframe(per_serve)

        st.write("Session summary:")
        st.json({
            "avg_knee_bend_deg": sum(d["knee_bend_deg"] for d in per_serve)/len(per_serve),
            "avg_contact_height_rel": sum(d["contact_height_rel"] for d in per_serve)/len(per_serve),
            "avg_x_factor_deg": sum(d["x_factor_deg"] for d in per_serve)/len(per_serve),
            "avg_score": sum(d["score"] for d in per_serve)/len(per_serve),
        })
