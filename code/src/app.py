"""
Streamlit dashboard for the TFG polyp-detection project.

Launch with:
    cd code
    streamlit run src/app.py
"""

import glob
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = SCRIPT_DIR
PROJ_DIR = os.path.join(SRC_DIR, "..")
CSV_PATH = os.path.join(SRC_DIR, "csv", "model_performances.csv")
LOSSES_DIR = os.path.join(PROJ_DIR, "out", "losses")
SAVED_MODELS_DIR = os.path.join(PROJ_DIR, "out", "saved_models")

sys.path.insert(0, SRC_DIR)

st.set_page_config(
    page_title="TFG — Polyp Detection Dashboard",
    page_icon=":microscope:",
    layout="wide",
)

st.title(":microscope: TFG — Polyp Detection Dashboard")
st.caption(
    "Bachelor's Thesis · FIB-UPC — deep learning polyp detection "
    "with generative data augmentation"
)

tab_perf, tab_infer, tab_losses = st.tabs(
    [":bar_chart: Performance Explorer", ":mag: Inference", ":chart_with_upwards_trend: Training Losses"]
)

# ── Tab 1: Performance Explorer ──────────────────────────────────────────────

with tab_perf:
    st.header("Model Performance Explorer")

    if not os.path.exists(CSV_PATH):
        st.warning(
            f"CSV not found at `{CSV_PATH}`. "
            "Run `python src/evaluate_models.py` first to generate results."
        )
    else:
        df = pd.read_csv(CSV_PATH)

        def _f1(row):
            ap, ar = row["AP_50_95_all"], row["AR_50_95_all_maxDets_100"]
            return 2 * ap * ar / (ap + ar) if (ap + ar) > 0 else 0.0

        df["F1"] = df.apply(_f1, axis=1)
        df["LR_fmt"] = df["LR"].map(lambda x: f"{x:.2e}")
        df["Config"] = (
            df["Model"]
            + " bs="
            + df["BATCH_SIZE"].astype(str)
            + " lr="
            + df["LR_fmt"]
            + " ep="
            + df["NUM_EPOCHS"].astype(str)
        )

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            batch_sizes = sorted(df["BATCH_SIZE"].unique())
            sel_bs = st.multiselect("Batch size", batch_sizes, default=batch_sizes)
        with col_f2:
            min_lr, max_lr = float(df["LR"].min()), float(df["LR"].max())
            lr_range = st.slider(
                "Learning rate range (log scale)",
                min_value=min_lr,
                max_value=max_lr,
                value=(min_lr, max_lr),
                format="%.2e",
            )
        with col_f3:
            epochs = sorted(df["NUM_EPOCHS"].unique())
            sel_ep = st.multiselect("Num epochs", epochs, default=epochs)

        mask = (
            df["BATCH_SIZE"].isin(sel_bs)
            & df["LR"].between(lr_range[0], lr_range[1])
            & df["NUM_EPOCHS"].isin(sel_ep)
        )
        filtered = df[mask].copy()

        if filtered.empty:
            st.info("No models match the current filters.")
        else:
            # KPI row
            best = filtered.loc[filtered["F1"].idxmax()]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best F1", f"{best['F1']:.4f}")
            k2.metric("Best AP@50", f"{best['AP_50_all']:.4f}")
            k3.metric("Best AR@100", f"{best['AR_50_95_all_maxDets_100']:.4f}")
            k4.metric("Configs shown", len(filtered))

            # Grouped bar chart
            metrics_to_plot = ["AP_50_95_all", "AP_50_all", "AP_75_all", "F1"]
            plot_df = filtered[["Config"] + metrics_to_plot].melt(
                id_vars="Config", var_name="Metric", value_name="Score"
            )
            fig = px.bar(
                plot_df,
                x="Config",
                y="Score",
                color="Metric",
                barmode="group",
                title="Detection metrics by configuration",
                height=450,
            )
            fig.update_layout(xaxis_tickangle=-35, margin=dict(b=140))
            st.plotly_chart(fig, use_container_width=True)

            # AP vs AR scatter
            fig2 = px.scatter(
                filtered,
                x="AP_50_95_all",
                y="AR_50_95_all_maxDets_100",
                size="F1",
                color="BATCH_SIZE",
                hover_data=["LR_fmt", "WEIGHT_DECAY", "NUM_EPOCHS"],
                title="AP vs AR (bubble size = F1)",
                labels={
                    "AP_50_95_all": "AP @[.50:.95]",
                    "AR_50_95_all_maxDets_100": "AR @[.50:.95] maxDets=100",
                },
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Full data table
            with st.expander("Full results table"):
                display_cols = [
                    "Config", "BATCH_SIZE", "LR", "WEIGHT_DECAY", "NUM_EPOCHS",
                    "AP_50_95_all", "AP_50_all", "AP_75_all",
                    "AR_50_95_all_maxDets_100", "F1",
                ]
                st.dataframe(
                    filtered[display_cols].sort_values("F1", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Tab 2: Inference ─────────────────────────────────────────────────────────

with tab_infer:
    st.header("Run Inference")

    try:
        import torch
        from torchvision import transforms
        from torchvision.utils import draw_bounding_boxes
        from PIL import Image

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

    if not TORCH_AVAILABLE:
        st.error(
            "PyTorch is not installed in this environment. "
            "Install it with `pip install torch torchvision` to enable inference."
        )
    else:
        try:
            from clases.model_utils import get_model, load_model_with_hyperparams
        except ImportError:
            st.error(
                "Could not import `clases.model_utils`. "
                "Make sure you run this app from `code/` with: "
                "`streamlit run src/app.py`"
            )
            st.stop()

        col_left, col_right = st.columns([1, 2])

        with col_left:
            model_arch = st.selectbox(
                "Model architecture", ["FasterRCNN", "RetinaNet", "SSD"]
            )
            confidence = st.slider(
                "Confidence threshold", 0.0, 1.0, 0.5, 0.05
            )

            model_dir = st.text_input(
                "Saved models directory",
                value=SAVED_MODELS_DIR,
            )
            available_models = []
            if os.path.isdir(model_dir):
                available_models = [
                    f
                    for f in os.listdir(model_dir)
                    if not f.startswith(".")
                ]
            if available_models:
                model_file = st.selectbox("Model weights file", available_models)
            else:
                st.info(
                    "No model files found. Train a model first or point to "
                    "the correct directory."
                )
                model_file = None

            uploaded = st.file_uploader(
                "Upload a colonoscopy image", type=["jpg", "jpeg", "png", "bmp"]
            )

        with col_right:
            if uploaded is not None and model_file is not None:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Uploaded image", use_container_width=True)

                if st.button("Run detection", type="primary"):
                    with st.spinner("Loading model and running inference..."):
                        num_classes = 2
                        model = get_model(model_arch, num_classes)
                        model, _, _ = load_model_with_hyperparams(
                            model, model_file, load_dir=model_dir
                        )
                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        model.to(device)
                        model.eval()

                        transform = transforms.Compose(
                            [transforms.Resize((560, 480)), transforms.ToTensor()]
                        )
                        img_tensor = transform(img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(img_tensor)[0]

                        keep = output["scores"] >= confidence
                        boxes = output["boxes"][keep].cpu()
                        scores = output["scores"][keep].cpu()
                        labels = output["labels"][keep].cpu()

                        img_byte = (
                            transform(img).mul(255).byte()
                        )
                        label_strings = [
                            f"polyp {s:.2f}" for s in scores.tolist()
                        ]
                        drawn = draw_bounding_boxes(
                            img_byte, boxes, labels=label_strings, colors="red", width=2
                        )
                        from torchvision.transforms.functional import to_pil_image

                        result_img = to_pil_image(drawn)
                        st.image(
                            result_img,
                            caption=f"{len(boxes)} detection(s) above {confidence:.0%} confidence",
                            use_container_width=True,
                        )

                        if len(boxes) > 0:
                            det_df = pd.DataFrame(
                                {
                                    "Score": scores.tolist(),
                                    "x_min": boxes[:, 0].tolist(),
                                    "y_min": boxes[:, 1].tolist(),
                                    "x_max": boxes[:, 2].tolist(),
                                    "y_max": boxes[:, 3].tolist(),
                                }
                            )
                            st.dataframe(det_df, hide_index=True)
                        else:
                            st.info("No detections above the confidence threshold.")
            elif uploaded is None:
                st.info("Upload a colonoscopy image on the left to get started.")
            else:
                st.info("Select a model weights file to enable inference.")

# ── Tab 3: Training Losses ───────────────────────────────────────────────────

with tab_losses:
    st.header("Training Loss Viewer")

    losses_dir = st.text_input(
        "Losses directory",
        value=LOSSES_DIR,
        key="losses_dir_input",
    )

    if not os.path.isdir(losses_dir):
        st.warning(
            f"Directory `{losses_dir}` not found. "
            "Train a model to generate loss files, or point to the correct path."
        )
    else:
        loss_files = sorted(glob.glob(os.path.join(losses_dir, "*_losses.txt")))
        if not loss_files:
            st.info("No loss files found in the directory.")
        else:
            basenames = [os.path.basename(f) for f in loss_files]

            epoch_files = [b for b in basenames if "epoch_losses" in b]
            batch_files = [b for b in basenames if "batch_losses" in b]

            loss_type = st.radio(
                "Loss granularity", ["Epoch losses", "Batch losses"], horizontal=True
            )
            file_list = epoch_files if loss_type == "Epoch losses" else batch_files

            if not file_list:
                st.info(f"No {loss_type.lower()} files found.")
            else:
                selected = st.multiselect(
                    "Select runs to compare",
                    file_list,
                    default=file_list[:3],
                )

                if selected:
                    fig = go.Figure()
                    for fname in selected:
                        fpath = os.path.join(losses_dir, fname)
                        with open(fpath, "r") as f:
                            values = [float(line.strip()) for line in f if line.strip()]
                        short = fname.replace("_epoch_losses.txt", "").replace(
                            "_batch_losses.txt", ""
                        )
                        fig.add_trace(
                            go.Scatter(
                                y=values,
                                mode="lines+markers",
                                name=short,
                                marker=dict(size=4),
                            )
                        )
                    x_label = "Epoch" if loss_type == "Epoch losses" else "Batch"
                    fig.update_layout(
                        title=f"{loss_type} over training",
                        xaxis_title=x_label,
                        yaxis_title="Loss",
                        height=500,
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
