"""
Traditional Dress Recognition & Trend Analytics Dashboard.

Run with:
    streamlit run app/streamlit_app.py

This dashboard is read-only with respect to training: it only displays
results that exist in models/ and reports/. If you haven't run the
training scripts yet, each tab clearly says so instead of inventing numbers.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from utils import DISPLAY_NAMES, MODELS_DIR, REPORTS_DIR, load_json  # noqa: E402
from train_classifier import build_model, IMAGE_SIZE  # noqa: E402

st.set_page_config(page_title="Traditional Dress Dashboard", layout="wide", page_icon="🧵")

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "myntra" / "processed" / "traditional_wear_clean.csv"


@st.cache_resource
def load_classifier_models():
    """Loads every trained CNN checkpoint found in models/."""
    leaderboard = load_json(REPORTS_DIR / "classifier_leaderboard.json")
    if not leaderboard:
        return {}, []
    loaded = {}
    class_names = None
    for row in leaderboard:
        name = row["model"]
        ckpt_path = MODELS_DIR / f"{name}_best.pt"
        report = load_json(REPORTS_DIR / f"classifier_{name}.json")
        if not ckpt_path.exists() or not report:
            continue
        class_names = report["class_names"]
        model = build_model(name, len(class_names))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        loaded[name] = model
    return loaded, class_names or []


@st.cache_data
def load_popularity_leaderboard():
    return load_json(REPORTS_DIR / "popularity_leaderboard.json")


@st.cache_data
def load_trend_data():
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)
    return None


EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(image: Image.Image, models_dict, class_names):
    tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    predictions = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            top_idx = int(np.argmax(probs))
            predictions[name] = {
                "label": class_names[top_idx],
                "confidence": float(probs[top_idx]),
                "all_probs": {class_names[i]: float(p) for i, p in enumerate(probs)},
            }
    return predictions


st.title("🧵 Traditional Dress Recognition & Trend Analytics")
st.caption("South Asian traditional wear — classification trained on IndoFashion, trend data from real e-commerce listings.")

tab1, tab2, tab3, tab4 = st.tabs([
    "👗 Dress Classifier", "🏆 Model Leaderboard", "📈 Trend & Popularity Explorer", "🔮 Popularity Predictor",
])

# ---------------------------------------------------------------------------
# TAB 1: Dress Classifier
# ---------------------------------------------------------------------------
with tab1:
    st.header("Upload a traditional garment photo")
    classifier_models, class_names = load_classifier_models()

    if not classifier_models:
        st.info(
            "**Demo mode** — no trained classifier found yet.\n\n"
            "Run `python src/train_classifier.py --data_dir data/indofashion/processed` "
            "(ideally on a GPU via Colab/Kaggle) to populate this tab with real predictions."
        )
    else:
        uploaded = st.file_uploader("Upload an image (saree, lehenga, kurta, sherwani, etc.)", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            col_img, col_pred = st.columns([1, 2])
            with col_img:
                st.image(image, caption="Uploaded image", use_container_width=True)

            with st.spinner("Running all trained models..."):
                predictions = predict_image(image, classifier_models, class_names)

            with col_pred:
                st.subheader("Predictions by model")
                rows = []
                for model_name, pred in predictions.items():
                    rows.append({
                        "Model": model_name,
                        "Predicted class": DISPLAY_NAMES.get(pred["label"], pred["label"]),
                        "Confidence": f"{pred['confidence']*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # majority vote
                votes = [p["label"] for p in predictions.values()]
                consensus = max(set(votes), key=votes.count)
                st.success(f"**Consensus prediction: {DISPLAY_NAMES.get(consensus, consensus)}**")

            st.subheader("Per-class confidence (first model)")
            first_model = next(iter(predictions.values()))
            probs_df = pd.DataFrame(
                [(DISPLAY_NAMES.get(k, k), v) for k, v in first_model["all_probs"].items()],
                columns=["Class", "Probability"],
            ).sort_values("Probability", ascending=True)
            fig = px.bar(probs_df, x="Probability", y="Class", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Model Leaderboard
# ---------------------------------------------------------------------------
with tab2:
    st.header("Classifier model comparison")
    leaderboard = load_json(REPORTS_DIR / "classifier_leaderboard.json")

    if not leaderboard:
        st.info("No training results yet. Run `src/train_classifier.py` to populate this leaderboard.")
    else:
        df = pd.DataFrame(leaderboard)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(df, x="model", y="test_accuracy", title="Test accuracy by model", text_auto=".3f")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion matrix viewer")
        selected_model = st.selectbox("Choose a model", df["model"].tolist())
        detail = load_json(REPORTS_DIR / f"classifier_{selected_model}.json")
        if detail:
            cm = np.array(detail["confusion_matrix"])
            labels = [DISPLAY_NAMES.get(c, c) for c in detail["class_names"]]
            fig_cm = px.imshow(
                cm, x=labels, y=labels, text_auto=True, aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Trend & Popularity Explorer
# ---------------------------------------------------------------------------
with tab3:
    st.header("Traditional wear trends (real e-commerce data)")
    trend_df = load_trend_data()

    if trend_df is None:
        st.info(
            "**Demo mode** — run `python src/prepare_myntra.py` on the Myntra Sales Dataset CSV "
            "to populate this tab with real trend data."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            cat_counts = trend_df["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts, x="category", y="count", title="Listings per traditional wear category")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "number_of_ratings" in trend_df.columns:
                pop_by_cat = trend_df.groupby("category")["number_of_ratings"].mean().reset_index()
                pop_by_cat = pop_by_cat.sort_values("number_of_ratings", ascending=False)
                fig2 = px.bar(pop_by_cat, x="category", y="number_of_ratings",
                              title="Average popularity (number of ratings) by category")
                st.plotly_chart(fig2, use_container_width=True)

        if "price" in trend_df.columns and "rating" in trend_df.columns:
            fig3 = px.scatter(
                trend_df, x="price", y="rating", color="category",
                size="number_of_ratings" if "number_of_ratings" in trend_df.columns else None,
                title="Price vs. rating by category", opacity=0.6,
            )
            st.plotly_chart(fig3, use_container_width=True)

        if "brand" in trend_df.columns:
            top_brands = trend_df["brand"].value_counts().head(15).reset_index()
            top_brands.columns = ["brand", "listings"]
            fig4 = px.bar(top_brands, x="brand", y="listings", title="Top 15 brands by number of listings")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Raw data sample")
        st.dataframe(trend_df.sample(min(50, len(trend_df))), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: Popularity Predictor
# ---------------------------------------------------------------------------
with tab4:
    st.header("Predict how well a garment will trend")
    pop_leaderboard = load_popularity_leaderboard()

    if not pop_leaderboard:
        st.info("No trained popularity model yet. Run `python src/train_popularity.py` first.")
    else:
        best_model_name = pop_leaderboard[0]["model"]
        st.caption(f"Using best model: **{best_model_name}** (highest test accuracy)")
        pipeline_path = MODELS_DIR / f"popularity_{best_model_name}.joblib"

        schema = load_json(REPORTS_DIR / "popularity_feature_schema.json")
        trend_df = load_trend_data()

        if pipeline_path.exists() and schema:
            pipeline = joblib.load(pipeline_path)

            col1, col2 = st.columns(2)
            with col1:
                categories = sorted(trend_df["category"].unique()) if trend_df is not None else ["saree", "lehenga", "kurta"]
                category = st.selectbox("Category", categories)
                brands = sorted(trend_df["brand"].dropna().unique()) if trend_df is not None else ["unknown"]
                brand = st.selectbox("Brand", brands)
            with col2:
                price = st.number_input("Price (₹/local currency)", min_value=0, value=1500)
                mrp = st.number_input("MRP", min_value=0, value=2500)
                discount_pct = st.number_input("Discount %", min_value=0, max_value=90, value=40)
                rating = st.slider("Average rating", 1.0, 5.0, 4.0, 0.1)

            input_row = pd.DataFrame([{
                "price": price, "mrp": mrp, "discount_pct": discount_pct,
                "rating": rating, "category": category, "brand": brand,
            }])
            input_row = input_row[[c for c in schema["feature_cols"] if c in input_row.columns]]

            if st.button("Predict popularity tier", type="primary"):
                pred = pipeline.predict(input_row)[0]
                tier_color = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}.get(pred, "")
                st.subheader(f"{tier_color} Predicted popularity tier: **{pred}**")

                if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                    probs = pipeline.predict_proba(input_row)[0]
                    classes = pipeline.named_steps["classifier"].classes_
                    prob_df = pd.DataFrame({"Tier": classes, "Probability": probs}).sort_values("Probability")
                    fig = px.bar(prob_df, x="Probability", y="Tier", orientation="h")
                    st.plotly_chart(fig, use_container_width=True)

            detail = load_json(REPORTS_DIR / f"popularity_{best_model_name}.json")
            if detail and detail.get("feature_importance"):
                st.subheader("What drives popularity (feature importance)")
                imp = pd.DataFrame(
                    list(detail["feature_importance"].items()), columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False).head(15)
                fig_imp = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Trained pipeline or feature schema missing — re-run src/train_popularity.py.")

st.divider()
st.caption(
    "Data: IndoFashion (Rajput & Aneja, CVPRW 2021) for classification · "
    "Myntra Sales Dataset (Kaggle) for trend/popularity. No Western fashion data used."
)
