import streamlit as st
import cv2
import numpy as np
import os
import json
import pandas as pd

# بنستورد الدوال من project.py مباشرة عشان منكررش الكود
from project import (
    preprocess_image, orient_foreground,
    method_otsu, method_kmeans, method_region_growing, method_watershed,
    post_process_and_extract_features, calculate_metrics
)

st.set_page_config(page_title="Skin Lesion Segmentation", layout="wide")

st.title("🔬 Medical Image Segmentation — Skin Lesion (ISIC 2016)")
st.caption("مقارنة 4 طرق Classical Segmentation: Otsu, K-Means, Region Growing, Marker Watershed")

DATA_DIR = os.path.join("ISBI2016_ISIC_Part1_Training_Data", "ISBI2016_ISIC_Part1_Training_Data")
GT_DIR = os.path.join("ISBI2016_ISIC_Part1_Training_GroundTruth", "ISBI2016_ISIC_Part1_Training_GroundTruth")
RESULTS_DIR = "results"

tab1, tab2, tab3 = st.tabs(["🖼️ Demo تفاعلي", "📊 نتائج الـ Batch", "⚠️ Failure Cases"])

# ==========================================
# TAB 1: Demo تفاعلي — صورة واحدة
# ==========================================
with tab1:
    st.subheader("جرب صورة من الـ dataset أو ارفع صورة بنفسك")
    source = st.radio("مصدر الصورة", ["اختار من الـ dataset", "ارفع صورة"], horizontal=True)

    img = None
    gt = None
    image_id = None

    if source == "اختار من الـ dataset":
        if os.path.isdir(DATA_DIR):
            files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")])
            selected = st.selectbox("اختار صورة", files)
            if selected:
                image_id = os.path.splitext(selected)[0]
                img = cv2.imread(os.path.join(DATA_DIR, selected))
                gt_path = os.path.join(GT_DIR, f"{image_id}_Segmentation.png")
                if os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        else:
            st.warning(
                "مجلد الـ dataset مش موجود. تأكد إنك شغال الـ app من نفس الفولدر اللي فيه "
                "ISBI2016_ISIC_Part1_Training_Data"
            )
    else:
        uploaded = st.file_uploader("ارفع صورة (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_id = uploaded.name

        gt_uploaded = st.file_uploader("ارفع ground truth mask (اختياري، لو عايز تحسب الـ metrics)", type=["png"])
        if gt_uploaded is not None:
            gt_bytes = np.asarray(bytearray(gt_uploaded.read()), dtype=np.uint8)
            gt = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        if gt is not None and gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        with st.spinner("بنعالج الصورة..."):
            preprocessed = preprocess_image(img)

            raw_methods = {
                "Otsu Threshold": orient_foreground(method_otsu(preprocessed)),
                "K-Means Clustering": orient_foreground(method_kmeans(preprocessed)),
                "Region Growing": orient_foreground(method_region_growing(preprocessed)),
                "Marker Watershed": orient_foreground(method_watershed(preprocessed))
            }

        st.markdown("### الصورة الأصلية والـ Preprocessing")
        cols = st.columns(3)
        cols[0].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        cols[1].image(preprocessed, caption="Preprocessed (CLAHE + Denoise)", use_container_width=True, clamp=True)
        if gt is not None:
            cols[2].image(gt, caption="Ground Truth", use_container_width=True, clamp=True)
        else:
            cols[2].info("مفيش Ground Truth متاح — هتتعرض النتائج من غير حساب الـ metrics")

        st.markdown("### نتائج الـ 4 Methods")
        method_cols = st.columns(4)
        metrics_table = []

        for i, (name, raw_mask) in enumerate(raw_methods.items()):
            cleaned_mask, features = post_process_and_extract_features(raw_mask)
            with method_cols[i]:
                st.image(cleaned_mask * 255, caption=name, use_container_width=True, clamp=True)
                if gt is not None:
                    dice, iou, sens, spec = calculate_metrics(gt, cleaned_mask)
                    st.metric("Dice", f"{dice:.3f}")
                    st.caption(f"IoU: {iou:.3f}  |  Sens: {sens:.3f}  |  Spec: {spec:.3f}")
                    metrics_table.append({
                        "Method": name, "Dice": round(dice, 4), "IoU": round(iou, 4),
                        "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
                        "Area": features["Area"], "Circularity": features["Circularity"]
                    })
                else:
                    st.caption(f"Area: {features['Area']}  |  Circularity: {features['Circularity']}")

        if metrics_table:
            st.markdown("### جدول المقارنة")
            st.dataframe(pd.DataFrame(metrics_table), use_container_width=True)

# ==========================================
# TAB 2: نتائج الـ Batch Processing (من ملفات CSV)
# ==========================================
with tab2:
    st.subheader("متوسط الأداء على كل صور الـ Dataset اللي اتعالجت")
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    detailed_path = os.path.join(RESULTS_DIR, "detailed_metrics.csv")

    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        st.dataframe(df_summary, use_container_width=True)
        st.markdown("#### مقارنة Dice و IoU بين الـ Methods")
        st.bar_chart(df_summary.set_index("method")[["avg_dice", "avg_iou"]])
    else:
        st.warning("لسه معملتش batch processing. شغل project.py الأول عشان تتولد ملفات النتائج في فولدر results/.")

    if os.path.exists(detailed_path):
        df_detailed = pd.read_csv(detailed_path)
        with st.expander(f"شوف كل التفاصيل ({len(df_detailed)} صف)"):
            st.dataframe(df_detailed, use_container_width=True)

# ==========================================
# TAB 3: Failure Cases
# ==========================================
with tab3:
    st.subheader("الحالات اللي فشلت (Dice تحت 0.5)")
    failures_path = os.path.join(RESULTS_DIR, "failure_cases.json")

    if os.path.exists(failures_path):
        with open(failures_path, encoding="utf-8") as f:
            failures = json.load(f)

        st.write(f"إجمالي الحالات الفاشلة: **{len(failures)}**")

        for fcase in failures:
            title = f"{fcase['image_id']} — {fcase['method']}  (Dice = {fcase['dice']})"
            with st.expander(title):
                st.write(f"**السبب:** {fcase['reason']}")
                st.json(fcase)
                fig_path = os.path.join(RESULTS_DIR, "figures", f"{fcase['image_id']}.png")
                if os.path.exists(fig_path):
                    st.image(fig_path, use_container_width=True)
    else:
        st.warning("لسه معملتش batch processing. شغل project.py الأول عشان تتولد ملفات النتائج.")