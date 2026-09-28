import sys
import os
import glob
import csv
import json
import cv2

# إصلاح مشكلة طباعة العربي على Windows terminal (cp1252 مش بيدعم يونيكود)
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage

# ==========================================
# 1. METRICS CALCULATION
# ==========================================
def calculate_metrics(gt, pred):
    """
    Calculate performance metrics:
    Dice Coefficient, IoU (Jaccard), Sensitivity, Specificity
    """
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    TP = np.sum((gt == 1) & (pred == 1))
    FP = np.sum((gt == 0) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FN = np.sum((gt == 1) & (pred == 0))

    dice = (2.0 * TP) / (2.0 * TP + FP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)

    return dice, iou, sensitivity, specificity

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
def preprocess_image(img):
    """
    Contrast Enhancement (CLAHE), Noise Removal, Illumination Correction (Top-Hat)
    """
    if len(img.shape) == 3:
        gray = img[:, :, 1]  # Extract Green channel
    else:
        gray = img.copy()

    # 1. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # ملحوظة: شلنا الـ Top-Hat filter اللي كان هنا.
    # Top-Hat بيمسح أي منطقة "كبيرة" وسايب بس تفاصيل صغيرة فاتحة (مناسب لخيوط شرايين رفيعة).
    # لكن في حالتنا (skin lesion)، الآفة نفسها منطقة كبيرة، فالـ Top-Hat كان بيمسحها هي كمان!
    # فبنكتفي بـ CLAHE + Denoising بس كخطوة تحضير نهائية.
    return denoised

def orient_foreground(mask):
    """
    الآفة الجلدية (lesion) المفروض تكون أصغر مساحة من باقي الصورة (الجلد).
    لو الـ mask طلعت أغلبها 1 (يعني أكتر من 50% من الصورة)، معناها الـ method
    اختارت الجلد (الخلفية) بالغلط بدل الآفة، فبنعكسها.
    """
    if np.mean(mask) > 0.5:
        return 1 - mask
    return mask

# ==========================================
# 3. POST-PROCESSING & FEATURE EXTRACTION
# ==========================================
def post_process_and_extract_features(binary_mask):
    """
    Morphological operations, Hole Filling, Component filtering, Shape Features
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Morphological Opening and Closing
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Hole filling
    cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

    # Remove small noise components (Using max_size to satisfy scikit-image 0.26+)
    cleaned = morphology.remove_small_objects(cleaned.astype(bool), max_size=30).astype(np.uint8)

    # Shape Features Extraction
    labeled_mask, _ = ndimage.label(cleaned)
    regions = measure.regionprops(labeled_mask)

    total_area = sum(r.area for r in regions)
    total_perimeter = sum(r.perimeter for r in regions)

    circularities = [(4 * np.pi * r.area) / (r.perimeter ** 2 + 1e-6) for r in regions if r.perimeter > 0]
    avg_circularity = np.mean(circularities) if circularities else 0.0

    features = {
        "Area": total_area,
        "Perimeter": round(total_perimeter, 2),
        "Circularity": round(avg_circularity, 4)
    }

    return cleaned, features

# ==========================================
# 4. THE 4 SEGMENTATION METHODS
# ==========================================

# Method 1: Otsu Thresholding
def method_otsu(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (thresh > 0).astype(np.uint8)

# Method 2: K-Means Clustering
def method_kmeans(img, k=2):
    pixel_vals = img.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    high_cluster = np.argmax(centers)
    segmented = (labels == high_cluster).astype(np.uint8)
    return segmented.reshape(img.shape)

# Method 3: Region Growing
def method_region_growing(img, seed_threshold=2.0):
    seed_val = np.percentile(img, 98)
    seed = (img >= seed_val)

    if not np.any(seed):
        return np.zeros_like(img, dtype=np.uint8)

    mean_val = np.mean(img[seed])
    std_val = np.std(img) + 1e-6
    segmented = np.abs(img.astype(np.float32) - mean_val) < (seed_threshold * std_val)
    return segmented.astype(np.uint8)

# Method 4: Marker-Controlled Watershed
def method_watershed(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    mask = np.zeros_like(img, dtype=np.uint8)
    mask[markers > 1] = 1
    return mask

def detect_failure_reason(dice, iou, sens, spec):
    """
    لو الأداء ضعيف، بترجع سبب محتمل (نص) بناءً على شكل الأرقام.
    لو الأداء كويس، بترجع None (يعني مش failure case).
    """
    DICE_THRESHOLD = 0.5  # أي صورة/method تحت الرقم ده تعتبر "فاشلة"

    if dice >= DICE_THRESHOLD:
        return None

    if spec < 0.3:
        return "الموديل اعتبر معظم الصورة foreground بالغلط (over-segmentation / فشل الـ watershed markers)"
    elif sens < 0.3:
        return "الموديل فشل يكتشف معظم الآفة (under-segmentation)"
    elif iou < 0.3:
        return "تداخل ضعيف جداً مع الـ ground truth (شكل الآفة غير منتظم أو contrast ضعيف)"
    else:
        return "أداء ضعيف عموماً (Dice تحت 0.5) بدون سبب واحد واضح"

# ==========================================
# SAVE RESULTS TO CSV
# ==========================================
def save_results_to_csv(results, output_dir="results"):
    """
    بتحفظ ملفين:
    1. detailed_metrics.csv -> صف لكل صورة/method (كل الأرقام بالتفصيل)
    2. summary_metrics.csv  -> صف واحد لكل method (المتوسط على كل الصور)
    """
    os.makedirs(output_dir, exist_ok=True)

    # -------- 1. detailed_metrics.csv --------
    detailed_path = os.path.join(output_dir, "detailed_metrics.csv")
    with open(detailed_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # صف العناوين (header)
        writer.writerow([
            "method", "image_id", "dice", "iou", "sensitivity",
            "specificity", "area", "perimeter", "circularity"
        ])
        # نلف على كل method، وجوه كل method نلف على كل صورة
        for method_name, entries in results.items():
            for e in entries:
                writer.writerow([
                    method_name, e["image_id"],
                    round(e["dice"], 4), round(e["iou"], 4),
                    round(e["sensitivity"], 4), round(e["specificity"], 4),
                    e["area"], e["perimeter"], e["circularity"]
                ])

    # -------- 2. summary_metrics.csv --------
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "avg_dice", "avg_iou", "avg_sensitivity", "avg_specificity", "num_images"])
        for method_name, entries in results.items():
            if not entries:
                continue
            avg_dice = np.mean([e["dice"] for e in entries])
            avg_iou = np.mean([e["iou"] for e in entries])
            avg_sens = np.mean([e["sensitivity"] for e in entries])
            avg_spec = np.mean([e["specificity"] for e in entries])
            writer.writerow([
                method_name, round(avg_dice, 4), round(avg_iou, 4),
                round(avg_sens, 4), round(avg_spec, 4), len(entries)
            ])

    print(f"\n[INFO] تم حفظ النتائج في:\n  - {detailed_path}\n  - {summary_path}")

# ==========================================
# SAVE FAILURE CASES TO JSON
# ==========================================
def save_failure_cases_json(failures, output_dir="results"):
    """
    بتحفظ كل الحالات الفاشلة (اللي Dice بتاعها تحت الـ threshold) في ملف JSON واحد،
    كل حالة فيها: الصورة، الـ method، الأرقام، والسبب المحتمل للفشل.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "failure_cases.json")

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"[INFO] تم حفظ {len(failures)} حالة فشل في:\n  - {json_path}")

# ==========================================
# SAVE PER-IMAGE COMPARISON FIGURE
# ==========================================
def save_comparison_figure(img, gt, preprocessed, cleaned_masks, image_id, output_dir="results/figures"):
    """
    بتحفظ صورة واحدة (PNG) فيها: Original + Ground Truth + Preprocessed + نتائج الـ 4 methods.
    Grid شكله 2x4: الصف الأول (Original, GT, Preprocessed, فاضي) والصف التاني (الـ 4 methods).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(preprocessed, cmap='gray')
    axes[0, 2].set_title("Preprocessed")
    axes[0, 2].axis('off')

    axes[0, 3].axis('off')  # خانة فاضية

    for i, (name, mask) in enumerate(cleaned_masks.items()):
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(name, fontsize=9)
        axes[1, i].axis('off')

    fig.suptitle(image_id)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{image_id}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)  # مهم جداً: بنقفل الـ figure عشان الذاكرة متمتلاش (خصوصاً لو هنشتغل على 900 صورة)

# ==========================================
# 6. BATCH PROCESSING ON THE FULL DATASET
# ==========================================
def run_batch_pipeline(data_dir, gt_dir, limit=None, output_dir="results", save_figures=True):
    """
    يشتغل على كل صور فولدر data_dir، ويجيب الـ mask المطابق من gt_dir.
    limit: لو محدد (زي 15)، بيشتغل على أول N صورة بس. لو None، بيشتغل على الكل.
    بيرجع dictionary فيه نتائج كل method لكل صورة.
    """
    # 1. هات كل مسارات الصور (jpg) مرتبة أبجدياً
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"[INFO] هيتم معالجة {len(image_paths)} صورة...")

    method_names = ["Otsu Threshold", "K-Means Clustering", "Region Growing", "Marker Watershed"]
    # dictionary فيه list فاضية لكل method، هنملاها بالنتائج
    results = {name: [] for name in method_names}
    # list هتتجمع فيها كل الحالات الفاشلة (من أي method)
    failures = []

    for idx, img_path in enumerate(image_paths, start=1):
        # 2. استخراج الـ image_id من اسم الملف (ISIC_0000000)
        filename = os.path.basename(img_path)
        image_id = os.path.splitext(filename)[0]

        # 3. بناء مسار الـ ground truth المطابق
        gt_path = os.path.join(gt_dir, f"{image_id}_Segmentation.png")

        if not os.path.exists(gt_path):
            print(f"[WARNING] مفيش ground truth للصورة {image_id}, هتتخطى")
            continue

        # 4. قراءة الصورة والـ mask
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gt is None:
            print(f"[WARNING] مشكلة في قراءة {image_id}, هتتخطى")
            continue

        # 5. لو مقاس الصورة والـ mask مش متطابقين، نظبط الـ mask
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 6. Preprocessing زي ما هو
        preprocessed = preprocess_image(img)

        raw_methods = {
            "Otsu Threshold": orient_foreground(method_otsu(preprocessed)),
            "K-Means Clustering": orient_foreground(method_kmeans(preprocessed)),
            "Region Growing": orient_foreground(method_region_growing(preprocessed)),
            "Marker Watershed": orient_foreground(method_watershed(preprocessed))
        }

        print(f"\n[{idx}/{len(image_paths)}] {image_id}")

        # dictionary هنجمع فيه الـ cleaned mask بتاعة كل method، عشان نرسمهم مع بعض في الآخر
        cleaned_masks_for_figure = {}

        # 7. لكل method: post-processing + metrics + تخزين النتيجة
        for name, raw_mask in raw_methods.items():
            cleaned_mask, features = post_process_and_extract_features(raw_mask)
            cleaned_masks_for_figure[name] = cleaned_mask
            dice, iou, sens, spec = calculate_metrics(gt, cleaned_mask)

            results[name].append({
                "image_id": image_id,
                "dice": dice,
                "iou": iou,
                "sensitivity": sens,
                "specificity": spec,
                "area": features["Area"],
                "perimeter": features["Perimeter"],
                "circularity": features["Circularity"]
            })

            print(f"   {name:<20} Dice={dice:.4f}  IoU={iou:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

            # فحص لو الحالة دي "فشل" ولو كده، نسجلها
            reason = detect_failure_reason(dice, iou, sens, spec)
            if reason is not None:
                failures.append({
                    "image_id": image_id,
                    "method": name,
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4),
                    "reason": reason
                })

        # 8. حفظ figure المقارنة للصورة دي (بعد ما خلصنا كل الـ 4 methods)
        if save_figures:
            save_comparison_figure(
                img, gt, preprocessed, cleaned_masks_for_figure,
                image_id, output_dir=os.path.join(output_dir, "figures")
            )

    # 8. حساب المتوسط (average) لكل method على كل الصور اللي اتعالجت
    print("\n" + "=" * 70)
    print("AVERAGE METRICS ACROSS ALL PROCESSED IMAGES")
    print("=" * 70)
    print(f"{'Method':<20} | {'Dice':<7} | {'IoU':<7} | {'Sens':<7} | {'Spec':<7}")
    print("-" * 70)

    for name in method_names:
        entries = results[name]
        if not entries:
            continue
        avg_dice = np.mean([e["dice"] for e in entries])
        avg_iou = np.mean([e["iou"] for e in entries])
        avg_sens = np.mean([e["sensitivity"] for e in entries])
        avg_spec = np.mean([e["specificity"] for e in entries])
        print(f"{name:<20} | {avg_dice:.4f}  | {avg_iou:.4f}  | {avg_sens:.4f}  | {avg_spec:.4f}")

    print("=" * 70)

    # حفظ كل النتائج في CSV
    save_results_to_csv(results, output_dir=output_dir)

    # حفظ الحالات الفاشلة في JSON
    save_failure_cases_json(failures, output_dir=output_dir)
    print(f"[INFO] عدد الحالات الفاشلة: {len(failures)} من أصل {len(image_paths) * len(method_names)} محاولة")

    return results

# ==========================================
# 7. EXPERIMENT PIPELINE & VISUALIZATION (single image demo)
# ==========================================
def run_pipeline(image_path=None, ground_truth_path=None):
    img = None
    gt = None

    if image_path:
        img = cv2.imread(image_path)

    if img is None:
        print("[INFO] No input image found. Creating synthetic retinal vessel image...")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 70, (10, 180, 10), -1)
        cv2.line(img, (30, 30), (270, 270), (10, 220, 10), 8)
        cv2.line(img, (270, 30), (30, 270), (10, 220, 10), 4)

    if ground_truth_path:
        gt = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        gt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.circle(gt, (150, 150), 70, 255, -1)
        cv2.line(gt, (30, 30), (270, 270), 255, 8)
        cv2.line(gt, (270, 30), (30, 270), 255, 4)

    preprocessed = preprocess_image(img)

    raw_methods = {
        "Otsu Threshold": orient_foreground(method_otsu(preprocessed)),
        "K-Means Clustering": orient_foreground(method_kmeans(preprocessed)),
        "Region Growing": orient_foreground(method_region_growing(preprocessed)),
        "Marker Watershed": orient_foreground(method_watershed(preprocessed))
    }

    cleaned_masks = {}

    print("\n" + "=" * 65)
    print(f"{'Method':<20} | {'Dice':<7} | {'IoU':<7} | {'Sens':<7} | {'Spec':<7}")
    print("=" * 65)

    for name, raw_mask in raw_methods.items():
        cleaned_mask, features = post_process_and_extract_features(raw_mask)
        cleaned_masks[name] = cleaned_mask

        dice, iou, sens, spec = calculate_metrics(gt, cleaned_mask)

        print(f"{name:<20} | {dice:.4f}  | {iou:.4f}  | {sens:.4f}  | {spec:.4f}")
        print(f"   -> Shape Features: {features}\n")
    print("=" * 65)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(preprocessed, cmap='gray')
    axes[0, 1].set_title("2. Preprocessed")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt, cmap='gray')
    axes[0, 2].set_title("3. Ground Truth")
    axes[0, 2].axis('off')

    plot_coords = [(1, 0), (1, 1), (1, 2)]
    for idx, (name, mask) in enumerate(cleaned_masks.items()):
        if idx < 3:
            r, c = plot_coords[idx]
            axes[r, c].imshow(mask, cmap='gray')
            axes[r, c].set_title(f"Method {idx+1}: {name}")
            axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # مسارات الـ dataset (نسبية لمكان ملف project.py)
    DATA_DIR = os.path.join(
        "ISBI2016_ISIC_Part1_Training_Data",
        "ISBI2016_ISIC_Part1_Training_Data"
    )
    GT_DIR = os.path.join(
        "ISBI2016_ISIC_Part1_Training_GroundTruth",
        "ISBI2016_ISIC_Part1_Training_GroundTruth"
    )

    # اختبار أولي على 15 صورة بس
    results = run_batch_pipeline(DATA_DIR, GT_DIR, limit=15)