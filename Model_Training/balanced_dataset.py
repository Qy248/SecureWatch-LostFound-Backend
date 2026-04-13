"""
YOLO Split Balancer (Train + Val) + YAML Class Names (Full Code, Debug-Friendly)

YOU ALREADY HAVE A SPLIT.
Input expected (Ultralytics standard):
merged_dataset/
  data.yaml
  images/train, images/val
  labels/train, labels/val

Output created:
balanced_merged_dataset/
  data.yaml (copied)
  images/train, images/val
  labels/train, labels/val

METHODS USED (for report):
1) Class Frequency Analysis (EDA): images-per-class + objects-per-class from YOLO labels
2) TRAIN balancing: augmentation-based oversampling (Albumentations)
3) VAL balancing: REAL-only capping/downsampling (no augmentation in val)
4) Multi-label aware counting: one image can contain multiple classes; update counts for all classes in image
5) Controlled train target selection: median_x2 with cap to prevent dataset explosion

IMPORTANT:
- If your VAL has very few samples for some classes (e.g. 6 images),
  "perfectly equal val" WITHOUT augmentation would force other classes down near 6.
  That's why VAL uses a cap-based balancing strategy.
"""

from __future__ import annotations

import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# =============================
# PART 0: Imports (with try/except)
# =============================
try:
    import cv2
except Exception as e:
    print("[FATAL] OpenCV import failed. Install: pip install opencv-python")
    print("Reason:", repr(e))
    sys.exit(1)

try:
    import albumentations as A
except Exception as e:
    print("[FATAL] Albumentations import failed. Install: pip install albumentations")
    print("Reason:", repr(e))
    sys.exit(1)

try:
    import yaml
except Exception as e:
    print("[FATAL] PyYAML import failed. Install: pip install pyyaml")
    print("Reason:", repr(e))
    sys.exit(1)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"]


# =============================
# PART 1: YAML class names
# =============================
def load_class_names(dataset_root: Path) -> Dict[int, str]:
    """
    Load class id -> name from dataset_root/data.yaml
    Supports:
      names: {0: backpack, 1: handbag, ...}
      names: [backpack, handbag, ...]
    """
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        print(f"[WARN] data.yaml NOT found at: {yaml_path.resolve()}")
        return {}

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}

        print("[WARN] Unsupported 'names' format in data.yaml")
        return {}
    except Exception as e:
        print("[WARN] Failed to parse data.yaml:", repr(e))
        return {}


def copy_data_yaml(dataset_root: Path, out_root: Path):
    """Copy data.yaml to output root for convenience."""
    try:
        src = dataset_root / "data.yaml"
        if src.exists():
            out_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_root / "data.yaml")
            print("[OK] Copied data.yaml to output:", (out_root / "data.yaml").resolve())
        else:
            print("[WARN] data.yaml not found to copy:", src.resolve())
    except Exception as e:
        print("[WARN] Failed to copy data.yaml:", repr(e))


# =============================
# PART 2: YOLO label helpers
# =============================
def read_yolo_label(label_path: Path):
    """Read YOLO txt -> list of (cls, x, y, w, h)."""
    try:
        boxes = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append((cls, x, y, w, h))
        return boxes
    except Exception as e:
        print(f"[WARN] read_yolo_label failed: {label_path} | {repr(e)}")
        return []


def write_yolo_label(label_path: Path, boxes):
    """Write YOLO label file."""
    try:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8") as f:
            for cls, x, y, w, h in boxes:
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        return True
    except Exception as e:
        print(f"[ERROR] write_yolo_label failed: {label_path} | {repr(e)}")
        return False


def get_classes_from_label(label_path: Path) -> Set[int]:
    return set(b[0] for b in read_yolo_label(label_path))


def yolo_to_albu(boxes):
    bboxes, class_labels = [], []
    for cls, x, y, w, h in boxes:
        bboxes.append([x, y, w, h])
        class_labels.append(cls)
    return bboxes, class_labels


def albu_to_yolo(bboxes, class_labels):
    out = []
    for bb, cls in zip(bboxes, class_labels):
        x, y, w, h = bb
        out.append((int(cls), float(x), float(y), float(w), float(h)))
    return out


# =============================
# PART 3: Image indexing helpers
# =============================
def build_image_index(images_dir: Path) -> Dict[str, Path]:
    """stem -> image path mapping."""
    try:
        idx = {}
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix in IMG_EXTS:
                idx[p.stem] = p
        return idx
    except Exception as e:
        print(f"[ERROR] build_image_index failed: {images_dir} | {repr(e)}")
        return {}


def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# =============================
# PART 4: Scan split stats
# =============================
def scan_split(images_dir: Path, labels_dir: Path):
    """
    Returns:
      obj_count[c] = total objects of class c
      img_count[c] = number of images containing class c
      class_to_images[c] = list of image paths containing class c
      matched_pairs = [(img, lab)]
    """
    obj_count = defaultdict(int)
    img_count = defaultdict(int)
    class_to_images = defaultdict(list)
    matched_pairs = []

    img_index = build_image_index(images_dir)
    label_files = list(labels_dir.glob("*.txt"))

    for lab in label_files:
        img = img_index.get(lab.stem)
        if img is None:
            continue

        boxes = read_yolo_label(lab)
        cls_set = set([b[0] for b in boxes])

        for b in boxes:
            obj_count[b[0]] += 1
        for c in cls_set:
            img_count[c] += 1
            class_to_images[c].append(img)

        matched_pairs.append((img, lab))

    return obj_count, img_count, class_to_images, matched_pairs


def print_summary(title: str, obj_count: dict, img_count: dict, class_names: Dict[int, str]):
    """
    Prints YAML names ONLY (no id).
    Falls back to 'class_{id}' only if yaml missing that id.
    """
    try:
        print(f"\n📌 {title}")
        print("=" * 70)
        print(f"{'Class':<28}{'Images':>10}{'Objects':>12}")
        print("-" * 70)

        for c in sorted(img_count.keys()):
            cname = class_names.get(c, f"class_{c}")
            print(f"{cname:<28}{img_count[c]:>10}{obj_count[c]:>12}")

        print("=" * 70)

        if img_count:
            smallest = min(img_count.values())
            largest = max(img_count.values())
            ratio = largest / smallest if smallest else float("inf")
            print(f"[INFO] Imbalance ratio (largest/smallest) = {ratio:.2f}x")

    except Exception as e:
        print("[WARN] print_summary failed:", repr(e))


# =============================
# PART 5: Copy utilities
# =============================
def copy_pairs(pairs: List[Tuple[Path, Path]], out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img, lab in pairs:
        try:
            shutil.copy2(img, out_images / img.name)
            shutil.copy2(lab, out_labels / lab.name)
            copied += 1
        except Exception as e:
            print(f"[WARN] copy failed: {img.name} / {lab.name} | {repr(e)}")
    return copied


# =============================
# PART 6: TRAIN Balancing (Augmentation Oversampling)
# =============================
def choose_train_target(img_count: dict, mode="median_x2", cap=1200):
    vals = sorted(img_count.values())
    if not vals:
        return 200
    median = vals[len(vals) // 2]
    if mode == "median_x2":
        return min(cap, median * 2)
    if isinstance(mode, int):
        return mode
    return min(cap, median * 2)


def balance_train_by_augmentation(
    out_train_images: Path,
    out_train_labels: Path,
    orig_train_labels_dir: Path,
    class_to_images: dict,
    target: int,
    seed=42
):
    random.seed(seed)

    # current counts from output
    out_counts = defaultdict(int)
    for lab in out_train_labels.glob("*.txt"):
        for c in get_classes_from_label(lab):
            out_counts[c] += 1

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Rotate(limit=10, p=0.35, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=0.15, p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.15),
    )

    # stem->label from ORIGINAL train labels
    label_by_stem = {p.stem: p for p in orig_train_labels_dir.glob("*.txt")}

    aug_id = 0
    for c in sorted(class_to_images.keys()):
        if not class_to_images[c]:
            continue

        while out_counts[c] < target:
            src_img = random.choice(class_to_images[c])
            src_lab = label_by_stem.get(src_img.stem)
            if not src_lab:
                continue

            image = cv2.imread(str(src_img))
            if image is None:
                continue

            boxes = read_yolo_label(src_lab)
            bboxes, class_labels = yolo_to_albu(boxes)
            if not bboxes:
                continue

            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception:
                continue

            if len(augmented["bboxes"]) == 0:
                continue

            new_name = f"{src_img.stem}_aug{aug_id:06d}"
            out_img_path = out_train_images / f"{new_name}.jpg"
            out_lab_path = out_train_labels / f"{new_name}.txt"

            cv2.imwrite(str(out_img_path), augmented["image"])
            new_boxes = albu_to_yolo(augmented["bboxes"], augmented["class_labels"])
            if not write_yolo_label(out_lab_path, new_boxes):
                continue

            for cc in set([b[0] for b in new_boxes]):
                out_counts[cc] += 1

            aug_id += 1

    return out_counts


# =============================
# PART 7: VAL Balancing (Real-only Capping / Downsampling)
# =============================
def balance_val_by_capping(
    val_pairs: List[Tuple[Path, Path]],
    target_val_per_class: int,
    seed=42
):
    random.seed(seed)

    items = []
    for img, lab in val_pairs:
        cls_set = get_classes_from_label(lab)
        if not cls_set:
            continue
        items.append((img, lab, cls_set))

    random.shuffle(items)

    selected = []
    counts = defaultdict(int)

    def can_take(cls_set):
        return any(counts[c] < target_val_per_class for c in cls_set)

    for img, lab, cls_set in items:
        if can_take(cls_set):
            selected.append((img, lab))
            for c in cls_set:
                counts[c] += 1

    return selected, counts


# =============================
# MAIN
# =============================
def main():
    dataset_root = Path(r"merged_dataset")
    out_root = Path(r"balanced_merged_dataset")

    # Tuning
    SEED = 42
    TRAIN_TARGET_MODE = "median_x2"
    TRAIN_TARGET_CAP = 1200

    # VAL: "cap_to_min" makes val as equal as possible (may shrink val)
    # Or set integer like 100 to cap big classes
    VAL_TARGET_MODE = 20  # or 100

    # ---- Stage 1: Path check ----
    try:
        tr_img = dataset_root / "images" / "train"
        tr_lab = dataset_root / "labels" / "train"
        va_img = dataset_root / "images" / "val"
        va_lab = dataset_root / "labels" / "val"

        if not tr_img.exists() or not tr_lab.exists():
            raise FileNotFoundError("Missing merged_dataset/images/train or labels/train")
        if not va_img.exists() or not va_lab.exists():
            raise FileNotFoundError("Missing merged_dataset/images/val or labels/val")

        print("[INFO] dataset_root:", dataset_root.resolve())
        print("[INFO] out_root:", out_root.resolve())

    except Exception as e:
        print("[FATAL] Stage 1 failed (paths).")
        print("Reason:", repr(e))
        return

    # ---- Load YAML class names ----
    class_names = load_class_names(dataset_root)
    print("[INFO] Loaded class names count:", len(class_names))
    if class_names:
        # show first few mappings
        print("[INFO] Sample class mapping:", dict(list(class_names.items())[:5]))
    else:
        print("[WARN] No class names loaded. Output will fallback to class_{id}.")

    # ---- Stage 2: Scan input splits ----
    tr_obj, tr_imgc, tr_class_to_imgs, tr_pairs = scan_split(tr_img, tr_lab)
    va_obj, va_imgc, _, va_pairs = scan_split(va_img, va_lab)

    print_summary("TRAIN (INPUT) SUMMARY", tr_obj, tr_imgc, class_names)
    print_summary("VAL (INPUT) SUMMARY", va_obj, va_imgc, class_names)

    if len(tr_pairs) == 0 or len(va_pairs) == 0:
        print("[FATAL] No (image,label) pairs found in train or val.")
        return

    # ---- Stage 3: Prepare output folders ----
    out_train_images = out_root / "images" / "train"
    out_train_labels = out_root / "labels" / "train"
    out_val_images = out_root / "images" / "val"
    out_val_labels = out_root / "labels" / "val"

    if out_root.exists():
        print("[WARN] Output folder exists. Recommended to delete first to avoid mixing:")
        print("       rmdir /s /q balanced_merged_dataset")

    # copy yaml to output
    copy_data_yaml(dataset_root, out_root)

    # ---- Stage 4: Balance VAL (real-only) ----
    if VAL_TARGET_MODE == "cap_to_min":
        if not va_imgc:
            print("[FATAL] VAL img_count empty.")
            return
        target_val = min(va_imgc.values())
    elif isinstance(VAL_TARGET_MODE, int):
        target_val = VAL_TARGET_MODE
    else:
        print("[FATAL] VAL_TARGET_MODE must be 'cap_to_min' or int.")
        return

    print(f"\n[VAL] target_val_per_class = {target_val} (real-only, no augmentation)")
    selected_val_pairs, _ = balance_val_by_capping(va_pairs, target_val_per_class=target_val, seed=SEED)

    print(f"[VAL] Selected {len(selected_val_pairs)} / {len(va_pairs)} val images after balancing.")
    copied_val = copy_pairs(selected_val_pairs, out_val_images, out_val_labels)
    print(f"[OK] Copied VAL: {copied_val}")

    # ---- Stage 5: Copy TRAIN originals ----
    copied_train = copy_pairs(tr_pairs, out_train_images, out_train_labels)
    print(f"\n[OK] Copied TRAIN originals: {copied_train}")

    # ---- Stage 6: Balance TRAIN by augmentation ----
    train_target = choose_train_target(tr_imgc, mode=TRAIN_TARGET_MODE, cap=TRAIN_TARGET_CAP)
    print(f"\n[TRAIN] target_images_per_class = {train_target} (augmentation oversampling)")

    _ = balance_train_by_augmentation(
        out_train_images=out_train_images,
        out_train_labels=out_train_labels,
        orig_train_labels_dir=tr_lab,
        class_to_images=tr_class_to_imgs,
        target=train_target,
        seed=SEED,
    )

    # ---- Stage 7: Final scan output summaries ----
    o_tr_obj, o_tr_imgc, _, _ = scan_split(out_train_images, out_train_labels)
    o_va_obj, o_va_imgc, _, _ = scan_split(out_val_images, out_val_labels)

    print_summary("TRAIN (OUTPUT) SUMMARY", o_tr_obj, o_tr_imgc, class_names)
    print_summary("VAL (OUTPUT) SUMMARY", o_va_obj, o_va_imgc, class_names)

    print("\n[DONE] Balanced dataset created at:", out_root.resolve())
    print("Output structure:")
    print(" - balanced_merged_dataset/data.yaml")
    print(" - balanced_merged_dataset/images/train")
    print(" - balanced_merged_dataset/labels/train")
    print(" - balanced_merged_dataset/images/val")
    print(" - balanced_merged_dataset/labels/val")
    print("\nNow run your count.py on balanced_merged_dataset to verify.")


if __name__ == "__main__":
    main()
