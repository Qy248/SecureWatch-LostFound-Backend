import os
import yaml
from collections import defaultdict

# ========= CHANGE THIS =========
DATASET_ROOT = "merged_dataset"
# ===============================

# merged_dataset only has train + val
SPLITS = ["train", "val"]


def load_class_names(dataset_root: str):
    """Load class names from data.yaml (YOLO format)."""
    yaml_path = os.path.join(dataset_root, r"C:\Users\admin\Documents\LostAnd\data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[WARN] data.yaml not found at: {yaml_path}")
        return {}

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {}


def summarize_split(labels_dir: str):
    """
    Returns:
      object_count[class_id] -> total number of boxes
      image_set[class_id] -> set of image base names that contain class
      total_label_files -> total label files scanned
    """
    object_count = defaultdict(int)
    image_set = defaultdict(set)
    total_label_files = 0

    if not os.path.isdir(labels_dir):
        return object_count, image_set, total_label_files

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue

        total_label_files += 1
        label_path = os.path.join(labels_dir, fname)
        image_id = os.path.splitext(fname)[0]

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue

        used_classes_in_image = set()

        for ln in lines:
            parts = ln.split()
            if len(parts) < 1:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                continue

            object_count[cid] += 1
            used_classes_in_image.add(cid)

        for cid in used_classes_in_image:
            image_set[cid].add(image_id)

    return object_count, image_set, total_label_files


def merge_counts(dst_obj, dst_imgset, src_obj, src_imgset):
    for cid, cnt in src_obj.items():
        dst_obj[cid] += cnt
    for cid, s in src_imgset.items():
        dst_imgset[cid].update(s)


def print_table(title, class_names, obj_count, img_set):
    print(f"\n📌 {title}")
    print("=" * 70)
    print(f"{'Class':<22} {'Images':>10} {'Objects':>10}   (id)")
    print("-" * 70)

    all_cids = sorted(set(obj_count) | set(img_set))
    for cid in all_cids:
        name = class_names.get(cid, f"class_{cid}")
        images = len(img_set.get(cid, set()))
        objects = obj_count.get(cid, 0)
        print(f"{name:<22} {images:>10} {objects:>10}   ({cid})")

    print("=" * 70)


def main():
    class_names = load_class_names(DATASET_ROOT)

    total_obj = defaultdict(int)
    total_imgset = defaultdict(set)

    for split in SPLITS:
        labels_dir = os.path.join(DATASET_ROOT, "labels", split)

        obj_count, img_set, nfiles = summarize_split(labels_dir)

        print(f"\n🔍 Split: {split}")
        print(f"Labels folder: {labels_dir}")
        print(f"Label files scanned: {nfiles}")

        print_table(f"{split.upper()} SUMMARY", class_names, obj_count, img_set)

        merge_counts(total_obj, total_imgset, obj_count, img_set)

    print_table("OVERALL (TRAIN + VAL)", class_names, total_obj, total_imgset)


if __name__ == "__main__":
    main()
