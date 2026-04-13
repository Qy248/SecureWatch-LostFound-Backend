from pathlib import Path
import shutil
import random
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(r"C:/Users/admin/Documents/LostAnd")
EXCEL_PATH = BASE_DIR / "Dataset Detail.xlsx"

OUT_ROOT = BASE_DIR / "merged_dataset"
IMG_TRAIN_OUT = OUT_ROOT / "images" / "train"
LBL_TRAIN_OUT = OUT_ROOT / "labels" / "train"
IMG_VAL_OUT   = OUT_ROOT / "images" / "val"
LBL_VAL_OUT   = OUT_ROOT / "labels" / "val"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

TARGET_NAMES = {
    0: "backpack",
    1: "handbag",
    2: "watch",
    3: "laptop",
    4: "tablet",
    5: "mobile_phone",
    6: "earphones",
    7: "powerbank",
    8: "water_bottle",
    9: "umbrella",
    10: "person",
    11: "usb_drive",
    12: "wallet",
    13: "card",
    14: "key",
    15: "headphone",
    16: "charger_adapter",
    17: "spectacles",
}

# split candidates (your folders may have train/valid/test only)
SPLIT_CANDIDATES = ["train", "valid", "val", "test"]

VAL_RATIO = 0.2
SEED = 42

# =========================================================
# HELPERS
# =========================================================

def ensure_dirs():
    for p in [IMG_TRAIN_OUT, LBL_TRAIN_OUT, IMG_VAL_OUT, LBL_VAL_OUT]:
        p.mkdir(parents=True, exist_ok=True)

def normalize_convert_value(v):
    """Return int label id or None if ignore/blank."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "" or s == "nan":
        return None
    if "ignore" in s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def build_maps_from_excel(excel_path: Path):
    """
    Build mapping dicts for dataset1..dataset4 from Dataset Detail.xlsx:
    - local_id is inferred by row order within each dataset block (0..n-1)
    - Convert to Label provides the NEW class id (0..17) or 'ignore'
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    df = df[df["Item List"].notna()].copy()
    df["Item List"] = df["Item List"].astype(str).str.strip()
    df = df[df["Item List"] != ""].copy()

    # Fill dataset number down
    df["Dataset"] = df["Dataset"].ffill()

    maps = {}  # {1: {local_id:new_id}, 2: {...}, ...}
    for d in [1, 2, 3, 4]:
        sub = df[df["Dataset"] == float(d)].copy()

        # remove summary rows like "Total"
        sub = sub[~sub["Item List"].str.lower().eq("total")].copy()

        id_map = {}
        local_id = 0
        for _, row in sub.iterrows():
            new_id = normalize_convert_value(row.get("Convert to Label"))
            if new_id is not None:
                # keep only 0..17
                if 0 <= new_id <= 17:
                    id_map[local_id] = new_id
            local_id += 1

        maps[d] = id_map

    return maps

def convert_one_label(src_lbl: Path, dst_lbl: Path, id_map: dict) -> bool:
    """
    Convert YOLO label file by remapping class ids using id_map.
    If no valid labels remain after ignoring => return False.
    """
    if not src_lbl.exists():
        return False

    out_lines = []
    with open(src_lbl, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(float(parts[0]))
            if old_id not in id_map:
                continue  # ignored item
            new_id = id_map[old_id]
            parts[0] = str(new_id)
            out_lines.append(" ".join(parts))

    if not out_lines:
        return False

    with open(dst_lbl, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    return True

def merge_split(ds_root: Path, split: str, id_map: dict, prefix: str) -> int:
    """
    Merge one split folder:
    ds_root/split/images + ds_root/split/labels
    """
    img_dir = ds_root / split / "images"
    lbl_dir = ds_root / split / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        return 0

    count = 0
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        stem = img_path.stem
        src_lbl = lbl_dir / f"{stem}.txt"

        dst_img = IMG_TRAIN_OUT / f"{prefix}_{split}_{stem}{img_path.suffix.lower()}"
        dst_lbl = LBL_TRAIN_OUT / f"{prefix}_{split}_{stem}.txt"

        if convert_one_label(src_lbl, dst_lbl, id_map):
            shutil.copy2(img_path, dst_img)
            count += 1
        else:
            # empty after ignoring => do not copy image
            if dst_lbl.exists():
                dst_lbl.unlink(missing_ok=True)

    return count

def merge_dataset(ds_index: int, ds_root: Path, id_map: dict):
    """
    Merge datasetN with all splits that exist.
    """
    total = 0
    for split in SPLIT_CANDIDATES:
        merged = merge_split(ds_root, split, id_map, prefix=f"ds{ds_index}")
        if merged > 0:
            print(f"[OK] dataset{ds_index} {split}: merged {merged}")
        total += merged
    print(f"[DONE] dataset{ds_index}: total merged {total}")
    return total

def create_val_split(val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    images = [p for p in IMG_TRAIN_OUT.iterdir() if p.suffix.lower() in IMG_EXTS]
    total = len(images)
    if total == 0:
        print("[SPLIT] No images in train to split.")
        return

    n_val = max(1, int(total * val_ratio))
    random.shuffle(images)
    val_imgs = images[:n_val]

    moved = 0
    for img_path in val_imgs:
        stem = img_path.stem
        lbl_path = LBL_TRAIN_OUT / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        shutil.move(str(img_path), str(IMG_VAL_OUT / img_path.name))
        shutil.move(str(lbl_path), str(LBL_VAL_OUT / lbl_path.name))
        moved += 1

    print(f"[SPLIT] Moved {moved}/{n_val} to val (ratio={val_ratio}).")

def write_data_yaml():
    yaml_path = OUT_ROOT / "data.yaml"
    lines = [
        f"path: {OUT_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "nc: 18",
        "",
        "names:",
    ]
    for k in range(18):
        lines.append(f"  {k}: {TARGET_NAMES[k]}")
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print("[YAML] Wrote:", yaml_path)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    ensure_dirs()

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Dataset Detail.xlsx not found at: {EXCEL_PATH}")

    maps = build_maps_from_excel(EXCEL_PATH)

    # dataset folders (must match your folder names)
    roots = {
        1: BASE_DIR / "dataset1",
        2: BASE_DIR / "dataset2",
        3: BASE_DIR / "dataset3",
        4: BASE_DIR / "dataset4",
    }

    print("[STEP] Merging 4 datasets following Dataset Detail.xlsx mappings...")
    for d in [1, 2, 3, 4]:
        if not roots[d].exists():
            print(f"[WARN] dataset{d} folder not found: {roots[d]}")
            continue
        merge_dataset(d, roots[d], maps[d])

    print("[STEP] Creating train/val split...")
    create_val_split(val_ratio=VAL_RATIO, seed=SEED)

    print("[STEP] Writing data.yaml...")
    write_data_yaml()

    print("[DONE] Output:", OUT_ROOT)
