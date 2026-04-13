# train_yolov8.py (or train_lostandfound.py)
# ✅ Robust + Windows-safe + YAML validation (TRY–EXCEPT protected)

from ultralytics import YOLO
import torch
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_lostandfound")


class DeviceManager:
    @staticmethod
    def get_device():
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ CUDA GPU detected: {gpu_name}")
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("✅ Apple MPS detected – using Apple GPU")
                return "mps"
        except Exception as e:
            logger.warning(f"⚠️ Device detection failed: {e}")

        logger.info("⚠️ No GPU detected – using CPU (slow)")
        return "cpu"


def validate_dataset(data_yaml: str) -> bool:
    """Validate data.yaml and resolved train/val paths."""
    try:
        # --- Load YAML safely (no ultralytics internal dependency) ---
        try:
            import yaml
        except Exception as e:
            raise RuntimeError(
                "PyYAML is not installed. Run: pip install pyyaml\n"
                f"Original import error: {e}"
            )

        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError("data.yaml is not a valid YAML mapping (dict).")

        root = cfg.get("path", "")
        train = cfg.get("train")
        val = cfg.get("val")

        if not train or not val:
            raise ValueError("data.yaml must contain 'train' and 'val' (pointing to IMAGE folders).")

        # Resolve train/val paths
        train_path = (Path(root) / train) if root else Path(train)
        val_path = (Path(root) / val) if root else Path(val)

        logger.info(f"📁 Dataset YAML: {data_yaml}")
        logger.info(f"🔍 RESOLVED train images path: {train_path}")
        logger.info(f"🔍 RESOLVED val images path:   {val_path}")

        if not train_path.exists():
            raise FileNotFoundError(f"Train images folder not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Val images folder not found: {val_path}")

        # Optional label folder check (warning only)
        guessed_root = Path(root) if root else Path(data_yaml).parent
        labels_train = guessed_root / "labels" / "train"
        labels_val = guessed_root / "labels" / "val"
        if not labels_train.exists() or not labels_val.exists():
            logger.warning(
                "⚠️ Labels folders not found at expected locations:\n"
                f"   {labels_train}\n"
                f"   {labels_val}\n"
                "   (If your labels are stored differently but training works, ignore.)"
            )

        logger.info("✅ Dataset validation passed")
        return True

    except Exception as e:
        logger.error("❌ Dataset validation failed")
        logger.error(e)
        return False


def train_lost_and_found(
    data_yaml: str,
    base_weights: str = "yolov8m.pt",
    epochs: int = 120,
    imgsz: int = 640,
    batch: int = 12,
    project: str = "runs",
    name: str = "lostandfound_global_v2",
):
    try:
        device = DeviceManager.get_device()

        if not validate_dataset(data_yaml):
            raise RuntimeError("Dataset check failed. Fix dataset before training.")

        logger.info(f"🧠 Base weights: {base_weights}")
        logger.info(f"🔢 Epochs={epochs}, imgsz={imgsz}, batch={batch}")
        logger.info(f"💾 Project={project}, Name={name}")
        logger.info(f"💻 Device={device}")

        # Load model
        try:
            model = YOLO(base_weights)
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO weights: {e}")

        # Train
        try:
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=project,
                name=name,
                exist_ok=True,
                workers=2,          # Windows-safe
                pretrained=True,

                # Accuracy-focused
                patience=25,
                close_mosaic=10,
                cos_lr=True,
                seed=42,
            )
        except Exception as e:
            raise RuntimeError(f"Training crashed: {e}")

        # Best model path
        try:
            best_path = str(model.trainer.best)
        except Exception:
            best_path = os.path.join(project, "detect", name, "weights", "best.pt")

        logger.info("🎉 Training completed successfully")
        logger.info(f"🏆 Best model saved at:\n    {best_path}")
        return best_path, results

    except Exception as e:
        logger.error("❌ TRAINING FAILED")
        logger.error(e)
        return None, None


if __name__ == "__main__":
    DATA_YAML = r"C:\Users\admin\Documents\LostAnd\data.yaml"

    # RTX 4050 suggestion:
    # - Start with yolov8m (stable)
    # - If you want higher accuracy later, try yolov8l with batch=8~12
    best_model, _ = train_lost_and_found(
        data_yaml=DATA_YAML,
        base_weights="yolov8m.pt",
        epochs=60,
        imgsz=512,
        batch=8,
        project="runs",
        name="lostandfound_global_v2",
    )

    if best_model:
        logger.info(f"✅ FINAL MODEL READY: {best_model}")
    else:
        logger.warning("⚠️ Training did not complete successfully")
