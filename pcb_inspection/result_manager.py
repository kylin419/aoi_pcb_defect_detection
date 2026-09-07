from pathlib import Path
from datetime import datetime
import cv2
import json

from .config import RESULTS_DIR


class ResultManager:
    """Manages saving inspection images, overlays, and JSON metadata."""

    def __init__(self, output_dir: str = RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_folder(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.output_dir / timestamp
        counter = 1
        orig_folder = folder
        while folder.exists():
            folder = Path(f"{orig_folder}_{counter}")
            counter += 1

        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_image(self, folder: Path, frame):
        if frame is not None:
            path = folder / "image.jpg"
            cv2.imwrite(str(path), frame)

    def save_overlay(self, folder: Path, frame):
        if frame is not None:
            path = folder / "overlay.jpg"
            cv2.imwrite(str(path), frame)

    def save_result(self, folder: Path, result: dict):
        if result is not None:
            path = folder / "result.json"
            # Ensure serializable
            clean_result = {}
            for k, v in result.items():
                if k in ("detections", "stats", "inference_ms", "ok"):
                    clean_result[k] = v
            clean_result["saved_at"] = datetime.now().isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(clean_result, f, indent=4)

    def save_all(self, frame, overlay, result: dict) -> str:
        """Helper to create directory and save all result artifacts at once."""
        folder = self.create_folder()
        self.save_image(folder, frame)
        self.save_overlay(folder, overlay)
        self.save_result(folder, result)
        return str(folder)
   


     