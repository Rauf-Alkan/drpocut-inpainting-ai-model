import os
import tempfile
import cv2
from cog import BasePredictor, Input, Path as CogPath
from propainter.inference import Propainter, get_device


class Predictor(BasePredictor):
    def setup(self):
        """Model bir kez yüklenir, sonra bellekte kalır."""
        self.device = get_device()
        self.propainter = Propainter(
            propainter_model_dir="weights/propainter",
            device=self.device
        )
        print(f"Model yüklendi. Device: {self.device}")

    def predict(
        self,
        video: CogPath = Input(description="İnpainting yapılacak video (.mp4)"),
        mask: CogPath = Input(description="Maske — tek png veya video (.png / .mp4)"),
        fp16: bool = Input(description="Yarı hassasiyet — daha hızlı, daha az VRAM", default=True),
        neighbor_length: int = Input(description="Lokal komşu sayısı (azalt = daha az VRAM)", default=10, ge=5, le=20),
        subvideo_length: int = Input(description="Alt video uzunluğu (azalt = daha az VRAM)", default=80, ge=20, le=150),
        max_seconds: int = Input(description="Maksimum işlenecek süre (saniye)", default=10, ge=1, le=60),
    ) -> CogPath:

        # Videodan otomatik boyut al
        cap = cv2.VideoCapture(str(video))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 8'in katı olmalı (ProPainter şartı)
        width = (orig_w // 8) * 8
        height = (orig_h // 8) * 8

        # Süre kontrolü
        duration = frame_count / fps if fps > 0 else 0
        if duration > 60:
            raise ValueError(f"Video çok uzun ({duration:.1f}s). Maksimum 60 saniye.")
        if frame_count < 5:
            raise ValueError("Video çok kısa, en az 5 frame gerekli.")

        print(f"Video: {orig_w}x{orig_h} → işlenecek: {width}x{height}, {duration:.1f}s, {frame_count} frame")

        output_path = tempfile.mktemp(suffix=".mp4")

        self.propainter.forward(
            video=str(video),
            mask=str(mask),
            output_path=output_path,
            video_length=max_seconds,
            width=width,
            height=height,
            fp16=fp16,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("ProPainter çıktı dosyası oluşturulamadı.")

        return CogPath(output_path)