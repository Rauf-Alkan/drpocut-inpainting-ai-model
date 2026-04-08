import os
import tempfile
from cog import BasePredictor, Input, Path as CogPath
from propainter.inference import Propainter, get_device


class Predictor(BasePredictor):
    def setup(self):
        self.device = get_device()
        self.propainter = Propainter(
            propainter_model_dir="weights/propainter",
            device=self.device
        )

    def predict(
        self,
        video: CogPath = Input(description="Maskelenecek video (.mp4)"),
        mask: CogPath = Input(description="Maske görüntüsü veya klasörü (.png veya .mp4)"),
        width: int = Input(description="Çıktı genişliği", default=640),
        height: int = Input(description="Çıktı yüksekliği", default=360),
        fp16: bool = Input(description="Yarı hassasiyet (daha hızlı, daha az VRAM)", default=True),
        neighbor_length: int = Input(description="Lokal komşu uzunluğu (azalt = daha az VRAM)", default=10),
        subvideo_length: int = Input(description="Alt video uzunluğu", default=80),
        video_length: int = Input(description="Maksimum video uzunluğu (saniye)", default=10),
    ) -> CogPath:

        output_path = tempfile.mktemp(suffix=".mp4")

        self.propainter.forward(
            video=str(video),
            mask=str(mask),
            output_path=output_path,
            video_length=video_length,
            width=width,
            height=height,
            fp16=fp16,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
        )

        return CogPath(output_path)