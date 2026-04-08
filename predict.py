import os
import torch
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        print("DiffuEraser yükleniyor...")
        from diffueraser.diffueraser import DiffuEraser
        from propainter.inference import get_device

        self.device = get_device()
        self.model = DiffuEraser(
            self.device,
            "/src/weights/stable-diffusion-v1-5",
            "/src/weights/sd-vae-ft-mse",
            "/src/weights/diffuEraser",
            ckpt="2-Step"
        )
        print("Hazır!")

    def predict(
        self,
        video: Path = Input(description="Orijinal video (mp4)"),
        mask: Path = Input(description="Mask video - beyaz=silinecek alan (mp4)"),
        video_length: int = Input(description="Max video uzunluğu (saniye)", default=60),
        max_img_size: int = Input(description="Max genişlik/yükseklik", default=960),
        mask_dilation_iter: int = Input(description="Mask genişleme miktarı", default=8),
    ) -> Path:

        os.makedirs("/tmp/results", exist_ok=True)
        output_path = "/tmp/results/output.mp4"
        priori_path = "/tmp/results/priori.mp4"

        from propainter.inference import Propainter
        propainter = Propainter("/src/weights/propainter", device=self.device)

        propainter.forward(
            str(video),
            str(mask),
            priori_path,
            video_length=video_length,
            ref_stride=10,
            neighbor_length=10,
            subvideo_length=50,
            mask_dilation=mask_dilation_iter
        )

        self.model.forward(
            str(video),
            str(mask),
            priori_path,
            output_path,
            max_img_size=max_img_size,
            video_length=video_length,
            mask_dilation_iter=mask_dilation_iter,
            guidance_scale=None
        )

        return Path(output_path)