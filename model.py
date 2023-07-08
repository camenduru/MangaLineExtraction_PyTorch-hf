from __future__ import annotations

import pathlib
import sys

import cv2
import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'MangaLineExtraction_PyTorch'
sys.path.insert(0, submodule_dir.as_posix())

from model_torch import res_skip

MAX_SIZE = 1000


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        ckpt_path = huggingface_hub.hf_hub_download(
            'public-data/MangaLineExtraction_PyTorch', 'erika.pth')
        state_dict = torch.load(ckpt_path)
        model = res_skip()
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if max(gray.shape) > MAX_SIZE:
            scale = MAX_SIZE / max(gray.shape)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        h, w = gray.shape
        size = 16
        new_w = (w + size - 1) // size * size
        new_h = (h + size - 1) // size * size

        patch = np.ones((1, 1, new_h, new_w), dtype=np.float32)
        patch[0, 0, :h, :w] = gray
        tensor = torch.from_numpy(patch).to(self.device)
        out = self.model(tensor)

        res = out.cpu().numpy()[0, 0, :h, :w]
        res = np.clip(res, 0, 255).astype(np.uint8)
        return res
