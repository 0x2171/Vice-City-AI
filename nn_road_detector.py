"""
Детекция дороги нейросетью SegFormer.

Алгоритм:
1. SegFormer находит полотно дороги (класс 'road' в Cityscapes).
2. Опционально вычитаем зелёные пиксели (трава, кусты), если сеть
   ошибочно захватила их.
3. Возвращаем маску дороги — она подсвечивается зелёным в main.py.

Никакой разметки (жёлтой/белой), никаких траекторий — только общий вид дороги.
"""
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from config import (
    NN_MODEL_NAME, NN_ROAD_CLASS_ID, USE_HALF_PRECISION, NN_INFER_EVERY_N,
    ROAD_FILTER_GREEN, GREEN_H_LOWER, GREEN_H_UPPER,
    GREEN_S_LOWER, GREEN_V_LOWER, GREEN_DILATE_KERNEL,
    DEBUG_MODE, DEBUG_MASK_STATS_EVERY,
)


class NNRoadDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"[NN ROAD] CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            print("[NN ROAD] CUDA недоступна, будет использоваться CPU.")

        self._loaded = False
        self._load_failed = False
        self.processor = None
        self.model = None
        self.use_half = False

        try:
            self.infer_every_n = max(1, int(NN_INFER_EVERY_N))
        except Exception:
            self.infer_every_n = 1

        self._frame_counter = 0
        self._cached_road_mask = None
        self._last_h = 0
        self._last_w = 0
        self._diag_counter = 0

    def load(self):
        """Явная загрузка модели в память."""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        print(f"[NN ROAD] Загрузка SegFormer: {NN_MODEL_NAME} ...")
        try:
            self.processor = SegformerImageProcessor.from_pretrained(NN_MODEL_NAME)
            self.model = SegformerForSemanticSegmentation.from_pretrained(NN_MODEL_NAME)
            self.model.eval()
            self.model.to(self.device)

            self.use_half = USE_HALF_PRECISION and self.device == "cuda"
            if self.use_half:
                self.model = self.model.half()
                print("[NN ROAD] Включён режим FP16")

            self._loaded = True
            print("[NN ROAD] Модель SegFormer успешно загружена в память.")

            if self.device == "cuda":
                dummy = np.zeros((256, 512, 3), dtype=np.uint8)
                try:
                    self._infer_road_mask(dummy)
                    print("[NN ROAD] Прогрев GPU завершён.")
                except Exception:
                    pass
            return True
        except Exception as e:
            self._load_failed = True
            print(f"[NN ROAD] Ошибка загрузки модели: {e}")
            return False

    @property
    def loaded(self):
        return self._loaded

    def detect(self, frame):
        """
        Возвращает (frame, road_mask, road_detected).

        frame           — исходный кадр (не изменяется).
        road_mask       — uint8-маска: 255 в пикселях дороги, 0 — фон.
        road_detected   — True, если маска непустая.
        """
        if not self._loaded:
            if not self.load():
                return frame, None, False

        h, w = frame.shape[:2]
        self._frame_counter += 1

        need_infer = (
            self._cached_road_mask is None
            or self._frame_counter % self.infer_every_n == 0
            or self._last_h != h or self._last_w != w
        )

        if need_infer:
            road_mask = self._infer_road_mask(frame)
            self._cached_road_mask = road_mask
            self._last_h, self._last_w = h, w
        else:
            road_mask = self._cached_road_mask

        if road_mask is None or road_mask.shape[:2] != (h, w):
            return frame, None, False

        self._maybe_print_mask_stats(road_mask, h)

        if ROAD_FILTER_GREEN:
            road_mask = self._filter_green(road_mask, frame)
            if road_mask is None or not road_mask.any():
                return frame, None, False

        return frame, road_mask, True

    # ==================================================================
    # ИНФЕРЕНС SEGFORMER
    # ==================================================================
    def _infer_road_mask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            outputs = self.model(**inputs)
            seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[frame.shape[:2]]
            )[0].cpu().numpy()

        road = (seg == NN_ROAD_CLASS_ID).astype(np.uint8) * 255
        return road if road.any() else None

    def _filter_green(self, mask, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            (GREEN_H_LOWER, GREEN_S_LOWER, GREEN_V_LOWER),
            (GREEN_H_UPPER, 255, 255)
        )
        k = max(1, int(GREEN_DILATE_KERNEL))
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            green = cv2.dilate(green, kernel, iterations=1)
        mask = mask.copy()
        mask[green > 0] = 0
        return mask

    # ==================================================================
    # ДИАГНОСТИКА
    # ==================================================================
    def _maybe_print_mask_stats(self, mask, h):
        if not DEBUG_MODE or DEBUG_MASK_STATS_EVERY <= 0:
            return
        self._diag_counter += 1
        if self._diag_counter % DEBUG_MASK_STATS_EVERY != 0:
            return
        third = h // 3
        top = int((mask[:third] > 0).sum())
        mid = int((mask[third:2 * third] > 0).sum())
        bot = int((mask[2 * third:] > 0).sum())
        total = top + mid + bot
        if total == 0:
            print("[MASK] маска пустая (сеть не видит дорогу!)")
        else:
            print(f"[MASK] пикселей дороги: верх={top}, центр={mid}, низ={bot}")