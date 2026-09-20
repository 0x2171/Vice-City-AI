"""
Детекция объектов через TensorFlow frozen graph.
"""
import cv2
import numpy as np
from config import (
    PATH_TO_MODEL, PATH_TO_LABELS, MIN_SCORE, INFER_W, INFER_H,
    NMS_IOU_THRESHOLD, NMS_CLASS_AGNOSTIC, SHOW_UNKNOWN_CLASSES, DEBUG_MODE,
)

class ObjectDetector:
    def __init__(self):
        self._loaded = False
        self._load_failed = False
        self._sess = None
        self._category_index = None
        self._tensors = None

    def load(self):
        """Явная загрузка модели в память."""
        if self._loaded: return True
        if self._load_failed: return False
        
        print(f"[OBJECT DETECTOR] Загрузка TensorFlow-модели: {PATH_TO_MODEL} ...")
        try:
            from model_loader import load_model
            self._sess, self._category_index, self._tensors = load_model()
            self._loaded = True
            known_classes = [v['name'] for v in self._category_index.values()]
            print(f"[OBJECT DETECTOR] Модель TF успешно загружена.")
            print(f"[OBJECT DETECTOR] Известные классы ({len(known_classes)}): {known_classes}")
            return True
        except Exception as e:
            self._load_failed = True
            print(f"[OBJECT DETECTOR] Ошибка загрузки модели: {e}")
            return False

    @property
    def loaded(self):
        return self._loaded

    def detect(self, frame):
        if not self._loaded:
            if not self.load():
                return frame, []

        image_tensor, boxes_t, scores_t, classes_t, num_t = self._tensors
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (INFER_W, INFER_H))
        inp = np.expand_dims(inp, axis=0)

        try:
            boxes, scores, classes, num = self._sess.run(
                [boxes_t, scores_t, classes_t, num_t],
                feed_dict={image_tensor: inp},
            )
        except Exception as e:
            print(f"[OBJECT DETECTOR] Ошибка инференса: {e}")
            return frame, []

        h, w = frame.shape[:2]
        detections = []
        skipped_unknown = 0

        for i in range(int(num[0])):
            score = float(scores[0][i])
            if score < MIN_SCORE: continue
            cls = int(classes[0][i])
            cat = self._category_index.get(cls, None)
            
            if cat is None:
                skipped_unknown += 1
                if not SHOW_UNKNOWN_CLASSES: continue
                label = f"unknown_{cls}"
            else:
                label = cat.get('name', f"class_{cls}")

            y1, x1, y2, x2 = boxes[0][i]
            x1 = max(0, min(w - 1, int(x1 * w)))
            y1 = max(0, min(h - 1, int(y1 * h)))
            x2 = max(0, min(w - 1, int(x2 * w)))
            y2 = max(0, min(h - 1, int(y2 * h)))

            if x2 <= x1 or y2 <= y1: continue

            detections.append({
                'box': (x1, y1, x2, y2),
                'label': label,
                'cls': cls,
                'score': score,
            })

        before_nms = len(detections)
        detections = self._apply_nms(detections, NMS_IOU_THRESHOLD, NMS_CLASS_AGNOSTIC)
        after_nms = len(detections)

        if DEBUG_MODE and (before_nms > 0 or skipped_unknown > 0):
            suppressed = before_nms - after_nms
            print(f"[OBJECT DETECTOR] Детекций: {before_nms} -> {after_nms} "
                  f"(NMS подавил: {suppressed}, неизвестных: {skipped_unknown})")

        return frame, detections

    @staticmethod
    def _compute_iou(box1, box2):
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        if union <= 0: return 0.0
        return inter / union

    @classmethod
    def _apply_nms(cls, detections, iou_threshold, class_agnostic):
        if not detections: return []
        detections = sorted(detections, key=lambda d: d['score'], reverse=True)
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            remaining = []
            for d in detections:
                if not class_agnostic and d['cls'] != best['cls']:
                    remaining.append(d)
                    continue
                iou = cls._compute_iou(best['box'], d['box'])
                if iou < iou_threshold:
                    remaining.append(d)
            detections = remaining
        return kept

    @staticmethod
    def draw(frame, detections):
        color_map = {
            'car': (255, 0, 0), 'person': (0, 255, 255),
            'bike': (255, 128, 0), 'item': (0, 255, 0),
        }
        for d in detections:
            x1, y1, x2, y2 = d['box']
            color = color_map.get(d['label'], (255, 0, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{d['label']} {d['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame