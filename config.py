"""
Конфигурация и настройки проекта (только детекция дороги + объектов).
"""
# ============ ОСНОВНЫЕ НАСТРОЙКИ ============
PROCESS_NAME   = "ViceCity.exe"
PATH_TO_MODEL  = "vice_city_frozen_inference_graph.pb"
PATH_TO_LABELS = "vice_city_labelmap.pbtxt"
MIN_SCORE      = 0.5
INFER_W        = 800
INFER_H        = 580
DEBUG_MODE     = True

# ============ ВИРТУАЛЬНЫЕ КОДЫ КЛАВИШ ============
# O — вкл/выкл детекцию объектов (TensorFlow)
KEY_OBJECT_DETECTION = ord('O')
# P — вкл/выкл детекцию дороги (SegFormer)
KEY_ROAD_DETECTION   = ord('P')

# ============ НЕЙРОСЕТЕВАЯ ДЕТЕКЦИЯ ДОРОГИ ============
USE_NN_ROAD_DETECTOR = True
NN_MODEL_NAME     = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
NN_ROAD_CLASS_ID  = 0     # в Cityscapes класс 0 — road
NN_MIN_AREA_RATIO = 0.02

# ============ РАНТАЙМ-ТУМБЛЕРЫ (стартовое состояние) ============
ENABLE_ROAD_DETECTION   = False
ENABLE_OBJECT_DETECTION = False

# ============ ОПТИМИЗАЦИИ FPS ============
USE_HALF_PRECISION = True
NN_INFER_EVERY_N = 3

# ============ ФИЛЬТР ЗЕЛЁНОГО (ТРАВА) ============
ROAD_FILTER_GREEN = True
GREEN_H_LOWER = 40
GREEN_H_UPPER = 80
GREEN_S_LOWER = 100
GREEN_V_LOWER = 60
GREEN_DILATE_KERNEL = 3

# ============ ДИАГНОСТИКА МАСКИ ============
DEBUG_MASK_STATS_EVERY = 10

# ============ NMS (подавление наложений боксов) ============
NMS_IOU_THRESHOLD = 0.45
NMS_CLASS_AGNOSTIC = True

# ============ НЕИЗВЕСТНЫЕ КЛАССЫ ============
SHOW_UNKNOWN_CLASSES = False