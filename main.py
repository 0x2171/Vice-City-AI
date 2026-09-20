"""
Основной скрипт: детекция дороги (SegFormer) + детекция объектов (TF).
Управление тумблерами:
P  — детекция дороги вкл/выкл
O  — детекция объектов вкл/выкл
F8 — старт/стоп записи окна
q  — выход
"""
import os
import time
import datetime
import cv2
import numpy as np
import mss
import win32api
import win32gui
from config import (
    PROCESS_NAME, DEBUG_MODE,
    KEY_OBJECT_DETECTION, KEY_ROAD_DETECTION,
    ENABLE_ROAD_DETECTION, ENABLE_OBJECT_DETECTION,
)
from window_utils import find_window_by_process, get_client_rect, focus_window
from nn_road_detector import NNRoadDetector
from object_detector import ObjectDetector


# VK_F8 = 0x77
KEY_TOGGLE_RECORD = 0x77

# FPS, с которым пишется видео (фиксированный, чтобы не «дёргалось»)
RECORD_FPS = 30.0


def key_pressed(key_code):
    """True, если клавиша нажата прямо сейчас (Win32 GetAsyncKeyState)."""
    return win32api.GetAsyncKeyState(key_code) & 0x8000 != 0


def _draw_status_panel(frame, road_on, obj_on):
    rd_color = (0, 255, 0) if road_on else (128, 128, 128)
    rd_text = "ROAD DETECTION: ON  (P)" if road_on else "ROAD DETECTION: OFF (P)"
    cv2.putText(frame, rd_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rd_color, 2)

    od_color = (255, 0, 255) if obj_on else (128, 128, 128)
    od_text = "OBJECT DETECTION: ON  (O)" if obj_on else "OBJECT DETECTION: OFF (O)"
    cv2.putText(frame, od_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, od_color, 2)


def main():
    road_detector = NNRoadDetector()
    object_detector = ObjectDetector()

    # ПРЕДЗАГРУЗКА МОДЕЛЕЙ ПРИ СТАРТЕ (Eager Loading)
    print("=" * 50)
    print("Предзагрузка нейросетей в память...")
    print("Это может занять несколько секунд, но уберёт фризы в игре.")
    road_detector.load()
    object_detector.load()
    print("Модели успешно загружены. Они неактивны, пока не нажаты P или O.")
    print("=" * 50)

    road_detection_enabled = ENABLE_ROAD_DETECTION
    object_detection_enabled = ENABLE_OBJECT_DETECTION

    hwnd = None
    monitor = None
    prev_time = time.time()
    frame_count = 0
    prev_obj_key_state = False
    prev_road_key_state = False

    # === Состояние записи ===
    prev_record_key_state = False
    recording = False
    video_writer = None
    record_path = None
    record_size = None  # (w, h), под который открыт writer

    print(f"Запуск. Ищем процесс: {PROCESS_NAME}")
    print("Управление:")
    print("  P  — детекция дороги вкл/выкл")
    print("  O  — детекция объектов вкл/выкл")
    print("  F8 — старт/стоп записи окна")
    print("  q  — выход")

    with mss.MSS() as sct:
        while True:
            # Пересоздаём monitor, если его нет, или если окно потеряно/закрыто
            if (hwnd is None
                    or monitor is None
                    or not win32gui.IsWindow(hwnd)):
                hwnd = find_window_by_process()
                if hwnd is None:
                    hwnd = None
                    monitor = None
                    time.sleep(1)
                    continue
                focus_window(hwnd)
                left, top, width, height = get_client_rect(hwnd)
                if width <= 0 or height <= 0:
                    monitor = None
                    time.sleep(0.1)
                    continue
                monitor = {"top": top, "left": left, "width": width, "height": height}

            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_count += 1

            # === КЛАВИША P ===
            road_key_state = key_pressed(KEY_ROAD_DETECTION)
            if road_key_state and not prev_road_key_state:
                road_detection_enabled = not road_detection_enabled
                state_text = "ВКЛЮЧЕНА" if road_detection_enabled else "ВЫКЛЮЧЕНА"
                print(f"\n[ROAD DETECTION] Детекция дороги {state_text}")
            prev_road_key_state = road_key_state

            # === КЛАВИША O ===
            obj_key_state = key_pressed(KEY_OBJECT_DETECTION)
            if obj_key_state and not prev_obj_key_state:
                object_detection_enabled = not object_detection_enabled
                state_text = "ВКЛЮЧЕНА" if object_detection_enabled else "ВЫКЛЮЧЕНА"
                print(f"\n[OBJECT DETECTOR] Детекция объектов {state_text}")
            prev_obj_key_state = obj_key_state

            # === КЛАВИША F8 (старт/стоп записи окна) ===
            record_key_state = key_pressed(KEY_TOGGLE_RECORD)
            if record_key_state and not prev_record_key_state:
                if not recording:
                    # --- СТАРТ ЗАПИСИ ---
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = frame.shape[:2]
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    record_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        f"recording_{ts}.mp4"
                    )
                    video_writer = cv2.VideoWriter(
                        record_path, fourcc, RECORD_FPS, (w, h)
                    )
                    if not video_writer.isOpened():
                        print(f"[REC] Не удалось открыть VideoWriter: {record_path}")
                        video_writer = None
                        record_path = None
                        record_size = None
                    else:
                        recording = True
                        record_size = (w, h)
                        print(f"[REC] Запись начата: {record_path}")
                else:
                    # --- СТОП ЗАПИСИ ---
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    recording = False
                    print(f"[REC] Запись остановлена: {record_path}")
                    record_path = None
                    record_size = None
            prev_record_key_state = record_key_state

            # === ДЕТЕКЦИЯ ДОРОГИ ===
            road_mask = None
            if road_detection_enabled:
                try:
                    frame, road_mask, road_detected = road_detector.detect(frame)
                except Exception as e:
                    print(f"[ERROR] Ошибка детекции дороги: {e}")

            # === ДЕТЕКЦИЯ ОБЪЕКТОВ ===
            if object_detection_enabled:
                try:
                    frame, detections = object_detector.detect(frame)
                    if detections:
                        frame = object_detector.draw(frame, detections)
                except Exception as e:
                    print(f"[ERROR] Ошибка детекции объектов: {e}")

            # === Подсветка дороги зелёным ===
            if road_detection_enabled and road_mask is not None:
                mask_overlay = np.zeros_like(frame)
                mask_overlay[road_mask > 0] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.25, 0)

            # === Панель статусов ===
            _draw_status_panel(frame,
                               road_on=road_detection_enabled,
                               obj_on=object_detection_enabled)

            # === FPS ===
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # === ЗАПИСЬ КАДРА (если включена) ===
            if recording and video_writer is not None:
                try:
                    h, w = frame.shape[:2]
                    if record_size is not None and (w, h) != record_size:
                        # Кадр изменил размер — пересоздаём writer, чтобы не потерять запись
                        video_writer.release()
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            record_path, fourcc, RECORD_FPS, (w, h)
                        )
                        if not video_writer.isOpened():
                            print(f"[REC] Не удалось пересоздать VideoWriter: {record_path}")
                            video_writer = None
                            recording = False
                            record_path = None
                            record_size = None
                        else:
                            record_size = (w, h)

                    if video_writer is not None:
                        video_writer.write(frame)
                except Exception as e:
                    print(f"[REC] Ошибка записи кадра: {e}")

            cv2.imshow("Vice City Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # === Корректно закрываем writer и сохраняем видео ===
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print(f"[REC] Запись сохранена: {record_path}")
        recording = False

    cv2.destroyAllWindows()
    print("Работа завершена.")


if __name__ == "__main__":
    main()