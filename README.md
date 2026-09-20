# 🚗 GTA: Vice City AI — Детекция объектов и дороги в реальном времени

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-SegFormer-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Object_Detection-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Проект **GTA-VC-AI** — это система компьютерного зрения, которая накладывает нейросетевую аналитику поверх окна игры *Grand Theft Auto: Vice City*. 

В отличие от традиционных модов, этот инструмент работает **внешне** (через захват экрана), используя современные модели для семантической сегментации дороги и детекции игровых объектов (машин, пешеходов) в реальном времени.

---

## 📸 Демонстрация (Screenshots)

### 🚙 Только детекция объектов (TensorFlow)
Распознавание автомобилей и главного героя (Tommy).
![Object Detection](screenshots/1.png)

### 🛣️ Детекция дороги + Объекты (SegFormer + TF)
Семантическая сегментация дорожного полотна с наложением маски и детекцией объектов.
![Road + Object Detection](screenshots/2.png)
![Road + Object Detection](screenshots/3.png)

### 📐 Анализ границ дороги
Визуализация границ проезжей части и трекинг объектов на трассе.
![Road Boundaries](screenshots/4.png)

---

## ✨ Возможности

*   **Детекция дороги (Road Detection):** Использует модель **SegFormer** (HuggingFace) для семантической сегментации. Определяет класс `road` из датасета Cityscapes.
*   **Фильтрация шума:** Встроенный HSV-фильтр отсекает зелёные пиксели (траву, кусты), чтобы маска дороги не "заливала" обочины.
*   **Детекция объектов (Object Detection):** Использует **TensorFlow Frozen Graph** для поиска машин, людей и других сущностей с применением NMS (Non-Maximum Suppression).
*   **Захват экрана:** Высокопроизводительный захват клиентской области окна через `mss` и `win32gui`.
*   **Оптимизация:** Поддержка FP16 (полуточность) и инференс не каждый кадр для повышения FPS.

---

## 🛠️ Технологический стек

*   **Язык:** Python 3.10+
*   **Нейросети:** 
    *   `PyTorch` + `transformers` (SegFormer-b0)
    *   `TensorFlow` (Object Detection API)
*   **Компьютерное зрение:** `OpenCV`, `NumPy`
*   **Системные API:** `mss` (захват экрана), `win32gui`, `psutil` (поиск окна игры)

---

## ⚙️ Установка и запуск

### 1. Требования
*   Установленная игра **GTA: Vice City** (запущенная в оконном или безрамочном режиме).
*   Видеокарта **NVIDIA** (рекомендуется для использования CUDA, но поддерживается и CPU).

### 2. Клонирование и зависимости
```bash
git clone https://github.com/0x2171/Vice-City-AI.git
cd Vice-City-AI
pip install -r requirements.txt
