import cv2
import numpy as np


def remove_background_with_bboxes(results, conf_thres=0.0):
    """
    Убирает фон, сохраняя:
    - оригинальные цвета внутри bbox (без деградации)
    - bbox
    - подписи классов

    Args:
        results: Results объект из YOLO11 (результат model.predict())
        conf_thres: порог уверенности для фильтрации детекций
    """

    # YOLO11: оригинальное изображение находится в results.orig_img
    img = results.orig_img
    h, w = img.shape[:2]

    objects_layer = np.zeros((h, w, 3), dtype=np.float32)
    draw_layer = np.zeros((h, w, 3), dtype=np.uint8)

    # YOLO11: используем результаты через results.boxes
    # Это специальный объект, удобнее, чем массив numpy
    for box in results.boxes:
        conf = float(box.conf[0])

        if conf < conf_thres:
            continue

        # YOLO11: xyxy - это просто атрибут box объекта
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])

        # Копируем оригинальные цвета внутри bbox
        objects_layer[y1:y2, x1:x2] = img[y1:y2, x1:x2].astype(np.float32)

        # Рисуем bbox
        cv2.rectangle(draw_layer, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Формируем подпись (class_name + confidence)
        label = f"{results.names[cls]} {conf:.2f}"

        # Размер текста
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Фон для текста
        cv2.rectangle(
            draw_layer,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            (0, 255, 0),
            -1,
        )

        # Сам текст
        cv2.putText(
            draw_layer,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Финальная обработка
    result = np.clip(objects_layer, 0, 255).astype(np.uint8)
    mask = draw_layer.any(axis=2)
    result[mask] = draw_layer[mask]

    return result
