"""
Custom YOLOv5 Detector for image inference with background removal capabilities.
Cross-platform (Linux/Windows) and compatible with PyTorch 2.6+.
"""

from pathlib import Path
import torch
import cv2
import numpy as np

from ultralytics.utils.plotting import Annotator, colors
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device, smart_inference_mode


class CustomDetector:
    def __init__(
        self,
        weights,
        source,
        conf_thres=0.6,
        iou_thres=0.45,
        max_det=1000,
        device="",
        imgsz=(640, 640),
        exist_ok=False,
        bg_output_path="predictions/",
        with_bg_output_path="bgs/",
    ):
        self.weights = weights
        self.source = source
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.imgsz = imgsz
        self.exist_ok = exist_ok
        self.bg_output_path = Path(bg_output_path)
        self.with_bg_output_path = Path(with_bg_output_path)

        # Создаем папки для сохранения
        self.bg_output_path.mkdir(parents=True, exist_ok=True)
        self.with_bg_output_path.mkdir(parents=True, exist_ok=True)

        # Выбираем устройство
        self.device = select_device(device)

        # Загружаем модель через DetectMultiBackend (кроссплатформенно)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, fp16=False)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt

        # Проверка размера изображения
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        # Warmup модели
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    @smart_inference_mode()
    def run(self):
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)

        for path, im, im0s, vid_cap, s in dataset:
            im_tensor = torch.from_numpy(im).to(self.device).float() / 255.0
            if im_tensor.ndim == 3:
                im_tensor = im_tensor.unsqueeze(0)

            # Инференс
            pred = self.model(im_tensor, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det
            )

            for det in pred:
                p = Path(path)
                im0 = im0s.copy() if isinstance(im0s, np.ndarray) else im0s
                output_name = p.name

                # С изображением фона
                im_with_bg = im0.copy()
                annotator_with_bg = Annotator(im_with_bg, line_width=2, example=str(self.names))

                # С черным фоном
                im_without_bg = np.zeros_like(im0)
                annotator_without_bg = Annotator(im_without_bg, line_width=2, example=str(self.names))

                if len(det):
                    det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f"{self.names[c]} {conf:.2f}"

                        # Сохраняем на обоих изображениях
                        annotator_with_bg.box_label(xyxy, label, color=colors(c, True))
                        annotator_without_bg.box_label(xyxy, label, color=colors(c, True))

                        # Выделяем только область объекта на черном фоне
                        x1, y1, x2, y2 = map(int, xyxy)
                        im_without_bg[y1:y2, x1:x2] = im0[y1:y2, x1:x2]

                # Сохраняем результаты
                cv2.imwrite(str(self.with_bg_output_path / output_name), annotator_with_bg.result())
                cv2.imwrite(str(self.bg_output_path / output_name), annotator_without_bg.result())

                print(f"Saved: {self.with_bg_output_path / output_name}")
                print(f"Saved: {self.bg_output_path / output_name}")
                print(f"Detections: {len(det) if len(det) else 0}")


def main():
    detector = CustomDetector(
        weights="models/wood-detection-model.pt",
        source="test_image.jpg",
        conf_thres=0.6,
        exist_ok=True,
        bg_output_path="predictions/",
        with_bg_output_path="bgs/",
    )
    detector.run()


if __name__ == "__main__":
    main()
