from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11s.pt')
    model.train(data='dataset.yaml', epochs=25, batch=0.8, imgsz=640)
