import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/ygwexp5/weights/best.pt') # select your model.pt path
    model.predict(source='F:/APP检测展示图/6.jpg',
                  imgsz=1024,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  classes=[0,1,2],
                  iou=0.6,
                  conf=0.6
                )

