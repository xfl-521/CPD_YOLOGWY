import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/yolov9/best.pt')
    model.val(data=r'D:/yolov9/ultralytics/cfg/models/Add/yolov8-p2-FASFFHead.yaml',
              split='val',
              imgsz=1024,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/test',
              name='exp',
              )