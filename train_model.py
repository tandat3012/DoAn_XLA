

import zipfile
import os

with zipfile.ZipFile('doan.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

if os.path.exists('/content/doan/data.yaml'):
    print("✅ Đã tìm thấy file data.yaml. Sẵn sàng Train!")
else:
    print("❌ Không tìm thấy file data.yaml. Hãy kiểm tra lại cấu trúc file zip của bạn.")

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Bắt đầu Train
# device=0 nghĩa là ép máy dùng GPU T4
model.train(data='/content/doan/data.yaml', epochs=50, imgsz=640, device=0)