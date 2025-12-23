import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ================== LOAD MODEL ==================
MODEL_PERSON_PATH = 'yolov8n.pt'
MODEL_HELMET_PATH = './best.pt'

PERSON_ID = 0
HELMET_ID = 8

model_person = YOLO(MODEL_PERSON_PATH)
model_helmet = YOLO(MODEL_HELMET_PATH)

# ================== FUNCTIONS ==================
def detect_person(frame):
    results = model_person(frame, conf=0.4, verbose=False)[0]
    persons = []

    for box in results.boxes:
        if int(box.cls[0]) == PERSON_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2, y2))

    return persons


def detect_helmet(person_crop):
    helmet_results = model_helmet(person_crop, conf=0.1, verbose=False)[0]
    helmets = []

    for box in helmet_results.boxes:
        if int(box.cls[0]) == HELMET_ID:
            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
            helmets.append((hx1, hy1, hx2, hy2))

    return helmets


def check_has_helmet(person_crop, helmet_boxes):
    h, w, _ = person_crop.shape
    head_limit = int(h * 0.35)
    for hx1, hy1, hx2, hy2 in helmet_boxes:
        if hy1 < head_limit:
            return True
    return False


def detec_image(image):
    """
    image: numpy array (RGB)
    """
    if image is None:
        return None

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (800, 500))

    persons = detect_person(frame)

    for (x1, y1, x2, y2) in persons:
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        helmets = detect_helmet(person_crop)
        has_helmet = check_has_helmet(person_crop, helmets)

        # ===== DRAW =====
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        text = "AN TOAN" if has_helmet else "KHONG DOI MU"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for hx1, hy1, hx2, hy2 in helmets:
            cv2.rectangle(
                frame,
                (hx1 + x1, hy1 + y1),
                (hx2 + x1, hy2 + y1),
                (0, 255, 0),
                2
            )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


#Gradio
with gr.Blocks(title="Phát hiện mũ bảo hộ") as demo:
    gr.Markdown("##Demo phát hiện mũ bảo hộ lao động")

    with gr.Row():
        input_image = gr.Image(label="Ảnh đầu vào" )
        output_image = gr.Image(label="Kết quả")

    analyze_btn = gr.Button("Phân tích")

    analyze_btn.click(
        fn=detec_image,
        inputs=input_image,
        outputs=output_image
    )

demo.launch()
