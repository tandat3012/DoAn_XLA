import cv2
from ultralytics import YOLO

# ================== LOAD ==================
VIDEO_PATH = r"D:\Xu_li_anh\dataset\mubaoho0.mp4"
MODEL_PERSON_PATH = 'yolov8n.pt'
MODEL_HELMET_PATH = r'./best.pt'

PERSON_ID = 0
HELMET_ID = 8

cap = cv2.VideoCapture(VIDEO_PATH)

model_person = YOLO(MODEL_PERSON_PATH)
model_helmet = YOLO(MODEL_HELMET_PATH)

def detect_person(frame):
    results = model_person(frame, conf=0.4, verbose=False)[0]
    persons = []

    for box in results.boxes:
        if int(box.cls[0]) == PERSON_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2, y2))

    return persons


def detect_helmet(person_crop):

    helmet_results = model_helmet(person_crop, conf=0.15, verbose=False)[0]
    helmets = []

    for box in helmet_results.boxes:
        if int(box.cls[0]) == HELMET_ID:
            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
            helmets.append((hx1, hy1, hx2, hy2))

    return helmets

def check_has_helmet(person_crop, helmet_boxes):
    h, w, _ = person_crop.shape
    head_limit = int(h * 0.4)  # 40% phía trên

    for hx1, hy1, hx2, hy2 in helmet_boxes:
        helmet_center_y = (hy1 + hy2) // 2

        if helmet_center_y < head_limit:
            return True

    return False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    persons = detect_person(frame)

    for (x1, y1, x2, y2) in persons:
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        helmets = detect_helmet(person_crop)
        has_helmet = check_has_helmet(person_crop, helmets)
 

        # ===== VẼ KẾT QUẢ =====
        if has_helmet:
            color = (0, 255, 0)      
        else:
            color = (0, 0, 255)
        text = "AN TOAN" if has_helmet else "KHONG DOI MU"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, 
                    text, 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    color, 
                    2)

        # Vẽ mũ
        for hx1, hy1, hx2, hy2 in helmets:
            cv2.rectangle(frame,
                          (hx1 + x1, hy1 + y1),
                          (hx2 + x1, hy2 + y1),
                          (0, 255, 0), 2)

    cv2.imshow("PHAT HIEN MU BAO HO", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

