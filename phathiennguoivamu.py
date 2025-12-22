import cv2
from ultralytics import YOLO

# ================== LOAD ==================
cap = cv2.VideoCapture(r'D:\tài liệu\doan\doan\congnhan1.mp4')

model_person = YOLO('yolov8n.pt')               # Người
model_helmet = YOLO(r'D:\tài liệu\doan\doan\best.pt')  # Mũ

PERSON_ID = 0
HELMET_ID = 8

# ================== LOOP ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    # ========= 1️⃣ DETECT NGƯỜI =========
    persons = model_person(frame, conf=0.4, verbose=False)[0]

    for box in persons.boxes:
        if int(box.cls[0]) != PERSON_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = frame[y1:y2, x1:x2]   # 👈 giống body_crop

        if person_crop.size == 0:
            continue

        # ========= 2️⃣ DETECT MŨ TRONG NGƯỜI =========
        helmet_results = model_helmet(person_crop, conf=0.15, verbose=False)[0]

        has_helmet = False

        for hbox in helmet_results.boxes:
            if int(hbox.cls[0]) == HELMET_ID:
                hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0])

                # Vẽ mũ (tọa độ cộng offset giống face + body)
                cv2.rectangle(frame,
                              (hx1 + x1, hy1 + y1),
                              (hx2 + x1, hy2 + y1),
                              (0, 255, 0), 2)

                has_helmet = True
                break

        # ========= 3️⃣ VẼ KẾT QUẢ =========
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        text = "AN TOAN" if has_helmet else "KHONG DOI MU"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("PHAT HIEN MU BAO HO", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
