import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Carregar o modelo YOLOv8
model = YOLO("yolov8n.pt")

# Carregar os nomes das classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar o vídeo
cap = cv2.VideoCapture("./videos_corrida/Paulo André Camilo Campeão dos 100m rasos FISU 2019 (CAMPEONATO MUNDIAL UNIVERSITÁRIO).mp4")

frame_skip = 5  # Processar a cada 5 frames
frame_count = 0

# Usar uma fila maior para suavizar a detecção ao longo de vários frames
positions_history = deque(maxlen=10)

def process_frame(frame):
    height, width, _ = frame.shape

    # Realizar a detecção com YOLOv8
    results = model(frame)

    class_ids = []
    confidences = []
    boxes = []

    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls)
            confidence = detection.conf
            if confidence > 0.5 and classes[class_id] == "person":
                x, y, w, h = detection.xywh
                x = int(x - w / 2)
                y = int(y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Ordenar os corredores pela posição horizontal (x) de forma crescente
    sorted_indexes = sorted(indexes, key=lambda i: boxes[i][0])

    # Determinar a posição dos corredores
    positions = {i: idx + 1 for idx, i in enumerate(sorted_indexes)}

    # Atualizar o histórico de posições
    positions_history.append(positions)

    # Calcular a posição média ao longo dos frames
    avg_positions = {i: int(np.mean([pos[i] for pos in positions_history if i in pos])) for i in sorted_indexes}

    for i in range(len(boxes)):
        if i in sorted_indexes:
            x, y, w, h = boxes[i]
            label = f"{avg_positions[i]}º lugar"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Reduzir a resolução do frame
    frame = cv2.resize(frame, (1280, 720))

    # Processar o frame
    frame = process_frame(frame)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
