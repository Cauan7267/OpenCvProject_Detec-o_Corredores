import cv2
import numpy as np
import threading
from collections import deque

# Carregar os arquivos de configuração e pesos do YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Verificar se a GPU está disponível e configurar para usar a GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

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
    height, width, channels = frame.shape

    # Preparar a imagem para a detecção
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ordenar os corredores pela posição horizontal (x) de forma decrescente
    sorted_boxes = sorted([boxes[i] for i in indexes], key=lambda b: b[0], reverse=True)

    # Determinar a posição dos corredores
    positions = {i: idx + 1 for idx, i in enumerate(indexes)}

    # Atualizar o histórico de posições
    positions_history.append(positions)

    # Calcular a posição média ao longo dos frames
    avg_positions = {i: int(np.mean([pos[i] for pos in positions_history if i in pos])) for i in indexes}

    for i in range(len(boxes)):
        if i in indexes:
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

    # Processar o frame em uma thread separada
    thread = threading.Thread(target=process_frame, args=(frame,))
    thread.start()
    thread.join()

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
