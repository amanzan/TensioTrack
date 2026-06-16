import os
import re
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Configuración
DIR_PATH = "../fotostension"
YOLO_MODEL_PATH = "../datasets/bp_digits_yolo_dataset/runs/detect/train/weights/best.pt"
CNN_MODEL_PATH = "tool/digit_classifier.h5"
IMG_SIZE = 28

# Cargar modelos
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"Error: Modelo YOLO no encontrado en {YOLO_MODEL_PATH}")
    exit(1)
if not os.path.exists(CNN_MODEL_PATH):
    print(f"Error: Modelo CNN no encontrado en {CNN_MODEL_PATH}")
    exit(1)
    
yolo = YOLO(YOLO_MODEL_PATH)
cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
print("Modelos YOLO y CNN cargados correctamente.")

def get_expected(filename):
    match = re.match(r'^\d+_([0-9]+)_([0-9]+)\.(jpe?g|png)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.match(r'^synth_([0-9]+)_([0-9]+)\.(jpe?g|png)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def group_rows(detections, row_y_threshold=0.6):
    if not detections:
        return []
        
    # Ordenar por coordenada Y
    sorted_det = sorted(detections, key=lambda d: d['cy'])
    
    # Calcular altura promedio
    avg_height = sum(d['h'] for d in sorted_det) / len(sorted_det)
    y_threshold = avg_height * row_y_threshold
    
    rows = []
    for d in sorted_det:
        placed = False
        for row in rows:
            # Calcular Y promedio de la fila
            row_cy = sum(rd['cy'] for rd in row) / len(row)
            if abs(d['cy'] - row_cy) < y_threshold:
                row.append(d)
                placed = True
                break
        if not placed:
            rows.append([d])
            
    # Ordenar cada fila de izquierda a derecha
    for row in rows:
        row.sort(key=lambda d: d['cx'])
        
    # Ordenar filas por su Y promedio de arriba a abajo
    rows.sort(key=lambda row: sum(rd['cy'] for rd in row) / len(row))
    return rows

def evaluate_image_hybrid(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Inferencia YOLO para detectar dígitos
    # Usamos conf=0.1 y NMS agnóstico a clase para emular la recuperación y evitar duplicados
    results = yolo.predict(img, conf=0.1, iou=0.45, agnostic_nms=True, verbose=False)
    
    detections = []
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dw = x2 - x1
        dh = y2 - y1
        
        detections.append({
            'x1': int(x1), 'y1': int(y1),
            'x2': int(x2), 'y2': int(y2),
            'cx': cx, 'cy': cy,
            'w': dw, 'h': dh
        })
        
    if not detections:
        return None, None
        
    # 2. Agrupar detecciones por filas (SYS y DIA)
    rows = group_rows(detections)
    if len(rows) < 2:
        return None, None
        
    # 3. Clasificar cada recorte con la CNN
    row_values = []
    for row in rows[:2]:  # Solo las primeras dos filas (SYS y DIA)
        row_digits = []
        for det in row:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            
            # Recortar de la imagen en escala de grises
            # Añadir un pequeño margen de padding
            pad = int(min(det['w'], det['h']) * 0.15)
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(w, x2 + pad)
            ry2 = min(h, y2 + pad)
            
            crop = gray[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
                
            # Binarización adaptativa
            crop_bin = cv2.adaptiveThreshold(
                crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 25, 9
            )
            
            # Redimensionar al tamaño del clasificador (28x28)
            crop_resized = cv2.resize(crop_bin, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # Normalizar y preparar entrada para la CNN (canal y batch)
            crop_norm = (255.0 - crop_resized) / 255.0  # Invertir para que fondo sea 0 y texto 1
            crop_input = np.expand_dims(crop_norm, axis=(0, -1))
            
            # Predecir
            preds = cnn.predict(crop_input, verbose=0)
            pred_digit = np.argmax(preds[0])
            row_digits.append(str(pred_digit))
            
        if row_digits:
            row_values.append(int("".join(row_digits)))
        else:
            row_values.append(0)
            
    sys_val = row_values[0] if len(row_values) > 0 else None
    dia_val = row_values[1] if len(row_values) > 1 else None
    
    return sys_val, dia_val

# Ejecutar sobre todo el directorio
files = [f for f in os.listdir(DIR_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
files.sort()

print(f"\nIniciando Benchmark Híbrido (YOLO Detección + CNN Clasificación) sobre {len(files)} imágenes...")
ok = 0
failed = 0

for file in files:
    expected = get_expected(file)
    if not expected:
        continue
        
    expected_sys, expected_dia = expected
    path = os.path.join(DIR_PATH, file)
    
    sys_val, dia_val = evaluate_image_hybrid(path)
    
    status = "OK" if (sys_val == expected_sys and dia_val == expected_dia) else "FAIL"
    if status == "OK":
        ok += 1
    else:
        failed += 1
        
    print(f"{file}: Expected={expected_sys}/{expected_dia}, Detected={sys_val}/{dia_val} -> {status}")

precision = (ok / len(files)) * 100
print(f"\nResumen Pipeline Híbrido (YOLO + CNN):")
print(f"Total: {len(files)}")
print(f"Correctas: {ok}")
print(f"Falladas: {failed}")
print(f"Precisión: {precision:.2f}%")
