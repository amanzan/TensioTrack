import os
import re
import cv2
import numpy as np
import tensorflow as tf

# Configuración
DIR_PATH = "../fotostension"
MODEL_PATH = "tool/digit_classifier.h5"
IMG_SIZE = 28

# Cargar modelo
if not os.path.exists(MODEL_PATH):
    print(f"Error: Modelo no encontrado en {MODEL_PATH}")
    exit(1)
    
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo CNN cargado correctamente.")

def get_expected(filename):
    match = re.match(r'^\d+_([0-9]+)_([0-9]+)\.(jpe?g|png)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.match(r'^synth_([0-9]+)_([0-9]+)\.(jpe?g|png)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def segment_and_classify_region(crop):
    if crop is None or crop.size == 0:
        return None
        
    h, w, _ = crop.shape
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Redimensionar 2x para mejorar resolución
    resized = cv2.resize(gray, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape
    
    # 2. Binarización adaptativa local (Bradley-Roth equivalente)
    binarized = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 9
    )
    
    # Invertir para tener texto blanco (255) y fondo negro (0)
    inverted = 255 - binarized
    
    # 3. Dilatación morfológica para conectar los 7 segmentos de cada número
    # Usamos un kernel ligeramente vertical/horizontal para conectar segmentos separados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    dilated = cv2.dilate(inverted, kernel, iterations=2)
    
    # 4. Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    for cnt in contours:
        x, y, dw, dh = cv2.boundingRect(cnt)
        
        # Filtros de tamaño
        # El dígito debe ocupar al menos el 35% del alto del recorte
        if dh < rh * 0.35:
            continue
        # Descartar elementos demasiado pequeños o marcos demasiado anchos
        if dw < 4 or dw > rw * 0.5:
            continue
            
        digit_boxes.append((x, y, dw, dh))
        
    # Ordenar de izquierda a derecha
    digit_boxes.sort(key=lambda b: b[0])
    
    # Si no detectamos dígitos, retornamos 0
    if not digit_boxes:
        return 0
        
    # Clasificar cada dígito detectado
    digits = []
    for idx, (bx, by, bw, bh) in enumerate(digit_boxes):
        # Recortar de la imagen INVERTIDA original (sin dilatación excesiva)
        # Añadir un pequeño margen de padding
        pad = int(min(bw, bh) * 0.15)
        y1 = max(0, by - pad)
        y2 = min(rh, by + bh + pad)
        x1 = max(0, bx - pad)
        x2 = min(rw, bx + bw + pad)
        
        digit_crop = inverted[y1:y2, x1:x2]
        if digit_crop.size == 0:
            continue
            
        # Redimensionar al tamaño del clasificador (28x28)
        digit_resized = cv2.resize(digit_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Normalizar y preparar entrada para la CNN (canal y batch)
        digit_norm = digit_resized.astype(np.float32) / 255.0
        digit_input = np.expand_dims(digit_norm, axis=(0, -1))
        
        # Predecir
        preds = model.predict(digit_input, verbose=0)
        pred_digit = np.argmax(preds[0])
        
        digits.append(str(pred_digit))
        
    if not digits:
        return 0
        
    return int("".join(digits))

def evaluate_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
        
    h, w, _ = img.shape
    
    # Recortes centrales basados en la posición física de la pantalla (layout estándar)
    sys_crop = img[int(h*0.22):int(h*0.39), int(w*0.35):int(w*0.65)]
    dia_crop = img[int(h*0.36):int(h*0.51), int(w*0.35):int(w*0.65)]
    
    sys_val = segment_and_classify_region(sys_crop)
    dia_val = segment_and_classify_region(dia_crop)
    
    return sys_val, dia_val

# Ejecutar sobre todo el directorio
files = [f for f in os.listdir(DIR_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
files.sort()

print(f"\nIniciando Benchmark Híbrido sobre {len(files)} imágenes...")
ok = 0
failed = 0

for file in files:
    expected = get_expected(file)
    if not expected:
        continue
        
    expected_sys, expected_dia = expected
    path = os.path.join(DIR_PATH, file)
    
    sys_val, dia_val = evaluate_image(path)
    
    status = "OK" if (sys_val == expected_sys and dia_val == expected_dia) else "FAIL"
    if status == "OK":
        ok += 1
    else:
        failed += 1
        
    print(f"{file}: Expected={expected_sys}/{expected_dia}, Detected={sys_val}/{dia_val} -> {status}")

precision = (ok / len(files)) * 100
print(f"\nResumen Pipeline Híbrido:")
print(f"Total: {len(files)}")
print(f"Correctas: {ok}")
print(f"Falladas: {failed}")
print(f"Precisión: {precision:.2f}%")
