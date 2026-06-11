import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuración
DATASET_DIR = "../datasets/bp_digits_yolo_dataset_v2"
IMG_SIZE = 28  # Tamaño de entrada similar a MNIST (28x28)

def extract_crops(subset):
    images_dir = os.path.join(DATASET_DIR, "images", subset)
    labels_dir = os.path.join(DATASET_DIR, "labels", subset)
    
    crops = []
    labels = []
    
    if not os.path.exists(images_dir):
        print(f"Directorio no encontrado: {images_dir}")
        return np.array([]), np.array([])
        
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(images_dir, filename)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        # Cargar imagen en escala de grises
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        h, w = img.shape
        
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                
                # Calcular coordenadas absolutas
                x1 = int(max(0, cx - bw / 2.0))
                y1 = int(max(0, cy - bh / 2.0))
                x2 = int(min(w, cx + bw / 2.0))
                y2 = int(min(h, cy + bh / 2.0))
                
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                    
                crop = img[y1:y2, x1:x2]
                
                # Binarización adaptativa para emular la binarización de la app
                crop_bin = cv2.adaptiveThreshold(
                    crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 25, 9
                )
                
                # Redimensionar al tamaño estándar
                crop_resized = cv2.resize(crop_bin, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                
                crops.append(crop_resized)
                labels.append(class_id)
                
    return np.array(crops), np.array(labels)

print("Cargando recortes del dataset...")
X_train_raw, y_train_raw = extract_crops("train")
X_val_raw, y_val_raw = extract_crops("val")

print(f"Recortes de train extraídos: {len(X_train_raw)}")
print(f"Recortes de val extraídos: {len(X_val_raw)}")

if len(X_train_raw) == 0:
    print("Error: No se pudieron extraer recortes para entrenamiento.")
    exit(1)

# Aumento de datos (Data Augmentation) manual para balancear y robustecer el dataset
def augment_image(img):
    augmented = []
    h, w = img.shape
    
    # Imagen original
    augmented.append(img)
    
    # 1. Rotación aleatoria (-12 a 12 grados)
    for angle in [-12, -8, -4, 4, 8, 12]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
        augmented.append(rotated)
        
    # 2. Desplazamiento horizontal y vertical
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h), borderValue=255)
        augmented.append(shifted)
        
    # 3. Zoom (ligero recorte y redimensión)
    # Acercar
    crop_in = img[1:-1, 1:-1]
    zoomed_in = cv2.resize(crop_in, (w, h), interpolation=cv2.INTER_LINEAR)
    augmented.append(zoomed_in)
    
    # Alejar
    padded = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    zoomed_out = cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)
    augmented.append(zoomed_out)
    
    # 4. Dilatación y erosión morfológica para emular variaciones de grosor
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.erode(img, kernel, iterations=1) # Erode en blanco = Dilatar negro (texto)
    eroded = cv2.dilate(img, kernel, iterations=1)  # Dilate en blanco = Erosionar negro (texto)
    augmented.append(dilated)
    augmented.append(eroded)
    
    return augmented

print("Aplicando Data Augmentation...")
X_train_aug = []
y_train_aug = []

for img, label in zip(X_train_raw, y_train_raw):
    aug_images = augment_image(img)
    X_train_aug.extend(aug_images)
    y_train_aug.extend([label] * len(aug_images))
    
X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)

# Para validación no aumentamos demasiado, solo normalizamos la estructura
X_val = X_val_raw
y_val = y_val_raw

# Normalizar píxeles a rango 0.0 - 1.0 (invertir para que el fondo sea 0 y el texto sea 1, ayuda en la convergencia de la CNN)
X_train = (255.0 - X_train) / 255.0
X_val = (255.0 - X_val) / 255.0

# Añadir canal
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)

print(f"Dataset final de entrenamiento: {X_train.shape}")
print(f"Dataset de validación: {X_val.shape}")

# Crear la arquitectura de la CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
epochs = 25
batch_size = 64

print("Iniciando entrenamiento de la CNN...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val) if len(X_val) > 0 else None,
    verbose=1
)

# Evaluar precisión final
if len(X_val) > 0:
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nPrecisión en Validación (Clasificación de Dígitos): {val_acc*100:.2f}%")

# Guardar modelo Keras
os.makedirs("tool", exist_ok=True)
model.save("tool/digit_classifier.h5")
print("Modelo guardado como tool/digit_classifier.h5")

# Exportar a TFLite
print("Convirtiendo modelo a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = "tool/digit_classifier.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
    
print(f"Modelo TFLite exportado correctamente a {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")
