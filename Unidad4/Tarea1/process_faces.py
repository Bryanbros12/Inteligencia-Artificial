import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

def process_images(input_dir, output_dir, augmentations_per_image=3):

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Definir transformaciones con probabilidades
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, 
            contrast_limit=0.3, 
            p=0.75
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=25, 
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])
    
    # Procesar cada imagen
    for emotion_dir in os.listdir(input_dir):
        emotion_path = os.path.join(input_dir, emotion_dir)
        
        if os.path.isdir(emotion_path):
            print(f"\nProcesando emoción: {emotion_dir}")
            output_emotion_dir = os.path.join(output_dir, emotion_dir)
            os.makedirs(output_emotion_dir, exist_ok=True)
            
            for img_name in tqdm(os.listdir(emotion_path)):
                img_path = os.path.join(emotion_path, img_name)
                
                try:
                    # Leer imagen
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Guardar imagen original
                    original_output = os.path.join(
                        output_emotion_dir, 
                        f"orig_{img_name}"
                    )
                    cv2.imwrite(
                        original_output, 
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
                    
                    # Crear múltiples versiones aumentadas
                    for i in range(augmentations_per_image):
                        transformed = transform(image=image)["image"]
                        
                        output_path = os.path.join(
                            output_emotion_dir, 
                            f"aug_{i}_{img_name}"
                        )
                        cv2.imwrite(
                            output_path, 
                            cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
                        )
                
                except Exception as e:
                    print(f"Error procesando {img_path}: {str(e)}")
                    continue

if __name__ == "__main__":
    # Configura tus rutas aquí
    input_directory = "/workspaces/Inteligencia-Artificial/Unidad4/fer2013/train"  # Ruta a las imágenes originales
    output_directory = "/workspaces/Inteligencia-Artificial/Unidad4/fer2013/augmented_train"  # Ruta para imágenes procesadas
    
    process_images(input_directory, output_directory, augmentations_per_image=3)
    print("\nProcesamiento completado exitosamente!")