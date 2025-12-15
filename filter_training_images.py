import os
from pathlib import Path
import random
import cv2
import numpy as np
# Importaciones de SAHI
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.slicing import slice_image
from PIL import Image

def filter_images_with_detections(input_dir, output_dir, model_path, n_images, slice_size, overlap_ratio, resize_factor):
    # ... (unchanged) ...

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: El directorio de entrada '{input_dir}' no existe.")
        return

    # 2. Cargar Modelo
    print(f"--- INICIANDO FILTRADO PARA ENTRENAMIENTO ---")
    print(f"Cargando modelo desde: {model_path}")
    print(f"Configuración Tiling: Slice={slice_size}x{slice_size}, Overlap={overlap_ratio*100}%")
    
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.5,
            device="gpu:0" 
        )
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # 3. Listar Imágenes
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    total_images = len(image_files)
    print(f"Se encontraron {total_images} imágenes en {input_dir}")

    if total_images == 0:
        print("No se encontraron imágenes válidas.")
        return

    # 4. Seleccionar Imágenes
    if n_images == -1 or n_images >= total_images:
        selected_images = image_files
        print(f"Procesando TODAS las {total_images} imágenes.")
    else:
        selected_images = random.sample(image_files, n_images)
        print(f"Procesando {n_images} imágenes seleccionadas aleatoriamente.")

    # 5. Ejecutar Inferencia y Filtrado
    saved_count = 0
    for i, img_file in enumerate(selected_images):
        print(f"\n[{i+1}/{len(selected_images)}] Analizando: {img_file.name} ...")

        try:
            # A. Leer imagen con OpenCV
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  -> Error: No se pudo leer el archivo.")
                continue
            
            # B. Redimensionar (Opcional)
            image_for_inference = image.copy()
            if resize_factor != 1.0:
                new_width = int(image.shape[1] * resize_factor)
                new_height = int(image.shape[0] * resize_factor)
                image_for_inference = cv2.resize(image, (new_width, new_height))
            
            # C. Convertir a RGB para SAHI
            image_rgb = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2RGB)

            # D. Cortar en Slices (Tiling)
            slice_image_result = slice_image(
                image=image_rgb,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio
            )
            
            num_slices = len(slice_image_result.images)
            print(f"  -> Generados {num_slices} slices para inferencia.")
            
            # E. Inferencia por Slice
            slices_saved_for_this_image = 0
            for slice_idx, slice_img in enumerate(slice_image_result.images):
                # slice_img es PIL Image por defecto cuando se usa slice_image con numpy array?
                # SAHI slice_image devuelve PIL Images en .images
                
                # Inferencia en el slice
                result = get_prediction(
                    slice_img,
                    detection_model,
                    verbose=0
                )
                
                num_detections = len(result.object_prediction_list)
                
                if num_detections > 0:
                    # Guardar el slice
                    save_filename = f"{img_file.stem}_slice_{slice_idx}.jpg"
                    save_path = output_path / save_filename
                    
                    # slice_img es numpy array (RGB), convertir a PIL
                    Image.fromarray(slice_img).save(save_path)
                    
                    # print(f"    -> Slice {slice_idx}: {num_detections} detecciones. Guardado.")
                    slices_saved_for_this_image += 1
                    saved_count += 1
            
            if slices_saved_for_this_image > 0:
                print(f"  -> Se guardaron {slices_saved_for_this_image} slices con detecciones.")
            else:
                print(f"  -> Negativo (0 slices con detecciones).")



        except Exception as e:
            print(f"  -> Error crítico procesando imagen: {e}")
            
    print(f"\n¡Listo! Se guardaron {saved_count} imágenes en: {output_dir}")


# ==========================================
#      BLOQUE DE EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURACIÓN ---
    
    # 1. Entrada
    MIS_IMAGENES = "/media/samuel/SSD/medellin_panoramas_recortados/area5/50m"
    
    # 2. Salida (Training Dataset)
    MI_SALIDA = "/media/samuel/SSD/medellin_panoramas_recortados/training"
    
    # 3. Modelo
    MI_MODELO = "model/best.pt" 
    
    # 4. Parámetros SAHI
    SLICE_SIZE = 1280
    RESIZE_FACTOR = 0.5
    
    # 5. Cantidad
    CANTIDAD = 50

    filter_images_with_detections(
        input_dir=MIS_IMAGENES,
        output_dir=MI_SALIDA,
        model_path=MI_MODELO,
        n_images=CANTIDAD,
        slice_size=SLICE_SIZE,
        overlap_ratio=0,
        resize_factor=RESIZE_FACTOR
    )
