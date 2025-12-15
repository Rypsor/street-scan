import os
from pathlib import Path
import random
import cv2
import numpy as np
# Importaciones de SAHI
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

def run_inference(input_root, areas_list, output_dir, model_path, n_images, slice_size, overlap_ratio, resize_factor):
    # 1. Configurar rutas
    root_path = Path(input_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not root_path.exists():
        print(f"Error: El directorio raíz '{input_root}' no existe.")
        return

    # 2. Cargar Modelo
    print(f"--- INICIANDO PROCESO ---")
    print(f"Cargando modelo desde: {model_path}")
    print(f"Configuración Tiling: Slice={slice_size}x{slice_size}, Overlap={overlap_ratio*100}%")
    
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.5,
            device="gpu" 
        )
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # 3. Cargar Detecciones Existentes (Deduplicación)
    import json
    detections_buffer = {} # {'clase': [lista_detecciones]}
    processed_files = set() # Set de rutas relativas ya procesadas
    processed_images_record = [] # Lista para guardar en el JSON de procesados

    # A. Cargar registro de imágenes procesadas (si existe)
    processed_json_path = output_path / "processed_images.json"
    if processed_json_path.exists():
        try:
            with open(processed_json_path, 'r') as f:
                processed_images_record = json.load(f)
                # Agregar al set de procesados para saltarlas
                for path in processed_images_record:
                    processed_files.add(path)
            print(f"Cargado registro de {len(processed_images_record)} imágenes ya procesadas.")
        except Exception as e:
            print(f"Error cargando processed_images.json: {e}")

    # B. Buscar archivos JSON de detecciones existentes (para mayor seguridad)
    existing_jsons = list(output_path.glob("detections_*.json"))
    print(f"Cargando {len(existing_jsons)} archivos JSON de detecciones existentes...")

    for json_file in existing_jsons:
        class_name = json_file.stem.replace("detections_", "")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                detections_buffer[class_name] = data
                # Registrar archivos ya procesados (por si acaso no estaban en el processed_images.json)
                for item in data:
                    if "relative_path" in item:
                        processed_files.add(item["relative_path"])
                    elif "image" in item:
                        processed_files.add(item["image"]) 
        except Exception as e:
            print(f"  -> Error leyendo {json_file.name}: {e}")

    print(f"Total de imágenes únicas ya procesadas (detectadas + log): {len(processed_files)}")

    # Función auxiliar para guardar
    def save_progress():
        print(f"  -> Guardando progreso parcial...")
        # 1. Guardar detecciones
        for class_name, detections in detections_buffer.items():
            json_filename = f"detections_{class_name}.json"
            json_path = output_path / json_filename
            try:
                with open(json_path, 'w') as f:
                    json.dump(detections, f, indent=4)
            except Exception as e:
                print(f"Error guardando JSON parcial de detecciones: {e}")
        
        # 2. Guardar registro de procesados
        try:
            with open(processed_json_path, 'w') as f:
                json.dump(processed_images_record, f, indent=4)
        except Exception as e:
            print(f"Error guardando processed_images.json: {e}")

    # 4. Iterar sobre Áreas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    try:
        for area_id in areas_list:
            area_folder = f"area{area_id}/50m"
            current_input_path = root_path / area_folder
            
            print(f"\n--- Analizando Área: {area_folder} ---")
            
            if not current_input_path.exists():
                print(f"  -> Advertencia: La carpeta {current_input_path} no existe. Saltando.")
                continue

            # Listar imágenes del área
            all_images = [f for f in current_input_path.iterdir() if f.suffix.lower() in valid_extensions]
            
            # Filtrar ya procesadas
            images_to_process = []
            skipped_count = 0
            
            for img in all_images:
                # Usamos la ruta relativa desde el root como identificador único
                rel_path = str(img.relative_to(root_path))
                # También chequeamos el nombre solo por compatibilidad con lo que cargamos antes si faltaba path
                if rel_path in processed_files or img.name in processed_files:
                    skipped_count += 1
                else:
                    images_to_process.append(img)

            print(f"  -> Imágenes totales: {len(all_images)}")
            print(f"  -> Ya procesadas (saltadas): {skipped_count}")
            print(f"  -> Pendientes por procesar: {len(images_to_process)}")

            if not images_to_process:
                continue

            # Selección de muestra (si aplica)
            if n_images != -1 and n_images < len(images_to_process):
                selected_images = random.sample(images_to_process, n_images)
                print(f"  -> Procesando muestra de {n_images} imágenes.")
            else:
                selected_images = images_to_process
                print(f"  -> Procesando TODAS las {len(images_to_process)} imágenes pendientes.")

            # 5. Ejecutar Inferencia en el Área actual
            for i, img_file in enumerate(selected_images):
                print(f"[{i+1}/{len(selected_images)}] {img_file.name} ...", end=" ", flush=True)

                try:
                    # A. Leer imagen
                    image = cv2.imread(str(img_file))
                    if image is None:
                        print(f"Error lectura.")
                        continue
                    
                    # B. Redimensionar
                    if resize_factor != 1.0:
                        new_width = int(image.shape[1] * resize_factor)
                        new_height = int(image.shape[0] * resize_factor)
                        image = cv2.resize(image, (new_width, new_height))

                    # C. Convertir a RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # D. Inferencia
                    result = get_sliced_prediction(
                        image_rgb,
                        detection_model,
                        slice_height=slice_size,
                        slice_width=slice_size,
                        overlap_height_ratio=overlap_ratio,
                        overlap_width_ratio=overlap_ratio,
                        verbose=0 # Menos ruido en consola
                    )
                    
                    num_detections = len(result.object_prediction_list)
                    
                    if num_detections > 0:
                        print(f"Detectados: {num_detections}")
                        for prediction in result.object_prediction_list:
                            class_name = prediction.category.name
                            bbox = prediction.bbox.to_xyxy()
                            score = prediction.score.value

                            if class_name not in detections_buffer:
                                detections_buffer[class_name] = []

                            detections_buffer[class_name].append({
                                "image": img_file.name,
                                "relative_path": str(img_file.relative_to(root_path)),
                                "bbox": bbox,
                                "score": score
                            })
                    else:
                        print(f"0 detecciones.")

                except Exception as e:
                    print(f"Error: {e}")
                
                # Registrar como procesada (independientemente del resultado)
                rel_path_str = str(img_file.relative_to(root_path))
                if rel_path_str not in processed_images_record:
                     processed_images_record.append(rel_path_str)
                
                # Guardar cada 10 imágenes
                if (i + 1) % 10 == 0:
                    save_progress()
            
            # Guardar al finalizar el área
            save_progress()

    except KeyboardInterrupt:
        print(f"\n\n!!! INTERRUPCIÓN DE USUARIO DETECTADA !!!")
        print(f"Guardando datos procesados hasta el momento...")
        save_progress()
        print(f"Saliendo de forma segura.")
        return

    print(f"\n¡Listo! Proceso finalizado.")


# ==========================================
#      BLOQUE DE EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURACIÓN DE USUARIO (EDITA AQUÍ) ---
    

    # 1. Carpeta RAÍZ donde están las áreas
    INPUT_ROOT = "/media/samuel/SSD/medellin_panoramas_recortados"
    
    # 2. Lista de áreas a procesar (IDs numéricos)
    # Ejemplo: [5, 6, 7] buscará en area5/50m, area6/50m, area7/50m
    AREAS_A_PROCESAR = [3,4] 
    
    
    # 3. ¿Dónde guardamos los resultados (JSONs)?
    MI_SALIDA = "/media/samuel/SSD/medellin_panoramas_recortados/inferencia/all_detections"
    
    # 4. ¿Dónde está tu modelo entrenado (.pt)?
    MI_MODELO = "model/best.pt" 
    
    # 5. Parámetros
    SLICE_SIZE = 2560    
    RESIZE_FACTOR = 0.5 
    
    # 6. Cuántas imágenes procesar POR ÁREA (-1 para todas)
    CANTIDAD_POR_AREA = -1

    # --- LLAMADA AUTOMÁTICA ---
    run_inference(
        input_root=INPUT_ROOT,
        areas_list=AREAS_A_PROCESAR,
        output_dir=MI_SALIDA,
        model_path=MI_MODELO,
        n_images=CANTIDAD_POR_AREA,
        slice_size=SLICE_SIZE,
        overlap_ratio=0.2,
        resize_factor=RESIZE_FACTOR
    )