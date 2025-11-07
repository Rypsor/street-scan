import streamlit as st
import pandas as pd
import os
import json
from ultralytics import YOLO
from PIL import Image
import random

# --- 1. Definición de Constantes y Rutas ---
MODEL_PATH = 'model/best.pt'
IMAGE_DIR = 'imagenes_medellin/imagenes_muestreadas'
METADATA_PATH = 'imagenes_medellin/metadata_muestra.json'

# --- 2. Funciones de Carga (con Caché) ---
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo desde {path}: {e}")
        return None

@st.cache_data
def load_metadata_map(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Convertir la lista de diccionarios en un mapa (dict) para búsqueda rápida
        metadata_map = {item['filename']: item for item in data}
        return metadata_map
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de metadatos en {path}")
        return None
    except Exception as e:
        st.error(f"Error al leer el archivo JSON: {e}")
        return None


def main():
    st.title("🗺️ Detector de Graffiti con Geo-localización")

    # --- Verificación de Archivos ---
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: No se encuentra el modelo en '{MODEL_PATH}'")
        return
    if not os.path.exists(IMAGE_DIR):
        st.error(f"Error: No se encuentra el directorio de imágenes en '{IMAGE_DIR}'")
        return
    if not os.path.exists(METADATA_PATH):
        st.error(f"Error: No se encuentra el JSON de metadatos en '{METADATA_PATH}'")
        return

    # --- Cargar Modelo y Metadatos ---
    model = load_yolo_model(MODEL_PATH)
    metadata_map = load_metadata_map(METADATA_PATH)

    if model is None or metadata_map is None:
        st.warning("La aplicación no puede iniciar por falta de archivos.")
        return
        
    # --- ¡MODIFICACIÓN! ---
    # Extraer los nombres de las clases del modelo
    # Asumimos que el modelo tiene {0: 'artistico', 1: 'vandalico'}
    try:
        class_names = model.names
        st.info(f"Modelo cargado. Clases detectadas: {class_names}")
    except Exception:
        st.warning("No se pudieron leer los nombres de las clases. Asumiendo {0: 'artistico', 1: 'vandalico'}")
        class_names = {0: 'artistico', 1: 'vandalico'}

    # --- Obtener la lista de imágenes ---
    try:
        image_filenames = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_filenames:
            st.warning(f"No se encontraron imágenes en '{IMAGE_DIR}'")
            return
    except Exception as e:
        st.error(f"Error al leer el directorio de imágenes: {e}")
        return

    total_available = len(image_filenames)
    st.info(f"Encontradas {total_available} imágenes en el directorio.")

    max_to_process = st.number_input(
        "Número máximo de imágenes a procesar",
        min_value=1,
        max_value=total_available,
        value=min(50, total_available), # Valor por defecto es 50 o el total, lo que sea menor
        step=1,
    )

    # --- Botón de Inicio ---
    if st.button("🚀 Iniciar Análisis de Imágenes"):
        total_images = min(int(max_to_process), total_available)
        
        # Usamos random.sample() para seleccionar N imágenes aleatoriamente
        image_filenames_to_process = random.sample(image_filenames, k=total_images)

        locations_with_detections = []
        images_with_detections = []
        
        # --- ¡MODIFICACIÓN! ---
        # Contadores para cada clase
        detection_counts = {class_names[0]: 0, class_names[1]: 0}
        # Un set para las ubicaciones únicas, para el mapa
        unique_locations = set()

        progress_bar = st.progress(0)
        status_text = st.empty()

        st.info(f"Se procesarán {total_images} imágenes seleccionadas al azar.")

        # --- El Bucle Principal ---
        for i, filename in enumerate(image_filenames_to_process):
            image_path = os.path.join(IMAGE_DIR, filename)

            status_text.text(f"Procesando {i+1}/{total_images}: {filename}...")

            # --- ¡MODIFICACIÓN! ---
            # Realizar la predicción
            # clases=[0, 1] -> Filtra para ambas clases
            results = model.predict(image_path, classes=[0, 1], conf=0.5, verbose=False)[0]

            if len(results.boxes) > 0:
                
                # --- ¡MODIFICACIÓN! ---
                # Contar cuántas instancias de CADA clase se encontraron
                try:
                    detected_class_ids = results.boxes.cls.cpu().numpy() # Obtener IDs (ej. [0, 0, 1])
                    detection_counts[class_names[0]] += int((detected_class_ids == 0).sum())
                    detection_counts[class_names[1]] += int((detected_class_ids == 1).sum())
                except Exception as e:
                    st.warning(f"Error contando clases en {filename}: {e}")

                metadata = metadata_map.get(filename)

                if metadata:
                    lat = metadata.get('lat')
                    lon = metadata.get('lon')

                    if lat and lon:
                        # Añadir la tupla (lat, lon) al set. Los sets manejan duplicados.
                        unique_locations.add((lat, lon))

                        # Obtener la imagen con las detecciones dibujadas
                        # results.plot() automáticamente usará diferentes colores para cada clase
                        plotted_image_bgr = results.plot(pil=False) 
                        images_with_detections.append((filename, plotted_image_bgr))
                else:
                    st.warning(f"¡Detección en {filename}, pero no se encontraron metadatos!")

            progress_bar.progress((i + 1) / total_images)

        # --- Limpiar y Mostrar Resultados ---
        status_text.empty()
        progress_bar.empty()

        # --- ¡MODIFICACIÓN! ---
        # Convertir el set de ubicaciones a una lista de dicts para el DataFrame
        locations_with_detections = [{'lat': lat, 'lon': lon} for lat, lon in unique_locations]
        total_instances = sum(detection_counts.values())

        if locations_with_detections:
            st.success(f"¡Análisis completo! Se encontraron {total_instances} instancias de graffiti en {len(locations_with_detections)} ubicaciones.")

            # --- ¡MODIFICACIÓN! ---
            # Mostrar los contadores en métricas
            st.subheader("Resumen de Detecciones (Instancias)")
            col1, col2 = st.columns(2)
            col1.metric(f"Graffiti {class_names[0].capitalize()}", f"{detection_counts[class_names[0]]} instancias")
            col2.metric(f"Graffiti {class_names[1].capitalize()}", f"{detection_counts[class_names[1]]} instancias")

            df_locations = pd.DataFrame(locations_with_detections)

            st.subheader("Mapa de Detecciones de Graffiti")
            st.map(df_locations, zoom=11, use_container_width=True)

            st.subheader("Galería de Detecciones")
            with st.expander("Ver todas las imágenes con detecciones"):
                for filename, image in images_with_detections:
                    # results.plot() ya dibujó ambas clases con diferentes colores
                    st.image(image, caption=filename, use_column_width=True, channels="BGR")
        else:
            st.info("Análisis completo. No se encontró graffiti en ninguna imagen.")


# --- Punto de entrada para ejecutar el script ---
if __name__ == "__main__":
    main()