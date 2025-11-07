import streamlit as st
import pandas as pd
import os
import json
from ultralytics import YOLO
from PIL import Image
import random
import torch
from huggingface_hub import hf_hub_download, list_repo_files
import folium
from streamlit_folium import st_folium

# --- 1. Definición de Constantes y Rutas ---
MODEL_PATH = 'model/best.pt' 
HF_REPO_ID = "Rypsor/calles-medellin"
HF_REPO_TYPE = "dataset"
METADATA_FILENAME = "metadata_muestra.json" 

# --- 2. Funciones de Carga (Sin cambios) ---
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"Error: No se encontró el modelo en {path}.")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo desde {path}: {e}")
        return None

@st.cache_data
def get_synced_metadata_list(repo_id, metadata_filename):
    st.info(f"Descargando metadatos (JSON) desde Hugging Face...")
    try:
        json_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=metadata_filename)
        with open(json_path, 'r') as f:
            full_metadata_list = json.load(f)
        st.success(f"¡Metadatos cargados! {len(full_metadata_list)} registros.")
    except Exception as e:
        st.error(f"Error fatal al descargar metadatos: {e}")
        return None, None

    st.info("Sincronizando con los archivos reales del repositorio...")
    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        available_images = set()
        for f in repo_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '/' not in f:
                available_images.add(f)
       # st.success(f"{len(available_images)} imágenes realmente encontradas en la raíz de HF.")
    except Exception as e:
        st.error(f"Error al listar archivos de Hugging Face: {e}")
        return None, None
        
    synced_metadata_list = [
        item for item in full_metadata_list 
        if item.get('filename') in available_images
    ]
    st.info(f"¡Sincronización completa! {len(synced_metadata_list)} imágenes disponibles.")
    
    if len(synced_metadata_list) == 0:
        st.error("Error: 0 imágenes disponibles.")
        return None, None
        
    avg_lat = sum(item['lat'] for item in synced_metadata_list if 'lat' in item) / len(synced_metadata_list)
    avg_lon = sum(item['lon'] for item in synced_metadata_list if 'lon' in item) / len(synced_metadata_list)
    map_center = [avg_lat, avg_lon]
        
    return synced_metadata_list, map_center

def filter_metadata_by_bounds(metadata_list, bounds):
    min_lon = bounds['_southWest']['lng']
    min_lat = bounds['_southWest']['lat']
    max_lon = bounds['_northEast']['lng']
    max_lat = bounds['_northEast']['lat']
    filtered_list = [
        item for item in metadata_list
        if (min_lat <= item['lat'] <= max_lat) and (min_lon <= item['lon'] <= max_lon)
    ]
    return filtered_list

# --- Inicializador de Estado de Sesión (Sin cambios) ---
def initialize_session_state(class_names):
    """Define las variables por defecto en la sesión de Streamlit."""
    if 'map_key' not in st.session_state:
        st.session_state.map_key = 0 
    if 'filtered_metadata' not in st.session_state:
        st.session_state.filtered_metadata = [] 
    if 'images_with_detections' not in st.session_state:
        st.session_state.images_with_detections = [] 
    if 'locations_with_detections' not in st.session_state:
        st.session_state.locations_with_detections = [] 
    if 'detection_counts' not in st.session_state:
        st.session_state.detection_counts = {name: 0 for name in class_names.values()}

def main():
    st.title("🗺️ Detector de Graffiti con Geo-localización")

    # --- Iniciar la app ---
    model = load_yolo_model(MODEL_PATH)
    metadata_list, map_center = get_synced_metadata_list(HF_REPO_ID, METADATA_FILENAME)

    if model is None or metadata_list is None:
        st.warning("La aplicación no puede iniciar. Fallo al cargar datos/modelo.")
        return
        
    try:
        class_names = model.names
    except Exception:
        class_names = {0: 'artistico', 1: 'vandalico'}
    st.info(f"Modelo cargado. Clases detectadas: {class_names}")

    initialize_session_state(class_names)

    # --- Paso 1: Mostrar el Mapa de Selección ---
    st.subheader("Paso 1: Selecciona un área en el mapa")
    
    col1_map, col2_map = st.columns([4, 1])
    with col1_map:
        st.info("Usa la herramienta (■) para dibujar un área. Dibuja de nuevo para cambiar.")
    with col2_map:
        if st.button("Limpiar Selección y Resultados"):
            st.session_state.map_key += 1 
            st.session_state.filtered_metadata = [] 
            st.session_state.images_with_detections = [] 
            st.session_state.locations_with_detections = [] 
            st.session_state.detection_counts = {name: 0 for name in class_names.values()} 
            st.rerun() 

    m = folium.Map(location=map_center, zoom_start=12)
    folium.plugins.Draw(
        export=False,
        draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True}
    ).add_to(m)

    map_data = st_folium(m, key=str(st.session_state.map_key), width=700, height=500)

    # --- Paso 2: Filtrar la lista de imágenes ---
    if map_data.get('all_drawings') and len(map_data['all_drawings']) > 0:
        last_drawing = map_data['all_drawings'][-1]
        if last_drawing['geometry']['type'] == 'Polygon':
            coords = last_drawing['geometry']['coordinates'][0]
            lons = [point[0] for point in coords]
            lats = [point[1] for point in coords]
            bounds = {
                '_southWest': {'lat': min(lats), 'lng': min(lons)},
                '_northEast': {'lat': max(lats), 'lng': max(lons)}
            }
            
            st.session_state.filtered_metadata = filter_metadata_by_bounds(metadata_list, bounds)
            
            st.session_state.images_with_detections = []
            st.session_state.locations_with_detections = []
            st.session_state.detection_counts = {name: 0 for name in class_names.values()}
            
            st.success(f"¡Área seleccionada! Se encontraron {len(st.session_state.filtered_metadata)} imágenes en esta zona.")
    else:
        st.session_state.filtered_metadata = metadata_list

    total_available = len(st.session_state.filtered_metadata)
    if total_available == 0:
        st.warning("No hay imágenes en el área seleccionada para procesar.")
    else:
        st.info(f"Encontradas {total_available} imágenes válidas y sincronizadas en el área.")

        # --- Paso 3: Configuración de Detección ---
        st.subheader("Paso 2: Configura el análisis")
        col1, col2 = st.columns(2)
        with col1:
            conf_artistico = st.slider(f"Umbral para '{class_names[0]}'", 0.0, 1.0, 0.5, 0.05)
        with col2:
            conf_vandalico = st.slider(f"Umbral para '{class_names[1]}'", 0.0, 1.0, 0.3, 0.05)
        
        threshold_map = { 0: conf_artistico, 1: conf_vandalico }
        PRE_FILTER_CONF = min(conf_artistico, conf_vandalico, 0.05)

        max_to_process = st.number_input(
            "Número de imágenes a procesar (al azar)",
            min_value=1,
            max_value=total_available,
            value=min(20, total_available),
            step=1,
        )

        # --- Paso 4: Botón de Inicio ---
        st.subheader("Paso 3: Iniciar análisis")
        if st.button("🚀 Iniciar Análisis de Imágenes"):
            total_images = min(int(max_to_process), total_available)
            items_to_process = random.sample(st.session_state.filtered_metadata, k=total_images)

            progress_bar = st.progress(0)
            status_text = st.empty()
            st.info(f"Se procesarán {total_images} imágenes al azar del área seleccionada...")
            
            st.session_state.images_with_detections = []
            st.session_state.locations_with_detections = []
            st.session_state.detection_counts = {name: 0 for name in class_names.values()}
            unique_locations = set()

            for i, item in enumerate(items_to_process):
                filename = item['filename']
                status_text.text(f"Procesando {i+1}/{total_images}: {filename} (Descargando...)")
                image_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}"
                
                try:
                    results = model.predict(image_url, classes=[0, 1], conf=PRE_FILTER_CONF, verbose=False, save = False)[0]
                except Exception as e:
                    st.warning(f"Error procesando {filename}. Saltando. Error: {e}")
                    continue

                if len(results.boxes) > 0:
                    indices_to_keep = [] 
                    for j in range(len(results.boxes)):
                        box = results.boxes[j]
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if confidence >= threshold_map.get(class_id, 0.0):
                            indices_to_keep.append(j)
                    results.boxes = results.boxes[indices_to_keep]
                
                if len(results.boxes) > 0:
                    try:
                        detected_class_ids = results.boxes.cls.cpu().numpy()
                        st.session_state.detection_counts[class_names[0]] += int((detected_class_ids == 0).sum())
                        st.session_state.detection_counts[class_names[1]] += int((detected_class_ids == 1).sum())
                    except Exception: pass
                    lat = item.get('lat')
                    lon = item.get('lon')
                    if lat and lon:
                        unique_locations.add((lat, lon))
                        plotted_image_bgr = results.plot(pil=False) 
                        
                        # --- ¡CAMBIO 1! ---
                        # Guardamos (filename, image, lat, lon) en lugar de solo (filename, image)
                        st.session_state.images_with_detections.append((filename, plotted_image_bgr, lat, lon))
                        # --------------------

                progress_bar.progress((i + 1) / total_images)

            st.session_state.locations_with_detections = [{'lat': lat, 'lon': lon} for lat, lon in unique_locations]
            status_text.empty()
            progress_bar.empty()
            st.success("¡Análisis completado!")

    # --- Mostrar Resultados ---
    
    total_instances = sum(st.session_state.detection_counts.values())

    if total_instances > 0:
        st.subheader("Resumen de Detecciones (Instancias)")
        col1, col2 = st.columns(2)
        col1.metric(f"Graffiti {class_names[0].capitalize()}", f"{st.session_state.detection_counts[class_names[0]]} instancias")
        col2.metric(f"Graffiti {class_names[1].capitalize()}", f"{st.session_state.detection_counts[class_names[1]]} instancias")
        
    if st.session_state.locations_with_detections:
        df_locations = pd.DataFrame(st.session_state.locations_with_detections)
        st.subheader("Mapa de Detecciones de Graffiti (en el área)")
        st.map(df_locations, zoom=11, use_container_width=True)
    

    # --- ¡CAMBIO 2! ---
    if st.session_state.images_with_detections:
        st.subheader("Galería de Detecciones")
        with st.expander("Ver todas las imágenes con detecciones"):
            
            # Desempaquetamos los 4 valores que guardamos
            for filename, image, lat, lon in st.session_state.images_with_detections:
                st.image(image, caption=filename, use_column_width=True, channels="BGR")
                
                # Mostramos las coordenadas
                st.caption(f"Coordenadas: {lat}, {lon}")
                
                # Creamos el enlace a Google Maps
                map_link = f"https://www.google.com/maps?q={lat},{lon}"
                st.markdown(f"[Ver ubicación en Google Maps]({map_link})")
                
                st.divider() # Un separador visual
    # --------------------

    
    if total_available > 0 and total_instances == 0:
        last_button = st.session_state.get('st.button.last_used_label', '')
        if last_button != "🚀 Iniciar Análisis de Imágenes" and last_button != "Limpiar Selección y Resultados":
            st.info("Usa los controles de arriba para iniciar un análisis.")
        elif last_button == "🚀 Iniciar Análisis de Imágenes":
            st.info("No se encontraron detecciones en la última búsqueda.")


# --- Punto de entrada para ejecutar el script ---
if __name__ == "__main__":
    main()