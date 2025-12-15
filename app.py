import streamlit as st
import pandas as pd
import os
import json
from ultralytics import YOLO
from PIL import Image
import random
import torch
# apfrom huggingface_hub import hf_hub_download, list_repo_files
import folium
from streamlit_folium import st_folium
import numpy as np

# --- 1. Definición de Constantes y Rutas ---
MODEL_PATH = 'model/best.pt' 
HF_REPO_ID = "Rypsor/calles-medellin"
HF_REPO_TYPE = "dataset"
METADATA_FILENAME = "metadata_muestra.json" 
EMBEDDINGS_PATH = "/media/samuel/SSD/medellin_panoramas_recortados/inferencia/crops_artistico/embeddings_cache.npz"
IMAGES_DIR = "/media/samuel/SSD/medellin_panoramas_recortados/inferencia/crops_artistico"
DETECTIONS_PATH = "/media/samuel/SSD/medellin_panoramas_recortados/inferencia/all_detections/detections_artistico.json" 

# --- 2. Funciones de Carga (Sin cambios) ---
# Eliminamos cache_resource para evitar problemas de estado entre ejecuciones
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

# --- Funciones de Carga Local ---
@st.cache_data
def load_embeddings(path):
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data['embeddings'], data['filenames']

@st.cache_data
def load_confidence_map(json_path):
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        conf_map = {}
        for i, item in enumerate(data):
            conf_map[i] = item.get('score', 0.0)
        return conf_map
    except Exception as e:
        st.error(f"Error loading confidence map: {e}")
        return {}

def main():
    st.title("🗺️ Detector de Graffiti con Geo-localización")

    # --- Iniciar la app ---
    model = load_yolo_model(MODEL_PATH)
    cached_embeddings, cached_filenames = load_embeddings(EMBEDDINGS_PATH)
    confidence_map = load_confidence_map(DETECTIONS_PATH)

    if model is None:
        st.error("No se pudo cargar el modelo YOLO.")
        st.stop()

    if cached_embeddings is None:
        st.error(f"No se encontraron los embeddings en {EMBEDDINGS_PATH}. Ejecuta primero 'generate_embeddings.py'.")
        st.stop()

    # --- Sección de Búsqueda ---
    uploaded_file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar imagen subida
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Imagen Subida", use_container_width=True)
        
        with col2:
            st.write("### Configuración de Búsqueda")
            # --- Threshold Slider ---
            # Paso del slider ajustado a 0.02 como se planeó (el usuario pidió 0.2 pero para el rango 0.7-1.0 es muy grande)
            min_db_conf = st.slider("Umbral de Confianza (Base de Datos)", 0.7, 1.0, 0.8, 0.02, help="Filtrar resultados que provienen de detecciones con baja confianza.")
            
            search_button = st.button("🔎 Buscar Similares", type="primary")

        if search_button:
            with st.spinner("Analizando y buscando..."):
                try:
                    # Generar embedding
                    # Primero intentamos detectar y recortar el graffiti
                    img_array = np.array(image)
                    
                    # Ejecutar inferencia en la imagen subida
                    # Usar umbral bajo para DEBUG, luego filtraremos
                    results_list = model.predict(image, conf=0.1, verbose=False)
                    results = results_list[0]
                    
                    cropped_image = image # Por defecto usamos la original
                    
                    if hasattr(results, 'boxes') and len(results.boxes) > 0:
                        # Filtrar solo clase 'artistico' (ID 0) Y confianza > 0.5
                        artistico_boxes = [
                            box for box in results.boxes 
                            if int(box.cls[0]) == 0 and float(box.conf[0]) >= 0.5
                        ]
                        
                        if artistico_boxes:
                            # Encontrar la caja con mayor confianza
                            best_box = max(artistico_boxes, key=lambda x: x.conf[0])
                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                            
                            # Recortar
                            cropped_image = image.crop((x1, y1, x2, y2))
                            with col1:
                                st.image(cropped_image, caption=f"Recorte Automático (Conf: {float(best_box.conf[0]):.2f})", use_container_width=True)
                            
                            # Actualizar array para el embedding
                            img_array = np.array(cropped_image)
                        else:
                            st.warning("No se detectó ningún graffiti 'artístico' con confianza > 0.5. Usando imagen completa.")
                    else:
                        st.warning("No se detectó ningún objeto con confianza > 0.5. Usando imagen completa.")
                    
                    query_embedding = model.embed(img_array)
                    
                    if isinstance(query_embedding, list):
                        query_embedding = query_embedding[0]
                    if isinstance(query_embedding, torch.Tensor):
                        query_embedding = query_embedding.cpu().numpy()
                    
                    query_embedding = query_embedding.flatten()
                    
                    # Calcular similitud
                    query_norm = query_embedding / np.linalg.norm(query_embedding)
                    db_norms = np.linalg.norm(cached_embeddings, axis=1, keepdims=True)
                    normalized_db = cached_embeddings / db_norms
                    similarities = np.dot(normalized_db, query_norm)
                    
                    # Filter by confidence
                    import re
                    valid_indices = []
                    
                    for idx, fname in enumerate(cached_filenames):
                        # Extract index from filename: ..._det{i}.jpg
                        match = re.search(r"_det(\d+)\.", fname)
                        if match:
                            det_idx = int(match.group(1))
                            conf = confidence_map.get(det_idx, 0.0)
                            if conf >= min_db_conf:
                                valid_indices.append(idx)
                        else:
                            # If no index found, include it
                            valid_indices.append(idx)
                    
                    if not valid_indices:
                        st.warning(f"No hay imágenes en la base de datos con confianza >= {min_db_conf}.")
                    else:
                        # Filter similarities
                        filtered_similarities = similarities[valid_indices]
                        filtered_indices = np.array(valid_indices)
                        
                        # Top 5 from filtered
                        top_k = min(5, len(filtered_indices))
                        # Get indices in the filtered array
                        top_k_local_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
                        # Map back to original indices
                        top_indices = filtered_indices[top_k_local_indices]

                        st.subheader("Resultados Similares")
                        st.divider()
                        
                        cols = st.columns(top_k)
                        map_data = [] # Initialize map_data here
                        
                        for i, idx in enumerate(top_indices):
                            score = similarities[idx]
                            fname = cached_filenames[idx]
                            img_path = os.path.join(IMAGES_DIR, fname)
                            
                            # Get original confidence
                            match = re.search(r"_det(\d+)\.", fname)
                            orig_conf = 0.0
                            if match:
                                orig_conf = confidence_map.get(int(match.group(1)), 0.0)
                            
                            # Parsear coordenadas del nombre del archivo
                            # Formato esperado: crop_LAT_LON_ID.jpg
                            lat, lon = None, None
                            try:
                                parts = fname.split('_')
                                if len(parts) >= 3:
                                    lat = float(parts[1])
                                    lon = float(parts[2])
                                    map_data.append({'lat': lat, 'lon': lon, 'name': fname, 'score': score})
                            except Exception:
                                pass
                            
                            with cols[i]:
                                if os.path.exists(img_path):
                                    st.image(img_path, use_container_width=True)
                                    st.markdown(f"**Similitud:** {score:.2f}")
                                    st.markdown(f"**Confianza:** {orig_conf:.2f}")
                                    if lat and lon:
                                        st.markdown(f"[📍 Ver en Google Maps](https://www.google.com/maps?q={lat},{lon})")
                                        st.caption(f"{lat:.5f}, {lon:.5f}")
                                else:
                                    st.warning(f"Imagen no encontrada: {fname}")

                        # Mostrar mapa si hay coordenadas
                        if map_data:
                            st.subheader("📍 Ubicación de los Graffitis Similares")
                            df_map = pd.DataFrame(map_data)
                            st.map(df_map, zoom=12, use_container_width=True)
                                    

                except Exception as e:
                    st.error(f"Error durante la búsqueda: {e}")
                    import traceback
                    st.text(traceback.format_exc())


# --- Punto de entrada para ejecutar el script ---
if __name__ == "__main__":
    main()