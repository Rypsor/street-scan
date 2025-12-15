import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import json
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
DEFAULT_DETECTIONS_DIR = "/media/samuel/SSD/medellin_panoramas_recortados/inferencia/all_detections"
DEFAULT_IMAGES_ROOT = "/media/samuel/SSD/medellin_panoramas_recortados"

st.set_page_config(layout="wide", page_title="Street Scan Visualization")

st.title("Street Scan - Visualización de Detecciones")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Configuración")
detections_dir = st.sidebar.text_input("Directorio de Detecciones (JSONs)", DEFAULT_DETECTIONS_DIR)
images_root = st.sidebar.text_input("Directorio Raíz de Imágenes", DEFAULT_IMAGES_ROOT)

# --- DATA LOADING ---
@st.cache_data
def load_data(det_path, img_root):
    det_path = Path(det_path)
    img_root = Path(img_root)
    
    if not det_path.exists():
        return None, f"El directorio de detecciones no existe: {det_path}"
    
    all_detections = []
    
    json_files = list(det_path.glob("detections_*.json"))
    if not json_files:
        return None, "No se encontraron archivos detections_*.json"
        
    for json_file in json_files:
        class_name = json_file.stem.replace("detections_", "")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for item in data:
                img_name = item.get("image", "")
                rel_path = item.get("relative_path", "")
                bbox = item.get("bbox", [])
                score = item.get("score", 0.0)
                
                try:
                    safe_name = Path(img_name).name
                    parts = safe_name.split('_')
                    if len(parts) >= 4:
                        lat = float(parts[1])
                        lon = float(parts[2])
                    else:
                        continue 
                except ValueError:
                    continue 

                full_path = None
                if rel_path:
                    full_path = img_root / rel_path

                all_detections.append({
                    "class": class_name,
                    "lat": lat,
                    "lon": lon,
                    "image_name": img_name,
                    "relative_path": rel_path,
                    "full_path": str(full_path) if full_path else None,
                    "bbox": bbox,
                    "score": score
                })
                
        except Exception as e:
            st.error(f"Error leyendo {json_file.name}: {e}")
            
    if not all_detections:
        return None, "No se encontraron detecciones válidas con coordenadas."
        
    return pd.DataFrame(all_detections), None

df, error = load_data(detections_dir, images_root)

if error:
    st.error(error)
    st.stop()

st.sidebar.success(f"Cargadas {len(df)} detecciones.")

# --- FILTERS ---
classes = df['class'].unique().tolist()
selected_classes = st.sidebar.multiselect("Filtrar por Clase", classes, default=classes)
max_images = st.sidebar.slider("Máximo de imágenes a mostrar en el área", min_value=1, max_value=1000, value=50)

# Botón para limpiar selección
if st.sidebar.button("🗑️ Limpiar área seleccionada"):
    st.session_state.selected_bounds = None
    st.rerun()

filtered_df = df[df['class'].isin(selected_classes)]

# --- MAP STATE INITIALIZATION ---
current_filter_hash = f"{sorted(selected_classes)}_{max_images}_{len(filtered_df)}"

if 'last_filter_hash' not in st.session_state or st.session_state.last_filter_hash != current_filter_hash:
    st.session_state.last_filter_hash = current_filter_hash
    st.session_state.selected_bounds = None 
    
    if not filtered_df.empty:
        # Default center (City view)
        st.session_state.map_center = [filtered_df['lat'].mean(), filtered_df['lon'].mean()]
        st.session_state.map_zoom = 13
    else:
        st.session_state.map_center = [6.2442, -75.5812]
        st.session_state.map_zoom = 12

# --- MAP RENDER ---
st.subheader("Mapa de Detecciones")

if not filtered_df.empty:
    
    # Creamos el mapa
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
    
    unique_images_in_area = pd.DataFrame()

    # LOGICA DE DIBUJO Y FILTRADO
    if st.session_state.selected_bounds:
        min_lat, max_lat, min_lon, max_lon = st.session_state.selected_bounds
        
        # 1. Dibujar el rectángulo guardado en el mapa (CLAVE: persistir el dibujo)
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color='#3388ff',
            fill=True,
            fillOpacity=0.2,
            weight=2
        ).add_to(m)
        
        # 2. Ajustar el mapa visualmente al área seleccionada
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        # 3. Filtrar datos
        area_df = filtered_df[
            (filtered_df['lat'] >= min_lat) & (filtered_df['lat'] <= max_lat) &
            (filtered_df['lon'] >= min_lon) & (filtered_df['lon'] <= max_lon)
        ]
        
        unique_images_in_area = area_df.drop_duplicates(subset=['image_name'])
        total_in_area = len(unique_images_in_area)
        
        if total_in_area > max_images:
            st.warning(f"Se encontraron {total_in_area} imágenes. Mostrando las primeras {max_images}.")
            unique_images_in_area = unique_images_in_area.head(max_images)
        else:
            st.info(f"Mostrando {total_in_area} imágenes en el área.")
        
        dets_grouped = area_df.groupby('image_name')

        for _, row in unique_images_in_area.iterrows():
            img_name = row['image_name']
            if img_name in dets_grouped.groups:
                img_dets = dets_grouped.get_group(img_name)
                det_summary = ", ".join([f"{r['class']} ({r['score']:.2f})" for _, r in img_dets.iterrows()])
            else:
                det_summary = "..."
            
            folium.Marker(
                [row['lat'], row['lon']],
                popup=f"<b>{img_name}</b><br>{det_summary}",
                tooltip=det_summary,
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

    # Draw control siempre visible
    draw = Draw(
        export=False,
        position='topleft',
        draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True},
        edit_options={'poly': {'allowIntersection': False}}
    )
    draw.add_to(m)

    # --- RENDERIZADO DEL MAPA ---
    # Mantenemos returned_objects limitado para evitar el refresh infinito al mover el mapa
    output = st_folium(
        m, 
        width=None, 
        height=500, 
        key="folium_map",
        returned_objects=["all_drawings", "last_object_clicked"] 
    )
    
    # --- EVENT HANDLING ---
    # Solo procesamos nuevos dibujos si NO hay bounds guardados (evita loops)
    if output and output.get('all_drawings') and not st.session_state.selected_bounds:
        last_drawing = output['all_drawings'][-1]
        geometry = last_drawing['geometry']
        
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            
            new_bounds = (min_lat, max_lat, min_lon, max_lon)
            st.session_state.selected_bounds = new_bounds
            st.rerun()

    # --- IMAGE VIEWER ---
    st.subheader("Vista de Detalle")
    
    selected_image_name = None
    
    if output and output.get('last_object_clicked'):
        clicked_lat = output['last_object_clicked']['lat']
        clicked_lon = output['last_object_clicked']['lng']
        
        if not unique_images_in_area.empty:
            selected_row = unique_images_in_area[
                (np.isclose(unique_images_in_area['lat'], clicked_lat, atol=1e-5)) & 
                (np.isclose(unique_images_in_area['lon'], clicked_lon, atol=1e-5))
            ]
            if not selected_row.empty:
                selected_image_name = selected_row.iloc[0]['image_name']
            
    if selected_image_name:
        st.write(f"**Imagen Seleccionada:** {selected_image_name}")
        img_detections = df[df['image_name'] == selected_image_name]
        
        rel_path = img_detections.iloc[0]['relative_path']
        full_path = Path(images_root) / rel_path if rel_path else None
        
        if not full_path or not full_path.exists():
            found = False
            for area_id in range(1, 9): 
                potential_path = Path(images_root) / f"area{area_id}/50m/{selected_image_name}"
                if potential_path.exists():
                    full_path = potential_path
                    found = True
                    break
            if not found:
                st.error(f"Imagen no encontrada: {selected_image_name}")
                full_path = None
        
        if full_path and full_path.exists():
            img = cv2.imread(str(full_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for _, row in img_detections.iterrows():
                    bbox = row['bbox']
                    cls = row['class']
                    conf = row['score']
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (255, 0, 0) 
                    if cls == 'artistico': color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 8)
                    cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                st.image(img, use_container_width=True)
                st.caption(f"Ruta: {full_path}")
            
    elif st.session_state.selected_bounds:
        st.info("Selecciona un marcador rojo en el mapa para ver la imagen.")
    else:
        st.info("Usa la herramienta de rectángulo para filtrar.")

else:
    st.warning("No hay datos para mostrar con los filtros actuales.")