# Street Scan 🗺️

Sistema de detección, clasificación y búsqueda de graffiti urbano en las calles de Medellín utilizando visión por computadora y geolocalización.

**Demo en línea:** https://rypsor-street-scan-app-mawxkf.streamlit.app/

---

## 📋 Descripción

Este proyecto implementa un pipeline completo para:

1. **Detección de graffiti** en imágenes de Street View usando YOLOv8
2. **Clasificación** entre graffiti artístico y vandálico
3. **Búsqueda por similitud** visual usando embeddings
4. **Visualización geoespacial** en mapas interactivos

---

## 🚀 Características

- ✅ Detección de dos tipos de graffiti: **artístico** y **vandálico**
- ✅ Precisión del modelo superior al **95%** (mAP@0.5: 0.966)
- ✅ Búsqueda de graffitis similares con puntuaciones >0.93
- ✅ Visualización en mapa con marcadores geolocalizados
- ✅ Enlaces directos a Google Maps para cada ubicación
- ✅ Interfaz web interactiva con Streamlit

---

## 📁 Estructura del Proyecto

```
street-scan/
├── app.py                      # Aplicación Streamlit principal
├── model/
│   ├── best.pt                 # Modelo YOLOv8 entrenado
│   └── test_results/           # Métricas y curvas de evaluación
├── test_images/                # Imágenes de prueba
│
├── # Scripts de Entrenamiento
├── entrenamiento-del-modelo.ipynb  # Notebook de entrenamiento
├── merge_datasets.py           # Combinar datasets
├── filter_training_images.py   # Filtrado de imágenes
│
├── # Scripts de Inferencia
├── inference_script.py         # Detección en imágenes
├── crop_graffiti.py            # Extracción de recortes
│
├── # Sistema de Embeddings
├── generate_embeddings.py      # Generación de embeddings
├── find_similar_graffiti.py    # Búsqueda por similitud
├── research_embedding.py       # Experimentación
│
├── # Visualización
├── visualize_map.py            # Mapa interactivo con Folium
├── visualize_classes.py        # Distribución de clases
│
├── # Documentación
├── INFORME_RESULTADOS.md       # Informe completo de resultados
├── INFORME_RESULTADOS.pdf      # Versión PDF del informe
└── README.md                   # Este archivo
```

---

## 🛠️ Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- libgl1-mesa-glx (para OpenCV)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/Rypsor/street-scan.git
cd street-scan
```

2. Crear entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## 💻 Uso

### Aplicación Web (Búsqueda por Similitud)

```bash
streamlit run app.py
```

1. Sube una imagen de graffiti
2. Ajusta el umbral de confianza
3. Obtén los 5 graffitis más similares con sus ubicaciones en el mapa

### Búsqueda por Línea de Comandos

```bash
python find_similar_graffiti.py imagen_query.jpg --top_k 5
```

### Generar Embeddings

```bash
python generate_embeddings.py --database /ruta/a/imagenes --force
```

### Ejecutar Inferencia

```bash
python inference_script.py --source /ruta/a/imagenes --output /ruta/salida
```

---

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| mAP@0.5 | **0.966** |
| F1 Score (óptimo) | **0.95** |
| Umbral óptimo | 0.743 |
| Precisión (artístico) | 0.980 |
| Precisión (vandálico) | 0.952 |

Para más detalles, consulta el [Informe de Resultados](INFORME_RESULTADOS.md).

---

## 🤖 Modelo de IA

El detector usa **YOLOv8** entrenado para identificar:

- 🎨 **Graffiti Artístico**: Murales, arte urbano, obras con valor estético
- ⚠️ **Graffiti Vandálico**: Tags, firmas, marcas sin autorización

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT.

---

## 🔗 Enlaces

| Recurso | URL |
|---------|-----|
| Demo en línea | https://rypsor-street-scan-app-mawxkf.streamlit.app/ |
| Dataset Medellín | https://huggingface.co/datasets/Rypsor/calles-medellin |
| Dataset Entrenamiento | https://app.roboflow.com/workspace-h90hn/graf-fxodj-bbro0/4 |
