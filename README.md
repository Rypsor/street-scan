# Street Scan 🗺️

Aplicación web para detectar y visualizar graffiti en las calles de Medellín usando modelo YOLOv8 e imágenes geolocalizadas.

## Requisitos del Sistema

- Python 3.8 o superior
- libgl1-mesa-glx (para OpenCV)
- Acceso a internet (para descargar imágenes de Hugging Face)

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/Rypsor/street-scan.git
cd street-scan
```

1. Crea un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

1. Inicia la aplicación:

```bash
streamlit run app.py
```

1. En la interfaz web:

   - Selecciona un área en el mapa
   - Ajusta los umbrales de detección
   - Elige cuántas imágenes procesar
   - Haz clic en "Iniciar Análisis"

## Funcionalidades

- Detección de dos tipos de graffiti: artístico y vandálico
- Visualización en mapa de las ubicaciones con detecciones
- Galería de imágenes con las detecciones marcadas
- Enlaces directos a Google Maps para cada ubicación
- Selección de área mediante herramienta de dibujo
- Muestreo aleatorio de imágenes del área seleccionada

## Estructura del Proyecto

```plaintext
street-scan/
├── app.py                 # Aplicación principal Streamlit
├── requirements.txt       # Dependencias de Python
├── packages.txt          # Dependencias del sistema
├── model/
│   └── best.pt          # Modelo YOLOv8 entrenado
├── imagenes_medellin/
│   ├── metadata_muestra.json     # Metadatos de imágenes
│   └── imagenes_muestreadas/     # Carpeta de imágenes
└── mapas/
    ├── mapa_enumerado_bboxes.html
    └── mapa_rectangulo.html
```

## Modelo de IA

El detector utiliza YOLOv8 entrenado para identificar:

- Graffiti artístico
- Graffiti vandálico


## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Enlaces
### Imágenes de Medellín
https://huggingface.co/datasets/Rypsor/calles-medellin

### Imágenes usadas en el entrenamiento
https://app.roboflow.com/workspace-h90hn/graf-fxodj-bbro0/4