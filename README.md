# Street Scan 🗺️

A web application for detecting and visualizing graffiti on the streets of Medellín using a YOLOv8 model and geolocated images.

Online demo: https://rypsor-street-scan-app-mawxkf.streamlit.app/

## System Requirements

- Python 3.8 or higher
- libgl1-mesa-glx (for OpenCV)
- Internet access (to download images from Hugging Face)

## Installation

1. Clone the repository:

git clone https://github.com/Rypsor/street-scan.git
cd street-scan

2. Create a virtual environment:

python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

3. Install dependencies:

pip install -r requirements.txt

## Usage

1. Start the application:

streamlit run app.py

2. In the web interface:

- Select an area on the map
- Adjust detection thresholds
- Choose how many images to process
- Click “Start Analysis”

## Features

- Detection of two types of graffiti: artistic and vandalism
- Map visualization of detection locations
- Image gallery with marked detections
- Direct links to Google Maps for each location
- Area selection using a drawing tool
- Random sampling of images from the selected area

## Project Structure

street-scan/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies
├── model/
│   └── best.pt            # Trained YOLOv8 model
├── imagenes_medellin/
│   ├── metadata_muestra.json     # Image metadata
│   └── imagenes_muestreadas/     # Folder with sampled images
└── mapas/
    ├── mapa_enumerado_bboxes.html
    └── mapa_rectangulo.html

## AI Model

The detector uses YOLOv8 trained to identify:

- Artistic graffiti
- Vandalism graffiti

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Links

### Medellín Images
https://huggingface.co/datasets/Rypsor/calles-medellin

### Training Images
https://app.roboflow.com/workspace-h90hn/graf-fxodj-bbro0/4
