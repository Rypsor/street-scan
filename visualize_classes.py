import cv2
import random
from pathlib import Path
import yaml

def visualize_dataset_classes(dataset_name, dataset_path, class_names, output_dir):
    print(f"Visualizing {dataset_name}...")
    
    images_dir = dataset_path / "train" / "images"
    labels_dir = dataset_path / "train" / "labels"
    
    # Encontrar 5 ejemplos por clase
    samples_per_class = 5
    found_counts = {i: 0 for i in range(len(class_names))}
    
    # Listar todos los labels y barajar para variedad
    label_files = list(labels_dir.glob("*.txt"))
    random.shuffle(label_files)
    
    for label_file in label_files:
        # Chequear si ya tenemos suficientes de todas las clases
        if all(c >= samples_per_class for c in found_counts.values()):
            break
            
        # Leer clases en este archivo
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        # Identificar qué clases útiles tiene este archivo
        useful_classes_in_file = set()
        for line in lines:
            try:
                cls_id = int(line.split()[0])
                if cls_id in found_counts and found_counts[cls_id] < samples_per_class:
                    useful_classes_in_file.add(cls_id)
            except:
                continue
        
        if not useful_classes_in_file:
            continue
            
        # Cargar imagen
        img_stem = label_file.stem
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            p = images_dir / f"{img_stem}{ext}"
            if p.exists():
                image_path = p
                break
        
        if image_path is None:
            continue
            
        img = cv2.imread(str(image_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Dibujar bboxes
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            
            # Coordenadas YOLO
            cx, cy, bw, bh = map(float, parts[1:5])
            
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            color = (0, 255, 0) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{class_names[cls_id]} ({cls_id})"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        # Guardar y actualizar contadores
        # Guardamos la imagen una vez, pero cuenta para todas las clases útiles que contiene
        # Para evitar duplicados en el reporte si una imagen tiene varias clases, 
        # podemos guardarla con un nombre único y referenciarla, o guardarla por clase.
        # Para simplificar, guardaremos una copia por cada clase que "contribuye" a completar.
        
        for cls_id in useful_classes_in_file:
            idx = found_counts[cls_id]
            out_name = f"vis_{dataset_name}_{class_names[cls_id]}_{idx}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), img)
            print(f"  Saved {out_name}")
            found_counts[cls_id] += 1

def main():
    artifacts_dir = Path("/home/samuel/.gemini/antigravity/brain/11844627-d367-44aa-af82-03b443eaed68")
    base_dir = Path("/home/samuel/Descargas")
    
    # Dataset 1: graffiti
    ds1_path = base_dir / "graffiti"
    ds1_names = ['Graffiti', 'artistico', 'vandalico']
    visualize_dataset_classes("graffiti", ds1_path, ds1_names, artifacts_dir)
    
    # Dataset 2: graffiti2
    ds2_path = base_dir / "graffiti2"
    ds2_names = ['artistico', 'vandalico']
    visualize_dataset_classes("graffiti2", ds2_path, ds2_names, artifacts_dir)

if __name__ == "__main__":
    main()
