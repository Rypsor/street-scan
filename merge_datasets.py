import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def merge_datasets():
    # Rutas base
    base_dir = Path("/home/samuel/Descargas")
    src1_dir = base_dir / "graffiti"
    src2_dir = base_dir / "graffiti2"
    dest_dir = base_dir / "graffiti_combined"

    print(f"Combinando datasets (CLEANED):")
    print(f"  Source 1: {src1_dir} (Filtering class 0, remapping 1->0, 2->1)")
    print(f"  Source 2: {src2_dir} (Copying as is)")
    print(f"  Target:   {dest_dir}")

    if dest_dir.exists():
        print(f"Limpiando directorio destino existente...")
        shutil.rmtree(dest_dir)
    
    dest_dir.mkdir(parents=True)

    # 1. Procesar Source 1 (graffiti)
    # Schema original: 0=Graffiti, 1=artistico, 2=vandalico
    # Target schema: 0=artistico, 1=vandalico
    print("\nProcesando Source 1 (graffiti)...")
    
    count_src1_imgs = 0
    count_src1_lbls = 0
    
    for split in ['train', 'valid', 'test']:
        src_split_imgs = src1_dir / split / "images"
        src_split_lbls = src1_dir / split / "labels"
        
        if not src_split_imgs.exists():
            continue
            
        dest_split_imgs = dest_dir / split / "images"
        dest_split_lbls = dest_dir / split / "labels"
        dest_split_imgs.mkdir(parents=True, exist_ok=True)
        dest_split_lbls.mkdir(parents=True, exist_ok=True)
        
        files = list(src_split_imgs.glob("*"))
        print(f"  Analizando {len(files)} imágenes en {split}...")
        
        for img_file in tqdm(files):
            label_file = src_split_lbls / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                # Si no tiene label, ¿la copiamos? Asumiremos que es background image o error.
                # Si queremos ser estrictos con "eliminar rastro de Graffiti", mejor chequeamos.
                # Pero si no tiene label, no tiene Graffiti. Copiamos.
                shutil.copy2(img_file, dest_split_imgs / img_file.name)
                continue
                
            # Procesar label
            new_lines = []
            has_valid_classes = False
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    cls_id = int(parts[0])
                    
                    # Filtrar clase 0 (Graffiti)
                    if cls_id == 0:
                        continue
                    
                    # Remapear
                    # 1 (artistico) -> 0
                    # 2 (vandalico) -> 1
                    if cls_id == 1:
                        new_id = 0
                    elif cls_id == 2:
                        new_id = 1
                    else:
                        # Clase desconocida?
                        continue
                        
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                    has_valid_classes = True
            
            # Si quedaron clases válidas, guardamos
            if has_valid_classes:
                dest_label_path = dest_split_lbls / label_file.name
                with open(dest_label_path, 'w') as f_out:
                    f_out.writelines(new_lines)
                
                # Copiar imagen
                shutil.copy2(img_file, dest_split_imgs / img_file.name)
                
                count_src1_imgs += 1
                count_src1_lbls += 1
            # Si no quedaron clases (solo tenía Graffiti), NO copiamos nada.

    print(f"  Source 1 procesado: {count_src1_imgs} imágenes conservadas.")

    # 2. Procesar Source 2 (graffiti2)
    # Schema original: 0=artistico, 1=vandalico
    # Target schema: 0=artistico, 1=vandalico (Match!)
    print("\nProcesando Source 2 (graffiti2)...")
    
    src2_train_imgs = src2_dir / "train" / "images"
    src2_train_lbls = src2_dir / "train" / "labels"
    
    dest_train_imgs = dest_dir / "train" / "images"
    dest_train_lbls = dest_dir / "train" / "labels"
    # Ya existen por el paso anterior
    
    files = list(src2_train_imgs.glob("*"))
    print(f"  Añadiendo {len(files)} imágenes de {src2_dir.name}...")
    
    count_src2 = 0
    for img_file in tqdm(files):
        # Prefijo para evitar colisiones
        new_name = f"src2_{img_file.name}"
        dest_img_path = dest_train_imgs / new_name
        shutil.copy2(img_file, dest_img_path)
        
        label_file = src2_train_lbls / f"{img_file.stem}.txt"
        if label_file.exists():
            dest_label_path = dest_train_lbls / f"src2_{img_file.stem}.txt"
            shutil.copy2(label_file, dest_label_path)
            
        count_src2 += 1
            
    print(f"  Añadidas {count_src2} imágenes de Source 2.")

    # 3. Crear data.yaml
    print("\nGenerando data.yaml...")
    data_config = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 2,
        'names': ['artistico', 'vandalico']
    }
    
    with open(dest_dir / "data.yaml", 'w') as f:
        yaml.dump(data_config, f, default_flow_style=None)
        
    print(f"Dataset combinado (CLEANED) creado exitosamente en: {dest_dir}")

if __name__ == "__main__":
    merge_datasets()
