import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import torch

def generate_embeddings(database_dir, model_path, force_recompute=False):
    """
    Generates embeddings for images in the directory and saves them to cache.
    """
    db_path = Path(database_dir)
    cache_path = db_path / "embeddings_cache.npz"
    
    if cache_path.exists() and not force_recompute:
        print(f"Cache already exists at {cache_path}.")
        print("Use --force to overwrite.")
        return

    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Scanning images in {database_dir}...")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in db_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print("No images found in database directory.")
        return

    embeddings_list = []
    filenames_list = []
    
    print(f"Generating embeddings for {len(image_files)} images...")
    for img_file in tqdm(image_files, desc="Embedding"):
        try:
            emb = model.embed(str(img_file))
            
            if isinstance(emb, list):
                emb = emb[0]
            
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
                
            emb = emb.flatten()
            
            embeddings_list.append(emb)
            filenames_list.append(img_file.name)
            
        except Exception as e:
            print(f"Error embedding {img_file.name}: {e}")

    if not embeddings_list:
        print("Failed to generate any embeddings.")
        return

    embeddings = np.stack(embeddings_list)
    filenames = np.array(filenames_list)
    
    print(f"Saving cache to {cache_path}...")
    np.savez(cache_path, embeddings=embeddings, filenames=filenames)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for graffiti images.")
    
    parser.add_argument("--database", type=str, 
                        default="/media/samuel/SSD/medellin_panoramas_recortados/inferencia/crops_artistico",
                        help="Directory containing the graffiti images.")
    
    parser.add_argument("--model", type=str, 
                        default="model/best.pt",
                        help="Path to the YOLOv8 model.")
    
    parser.add_argument("--force", action="store_true", help="Force recomputation of embeddings.")
    
    args = parser.parse_args()
    
    generate_embeddings(args.database, args.model, args.force)
