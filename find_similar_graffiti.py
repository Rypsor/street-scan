import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch

def find_similar(query_image_path, database_dir, model_path, top_k=5):
    db_path = Path(database_dir)
    cache_path = db_path / "embeddings_cache.npz"
    
    # 1. Load Embeddings
    if not cache_path.exists():
        print(f"Error: Embeddings cache not found at {cache_path}")
        print("Please run 'generate_embeddings.py' first.")
        return

    print(f"Loading embeddings from cache: {cache_path}")
    try:
        data = np.load(cache_path)
        db_embeddings = data['embeddings']
        db_filenames = data['filenames']
    except Exception as e:
        print(f"Error loading cache: {e}")
        return
        
    if len(db_embeddings) == 0:
        print("Database embeddings are empty.")
        return

    # 2. Load Model (only needed for query embedding)
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Get Query Embedding
    print(f"Processing query image: {query_image_path}")
    try:
        query_emb = model.embed(query_image_path)
        if isinstance(query_emb, list):
            query_emb = query_emb[0]
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.cpu().numpy()
        query_emb = query_emb.flatten()
    except Exception as e:
        print(f"Error processing query image: {e}")
        return

    # 4. Compute Cosine Similarity
    norm_db = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    norm_query = np.linalg.norm(query_emb)
    
    norm_db[norm_db == 0] = 1e-10
    if norm_query == 0:
        norm_query = 1e-10
        
    normalized_db = db_embeddings / norm_db
    normalized_query = query_emb / norm_query
    
    similarities = np.dot(normalized_db, normalized_query)
    
    # 5. Get Top K
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print(f"\n--- Top {top_k} Similar Images ---")
    results = []
    for idx in top_indices:
        score = similarities[idx]
        fname = db_filenames[idx]
        print(f"{fname}: Score {score:.4f}")
        results.append((fname, score))
        
    # 6. Visualize
    visualize_results(query_image_path, database_dir, results)

def visualize_results(query_path, db_dir, results):
    try:
        query_img = cv2.imread(query_path)
        if query_img is None:
            print("Could not load query image for visualization.")
            return

        target_h = 300
        h, w = query_img.shape[:2]
        scale = target_h / h
        query_img = cv2.resize(query_img, (int(w * scale), target_h))
        
        query_img = cv2.copyMakeBorder(query_img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(query_img, "Query", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        match_imgs = []
        for fname, score in results:
            path = os.path.join(db_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                scale = target_h / h
                img = cv2.resize(img, (int(w * scale), target_h))
                
                img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(img, f"{score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                match_imgs.append(img)
        
        if match_imgs:
            matches_concat = np.hstack(match_imgs)
            total_w = max(query_img.shape[1], matches_concat.shape[1])
            
            pad_w = total_w - query_img.shape[1]
            if pad_w > 0:
                query_img = cv2.copyMakeBorder(query_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                
            pad_w = total_w - matches_concat.shape[1]
            if pad_w > 0:
                matches_concat = cv2.copyMakeBorder(matches_concat, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            final_vis = np.vstack([query_img, matches_concat])
            
            out_path = "similarity_result.jpg"
            cv2.imwrite(out_path, final_vis)
            print(f"Visualization saved to {out_path}")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar graffitis using pre-computed embeddings.")
    
    parser.add_argument("query_image", type=str, help="Path to the query image.")
    
    parser.add_argument("--database", type=str, 
                        default="/media/samuel/SSD/medellin_panoramas_recortados/inferencia/crops_artistico",
                        help="Directory containing the graffiti database.")
    
    parser.add_argument("--model", type=str, 
                        default="model/best.pt",
                        help="Path to the YOLOv8 model (for query embedding).")
    
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar images to return.")
    
    args = parser.parse_args()
    
    find_similar(args.query_image, args.database, args.model, args.top_k)
