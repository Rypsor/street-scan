import json
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import os

def crop_detections(detections_path, images_root, output_dir, resize_factor=0.5, target_class="artistico"):
    """
    Crops detections from images based on JSON output.
    
    Args:
        detections_path (str): Path to the JSON file or directory containing detections.
        images_root (str): Root directory to resolve relative paths.
        output_dir (str): Directory to save cropped images.
        resize_factor (float): Factor by which images were resized during inference. 
                               Used to scale coordinates back to original size.
        target_class (str): Class name to filter for (e.g., 'artistico'). 
                            If detections_path is a directory, tries to find 'detections_{target_class}.json'.
    """
    
    det_path = Path(detections_path)
    root_path = Path(images_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which file to load
    json_file = None
    if det_path.is_file():
        json_file = det_path
    elif det_path.is_dir():
        candidate = det_path / f"detections_{target_class}.json"
        if candidate.exists():
            json_file = candidate
        else:
            # Fallback: try to find any json that looks relevant or just fail
            print(f"No specific 'detections_{target_class}.json' found in {det_path}.")
            print("Looking for any JSON file...")
            jsons = list(det_path.glob("*.json"))
            if not jsons:
                print("Error: No JSON files found.")
                return
            json_file = jsons[0] # Pick the first one
            print(f"Using {json_file}")
            
    if not json_file:
        print(f"Error: Could not locate a valid detections JSON file.")
        return

    print(f"Loading detections from: {json_file}")
    try:
        with open(json_file, 'r') as f:
            detections = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Found {len(detections)} detections. Processing...")
    
    success_count = 0
    error_count = 0
    
    # Calculate scale factor to go from Inference Size -> Original Size
    # If inference was at 0.5x, we need to multiply coords by 1/0.5 = 2.0
    scale_factor = 1.0 / resize_factor
    print(f"Scaling coordinates by {scale_factor} (Resize Factor: {resize_factor})")

    for i, item in tqdm(enumerate(detections), total=len(detections)):
        try:
            # Resolve image path
            # The JSON contains "relative_path" which is relative to INPUT_ROOT
            rel_path = item.get("relative_path")
            if not rel_path:
                # Fallback to "image" name if relative path missing (older version compatibility)
                rel_path = item.get("image")
                
            if not rel_path:
                print(f"Skipping item {i}: No path info.")
                error_count += 1
                continue
                
            image_path = root_path / rel_path
            
            if not image_path.exists():
                # Try to find it recursively if path structure doesn't match exactly? 
                # For now, just report error.
                # print(f"Image not found: {image_path}")
                error_count += 1
                continue
            
            # Load Image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Failed to load image: {image_path}")
                error_count += 1
                continue
            
            # Get BBox and Scale
            # bbox format in JSON is [x_min, y_min, x_max, y_max]
            bbox = item["bbox"]
            x_min, y_min, x_max, y_max = bbox
            
            # Scale to original resolution
            x_min = int(x_min * scale_factor)
            y_min = int(y_min * scale_factor)
            x_max = int(x_max * scale_factor)
            y_max = int(y_max * scale_factor)
            
            # Clip to image dimensions
            h, w = img.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                print(f"Invalid bbox after scaling/clipping: {x_min},{y_min},{x_max},{y_max}")
                error_count += 1
                continue
                
            # Crop
            crop = img[y_min:y_max, x_min:x_max]
            
            # Save
            # Create a unique name: {original_name_no_ext}_det{i}.jpg
            original_stem = Path(rel_path).stem
            save_name = f"{original_stem}_det{i}.jpg"
            save_path = out_path / save_name
            
            cv2.imwrite(str(save_path), crop)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing detection {i}: {e}")
            error_count += 1

    print(f"\n--- Finished ---")
    print(f"Successfully cropped: {success_count}")
    print(f"Errors/Skipped: {error_count}")
    print(f"Output directory: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop graffiti detections from images.")
    
    parser.add_argument("--detections", type=str, 
                        default="/media/samuel/SSD/medellin_panoramas_recortados/inferencia/all_detections",
                        help="Path to detections JSON file or directory.")
    
    parser.add_argument("--images_root", type=str, 
                        default="/media/samuel/SSD/medellin_panoramas_recortados",
                        help="Root directory for images.")
    
    parser.add_argument("--output", type=str, 
                        default="/media/samuel/SSD/medellin_panoramas_recortados/inferencia/crops_artistico",
                        help="Directory to save cropped images.")
    
    parser.add_argument("--resize_factor", type=float, default=0.5,
                        help="Resize factor used in inference (default 0.5). Coords will be scaled by 1/factor.")
    
    parser.add_argument("--class_name", type=str, default="artistico",
                        help="Class name to look for if providing a directory (default: artistico).")

    args = parser.parse_args()
    
    crop_detections(
        detections_path=args.detections,
        images_root=args.images_root,
        output_dir=args.output,
        resize_factor=args.resize_factor,
        target_class=args.class_name
    )
