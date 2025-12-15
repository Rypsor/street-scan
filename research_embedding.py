from ultralytics import YOLO
import torch
import numpy as np
import sys

print("Ultralytics version:", sys.modules['ultralytics'].__version__ if 'ultralytics' in sys.modules else "Unknown")

try:
    # Try to load the user's model, fallback to nano
    try:
        model = YOLO("model/best.pt")
        print("Loaded model/best.pt")
    except Exception as e:
        print(f"Could not load model/best.pt: {e}")
        model = YOLO("yolov8n.pt")
        print("Loaded yolov8n.pt")

    # Dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Check for embed method
    if hasattr(model, 'embed'):
        print("model.embed() method EXISTS.")
        try:
            emb = model.embed(img)
            print(f"model.embed() output shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
        except Exception as e:
            print(f"model.embed() execution failed: {e}")
    else:
        print("model.embed() method does NOT exist.")

    # Check predict with embed arg (older versions or specific tasks)
    try:
        res = model.predict(img, embed=[1, 2], verbose=False)
        print("model.predict(embed=...) ran without error (but might ignore it).")
    except Exception as e:
        print(f"model.predict(embed=...) failed: {e}")

except Exception as e:
    print(f"General error: {e}")
