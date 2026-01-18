import astrolign as al
import numpy as np
from PIL import Image
import sys

def align():
    if len(sys.argv) < 4:
        return
    
    ref_path, target_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    try:
        # Ouverture en 16-bit
        ref_img = np.array(Image.open(ref_path)).astype(np.float32)
        target_img = np.array(Image.open(target_path)).astype(np.float32)

        # Alignement astrométrique
        aligned_img, footprint = al.register(target_img, ref_img, fill_value=0)
        
        # Sauvegarde uint16
        output = Image.fromarray(aligned_img.astype(np.uint16))
        output.save(out_path)
    except Exception as e:
        print(f"Erreur alignement {target_path}: {e}")

if __name__ == "__main__":
    align()

