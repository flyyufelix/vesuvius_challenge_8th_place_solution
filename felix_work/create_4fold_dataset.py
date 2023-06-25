"""
Script to create dataset for 4fold CV by partitioning Fragment 2 into two fragments by height
"""

import cv2
import gc
import numpy as np
import os
import sys
import pandas as pd
import PIL.Image as Image
import tifffile as tiff

from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm

sys.path.extend([".",".."])
from paths import project_path, dataset_path

# === Dir === 

input_dir = Path(dataset_path) / "train" / "2"
inklabels_img = np.array(Image.open(str(input_dir / "inklabels.png")))
mask_img = np.array(Image.open(str(input_dir / "mask.png")))
ir_img = np.array(Image.open(str(input_dir / "ir.png")))

# Create New Fragments Folders
new_fragments = ["4","5"]
base_dir = Path(dataset_path) / "train"
frag_dirs = {}
for frag in new_fragments:

    frag_dir = Path(base_dir / f"{frag}")
    if not frag_dir.exists():
        frag_dir.mkdir()

    surface_vol_dir = Path(frag_dir / "surface_volume")
    if not surface_vol_dir.exists():
        surface_vol_dir.mkdir()

    frag_dirs[frag] = frag_dir

# === Fragment 4 ===
inklabels_roi = inklabels_img[:7500, :]
inklabels_path = str(Path(frag_dirs["4"] / "inklabels.png"))
cv2.imwrite(inklabels_path, (inklabels_roi * 255).astype(np.uint8))

mask_roi = mask_img[:7500, :]
mask_path = str(Path(frag_dirs["4"] / "mask.png"))
cv2.imwrite(mask_path, (mask_roi * 255).astype(np.uint8))

ir_roi = ir_img[:7500, :]
ir_path = str(Path(frag_dirs["4"] / "ir.png"))
cv2.imwrite(ir_path, ir_roi)

# === Fragment 5 ===
inklabels_roi = inklabels_img[7500:, :]
inklabels_path = str(Path(frag_dirs["5"] / "inklabels.png"))
cv2.imwrite(inklabels_path, (inklabels_roi * 255).astype(np.uint8))

mask_roi = mask_img[7500:, :]
mask_path = str(Path(frag_dirs["5"] / "mask.png"))
cv2.imwrite(mask_path, (mask_roi * 255).astype(np.uint8))

ir_roi = ir_img[7500:, :]
ir_path = str(Path(frag_dirs["5"] / "ir.png"))
cv2.imwrite(ir_path, ir_roi)


image_path_list = sorted(list(Path(input_dir / "surface_volume").glob('*.tif')))
for i, image_path in tqdm(enumerate(image_path_list), total=len(image_path_list)):

    img_name = image_path.parts[-1]

    img = tiff.imread(str(image_path))

    # Top portion of Fragment 2 (0 - 7500)
    image_roi = img[:7500, :]
    img_path = str(Path(frag_dirs["4"] / "surface_volume" / img_name))
    tiff.imwrite(img_path, image_roi, dtype=np.uint16)

    # Bottom portion of Fragment 2 (7500 - 14830)
    image_roi = img[7500:, :]
    img_path = str(Path(frag_dirs["5"] / "surface_volume" / img_name))
    tiff.imwrite(img_path, image_roi, dtype=np.uint16)








