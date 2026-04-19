import os
import glob
from pathlib import Path
import cv2
import numpy as np

def get_largest_bbox(mask: np.ndarray, margin: int = 20) -> tuple:
    if mask is None:
        return None
        
    h_img, w_img = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return None
        
    largest_idx = np.argmax(areas)
    if areas[largest_idx] < (h_img * w_img * 0.01):
        return None
    
    mask_filtered = (labels == largest_idx + 1)
    ys, xs = np.where(mask_filtered)

    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x = int(max(0, x_min - margin))
    y = int(max(0, y_min - margin))
    w = int(min(w_img - x, (x_max - x_min) + 2 * margin))
    h = int(min(h_img - y, (y_max - y_min) + 2 * margin))

    return x, y, w, h


def make_skew_mask(cv_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    s = 15
    if s > 0:
        closed[:s, :]  = 0; closed[-s:, :] = 0
        closed[:, :s]  = 0; closed[:, -s:] = 0
    return closed


def make_otsu_mask(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    
    blurred = cv2.GaussianBlur(L, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h, w = mask.shape
    center_val = np.median(mask[h//3:2*h//3, w//3:2*w//3])
    if center_val < 128:
        mask = cv2.bitwise_not(mask)
        
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    s = 30
    mask[:s, :] = 0; mask[-s:, :] = 0
    mask[:, :s] = 0; mask[:, -s:] = 0
    
    return mask


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 80), (0,0,0), -1)
    cv2.putText(out, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return out


def resize_height(img: np.ndarray, target_h: int) -> np.ndarray:
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h))


def main():
    input_dir = Path("input_images")
    output_dir = Path("cropping_experiments_combined_final")
    output_dir.mkdir(exist_ok=True)
    
    images = glob.glob(str(input_dir / "*.png")) + glob.glob(str(input_dir / "*.jpg"))
    if not images:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(images)} images to test.")
    
    for impath in images:
        name = Path(impath).name
        print(f"Processing Combined Strategy for {name}...")
        
        img = cv2.imread(impath)
        if img is None:
            continue
            
        res_orig = add_title(img, "Original")
            
        s_mask = make_skew_mask(img)
        res_s_mask = add_title(cv2.cvtColor(s_mask, cv2.COLOR_GRAY2BGR), "Skew Mask (Adaptive)")
        
        o_mask = make_otsu_mask(img)
        res_o_mask = add_title(cv2.cvtColor(o_mask, cv2.COLOR_GRAY2BGR), "Otsu Mask (LAB)")
        
        # Original simple Combination with no destructive opening
        c_mask = cv2.bitwise_or(s_mask, o_mask)
        kernel = np.ones((11, 11), np.uint8)
        c_mask_bridged = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
        
        res_c_mask = add_title(cv2.cvtColor(c_mask_bridged, cv2.COLOR_GRAY2BGR), "Combined Mask")
        
        box = get_largest_bbox(c_mask_bridged, margin=20)
        res_box = img.copy()
        crop_img = img.copy()
        
        if box:
            x, y, w, h = box
            cv2.rectangle(res_box, (x, y), (x+w, y+h), (0, 255, 0), 8)
            crop_img = img[y:y+h, x:x+w]
            
        res_box = add_title(res_box, "Final Box")
        res_crop = add_title(crop_img, "Final Crop")
        
        target_height = 800
        row1 = np.hstack([
            resize_height(res_orig, target_height),
            resize_height(res_s_mask, target_height),
            resize_height(res_o_mask, target_height)
        ])
        
        row2 = np.hstack([
            resize_height(res_c_mask, target_height),
            resize_height(res_box, target_height),
            resize_height(res_crop, target_height)
        ])
        
        row2 = cv2.resize(row2, (row1.shape[1], row1.shape[0]))
        
        grid = np.vstack([row1, row2])
        out_path = output_dir / f"final_{name}"
        cv2.imwrite(str(out_path), grid)
        
    print(f"\nSaved {len(images)} results to {output_dir}/")

if __name__ == "__main__":
    main()
