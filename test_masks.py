import glob
from pathlib import Path
import cv2
import numpy as np

# ---------------- MASKS ----------------
def make_skew_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    s = 15
    closed[:s, :] = 0; closed[-s:, :] = 0
    closed[:, :s] = 0; closed[:, -s:] = 0
    return closed

def make_otsu_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    blurred = cv2.GaussianBlur(L, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = mask.shape
    if np.median(mask[h//3:2*h//3, w//3:2*w//3]) < 128:
        mask = cv2.bitwise_not(mask)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    s = 30
    mask[:s, :] = 0; mask[-s:, :] = 0
    mask[:, :s] = 0; mask[:, -s:] = 0
    return mask

def make_lab_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # B > 135 effectively targets yellow/brown hues of aging parchment
    mask = cv2.inRange(B, 135, 255)
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def make_hsv_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    # Saturation threshold isolates colored objects from gray backgrounds
    blurred = cv2.GaussianBlur(S, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ---------------- VIS HELPERS ----------------
def add_title(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return out

def resize_h(img, h=600):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*scale), h))

def to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# ---------------- MAIN ----------------
def main():
    input_dir = Path("input_images")
    output_dir = Path("mask_permutations")
    output_dir.mkdir(exist_ok=True)

    images = glob.glob(str(input_dir / "*.jpg")) + glob.glob(str(input_dir / "*.png"))

    for impath in images:
        name = Path(impath).name
        print(f"Processing: {name}")
        img = cv2.imread(impath)
        if img is None: continue

        # 1. Base Masks
        m_skew = make_skew_mask(img)
        m_otsu = make_otsu_mask(img)
        m_lab = make_lab_mask(img)
        m_hsv = make_hsv_mask(img)

        # 2. P&C Combine Logic
        def combine(*masks):
            c = masks[0]
            for m in masks[1:]:
                c = cv2.bitwise_or(c, m)
            return cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        comb_all = combine(m_skew, m_otsu, m_lab, m_hsv)

        # 3. Assemble Output Grid
        r1 = np.hstack([
            resize_h(add_title(img, "Original")),
            resize_h(add_title(to_bgr(m_skew), "Skew")),
            resize_h(add_title(to_bgr(m_otsu), "Otsu"))
        ])
        
        r2 = np.hstack([
            resize_h(add_title(to_bgr(m_lab), "LAB (B-channel)")),
            resize_h(add_title(to_bgr(m_hsv), "HSV (S-channel)")),
            resize_h(add_title(to_bgr(comb_all), "All 4 COMBINED"))
        ])

        r2 = cv2.resize(r2, (r1.shape[1], r1.shape[0]))
        grid = np.vstack([r1, r2])

        cv2.imwrite(str(output_dir / f"masks_{name}"), grid)

    print("\n[SUCCESS] Check the 'mask_permutations' directory for visual outputs!")

if __name__ == "__main__":
    main()
