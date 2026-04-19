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
    return mask

def make_lab_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # B > 135 effectively targets yellow/brown hues of aging parchment
    mask = cv2.inRange(B, 135, 255)
    kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ---------------- STRATEGIES ----------------
def bbox_largest_cc(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = stats[idx, :4]
    return int(x), int(y), int(w), int(h)

def bbox_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x), int(y), int(w), int(h)

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

def draw_boxes(original_img, mask):
    out = original_img.copy()
    
    box_cc = bbox_largest_cc(mask)
    box_cnt = bbox_contour(mask)

    if box_cc:
        xc, yc, wc, hc = box_cc
        # Draw CC box in Green (inner)
        cv2.rectangle(out, (xc, yc), (xc+wc, yc+hc), (0, 255, 0), 12)
        
    if box_cnt:
        xk, yk, wk, hk = box_cnt
        # Draw Contour box in Red (outer)
        cv2.rectangle(out, (xk, yk), (xk+wk, yk+hk), (0, 0, 255), 6)
        
    return out

# ---------------- MAIN ----------------
def main():
    input_dir = Path("input_images")
    output_dir = Path("bbox_strategies_vis")
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

        # 2. Add bounding boxes natively onto original image for each mask
        drawn_skew = draw_boxes(img, m_skew)
        drawn_otsu = draw_boxes(img, m_otsu)
        drawn_lab = draw_boxes(img, m_lab)

        # 3. Assemble Output Grid
        # Row 1: The Raw Masks
        r1 = np.hstack([
            resize_h(add_title(to_bgr(m_skew), "Skew Mask")),
            resize_h(add_title(to_bgr(m_otsu), "Otsu Mask")),
            resize_h(add_title(to_bgr(m_lab), "LAB B Mask")),
        ])
        
        # Row 2: Bounding Box Outcomes (Green=CC, Red=Contour)
        r2 = np.hstack([
            resize_h(add_title(drawn_skew, "Skew Bounds (G=CC, R=Contour)")),
            resize_h(add_title(drawn_otsu, "Otsu Bounds (G=CC, R=Contour)")),
            resize_h(add_title(drawn_lab, "LAB Bounds (G=CC, R=Contour)")),
        ])

        r2 = cv2.resize(r2, (r1.shape[1], r1.shape[0]))
        grid = np.vstack([r1, r2])

        cv2.imwrite(str(output_dir / f"boxes_{name}"), grid)

    print("\n[SUCCESS] Check the 'bbox_strategies_vis' directory for visual outputs!")

if __name__ == "__main__":
    main()
