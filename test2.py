import glob
from pathlib import Path
import cv2
import numpy as np


# ---------------- MASKS ----------------

def make_skew_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


# ---------------- BBOX ----------------

def get_largest_bbox(mask, margin=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None

    h_img, w_img = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas)

    if areas[largest_idx] < (h_img * w_img * 0.01):
        return None

    mask_filtered = (labels == largest_idx + 1)
    ys, xs = np.where(mask_filtered)

    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x = max(0, x_min - margin)
    y = max(0, y_min - margin)
    w = min(w_img - x, (x_max - x_min) + 2 * margin)
    h = min(h_img - y, (y_max - y_min) + 2 * margin)

    return int(x), int(y), int(w), int(h)


def bbox_largest_cc(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return tuple(stats[idx, :4])


def bbox_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(cnt)


# ---------------- VIS ----------------

def draw_box(img, box, color):
    out = img.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 5)
    return out


def crop(img, box):
    if not box:
        return img
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def add_title(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return out


def resize_h(img, h=700):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*scale), h))


# ---------------- MAIN ----------------

def main():
    input_dir = Path("input_images")
    output_dir = Path("final_comparison_grid")
    output_dir.mkdir(exist_ok=True)

    images = glob.glob(str(input_dir / "*.jpg")) + glob.glob(str(input_dir / "*.png"))

    for impath in images:
        name = Path(impath).name
        print("Processing:", name)

        img = cv2.imread(impath)
        if img is None:
            continue

        # masks
        s_mask = make_skew_mask(img)
        o_mask = make_otsu_mask(img)
        combined = cv2.bitwise_or(s_mask, o_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

        # bounding boxes
        b_robust = get_largest_bbox(combined)
        b_cc = bbox_largest_cc(combined)
        b_cnt = bbox_contour(combined)

        # ---------------- ROW 1 (Original Script) ----------------
        r1 = np.hstack([
            resize_h(add_title(img, "Original")),
            resize_h(add_title(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), "Combined Mask")),
            resize_h(add_title(draw_box(img, b_robust, (0,255,0)), "Robust CC Box")),
            resize_h(add_title(crop(img, b_robust), "Robust Crop"))
        ])

        # ---------------- ROW 2 (New Script) ----------------
        r2 = np.hstack([
            resize_h(add_title(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), "Mask")),
            resize_h(add_title(draw_box(img, b_cc, (0,255,0)), "CC Box")),
            resize_h(add_title(draw_box(img, b_cnt, (255,0,0)), "Contour Box")),
            resize_h(add_title(np.hstack([
                resize_h(crop(img, b_cc), 300),
                resize_h(crop(img, b_cnt), 300)
            ]), "CC vs Contour Crop"))
        ])

        r2 = cv2.resize(r2, (r1.shape[1], r1.shape[0]))
        grid = np.vstack([r1, r2])

        cv2.imwrite(str(output_dir / f"compare_{name}"), grid)

    print("\nDone. Saved to final_comparison_grid/")


if __name__ == "__main__":
    main()