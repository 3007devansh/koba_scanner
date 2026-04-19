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
    output_dir = Path("bbox_compare_focus")
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

        # boxes
        b_cc = bbox_largest_cc(combined)
        b_cnt = bbox_contour(combined)

        # visuals (ROW 1)
        v_orig = add_title(img, "Original")
        v_cc = add_title(draw_box(img, b_cc, (0,255,0)), "Largest CC")
        v_cnt = add_title(draw_box(img, b_cnt, (255,0,0)), "Contour")

        row1 = np.hstack([
            resize_h(v_orig),
            resize_h(v_cc),
            resize_h(v_cnt)
        ])

        # crops (ROW 2)
        def crop(img, box):
            if not box:
                return img
            x,y,w,h = box
            return img[y:y+h, x:x+w]

        crop_cc = add_title(crop(img, b_cc), "Crop CC")
        crop_cnt = add_title(crop(img, b_cnt), "Crop Contour")
        mask_vis = add_title(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), "Combined Mask")

        row2 = np.hstack([
            resize_h(mask_vis),
            resize_h(crop_cc),
            resize_h(crop_cnt)
        ])

        # match widths
        row2 = cv2.resize(row2, (row1.shape[1], row1.shape[0]))

        grid = np.vstack([row1, row2])

        cv2.imwrite(str(output_dir / f"compare_{name}"), grid)

    print("\nDone. Results saved to bbox_compare_focus/")


if __name__ == "__main__":
    main()