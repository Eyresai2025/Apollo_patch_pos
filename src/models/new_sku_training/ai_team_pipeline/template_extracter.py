import cv2

# =============================
# PATHS
# =============================
IMG_PATH = r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_defect\1.png"
SAVE_IMG = r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_defect\roi.png"

# =============================
# LOAD IMAGE
# =============================
img = cv2.imread(IMG_PATH)

if img is None:
    raise ValueError("Image not found")

H, W = img.shape[:2]
print("Image shape:", (H, W))

# =============================
# VIEW SETTINGS
# =============================
win_h, win_w = 800, 800
x_off, y_off = 0, 0
zoom = 1.0

drawing = False
x1 = y1 = x2 = y2 = 0

# =============================
# MOUSE FUNCTION (AUTO SAVE)
# =============================
def draw_roi(event, x, y, flags, param):
    global drawing, x1, y1, x2, y2

    ix = int(x / zoom + x_off)
    iy = int(y / zoom + y_off)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = ix, iy

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = ix, iy

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = ix, iy

        # =============================
        # AUTO SAVE ROI HERE
        # =============================
        if x1 != x2 and y1 != y2:
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            roi = img[y_min:y_max, x_min:x_max]

            cv2.imwrite(SAVE_IMG, roi)

            print("✅ ROI saved:", SAVE_IMG)
            print("ROI shape:", roi.shape)
        else:
            print("❌ Invalid ROI")

# =============================
# WINDOW
# =============================
cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Viewer", draw_roi)

print("\nControls:")
print("W/A/S/D → move")
print("+ / -   → zoom")
print("Mouse drag → ROI (auto-saves on release)")
print("Q → quit\n")

# =============================
# MAIN LOOP
# =============================
while True:

    # keep offsets valid
    x_off = max(0, min(x_off, W - win_w))
    y_off = max(0, min(y_off, H - win_h))

    view = img[y_off:y_off+win_h, x_off:x_off+win_w]
    display = cv2.resize(view, None, fx=zoom, fy=zoom)

    # draw rectangle live
    if drawing and x1 != x2 and y1 != y2:
        dx1 = int((x1 - x_off) * zoom)
        dy1 = int((y1 - y_off) * zoom)
        dx2 = int((x2 - x_off) * zoom)
        dy2 = int((y2 - y_off) * zoom)

        cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0,255,0), 2)

    cv2.imshow("Viewer", display)

    key = cv2.waitKey(1) & 0xFF

    # PAN
    if key == ord('w'): y_off -= 200
    elif key == ord('s'): y_off += 200
    elif key == ord('a'): x_off -= 200
    elif key == ord('d'): x_off += 200

    # ZOOM
    elif key == ord('+') or key == ord('='): zoom *= 1.2
    elif key == ord('-'): zoom /= 1.2

    elif key == ord('q'):
        break

cv2.destroyAllWindows()