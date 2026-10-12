# Assignment 9: Object detection from CCTV footage using YOLOv8
# Install first:  pip install ultralytics opencv-python
import cv2
from collections import Counter
from ultralytics import YOLO

VIDEO_IN = "cctv_footage.mp4"          # path to CCTV video (or 0 for a live camera / RTSP URL)
VIDEO_OUT = "cctv_detected.mp4"
CONF_THRESHOLD = 0.40                  # minimum confidence
TARGET_CLASSES = None                  # e.g. [0, 2] -> only person & car (COCO ids)

# 1. Load a pre-trained COCO model (downloads weights on first run)
model = YOLO("yolov8n.pt")

# 2. Open the video and prepare the writer
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

total_counts = Counter()
frame_no = 0

# 3. Frame-by-frame detection (with tracking so every object keeps an ID)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    results = model.track(frame, persist=True, conf=CONF_THRESHOLD,
                          classes=TARGET_CLASSES, verbose=False)[0]

    # Count objects found in this frame
    names = [model.names[int(c)] for c in results.boxes.cls]
    total_counts.update(names)
    frame_counts = Counter(names)

    # Draw boxes + labels + IDs
    annotated = results.plot()
    summary = ", ".join(f"{k}: {v}" for k, v in frame_counts.items()) or "no objects"
    cv2.putText(annotated, f"Frame {frame_no} | {summary}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    writer.write(annotated)
    cv2.imshow("CCTV Object Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):      # press q to stop
        break

# 4. Release resources and print the report
cap.release(); writer.release(); cv2.destroyAllWindows()
print("Frames processed:", frame_no)
print("Detections per class (all frames):")
for cls, n in total_counts.most_common():
    print(f"  {cls:12s}: {n}")
print("Annotated video saved as", VIDEO_OUT)
