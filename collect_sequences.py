import cv2
import os
import time

SIGN = "HI"
SEQUENCE_LENGTH = 30
SAVE_DIR = os.path.join("sequences", SIGN)

os.makedirs(SAVE_DIR, exist_ok=True)

# Count existing recordings
existing = [
    d for d in os.listdir(SAVE_DIR)
    if d.startswith("seq_") and os.path.isdir(os.path.join(SAVE_DIR, d))
]

sequence_number = len(existing)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

print("\n================================")
print("ISHARA - HI MOTION DATA")
print("================================")
print("Press SPACE to record one HI gesture.")
print("Press Q to quit.")
print(f"Each recording = {SEQUENCE_LENGTH} frames")
print("")

recording = False
frames = []
countdown_start = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame not received.")
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # Hand region
    box_size = 400
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    # Countdown before recording
    if countdown_start is not None and not recording:
        elapsed = time.time() - countdown_start
        remaining = 1.0 - elapsed

        if remaining > 0:
            cv2.putText(
                frame,
                f"GET READY: {remaining:.1f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        else:
            countdown_start = None
            recording = True
            frames = []

    if recording:

        # Crop hand region
        roi = frame[y1:y2, x1:x2].copy()

        frames.append(roi)

        cv2.putText(
            frame,
            f"RECORDING {len(frames)}/{SEQUENCE_LENGTH}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Recording finished
        if len(frames) == SEQUENCE_LENGTH:

            folder = os.path.join(
                SAVE_DIR,
                f"seq_{sequence_number:04d}"
            )

            os.makedirs(folder, exist_ok=True)

            for i, img in enumerate(frames):
                filename = os.path.join(
                    folder,
                    f"{i:03d}.jpg"
                )
                cv2.imwrite(filename, img)

            print(f"Saved sequence {sequence_number}")

            sequence_number += 1
            recording = False
            frames = []

            cv2.putText(
                frame,
                "SAVED!",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

    else:
        cv2.putText(
            frame,
            "SPACE = record HI",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    cv2.imshow("ISHARA - HI Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        if not recording and countdown_start is None:
            countdown_start = time.time()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
