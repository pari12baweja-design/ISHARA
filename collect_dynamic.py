import cv2
import os
import time

# =========================
# ISHARA DYNAMIC DATASET
# =========================

SIGNS = [
    "HI",
    "THANK_YOU",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "HELP",
    "STOP",
    "GOOD",
    "HOW_ARE_YOU"
]

SEQUENCE_LENGTH = 40
SEQUENCES_PER_SIGN = 50
FRAME_SIZE = 400
COUNTDOWN = 2

os.makedirs("sequences", exist_ok=True)

for sign in SIGNS:
    os.makedirs(os.path.join("sequences", sign), exist_ok=True)


def get_existing_count(sign):
    folder = os.path.join("sequences", sign)

    count = 0

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isdir(path) and name.startswith("seq_"):
            count += 1

    return count


def record_sequence(cap, sign, sequence_number):
    sequence_folder = os.path.join(
        "sequences",
        sign,
        f"seq_{sequence_number:04d}"
    )

    os.makedirs(sequence_folder, exist_ok=True)

    # Countdown
    start_time = time.time()

    while time.time() - start_time < COUNTDOWN:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        x1 = max((w - FRAME_SIZE) // 2, 0)
        y1 = max((h - FRAME_SIZE) // 2, 0)

        x2 = min(x1 + FRAME_SIZE, w)
        y2 = min(y1 + FRAME_SIZE, h)

        roi = frame[y1:y2, x1:x2]

        remaining = COUNTDOWN - int(time.time() - start_time)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"GET READY: {remaining}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow("ISHARA Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    # Record frames
    for frame_number in range(SEQUENCE_LENGTH):

        ret, frame = cap.read()

        if not ret:
            print("Camera frame could not be read.")
            return False

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        x1 = max((w - FRAME_SIZE) // 2, 0)
        y1 = max((h - FRAME_SIZE) // 2, 0)

        x2 = min(x1 + FRAME_SIZE, w)
        y2 = min(y1 + FRAME_SIZE, h)

        roi = frame[y1:y2, x1:x2]

        filename = os.path.join(
            sequence_folder,
            f"{frame_number:03d}.jpg"
        )

        cv2.imwrite(filename, roi)

        # Display progress
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{sign}  {frame_number + 1}/{SEQUENCE_LENGTH}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "PERFORM THE SIGN",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("ISHARA Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

    print(f"Saved {sign} sequence {sequence_number}")

    return True


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print()
print("====================================")
print("       ISHARA DATA COLLECTION")
print("====================================")
print()
print("Controls:")
print("  SPACE = record selected sign")
print("  Q     = quit")
print()
print("IMPORTANT:")
print("Perform the FULL movement during recording.")
print("Do not hold a still pose.")
print()

current_sign_index = 0

while True:

    sign = SIGNS[current_sign_index]

    existing = get_existing_count(sign)

    if existing >= SEQUENCES_PER_SIGN:
        current_sign_index += 1

        if current_sign_index >= len(SIGNS):
            print()
            print("====================================")
            print("ALL DATA COLLECTION COMPLETE!")
            print("====================================")
            break

        continue

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    x1 = max((w - FRAME_SIZE) // 2, 0)
    y1 = max((h - FRAME_SIZE) // 2, 0)

    x2 = min(x1 + FRAME_SIZE, w)
    y2 = min(y1 + FRAME_SIZE, h)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"SIGN: {sign}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sequences: {existing}/{SEQUENCES_PER_SIGN}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = RECORD",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = QUIT",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("ISHARA Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == 32:  # SPACE

        success = record_sequence(
            cap,
            sign,
            existing
        )

        if not success:
            break


cap.release()
cv2.destroyAllWindows()
