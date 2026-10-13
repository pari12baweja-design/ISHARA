import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import subprocess
import os

from torchvision.models import mobilenet_v3_small


# ==========================================
# SETTINGS
# ==========================================

SIGNS = [
    "HI",
    "THANK YOU",
    "YES",
    "NO",
    "PLEASE",
    "SORRY",
    "HELP",
    "STOP",
    "GOOD"
]

SEQUENCE_LENGTH = 40
FRAME_SIZE = 400
IMAGE_SIZE = 224

EXAMPLES_PER_SIGN = 3
COUNTDOWN = 2

DEVICE = torch.device("cpu")


# ==========================================
# MODEL
# ==========================================

class ISHARA_Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = mobilenet_v3_small(weights=None)
        self.cnn.classifier = nn.Identity()

        self.gru = nn.GRU(
            input_size=576,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 9)
        )

    def features(self, x):

        b, s, c, h, w = x.shape

        x = x.reshape(
            b * s,
            c,
            h,
            w
        )

        with torch.no_grad():

            cnn_features = self.cnn(x)

            cnn_features = cnn_features.reshape(
                b,
                s,
                576
            )

            output, _ = self.gru(
                cnn_features
            )

            embedding = output[:, -1, :]

        return embedding


print("Loading ISHARA...")

model = ISHARA_Model()

checkpoint = torch.load(
    "ishara_model.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded!")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():

    print("ERROR: Camera failed.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ==========================================
# FRAME PREPARATION
# ==========================================

def prepare(frame):

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    x1 = max((w - FRAME_SIZE) // 2, 0)
    y1 = max((h - FRAME_SIZE) // 2, 0)

    x2 = min(x1 + FRAME_SIZE, w)
    y2 = min(y1 + FRAME_SIZE, h)

    roi = frame[y1:y2, x1:x2]

    resized = cv2.resize(
        roi,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        rgb
    ).permute(2, 0, 1)

    return frame, tensor, (x1, y1, x2, y2)


# ==========================================
# CAPTURE ONE SEQUENCE
# ==========================================

def capture_sequence(label):

    frames = []

    print()
    print("Preparing:", label)

    start = time.time()

    while time.time() - start < COUNTDOWN:

        ret, raw = cap.read()

        if not ret:
            continue

        frame, tensor, box = prepare(raw)

        x1, y1, x2, y2 = box

        remaining = int(
            np.ceil(
                COUNTDOWN -
                (time.time() - start)
            )
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            label,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "GET READY " + str(remaining),
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISHARA",
            frame
        )

        cv2.waitKey(1)


    print("CAPTURE!")

    while len(frames) < SEQUENCE_LENGTH:

        ret, raw = cap.read()

        if not ret:
            continue

        frame, tensor, box = prepare(raw)

        frames.append(tensor)

        x1, y1, x2, y2 = box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "DO: " + label,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "{}/40".format(len(frames)),
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISHARA",
            frame
        )

        cv2.waitKey(1)


    sequence = torch.stack(
        frames
    ).unsqueeze(0)

    with torch.no_grad():

        embedding = model.features(
            sequence
        )

    embedding = embedding.squeeze(0)

    # Normalize for cosine similarity
    embedding = embedding / (
        torch.norm(embedding) + 1e-8
    )

    return embedding


# ==========================================
# CALIBRATION
# ==========================================

print()
print("==========================================")
print("       ISHARA LIVE CALIBRATION")
print("==========================================")
print()
print("You will perform each sign 3 times.")
print("Do the sign naturally and completely.")
print()


calibration = {}


for sign in SIGNS:

    calibration[sign] = []

    for example in range(EXAMPLES_PER_SIGN):

        print(
            "{} - example {}/{}".format(
                sign,
                example + 1,
                EXAMPLES_PER_SIGN
            )
        )

        embedding = capture_sequence(
            sign
        )

        calibration[sign].append(
            embedding
        )

        print("Saved.")

        time.sleep(0.5)


print()
print("==========================================")
print("       CALIBRATION COMPLETE")
print("==========================================")
print()
print("Now the system will detect signs")
print("WITHOUT asking you which sign it is.")
print()
print("SPACE = Detect")
print("1 = Display")
print("2 = Speak")
print("R = Reset")
print("Q = Quit")
print()


# ==========================================
# DETECTION
# ==========================================

detected = "NONE"
confidence = 0.0


while True:

    ret, raw = cap.read()

    if not ret:
        break

    frame, tensor, box = prepare(raw)

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = DETECT",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Detected: " + detected,
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if detected != "NONE":

        cv2.putText(
            frame,
            "Match: {:.1f}%".format(
                confidence * 100
            ),
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "ISHARA",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("r"):

        detected = "NONE"
        confidence = 0.0


    elif key == ord(" ") :

        # Capture without asking for sign
        frames = []

        print()
        print("Starting detection...")

        start = time.time()

        while time.time() - start < COUNTDOWN:

            ret2, raw2 = cap.read()

            if not ret2:
                continue

            frame2, tensor2, box2 = prepare(
                raw2
            )

            remaining = int(
                np.ceil(
                    COUNTDOWN -
                    (time.time() - start)
                )
            )

            x12, y12, x22, y22 = box2

            cv2.rectangle(
                frame2,
                (x12, y12),
                (x22, y22),
                (255, 255, 255),
                3
            )

            cv2.putText(
                frame2,
                "GET READY " + str(remaining),
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ISHARA",
                frame2
            )

            cv2.waitKey(1)


        while len(frames) < SEQUENCE_LENGTH:

            ret2, raw2 = cap.read()

            if not ret2:
                continue

            frame2, tensor2, box2 = prepare(
                raw2
            )

            frames.append(tensor2)

            x12, y12, x22, y22 = box2

            cv2.rectangle(
                frame2,
                (x12, y12),
                (x22, y22),
                (255, 255, 255),
                3
            )

            cv2.putText(
                frame2,
                "DETECTING",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame2,
                "{}/40".format(len(frames)),
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ISHARA",
                frame2
            )

            cv2.waitKey(1)


        sequence = torch.stack(
            frames
        ).unsqueeze(0)

        with torch.no_grad():

            embedding = model.features(
                sequence
            ).squeeze(0)

        embedding = embedding / (
            torch.norm(embedding) + 1e-8
        )


        # ==================================
        # NEAREST CALIBRATED EXAMPLE
        # ==================================

        best_sign = None
        best_similarity = -999

        for sign in SIGNS:

            for reference in calibration[sign]:

                similarity = torch.dot(
                    embedding,
                    reference
                ).item()

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_sign = sign


        detected = best_sign

        # Convert cosine similarity to readable score
        confidence = max(
            0.0,
            min(
                1.0,
                (best_similarity + 1.0) / 2.0
            )
        )

        print()
        print(
            "DETECTED:",
            detected
        )

        print(
            "MATCH:",
            "{:.1f}%".format(
                confidence * 100
            )
        )


    elif key == ord("1"):

        print(
            "DISPLAY:",
            detected
        )


    elif key == ord("2"):

        if detected != "NONE":

            spoken = detected.replace(
                "_",
                " "
            )

            print(
                "SPEAK:",
                spoken
            )

            subprocess.Popen(
                ["say", spoken]
            )


cap.release()
cv2.destroyAllWindows()

print("ISHARA closed.")
