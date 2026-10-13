import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import subprocess
from torchvision.models import mobilenet_v3_small

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
IMAGE_SIZE = 224
FRAME_SIZE = 400
COUNTDOWN = 2

DEVICE = torch.device("cpu")


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

    def forward(self, x):

        b, s, c, h, w = x.shape

        x = x.reshape(
            b * s,
            c,
            h,
            w
        )

        features = self.cnn(x)

        features = features.reshape(
            b,
            s,
            576
        )

        output, _ = self.gru(features)

        return self.classifier(output[:, -1, :])


print("Loading ISHARA model...")

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

print("Model loaded successfully!")
print("Classes:", SIGNS)


cap = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def get_frame(frame):

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    x1 = (w - FRAME_SIZE) // 2
    y1 = (h - FRAME_SIZE) // 2

    x2 = x1 + FRAME_SIZE
    y2 = y1 + FRAME_SIZE

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


def classify(frames):

    print("Processing sequence...")

    sequence = torch.stack(frames)

    sequence = sequence.unsqueeze(0)

    with torch.no_grad():

        b, s, c, h, w = sequence.shape

        flat = sequence.reshape(
            s,
            c,
            h,
            w
        )

        features = []

        for i in range(0, s, 8):

            batch = flat[i:i + 8]

            features.append(
                model.cnn(batch)
            )

        features = torch.cat(
            features,
            dim=0
        )

        features = features.unsqueeze(0)

        output, _ = model.gru(
            features
        )

        logits = model.classifier(
            output[:, -1, :]
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        SIGNS[prediction.item()],
        confidence.item()
    )


print()
print("======================================")
print("          ISHARA LIVE DEMO")
print("======================================")
print("SPACE = Start")
print("1 = Display")
print("2 = Speak")
print("R = Reset")
print("Q = Quit")
print()
print("After countdown, START MOVING YOUR HAND.")
print()


state = "IDLE"

start_time = 0
frames = []

detected = "NONE"
confidence = 0.0


while True:

    ret, raw = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    frame, tensor, box = get_frame(raw)

    x1, y1, x2, y2 = box

    h, w = frame.shape[:2]


    # ==========================
    # IDLE
    # ==========================

    if state == "IDLE":

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "PRESS SPACE",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
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
                "Confidence: {:.1f}%".format(
                    confidence * 100
                ),
                (25, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


    # ==========================
    # COUNTDOWN
    # ==========================

    elif state == "COUNTDOWN":

        elapsed = time.time() - start_time

        remaining = COUNTDOWN - elapsed

        number = int(
            np.ceil(remaining)
        )

        if number <= 0:

            frames = []

            state = "CAPTURING"

        else:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "GET READY",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                str(number),
                (w // 2 - 30, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                5
            )


    # ==========================
    # CAPTURE
    # ==========================

    elif state == "CAPTURING":

        frames.append(tensor)

        n = len(frames)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "CAPTURING",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "{}/40".format(n),
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if n >= SEQUENCE_LENGTH:

            state = "PROCESSING"


    # ==========================
    # PROCESSING
    # ==========================

    elif state == "PROCESSING":

        cv2.putText(
            frame,
            "PROCESSING...",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISHARA - Let Hands Be Heard",
            frame
        )

        cv2.waitKey(1)

        detected, confidence = classify(
            frames
        )

        print()
        print(
            "Prediction:",
            detected
        )
        print(
            "Confidence:",
            "{:.2f}%".format(
                confidence * 100
            )
        )

        frames = []

        state = "IDLE"


    cv2.imshow(
        "ISHARA - Let Hands Be Heard",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("r"):

        detected = "NONE"
        confidence = 0.0
        frames = []
        state = "IDLE"

        print("Reset.")


    elif key == ord(" ") and state == "IDLE":

        detected = "NONE"
        confidence = 0.0

        start_time = time.time()

        state = "COUNTDOWN"

        print("Starting sign capture...")


    elif key == ord("1"):

        if detected != "NONE":

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
                "SPEAKING:",
                spoken
            )

            subprocess.Popen(
                ["say", spoken]
            )


cap.release()
cv2.destroyAllWindows()

print("ISHARA demo closed.")
