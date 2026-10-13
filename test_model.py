import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# ISHARA - MODEL DIAGNOSTIC TEST
# ============================================================

SIGNS = [
    "HI",
    "THANK_YOU",
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

# Same model architecture used during training
class IsharaModel(nn.Module):

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn

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
            nn.Linear(128, len(SIGNS))
        )

    def forward(self, x):

        batch_size, sequence_length, channels, height, width = x.shape

        x = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        x = self.cnn(x)

        x = x.reshape(
            batch_size,
            sequence_length,
            576
        )

        x, hidden = self.gru(x)

        forward_last = hidden[-2]
        backward_last = hidden[-1]

        x = torch.cat(
            (forward_last, backward_last),
            dim=1
        )

        return self.classifier(x)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("========================================")
print("       ISHARA MODEL DIAGNOSTIC")
print("========================================")
print()

print("Loading MobileNetV3...")

from torchvision.models import mobilenet_v3_small

cnn = mobilenet_v3_small(weights=None)

cnn.classifier = nn.Identity()

model = IsharaModel(cnn)

checkpoint = torch.load(
    "ishara_model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# LOAD ONE SEQUENCE
# ============================================================

def load_sequence(folder):

    frame_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    frame_files.sort()

    if len(frame_files) == 0:
        return None

    indices = np.linspace(
        0,
        len(frame_files) - 1,
        SEQUENCE_LENGTH
    ).round().astype(int)

    frames = []

    for index in indices:

        path = os.path.join(
            folder,
            frame_files[index]
        )

        frame = cv2.imread(path)

        if frame is None:
            return None

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        frame = frame.astype(
            np.float32
        ) / 255.0

        frame = torch.from_numpy(frame)

        frame = frame.permute(2, 0, 1)

        frames.append(frame)

    return torch.stack(frames)


# ============================================================
# BUILD TEST SET
# Same 70/15/15 split used during training
# ============================================================

import random

random.seed(42)

test_samples = []

for label, sign in enumerate(SIGNS):

    folder = os.path.join(
        "sequences",
        sign
    )

    sequences = []

    for name in os.listdir(folder):

        path = os.path.join(
            folder,
            name
        )

        if os.path.isdir(path) and name.startswith("seq_"):
            sequences.append(path)

    sequences.sort()

    random.shuffle(sequences)

    total = len(sequences)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    test_sequences = sequences[val_end:]

    for path in test_sequences:
        test_samples.append(
            (path, label)
        )


# ============================================================
# TEST
# ============================================================

correct = 0
total = 0

per_class_correct = [0] * len(SIGNS)
per_class_total = [0] * len(SIGNS)

confusion = [
    [0 for _ in SIGNS]
    for _ in SIGNS
]

print("Testing", len(test_samples), "unseen sequences...")
print()

with torch.no_grad():

    for number, (folder, true_label) in enumerate(
        test_samples,
        start=1
    ):

        frames = load_sequence(folder)

        if frames is None:
            print(
                "Could not read:",
                folder
            )
            continue

        frames = frames.unsqueeze(0)

        outputs = model(frames)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        predicted_label = prediction.item()

        confidence = confidence.item() * 100

        true_sign = SIGNS[true_label]

        predicted_sign = SIGNS[predicted_label]

        is_correct = (
            predicted_label == true_label
        )

        if is_correct:

            correct += 1

            per_class_correct[true_label] += 1

        per_class_total[true_label] += 1

        total += 1

        confusion[true_label][predicted_label] += 1

        status = "✓" if is_correct else "✗"

        print(
            f"{number:02d}/72  "
            f"{status}  "
            f"True: {true_sign:12s} "
            f"Predicted: {predicted_sign:12s} "
            f"Confidence: {confidence:5.1f}%"
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print("          OVERALL RESULT")
print("========================================")
print()

accuracy = (
    100.0 * correct / total
)

print(
    f"Correct:  {correct}/{total}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print()

print("========================================")
print("         PER-SIGN ACCURACY")
print("========================================")
print()

for i, sign in enumerate(SIGNS):

    if per_class_total[i] > 0:

        acc = (
            100.0 *
            per_class_correct[i] /
            per_class_total[i]
        )

    else:
        acc = 0

    print(
        f"{sign:12s}: "
        f"{per_class_correct[i]}/"
        f"{per_class_total[i]} "
        f"({acc:.1f}%)"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("========================================")
print("          CONFUSION MATRIX")
print("========================================")
print()

print(
    "Rows = actual sign"
)

print(
    "Columns = predicted sign"
)

print()

print(
    " " * 14 +
    " ".join(
        f"{sign[:7]:>8s}"
        for sign in SIGNS
    )
)

for i, sign in enumerate(SIGNS):

    print(
        f"{sign[:12]:12s} " +
        " ".join(
            f"{confusion[i][j]:8d}"
            for j in range(len(SIGNS))
        )
    )

print()
print("========================================")
print("          DIAGNOSTIC COMPLETE")
print("========================================")
