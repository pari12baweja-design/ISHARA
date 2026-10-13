import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ============================================================
# ISHARA - CNN + GRU TRAINING
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

NUM_CLASSES = len(SIGNS)

SEQUENCE_LENGTH = 40
IMAGE_SIZE = 224

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATA SPLIT
# ============================================================

def get_sequences(sign):
    folder = os.path.join("sequences", sign)

    sequences = []

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isdir(path) and name.startswith("seq_"):
            sequences.append(path)

    sequences.sort()

    return sequences


train_samples = []
val_samples = []
test_samples = []

print()
print("========================================")
print("        ISHARA DATASET PREPARATION")
print("========================================")
print()

for label, sign in enumerate(SIGNS):

    sequences = get_sequences(sign)

    random.shuffle(sequences)

    total = len(sequences)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_sequences = sequences[:train_end]
    val_sequences = sequences[train_end:val_end]
    test_sequences = sequences[val_end:]

    for path in train_sequences:
        train_samples.append((path, label))

    for path in val_sequences:
        val_samples.append((path, label))

    for path in test_sequences:
        test_samples.append((path, label))

    print(
        f"{sign:12s} "
        f"Train: {len(train_sequences):2d}  "
        f"Val: {len(val_sequences):2d}  "
        f"Test: {len(test_sequences):2d}"
    )

print()
print("Total training sequences:", len(train_samples))
print("Total validation sequences:", len(val_samples))
print("Total test sequences:", len(test_samples))
print()


# ============================================================
# DATASET
# ============================================================

class SignDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def load_sequence(self, folder):

        frame_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        frame_files.sort()

        frames = []

        # ----------------------------------------
        # Temporal normalization to 40 frames
        # ----------------------------------------

        indices = np.linspace(
            0,
            len(frame_files) - 1,
            SEQUENCE_LENGTH
        ).round().astype(int)

        for index in indices:

            filename = frame_files[index]

            path = os.path.join(folder, filename)

            frame = cv2.imread(path)

            if frame is None:
                raise RuntimeError(
                    f"Could not read image: {path}"
                )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            frame = frame.astype(np.float32) / 255.0

            frame = torch.from_numpy(frame)

            frame = frame.permute(2, 0, 1)

            frames.append(frame)

        return torch.stack(frames)

    def __getitem__(self, index):

        folder, label = self.samples[index]

        frames = self.load_sequence(folder)

        return frames, torch.tensor(label, dtype=torch.long)


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = SignDataset(train_samples)
val_dataset = SignDataset(val_samples)
test_dataset = SignDataset(test_samples)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

print("Loading pretrained MobileNetV3...")
print()

weights = MobileNet_V3_Small_Weights.DEFAULT

cnn = mobilenet_v3_small(weights=weights)

cnn.classifier = nn.Identity()

cnn.eval()

for parameter in cnn.parameters():
    parameter.requires_grad = False


# ============================================================
# CNN + GRU MODEL
# ============================================================

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
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):

        batch_size, sequence_length, channels, height, width = x.shape

        # ----------------------------------------
        # Process frames through CNN
        # ----------------------------------------

        x = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        x = self.cnn(x)

        # ----------------------------------------
        # Restore sequence structure
        # ----------------------------------------

        x = x.reshape(
            batch_size,
            sequence_length,
            576
        )

        # ----------------------------------------
        # GRU learns the movement over time
        # ----------------------------------------

        x, hidden = self.gru(x)

        # Use final forward + backward states
        forward_last = hidden[-2]
        backward_last = hidden[-1]

        x = torch.cat(
            (forward_last, backward_last),
            dim=1
        )

        # ----------------------------------------
        # Classification
        # ----------------------------------------

        x = self.classifier(x)

        return x


model = IsharaModel(cnn)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

model = model.to(device)

print("Using:", device)
print()


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.gru.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    # Keep CNN frozen and in evaluation mode
    model.cnn.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch_index, (frames, labels) in enumerate(train_loader):

        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (batch_index + 1) % 10 == 0:
            print(
                f"  Batch {batch_index + 1}/{len(train_loader)}"
            )

    accuracy = 100.0 * correct / total

    return total_loss / len(train_loader), accuracy


# ============================================================
# VALIDATION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for frames, labels in loader:

            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return total_loss / len(loader), accuracy


# ============================================================
# TRAIN
# ============================================================

print("========================================")
print("          STARTING TRAINING")
print("========================================")
print()

best_val_accuracy = 0.0
patience = 6
patience_counter = 0

for epoch in range(EPOCHS):

    print(
        f"--------------- Epoch {epoch + 1}/{EPOCHS} ---------------"
    )

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = evaluate(val_loader)

    print()
    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # ----------------------------------------
    # Save best model
    # ----------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": SIGNS,
                "sequence_length": SEQUENCE_LENGTH,
                "image_size": IMAGE_SIZE
            },
            "ishara_model.pth"
        )

        print("★ BEST MODEL SAVED")

    else:

        patience_counter += 1

        print(
            f"No improvement ({patience_counter}/{patience})"
        )

    print()

    if patience_counter >= patience:

        print("Early stopping.")

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("========================================")
print("        LOADING BEST MODEL")
print("========================================")
print()

checkpoint = torch.load(
    "ishara_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_accuracy = evaluate(test_loader)

print()
print("========================================")
print("          FINAL TEST RESULTS")
print("========================================")
print()

print(
    f"Test Loss:     {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print()

print("Classes:")

for i, sign in enumerate(SIGNS):

    print(
        f"  {i}: {sign}"
    )

print()

print("Model saved as:")
print("  ishara_model.pth")

print()
print("========================================")
print("           TRAINING COMPLETE")
print("========================================")
