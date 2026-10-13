# ISHARA — Let Hands Be Heard

## Project Overview

ISHARA is an AI-assisted project exploring Indian Sign Language (ISL) recognition to help make communication more accessible.

The project aims to recognise signs using a camera and provide an output that can be displayed as text and, where integrated, spoken aloud.

## Objectives

* Explore camera-based sign recognition using machine learning.
* Develop and test a sign-recognition model.
* Work towards displaying recognised signs and providing audio output.
* Explore an affordable, offline assistive-device prototype.

## Technology

* Python
* PyTorch
* MobileNetV3
* OpenCV
* MediaPipe (used in an earlier hand-tracking implementation)

## Repository Contents

* `live_demo.py` — live recognition demo using the PyTorch model.
* `calibrated_demo.py` — experimental recognition and calibration demo.
* `train_model.py` — model-training script.
* `test_model.py` — model-testing script.
* `collect_dynamic.py` and `collect_sequences.py` — data-collection scripts.
* `camera_test.py` — camera testing utility.

## Current Development Status

ISHARA is a work in progress. The recognition pipeline, supported signs, accuracy, and hardware integration are still being tested. The final prototype aims to support recognised-sign display and audio output.

## Running the Project

The Python environment and dependencies must be configured before running the scripts. The required model checkpoint and compatible dataset must also be available.

See the project code and training scripts for implementation details.

## Team

Developed as a student project by the ISHARA team.
