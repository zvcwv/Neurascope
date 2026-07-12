<img width="2228" height="1243" alt="image" src="https://github.com/user-attachments/assets/4e85df10-7dac-45c0-99e6-fe20f4e3f344" />
# Neurascope

A microscope for a neural network's mind — draw a digit and watch a **real**
feed-forward network read it, layer by layer, in your browser.

Live demo: single-file React component, no backend required.

## What's real here

- A 4-layer MLP (`784 → 64 → 32 → 16 → 10`, Dense+ReLU ×3, Softmax output)
  trained with **scikit-learn** on a 5,000-image slice of real MNIST digits
  (~93.6% held-out test accuracy).
- The trained weights (~53k parameters) are embedded directly in the file and
  run through a hand-written JS forward pass — verified to match `sklearn`'s
  `predict_proba` output exactly.
- Every visual (particle flow, neuron glow, connection weights, probability
  bars, "detector" tiles) is driven by that real computation. Nothing is
  pre-scripted; the animation just paces the reveal for teaching purposes.

<img width="2215" height="1249" alt="image" src="https://github.com/user-attachments/assets/d9a128f4-9ebd-44b1-b111-0ca334146224" />

## Features

- Draw with mouse / touch / stylus on a 400×400 canvas, real MNIST-style
  preprocessing (bounding box → 20×20 → centered in 28×28)
- Animated pipeline: image → particles → tensor → hidden layers → softmax → result
- Layer-1 "detector" view — real first-layer weights reshaped back into 28×28
- Step-by-step mode with the math formula for each stage
- Speed control (0.25×–4×), live diagnostics sidebar (FPS, params, activations, etc.)
- "Real example" button loads an actual MNIST test image if freehand drawing
  doesn't classify well (small training set — see Limitations)

<img width="2208" height="1237" alt="image" src="https://github.com/user-attachments/assets/59c70b23-1fb8-475d-a35b-3fcddbddb36e" />

## Not included (yet)

- **EMNIST** (letters) — toggle is in the UI, disabled for now
- **Backend / training dashboard** — real-time GPU/CPU monitoring, confusion
  matrix, ROC curves need a FastAPI + PyTorch service; this repo is client-only
- **Live retraining** in the browser — button is present but disabled

## Tech

React (single component) · Canvas 2D · plain JS matrix math (no ML framework
needed at inference time) · model trained with Python (`scikit-learn`,
`mlxtend` for the bundled MNIST subset)

## Limitations

Trained on only 5,000 samples (vs. the full 60k MNIST), so accuracy on messy
freehand drawings is lower than on clean, centered test digits. Good enough
for a teaching demo, not for production OCR.

## Roadmap

1. FastAPI + PyTorch backend: full-dataset training, live training dashboard
2. EMNIST support for letters
3. Real in-browser retraining



