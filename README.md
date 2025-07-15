# ✨ Pix2Pix Image-to-Image Translation using PyTorch 🧠🖼️

This project implements a simple version of the **pix2pix conditional GAN** (cGAN) model using **PyTorch**, designed to perform image-to-image translation.

You can train it on simple paired images — like turning sketches into colored shapes, or edges into real photos!

---

### 📂 Project Structure (Table Format)

| File / Folder        | Description                                  |
|----------------------|----------------------------------------------|
| `main.py`            | Main pix2pix model and training code         |
| `create_samples.py`  | Script to generate sample paired images      |
| `data/`              | Folder containing training image pairs       |
| ├─ `sample1.jpg`     | Black rectangle + red filled rectangle       |
| ├─ `sample2.jpg`     | Black circle + blue filled circle            |
| └─ `sample3.jpg`     | Black triangle + green filled triangle       |

---

## 🔧 Installation

Make sure Python 3.12+ is installed, then run:
pip install torch torchvision pillow
🖼️ Create Sample Training Images
Run this to generate simple shape-based input/target images:

python create_samples.py
This creates 3 images in the data/ folder like:

sample1.jpg: black rectangle → red filled rectangle

sample2.jpg: black circle → blue filled circle

sample3.jpg: black triangle → green filled triangle

Each image is 512×256, with:

Left half (256×256): Input (e.g. outline)

Right half (256×256): Target (e.g. color)

---
## 🧠 How the Code Works
main.py
This file contains:

PairedImageDataset: Loads paired images by splitting each image into input & target

Generator: A small U-Net style generator network

Discriminator: A PatchGAN discriminator to distinguish real vs fake pairs

train(): Trains the Generator and Discriminator together using cGAN + L1 loss

generate_sample(): Generates an output image using a trained Generator

---

## 🚀 Training the Model
To train the model using the generated samples:

dataset = PairedImageDataset("data")
loader = DataLoader(dataset, batch_size=1, shuffle=True)
G = Generator()
D = Discriminator()

train(G, D, loader, epochs=10)           # Train model
generate_sample(G, dataset[0][0])        # Predict using sample1
After running, a file predicted.jpg will be created — this is the model's generated output from the input sketch.

---

## 📷 Example Workflow
Generate training data: python create_samples.py

Train the model: Run main.py

View the output: Open predicted.jpg

---

## 📌 Features

✅ No TensorFlow, pure PyTorch

✅ Compatible with Python 3.12+

✅ Easy to modify & extend

✅ Tiny dataset included (for testing)

✅ Real-world dataset support (e.g. facades, edges2shoes)

---

## 💡 Future Ideas
Use real datasets from pix2pix benchmarks

Improve generator with skip connections

Save checkpoints during training

Add side-by-side visualization
