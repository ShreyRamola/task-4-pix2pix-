from PIL import Image, ImageDraw
import os

os.makedirs("data", exist_ok=True)

def create_paired_image(filename, shape="rectangle", color="red"):
    img = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(img)

    if shape == "rectangle":
        draw.rectangle([50, 50, 200, 200], outline="black", width=5)
        draw.rectangle([306, 50, 456, 200], fill=color)
    elif shape == "circle":
        draw.ellipse([50, 50, 200, 200], outline="black", width=5)
        draw.ellipse([306, 50, 456, 200], fill=color)
    elif shape == "triangle":
        draw.polygon([50, 200, 125, 50, 200, 200], outline="black", width=5)
        draw.polygon([306, 200, 381, 50, 456, 200], fill=color)

    img.save(f"data/{filename}")

create_paired_image("sample1.jpg", "rectangle", "red")
create_paired_image("sample2.jpg", "circle", "blue")
create_paired_image("sample3.jpg", "triangle", "green")
