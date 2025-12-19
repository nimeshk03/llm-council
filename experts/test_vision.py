#!/usr/bin/env python3
from pathlib import Path
from .supervisor import Supervisor, ExpertType

if __name__ == "__main__":
    sup = Supervisor()
    img_path = input("Path to image file: ").strip()
    if not Path(img_path).exists():
        print("File not found")
        exit(1)

    question = "Describe this image and extract any visible text or equations."
    answer = sup.ask(question, image_path=img_path)
    print("\nAssistant:\n", answer)
