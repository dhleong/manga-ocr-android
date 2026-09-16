#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=1.31.0",
#     "pillow>=12.3.0",
#     "rfdetr==1.7.0",
#     "safetensors>=0.5",
#     "ultralytics>=8.4.153",
# ]
# ///
import importlib.util

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics.utils.plotting import Annotator, colors

print("fetching models..")
weights = hf_hub_download(
    repo_id="mayocream/koharu-layout-rfdetr-seg-2xl-1152",
    filename="model.safetensors",
)
loader_path = hf_hub_download(
    repo_id="mayocream/koharu-layout-rfdetr-seg-2xl-1152",
    filename="load_model.py",
)
spec = importlib.util.spec_from_file_location("koharu_layout_loader", loader_path)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
model = loader.load_model(weights)

print("detecting...")
image = Image.open("../manga-ocr-android/Witch_Hat_Atelier_-_Ch._3_-_6.jpg").convert(
    "RGB"
)
detections = model.predict(
    image,
    threshold=0.20,
    shape=(1152, 1152),
    include_source_image=False,
)

class_thresholds = {0: 0.25, 1: 0.20, 2: 0.50, 3: 0.50}
keep = np.asarray(
    [
        score >= class_thresholds[int(class_id)]
        for class_id, score in zip(detections.class_id, detections.confidence)
    ]
)
detections = detections[keep]

print(detections.xyxy)  # bounding boxes
print(detections.mask)  # instance masks
print(detections.class_id)  # 0=text, 1=COO, 2=bubble, 3=panel
print(detections.confidence)

id_to_label = {0: "text", 1: "COO", 2: "bubble", 3: "panel"}

print("building annotated image")
annotator = Annotator(image)
for i, box in enumerate(detections.xyxy):
    class_id = detections.class_id[i]
    label = id_to_label[class_id]
    if label == "panel":
        continue
    annotator.box_label(box, f"{label}#{i}", colors(class_id, True))
annotator.show()
