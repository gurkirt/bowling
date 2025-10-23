
from typing import Any, Tuple, TypeAlias, List

import altair as alt
import dataclasses
import enum
import json
import numpy as np
import pandas as pd
import pathlib
import PIL 
import sys
import timm
import torch
import torchvision.transforms as transforms

Frame: TypeAlias = np.ndarray 


class Device(str, enum.Enum):
    CPU = 'cpu'
    CUDA = 'cuda'

    @classmethod
    def from_string(cls, value: str) -> "Device" :
        match value:
            case ""|"cpu":
                return cls.CPU
            case "cuda" | "gpu":
                return cls.CUDA
            case _:
                raise ValueError(f"Invalid device '{value}'. Choose from: {', '.join(d.value for d in cls)}")

class Model:
    def __init__(self, model_path: pathlib.Path, device: Device):
        self.model, h, w = _load_model(model_path, device)
        self.transform = _get_transforms(h, w)
        self.device = device

    def __call__(self, image: PIL.Image.Image) -> bool:
        return _predict_is_in_action(self.model, image, self.transform, self.device)


def _load_model(model_path: pathlib.Path, device: Device) -> Tuple[Any, int , int]:
    """Load trained model from checkpoint"""    
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    model = timm.create_model(checkpoint['args'].model_name, pretrained=False, num_classes=checkpoint['args'].num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(torch.device(device.value))
    model.eval()
    return model, checkpoint['args'].input_height, checkpoint['args'].input_width 


def _get_transforms(input_height=320, input_width=128) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _predict_is_in_action(model, image: PIL.Image.Image, transform, device: Device) -> bool:
    """Return True if the model predicts the action class for a given PIL RGB image."""
    device_t = torch.device(device.value) if isinstance(device, Device) else device
    input_tensor = transform(image).unsqueeze(0).to(device_t)
    with torch.no_grad():
        outputs = model(input_tensor)
        outputs = outputs.softmax(dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
    answer = predicted_class == 1
    sys.stdout.flush()
    return answer, outputs[0,1].item()


def crop(frame: Frame) -> Frame:
    """Remove 32% from the top vertically, crop the frame to a square centered horizontally."""
    h, w, _ = frame.shape
    top_offset = int(h * 0.32)
    new_h = h - top_offset
    side = min(new_h, w)
    center_x = w // 2
    center_y = top_offset + new_h // 2
    left = max(center_x - side // 2, 0)
    right = left + side
    top = max(center_y - side // 2, top_offset)
    bottom = top + side
    return frame[top:bottom, left:right]



@dataclasses.dataclass
class FrameStat:
    index: int
    correct: bool
    prediction: bool
    confidence: float

def read_start_end(video: pathlib.Path) -> List[Tuple[int, int]]:
    with open(video.with_suffix('.json')) as f:
        events = [(e["start_frame"], e["end_frame"]) for e in json.load(f)['temporal_events']]
    return events


class Stats:
    def __init__(self, video: pathlib.Path, start_ends:List[Tuple[int, int]]):
        self.stats = []
        self.events = start_ends
        self.video = video

    def add(self, index:int, prediction: bool, confidence: float):
        correct = any(start <= index <= end for (start, end) in self.events)
        self.stats.append(FrameStat(index, correct, prediction, confidence))

    def to_chart(self, filename) -> pd.DataFrame:
        points  = [(s.index, int(s.prediction), 'prediction') for s in self.stats]
        points.extend([(s.index, int(s.correct), 'correct') for s in self.stats])
        lines  = [(s.index, s.confidence, 'confidence') for s in self.stats]
       
        dfpoints = pd.DataFrame(points, columns=['index', 'probability', 'type'])
        dflines = pd.DataFrame(lines, columns=['index', 'probability', 'type'])
        
        group = ['prediction', 'correct']
        shapes = ['wedge', 'cross']

        pointschart = alt.Chart(dfpoints).mark_point().encode(
            x='index',
            y='probability',
            shape=alt.Shape('type:N', scale=alt.Scale(domain=group, range=shapes)),
            color="type:N",
            tooltip=['index', 'probability', 'type'])

        lineschart = alt.Chart(dflines).mark_line().encode(
            x='index',
            y='probability',
            color=alt.value('green'),
            tooltip=['index', 'probability', 'type'])

        (pointschart + lineschart).properties(
            title=self.video.stem + "\n" + ','.join(f"{s}-{e}" for s, e in self.events)
        ).save(filename)
