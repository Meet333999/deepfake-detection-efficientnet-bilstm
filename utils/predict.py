import torch
from torchvision import transforms
from PIL import Image
import os

from models.model import DeepfakeDetector
from utils.video_processing import extract_frames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepfakeDetector()
model.load_state_dict(torch.load("models/DFM_30700.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def predict_video(video_path):

    frame_paths = extract_frames(video_path, "static/frames")

    frames = []
    web_frames = []

    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        frames.append(img)
        
        # Create relative path for web template (e.g., 'frames/frame_0.jpg')
        # We need to just get the part after 'static/'
        filename = os.path.basename(path)
        web_frames.append(f"frames/{filename}")

    frames = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(frames)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "REAL" if pred.item() == 0 else "FAKE"

    return label, confidence.item(), web_frames