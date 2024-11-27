# flask_app.py

from flask import Flask, request, jsonify
import os
import cv2
import torch
import torchvision.models as models
from torchvision.transforms import v2
from torch import nn
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from datetime import datetime
from tqdm import tqdm
import hashlib
import time
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

client = QdrantClient(url="http://localhost:6333")

video_emb_dim = 500
distance = Distance.EUCLID

client.delete_collection('video')
client.create_collection(
    collection_name="video",
    vectors_config=VectorParams(size=video_emb_dim, distance=distance)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.677

# Load AlexNet model and extract convolutional layers
alexnet = models.alexnet(weights="AlexNet_Weights.DEFAULT").to(device)
conv_layers = nn.Sequential(*list(alexnet.features.children()))
alexnet.eval()

# Define transformations
transforms = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def retrieve_frames(video_path, start_time, fps=1):
    """
    Retrieves frames from a video at a specified frame rate between a start and end time
    and returns them as a list of torch tensors.

    :param video_path: Path to the video file
    :param start_time: Start time in seconds
    :param end_time: End time in seconds
    :param fps: Frame rate at which to extract frames (default is 2 fps)
    :return: List of frames as NumPy ndarrays
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return torch.empty()

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / video_fps

    # Convert start and end times to frame numbers
    start_frame = int(start_time * video_fps)
    end_frame = total_duration

    # Set the video capture to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Calculate the interval between frames to extract
    interval = int(video_fps / fps)

    # Extract frames
    frames = []
    frame_number = start_frame
    while cap.isOpened() and frame_number <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Append the frame to the list
        frame = torch.from_numpy(frame).permute(2, 0, 1) / 255
        frames.append(transforms(frame))

        # Move to the next frame to extract
        frame_number += interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Release the video capture object
    cap.release()
    frames = torch.stack(frames)
    return frames.detach().cpu()

class MaxPoolConvOutputs(nn.Module):
    def __init__(self, conv_layers):
        super(MaxPoolConvOutputs, self).__init__()
        self.conv_layers = conv_layers
        self.pool = nn.AdaptiveMaxPool2d((1, 1))  # Pool to a single value per channel

    def forward(self, x):
        outputs = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):  # Apply max pooling only on Conv2d outputs
                pooled_output = self.pool(x).squeeze(-1).squeeze(-1)  # Remove spatial dims
                outputs.append(pooled_output)
        return outputs

def get_frames_emb(video):
    max_pool_extractor = MaxPoolConvOutputs(conv_layers)

    with torch.no_grad():
        max_pooled_outputs = max_pool_extractor(video.to(device))

    result = torch.cat(max_pooled_outputs, dim=1)

    return result

def normalize_frames(video):
    # Step 1: Average across frames (mean along the first dimension)
    avg_emb = torch.mean(video, dim=0)

    # Step 2: Zero-mean normalization (subtract the mean of the vector)
    mean_value = torch.mean(avg_emb)
    zero_mean_emb = avg_emb - mean_value

    # Step 3: ℓ2-normalization (normalize by the L2 norm)
    l2_norm = torch.norm(zero_mean_emb, p=2)
    l2_normalized_emb = zero_mean_emb / l2_norm

    return l2_normalized_emb


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(1152, 1500)  # Assuming input images are 28x28
        self.fc2 = nn.Linear(1500, 1000)
        self.fc3 = nn.Linear(1000, 500)  # Output embedding of size 500

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output embedding
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings to have unit norm
        return x

base_model = EmbeddingNet().to(device)
base_model.load_state_dict(torch.load('api/model_77.pt', map_location=device))
base_model.eval()


@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get the video file from the request
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No files"})

        video_file = request.files['file']

        # Save the uploaded file to a temporary location
        input_file_path = f"data/video.mp4"
        video_file.save(input_file_path)


        # Process the video and get the embedding tensor
        frames = retrieve_frames(input_file_path, start_time=0, fps=2)
        emb = get_frames_emb(frames)
        embedding_tensor = normalize_frames(emb)
        # return jsonify({"success": "true", "desc": embedding_tensor})
        with torch.no_grad():
            descriptor = base_model(embedding_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        # return jsonify({"success": True, "desc": descriptor})
        search_result = client.query_points(
            collection_name="video",
            query=descriptor,
            with_payload=True,
            limit=1
        ).points
        print(search_result)
        if len(search_result) > 0:
            name2 = search_result[0].payload['name']
            if search_result[0].score < THRESHOLD:
                return jsonify({
                    "success": True,
                    "video_name": name2
                })
                
        # Convert the embedding tensor to a list for JSON serialization
        current_timestamp = str(time.time())
        hashed_hex = hashlib.sha256(current_timestamp.encode()).hexdigest()
        hashed_id = int(hashed_hex, 16) % (2**64)
        client.upsert(
            collection_name="video",
            points=[PointStruct(id=hashed_id, vector=descriptor, payload={'name': f'video_{hashed_id}'})]
        )
        # Return success response with the embedding tensor
        
        return jsonify({
            "success": False,
            "error": "Duplicate not found"
        })
        
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)