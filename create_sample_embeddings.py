import os
import json
from PIL import Image
from collections import OrderedDict
import numpy as np

import torch
from torchvision.models import resnet50
from torchvision import transforms

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor  = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, channels, height, width]
    return image_tensor 

def encoder(model, input_tensor):
    """This function extracts features from frames using ResNet50 PlantNet"""
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        embedding = model(input_tensor)  # Get features from the encoder
    return embedding

if __name__ == "__main__":
    json_data = []
    directory_samples = "samples"
    samples = sorted(os.listdir(directory_samples), key=lambda x: int(x.split('.')[0]))
    print(f"Samples {samples}: len {len(samples)}")
    
    # Initialize model
    print("Initializing model...")
    checkpoint_path = 'resnet50_weights_best_acc.tar'  # Path to the PyTorch checkpoint file
    dir_path_model = 'models/'
    num_classes = 1081  # Number of classes for the Pl@ntNet-300K dataset

    model = resnet50(num_classes=num_classes)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Keep all layers except the last one

    # Load the checkpoint
    try:
        checkpoint = torch.load(dir_path_model + checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)  # Use 'state_dict' if it exists, otherwise the checkpoint itself

        # Adjust state_dict keys if necessary (e.g., remove 'module.' prefix)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')  # Adjust this as needed based on the key inspection
            new_state_dict[new_key] = v

        # Load the adjusted state_dict into the model
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    print("Encoding frames...")
    for i, filename in enumerate(samples):
        frame = Image.open("samples/" + filename)
        input_tensor = preprocess_image(frame)
        features = encoder(model, input_tensor)

        # write to json
        data = {
            "filename": f"{i}.jpg",
            "feature": features.cpu().numpy().flatten().tolist()
        }
        json_data.append(data)
    
    print("Writing data to json...")
    with open("embeddings.json", 'w') as json_file:
        json.dump(json_data, json_file, indent=4)  # Write with pretty printing

    