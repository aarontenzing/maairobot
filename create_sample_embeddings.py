import os
import json
from PIL import Image
import cv2 as cv
from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA
import pickle

import torch
from torchvision.models import resnet18
from torchvision import transforms

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),                # Resize for more flexibility in cropping
        transforms.CenterCrop(224),            # Center crop to 224x224 (matches ResNet50)
        transforms.ToTensor(),                 # Convert to tensor
        transforms.Normalize(                   # Normalize with ResNet50's mean and std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
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
    embeddings = []
    directory_samples = "samples"
    samples = sorted(os.listdir(directory_samples), key=lambda x: int(x.split('.')[0]))
    print(f"Samples {samples[:5]}: len {len(samples)}")
    
    # Initialize model
    print("Initializing model...")
    checkpoint_path = 'resnet18_weights_best_acc.tar'  # Path to the PyTorch checkpoint file
    dir_path_model = 'models/'
    num_classes = 1081  # Number of classes for the Pl@ntNet-300K dataset

    model = resnet18(num_classes=num_classes)
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
        width, height = frame.size
        cropped_img = frame.crop((0, 200, width, height))  # (left, upper, right, lower) bounds
        input_tensor = preprocess_image(cropped_img)
        features = encoder(model, input_tensor)
        embeddings.append(features.cpu().numpy().flatten().tolist())

    print("Fitting PCA model...")
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    # Save the PCA model to a file
    with open('pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    
    print("PCA model saved successfully.")

    