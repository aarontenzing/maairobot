from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import os
import sys
import torch
from torchvision.models import resnet50
from torchvision import transforms
from collections import OrderedDict
import json


app = Flask(__name__)

model = None
embeddings = [] # Store the embeddings of the frames
length_intial_embeddings = 0
CHECKPOINT_PATH = 'models/resnet50_weights_best_acc.tar'  # Path to the PyTorch checkpoint file
NUM_CLASSES = 1081  # Number of classes for the Pl@ntNet-300K dataset
UPLOAD_FOLDER = 'frames' # Folder to store the uploaded frames
toggle_samples = False

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def clear_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')

def encoder(model, input_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        embedding = model(input_tensor)  # Get features from the encoder

    return embedding.cpu().numpy().flatten()

def load_model():
    # Initialize model
    model = resnet50(num_classes=NUM_CLASSES)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Keep all layers except the last one
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)
            
        # Adjust state_dict keys
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
            
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully.")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
        
    return model

def preprocess_image(image):
    """
    Preprocess the input image for ResNet-50.
    """
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
        
def load_img(img_path, target_size=None):
    img = Image.open(img_path)
    if target_size:
        img = img.resize(target_size)
    return np.array(img)  # Convert to a numpy array for matplotlib
        
def visualize_embeddings(embeddings, samples_vis=False):
    print("Visualize embeddings...")
    embeddings = np.array(embeddings) # nodig om tsne toe te passen
    features = tsne.fit_transform(embeddings) # Fit the t-SNE model to the embeddings
    embeddings = embeddings.tolist() # terug naar lijst

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    if samples_vis:
        samples = sorted(os.listdir("samples"), key=lambda x: int(x.split('.')[0]))
        print("Images in sample directory: ", samples)

        # Loop through each point and add the corresponding thumbnail
        for idx, point in enumerate(features[:length_intial_embeddings]):
                x, y = point[0], point[1]
                    
                plt.scatter(x, y, alpha=0)  # Hide the original points
                    
                # Load and create thumbnail from images in samples directory
                img = load_img(os.path.join("samples", samples[idx]), target_size=(50, 50))  # Resize thumbnail
                imagebox = OffsetImage(img, zoom=0.8)  # Adjust zoom as needed
                
                # Create an annotation box with the thumbnail
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.1)
                
                # Add the annotation box to the plot
                plt.gca().add_artist(ab)

    # Do for captured frames    
    frames = sorted(os.listdir(UPLOAD_FOLDER), key=lambda x: int(x.split('.')[0]))
    print("Images in frames directory: ", frames)

    # Loop through each point and add the corresponding thumbnail
    for idx, point in enumerate(features[length_intial_embeddings:]):
            x, y = point[0], point[1]
                
            plt.scatter(x, y, alpha=0)  # Hide the original points
                
            # Load and create thumbnail for the image
            img = load_img(os.path.join("frames", frames[idx]), target_size=(50, 50))  # Resize thumbnail
            imagebox = OffsetImage(img, zoom=0.8)  # Adjust zoom as needed
            
            # Create an annotation box with the thumbnail
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.1)
            
            # Add the annotation box to the plot
            plt.gca().add_artist(ab)
    
    plt.title("Image Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.autoscale(True)
    plt.grid()
    
    plot_image_path = os.path.join('static/images', 'embeddings_plot.png')  # Save in the static folder
    plt.savefig(plot_image_path)
    plt.close()
        
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/toggle", methods=['POST'])
def toggle():
    global toggle_samples
    global embeddings
    toggle_samples = not toggle_samples
    print(f"Toggled: is_on is now {toggle_samples}")
    visualize_embeddings(embeddings, samples_vis=toggle_samples)
    return redirect(url_for('home'))   


@app.route('/upload', methods=['POST', 'GET'])
def plot():
    global toggle_samples
    global embeddings
    if 'frame' not in request.files:
        print("No file part") # Returns an error if 'frame' is not in files
    else: 
        frame = request.files['frame']
        filename = str(len(embeddings)-length_intial_embeddings) + ".jpg" 
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        frame.save(file_path)
        frame = Image.open(file_path)
        frame = preprocess_image(frame)
        embedding = encoder(model, frame)
        embeddings.append(embedding) # Pass frame (image) through the PlantNet network and retrieve the embedding

    visualize_embeddings(embeddings, samples_vis=toggle_samples) 
    
    return redirect(url_for('home'))       

@app.route('/reset', methods=['POST'])
def reset():
    global embeddings
    # Remove embeddings from list
    print(f"Resizing embeddings vector with length {len(embeddings)}, to initial length of {length_intial_embeddings}.")
    embeddings = embeddings[:length_intial_embeddings]
    clear_images_in_directory(UPLOAD_FOLDER) # Clear directory of frames
    visualize_embeddings(embeddings, samples_vis=toggle_samples) 
    return redirect(url_for('home'))       

if __name__ == "__main__":
    
    model = load_model() # Load the PlantNet model and weights
    sample_embeddings = "embeddings.json"
    
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, max_iter=1000) # Initialize tsne

    # Load sample datapoints voor tsne
    with open(sample_embeddings, 'r') as json_file:
        json_data = json.load(json_file)

    # Check available smaples 
    embeddings = [json_data[idx]['feature'] for idx in range(len(os.listdir("samples")))] 

    length_intial_embeddings = len(embeddings)
    print(f"Length of initial embedding vector with samples: {length_intial_embeddings}")

    # size = sys.getsizeof(embeddings)
    # print("Total size list in bytes:", size)

    # Remove captured frames
    clear_images_in_directory(UPLOAD_FOLDER)

    visualize_embeddings(embeddings, samples_vis=False) 
    
    app.run(debug=True)