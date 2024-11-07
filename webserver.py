from flask import Flask, flash, request, render_template, redirect, url_for
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
import plotly.graph_objs as go
import plotly.offline as pyo


# curl -X POST http://127.0.0.1:5000/upload -F "frame=@/home/tenzing/Pictures/flower2.jpg"
# op de client run: sudo python3 run.py resnet50_retrained_grass_flower.rknn 0


app = Flask(__name__)

model = None
CHECKPOINT_PATH = 'models/resnet50_weights_best_acc.tar'  # Path to the PyTorch checkpoint file

embeddings = [] # Store the embeddings of the frames
class_names = [] # Store the class names of the frames (0 is flower and 1 is grass)
length_intial_embeddings = 0

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
                img = load_img(os.path.join("samples", samples[idx]), target_size=(60, 60))  # Resize thumbnail
                imagebox = OffsetImage(img, zoom=0.8)  # Adjust zoom as needed
                
                # Create an annotation box with the thumbnail
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.1)
                
                # Add the annotation box to the plot
                plt.gca().add_artist(ab)

    # Do for captured frames    
    frames = sorted(os.listdir(UPLOAD_FOLDER), key=lambda x: int(x.split('.')[0]))
    print("Images in frames directory: ", frames)

    if len(frames) is not 0:
        # Loop through each point and add the corresponding thumbnail
        for idx, point in enumerate(features[length_intial_embeddings:]):
                x, y = point[0], point[1]
                    
                plt.scatter(x, y, alpha=0)  # Hide the original points
                    
                # Load and create thumbnail for the image
                img = load_img(os.path.join("frames", frames[idx]), target_size=(60, 60))  # Resize thumbnail
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
        class_name = int(request.form['class']) # Get the class name from the form
        class_names.append(class_name)

        frame = request.files['frame']
        print(f"Class name: {class_name}")
    
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
    global class_names
    
    if len(embeddings) == length_intial_embeddings:
        print("No embeddings to reset.")
        return redirect(url_for('home'))
    
    # Resetting the embeddings and class names
    print(f"Resizing embeddings vector with length {len(embeddings)}, to initial length of {length_intial_embeddings}.")
    embeddings = embeddings[:length_intial_embeddings]
    class_names = []
    clear_images_in_directory(UPLOAD_FOLDER) # Clear directory of frames
    visualize_embeddings(embeddings, samples_vis=toggle_samples) 
    return redirect(url_for('home'))       

@app.route('/visualize_3d', methods=['GET'])
def visualize_3d():
    print("Visualize embeddings...")
    global embeddings
    global class_names

    if len(class_names) == 0:
        print("No embeddings to visualize.")
        return redirect(url_for('home'))  

    embeddings = np.array(embeddings)  # This line ensures embeddings is a NumPy array

    features = tsne_3d.fit_transform(embeddings) # Fit the t-SNE model to the embeddings
    features = features[length_intial_embeddings:]  # Exclude the initial embeddings

    embeddings = embeddings.tolist()  # Convert embeddings back to a list

    class_names = np.array(class_names)  # This line ensures class_names is a NumPy array
    
    # Separate features based on labels
    flower_points = features[class_names == 0]  # Red points (Flower)
    grass_points = features[class_names == 1]  # Green points (Grass)

    class_names = class_names.tolist()  # Convert class_names back to a list

    # Create a 3D scatter plot using the transformed features
    fig = go.Figure()

    # Add trace for "Grass" points
    fig.add_trace(go.Scatter3d(
        x=grass_points[:, 0],
        y=grass_points[:, 1],
        z=grass_points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(0, 255, 0, 0.5)',  # Green color for grass
            line=dict(width=1)
        ),
        name='Grass'  # Label in the legend
    ))

    # Add trace for "Flower" points
    fig.add_trace(go.Scatter3d(
        x=flower_points[:, 0],
        y=flower_points[:, 1],
        z=flower_points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(255, 0, 0, 0.5)',  # Red color for flowers
            line=dict(width=1)
        ),
        name='Flower'  # Label in the legend
    ))

    # Update layout to add title and axis labels
    fig.update_layout(
        title="3D Visualization of Embeddings: Grass vs. Flower",  # Plot title
        scene=dict(
            xaxis_title="Dimension 1",  # X-axis label
            yaxis_title="Dimension 2",  # Y-axis label
            zaxis_title="Dimension 3"   # Z-axis label
        ),
        width=1500,  # Set figure width
        height=900,  # Set figure height
        showlegend=True,  # Ensure the legend is displayed
        legend=dict(
            title="Legend",
            itemsizing="constant"
        )
    )

    # Generate HTML for the plot
    plot_html = fig.to_html(full_html=False)

    return render_template('3d_plot.html', plot_html=plot_html)

if __name__ == "__main__":
    
    model = load_model() # Load the PlantNet model and weights
    sample_embeddings = "embeddings.json"
    
    tsne = TSNE(n_components=2, perplexity=25, random_state=42)
    tsne_3d = TSNE(n_components=3, perplexity=25, random_state=42)

    # Load sample datapoints voor tsne
    with open(sample_embeddings, 'r') as json_file:
        json_data = json.load(json_file)

    # Check available smaples 
    embeddings = [json_data[idx]['feature'] for idx in range(len(os.listdir("samples")))] 

    length_intial_embeddings = len(embeddings)
    print(f"Length of initial embedding vector with samples: {length_intial_embeddings}")
    
    clear_images_in_directory(UPLOAD_FOLDER) # Remove captured frames from the directory

    visualize_embeddings(embeddings, samples_vis=False) # Visualize the embeddings first
    
    app.run(host="0.0.0.0", debug=True)
