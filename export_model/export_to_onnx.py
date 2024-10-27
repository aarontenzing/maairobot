import torch
from torchvision import models
from collections import OrderedDict

# Define paths and parameters
checkpoint_path = 'resnet50_weights_best_acc.tar'  # Path to the PyTorch checkpoint file
onnx_model_path = 'export_model\resnet50_weights_best_acc.onnx'  # Desired output path for the ONNX model
num_classes = 1081  # Number of classes for the Pl@ntNet-300K dataset

# Load the model (ResNet-50) with the appropriate number of classes
model = models.resnet50(num_classes=num_classes)

# Load the checkpoint
try:
    checkpoint = torch.load(checkpoint_path, map_location='gpu')
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

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor with the shape that ResNet expects: (batch_size, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
try:
    torch.onnx.export(model, dummy_input, onnx_model_path, do_constant_folding=True)   
    print(f"Model has been successfully converted to ONNX format and saved at '{onnx_model_path}'")
except Exception as e:
    print(f"Error during ONNX export: {e}")
    exit(1)


