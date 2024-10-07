import torch
from torchvision import transforms
from PIL import Image
import numpy as np
# Load the saved model
# Load the saved model
model_path = "federated_model_approach1.keras"
model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model on CPU
class_names = ['benign', 'malignant', 'normal']

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess the input image
input_image_path = "normal (15).png"
input_image = Image.open(input_image_path)
input_tensor = transform(input_image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    model.eval()
    output = model(input_tensor)

# Get the predicted class
predicted_class_index = np.argmax(output.numpy())
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")