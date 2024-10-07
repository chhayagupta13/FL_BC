import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st

# Load the centralized server trained base model
base_model_path = "vgg_approach1.keras"
base_model = torch.load(base_model_path, map_location=torch.device('cpu'))  # Load the base model on CPU

# Load the federated learned model
federated_model_path = "federated_model_approach1.keras"
federated_model = torch.load(federated_model_path, map_location=torch.device('cpu'))  # Load the federated model on CPU

class_names = ['benign', 'malignant', 'normal']

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to make prediction
def predict(image, model):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
    predicted_class_index = np.argmax(output.numpy())
    predicted_class = class_names[predicted_class_index]
    return predicted_class

# Streamlit app
st.title("Breast Cancer Prediction using VGG 16 Algorithm")
st.write("Please select the model:")
model_selection = st.radio("", ("Use federated learnt model", "Use centralized server trained base model"))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    if model_selection == "Use federated learnt model":
        predicted_class = predict(image, federated_model)
        st.markdown(f"**Predicted class using federated learnt model:** {predicted_class}")
    else:
        predicted_class = predict(image, base_model)
        st.markdown(f"**Predicted class using centralized server trained base model:** {predicted_class}")
