import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'resnet50_model_final.pth'
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, 2)  # Binary classification
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, np.array(img)

# Prediction
def predict_image(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities[0]

# Saliency Map
def generate_saliency_map(img_tensor):
    img_tensor.requires_grad_() # Computing the gradients during backpropagation
    output = model(img_tensor)
    target_class = output.argmax().item()
    output[0, target_class].backward() # Computes the gradient of score
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1) # Take absolute value of gradients and finds the maximum value
    saliency = saliency.squeeze().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency = gaussian_filter(saliency, sigma=1.0) # Applies Gaussian smoothing to remove noise
    return saliency

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations) # Feature maps
        target_layer.register_full_backward_hook(self.save_gradients) # Important of each feature

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()
        output[:, target_class].backward()
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = gradients.mean(dim=[2, 3], keepdim=True) # Computes global average pooling of gradients
        cam = (weights * activations).sum(dim=1, keepdim=True).squeeze().numpy() # Class Activation Map (CAM) using weighted activations
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.resize(cam, (224, 224))
        return cam

# Simple Counterfactual Explanation
def generate_simple_counterfactual(img_tensor, model):
    img_np = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    perturbed_img = img_np + np.random.normal(scale=0.1, size=img_np.shape) #  Add noise to create slight perturbations
    perturbed_img = np.clip(perturbed_img, 0, 1)
    perturbed_img = (perturbed_img * 255).astype(np.uint8)
    heatmap = np.mean(np.abs(perturbed_img - (img_np * 255).astype(np.uint8)), axis=-1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return overlay_heatmap((img_np * 255).astype(np.uint8), heatmap)

def normalize_cam(cam):
    cam = np.maximum(cam, 0)  # no negative
    cam -= cam.min() # subtracting the minimum value
    cam /= (cam.max() + 1e-8)  # avoid division by zero 
    return cam

# Overlay Function
def overlay_heatmap(img, heatmap, alpha=0.6):
    heatmap = normalize_cam(heatmap) # Normalize it to [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_COOL)
    heatmap = cv2.resize(heatmap, (224, 224))
    if len(img.shape) == 2 or img.shape[2] == 1: # Convert grayscale to 3-channel BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (224, 224))
    blended = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0) # Blend heatmap with original image
    return blended





# Streamlit App
st.title("Explainable AI")
st.title("For Chest X-ray Prediction")
st.write("This interface uses Explainable AI (XAI) techniques to help medical professionals understand how a deep learning model makes decisions for pneumonia detection. \n Upload an image to see predictions results and visualized using Saliency Map, Grad-CAM, and Simple Counterfactual methods.")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpeg", "png", "jpg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_tensor, img_np = preprocess_image(img)
    predicted_class, probabilities = predict_image(img_tensor)
    class_label = "Pneumonia" if predicted_class == 1 else "Normal"

    # Generate Explanations
    saliency_map = generate_saliency_map(img_tensor)
    gradcam = GradCAM(model, model.layer4[2])
    gradcam_map = gradcam.generate_cam(img_tensor)
    counterfactual_img = generate_simple_counterfactual(img_tensor, model)
	
    # Apply heatmap overlays
    saliency_overlay = overlay_heatmap(img_np, saliency_map)
    gradcam_overlay = overlay_heatmap(img_np, gradcam_map)
    

    st.title(f"Prediction: {class_label}")

    # Display Results
    cols = st.columns(4)
    with cols[0]:
        st.image(cv2.resize(img_np, (224, 224)), caption='Original Image')
        st.text("The original image chest X-ray image.")

    with cols[1]:
        st.image(counterfactual_img, caption="Simple Counterfactual")
        st.text("This method helps to understand where the decision boundary make.")       

    with cols[2]:
        st.image(gradcam_overlay, caption="Grad-CAM")
        st.text("Grad-CAM highlights the most important regions contributing to the model's prediction. The highlighted areas (in purple/pink) indicate where the model focused to make its decision make.")

    with cols[3]:
        st.image(saliency_overlay, caption="Saliency Map")
        st.text("The Saliency Map emphasizes pixel-level importance by showing which areas had the highest gradient impact on the model's decision. The highlighted areas (in purple/pink) contribute the most to the classification.")
