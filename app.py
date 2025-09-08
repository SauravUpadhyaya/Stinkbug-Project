import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO # Import YOLO from ultralytics

# import gspread
# from google.oauth2.service_account import Credentials
# import datetime

# scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
# client = gspread.authorize(creds)

# Set the title of the app
st.title("Stinkbug Detection App")

# Add user guidance
st.markdown("This web application counts the total number of redbanded stinkbugs in the uploaded image and outlines the detected polygons.")

st.markdown("##### How to Use:")
st.markdown("1. Upload an image file (PNG, JPG, or JPEG) using the file uploader below.")
st.markdown("2. The app processes the image and displays the count of detected redbanded stinkbugs.")

# Implement file upload
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

# Load the .pt model
# model_path = './saved_models_of_1008_images/yolov8s_best.pt' # Use the correct path provided in the context.
model_path = 'saved_models/saved_models_thousand_eight_images/yolov8s_best.pt'
model = None # Initialize model to None
try:
    # Load the model using ultralytics
    model = YOLO(model_path)
    st.success("Model loaded successfully!") # Optional: uncomment to show success message
    # No need to call model.eval() explicitly for YOLO models loaded this way
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Process the uploaded image
def process_image(uploaded_file, model):
    """
    Processes an uploaded image using a YOLOv8 model to detect objects
    and draws bounding boxes or masks on the image.

    Args:
        uploaded_file: The file object uploaded by the user.
        model: The loaded YOLOv8 model.

    Returns:
        A tuple containing:
        - A Pillow Image object with detected objects outlined, or None if an error occurs.
        - The number of detected objects, or 0 if none or an error occurs.
    """
    if uploaded_file is not None and model is not None:
        try:
            # Read the image data
            image_data = uploaded_file.read()
            # Open the image using Pillow
            img = Image.open(io.BytesIO(image_data))

            # Convert Pillow image to NumPy array (HWC format)
            img_np = np.array(img)

            # Perform inference
            # Assuming the model is a YOLOv8 model and its predict method handles NumPy arrays
            # or can be adapted. If the model expects a different format (e.g., PyTorch tensor),
            # conversion will be needed here. YOLOv8 models from Ultralytics usually work with
            # PIL Images or NumPy arrays directly.
            results = model(img_np)

            # Get the number of detected objects
            # Assuming results[0] contains the detection results
            num_detections = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
            # You might need to adjust this based on whether your model outputs masks or boxes
            # For segmentation models, you might check len(results[0].masks)

            # Process results and draw on the image
            # The exact method for drawing depends on the results format.
            # Assuming results is an object with a .plot() method (common in Ultralytics YOLO)
            # which returns an image with detections drawn.
            results_image_np = results[0].plot()

            # Convert the result NumPy array back to a Pillow Image
            results_image_pil = Image.fromarray(results_image_np)

            return results_image_pil, num_detections

        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, 0
    return None, 0

# Display the output
if uploaded_file is not None and model is not None:
    # Process the uploaded image
    results_image_pil, num_detections = process_image(uploaded_file, model)

    # Display the results
    if results_image_pil is not None:
        st.image(results_image_pil, caption="Redbanded Stinkbug Detection", use_container_width=True)
        st.subheader(f"Number of redbanded stinkbugs detected: {num_detections}")

# Add instructions on how to run locally (optional for cloud deployment but good for development)
st.markdown("---")


# # Open your sheet
# sheet = client.open("Stinkbug-Counting-Logs").sheet1

# # Example data to log
# timestamp = str(datetime.datetime.now())
# user = st.text_input("Enter your name:")
# detection_count = num_detections  # or whatever logic you're using

# # Button to submit
# if st.button("Log Detection"):
#     sheet.append_row([timestamp, user, detection_count])
#     st.success("Logged successfully!")