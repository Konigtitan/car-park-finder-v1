import os
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import time

import requests

def download_model():
    model_url = "https://drive.google.com/file/d/1gsFMpMvQRQLOHyi5I5bNmN7jDMQvrNzf/view?usp=sharing"
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")

def load_model():
    download_model()
    try:
        model = YOLO("best.pt")
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Process the image using the YOLOv8 model
def process_image(model, image, confidence_threshold):
    if model is None:
        return None

    # Resize the image to 640x640
    resized_image = cv2.resize(image, (640, 640))

    # Perform YOLOv8 inference
    results = model(resized_image, conf=confidence_threshold, verbose=False)

    # Process results
    processed_frame, vacant_count, occupied_count, total_count = process_results(results, resized_image)

    return processed_frame, vacant_count, occupied_count, total_count

# Process the video using the YOLOv8 model
def process_video(model, video_path, confidence_threshold, stop_button, pause_button):
    if model is None:
        return

    # Open the video file or camera
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        st.error("Failed to open the video or camera.")
        return

    # Placeholder for the video frame
    video_placeholder = st.empty()

    while not stop_button:
        if not pause_button:
            ret, frame = video_capture.read()
            if not ret:
                st.warning("End of video or failed to read frame.")
                break

            # Process the frame
            resized_frame = cv2.resize(frame, (640, 640))
            results = model(resized_frame, conf=confidence_threshold, verbose=False)
            processed_frame, vacant_count, occupied_count, total_count = process_results(results, resized_frame)

            # Display the processed frame
            video_placeholder.image(processed_frame, channels="RGB", use_container_width=True)

    # Release the video capture object
    video_capture.release()

# Process YOLOv8 results and draw bounding boxes
def process_results(results, frame):
    if len(results) == 0:
        return frame, 0, 0, 0

    # Access the first result
    result = results[0]

    # Initialize counters for vacant and occupied spots
    vacant_spots = 0
    occupied_spots = 0

    # Get class mapping from the model
    class_names = result.names

    # Process detections
    boxes = []
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # Get coordinates
            xyxy = box.xyxy[0].tolist()

            # Count based on class ID
            class_name = class_names[cls_id].lower()

            # Check different possible class name variations
            if 'empty' in class_name or 'vacant' in class_name or cls_id == 0:
                vacant_spots += 1
            else:
                occupied_spots += 1

            boxes.append([*xyxy, conf, cls_id])

    # Calculate total spots
    total_spots = vacant_spots + occupied_spots

    # Draw bounding boxes and labels on the frame
    for det in boxes:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Determine class name and color
        class_name = class_names[cls_id]
        is_vacant = 'empty' in class_name.lower() or 'vacant' in class_name.lower() or cls_id == 0
        color = (0, 255, 0) if is_vacant else (0, 0, 255)  # Green for vacant, Red for occupied

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{'Vacant' if is_vacant else 'Occupied'}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the frame to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = Image.fromarray(frame_rgb)

    return processed_frame, vacant_spots, occupied_spots, total_spots

# Main Streamlit app
def main():
    st.title("Parking Spot Detection")

    # Load the model
    model = load_model()

    # Sidebar controls
    st.sidebar.title("Controls")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.01)

    # Upload image or video
    uploaded_file = st.sidebar.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file to a permanent location
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type.startswith("image"):
            # Process image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.sidebar.button("Process Image"):
                processed_frame, vacant_count, occupied_count, total_count = process_image(model, np.array(image), confidence_threshold)
                if processed_frame is not None:
                    st.image(processed_frame, caption="Processed Image", use_container_width=True)
                    st.sidebar.text(f"Vacant: {vacant_count}/{total_count} spaces")
                    st.sidebar.text(f"Occupied: {occupied_count}/{total_count} spaces")

        elif uploaded_file.type.startswith("video"):
            # Process video
            stop_button = st.sidebar.button("Stop Video")
            pause_button = st.sidebar.button("Pause Video")
            process_video(model, file_path, confidence_threshold, stop_button, pause_button)

        # Clean up the temporary file
        os.remove(file_path)

    # Start camera
    if st.sidebar.button("Start Camera"):
        st.warning("Camera access is not supported on Streamlit Sharing. Please upload a video file.")

if __name__ == "__main__":
    main()