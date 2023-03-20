import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/surya/Desktop/tflite_models/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the labels for the objects that can be detected
LABELS = ['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']

# Define the confidence threshold and non-maximum suppression threshold
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Define a function to run the inference on a single image
def detect_objects(image):
    # Preprocess the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (640, 640))
    image_normalized = image_resized / 255.0
    image_float32 = image_normalized.astype(np.float32)
    image_expanded = np.expand_dims(image_float32, axis=0)

    # Run the inference
    interpreter.set_tensor(input_details[0]['index'], image_expanded)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = interpreter.get_tensor(output_details[3]['index'])[0]

    # Postprocess the output
    for i in range(int(num_detections)):
        class_id = int(classes[i])
        score = float(scores[i])
        bbox = boxes[i]
        if score >= CONFIDENCE_THRESHOLD and LABELS[class_id] in LABELS:
            x1 = int(bbox[1] * image.shape[1])
            y1 = int(bbox[0] * image.shape[0])
            x2 = int(bbox[3] * image.shape[1])
            y2 = int(bbox[2] * image.shape[0])
            label = LABELS[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image



# Define the Streamlit app
def app():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Object Detection with YOLOv4-Tiny")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run the object detection inference
        image = np.array(image)
        image_with_boxes = detect_objects(image)
        st.image(image_with_boxes, caption="Output Image", use_column_width=True)

# Run the app
if __name__ == "__main__":
    app()

