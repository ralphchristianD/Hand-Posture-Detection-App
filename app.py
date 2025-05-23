import streamlit as st
from PIL import Image
from ultralytics import YOLO, RTDETR

# Label and color mapping
label_map = {0: "proper", 1: "improper"}

# Load models
yolo_model = YOLO("yolo11l_posture.pt")
rtdetr_model = RTDETR("rtdetrl_posture.pt")
rtdetr_model_fingers = RTDETR("rtdetr_fingers.pt")

# App UI
st.title('Hand Posture Detection Using Deep Learning from Image')
st.subheader('A Streamlit Web App for Hand Posture Detection')
st.write('Improper hand typing posture, such as using excessive force or misaligned fingers placement over and over may strain the muscles in your hand, leading to repetitive strain injuries.')

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose a model", ["YOLOv11", "RT-DETR"])

# Initialize session state for rotated image
if 'rotated_image' not in st.session_state:
    st.session_state.rotated_image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # If rotated, use that image
    if st.session_state.rotated_image is not None:
        image = st.session_state.rotated_image

    st.image(image, caption="Uploaded Image")

    # Rotate right button
    if st.button("🔄 Rotate Right (90° Clockwise)"):
        image = image.rotate(-90, expand=True)
        st.session_state.rotated_image = image
        st.experimental_rerun()  # Refresh to apply rotation

    threshold = 0.5

    if st.button("Detect Hand Posture"):
        warning = False
        if model_choice == "YOLOv11":
            results = yolo_model.predict(image)
            pred_img = results[0].plot(labels=True, boxes=True)
            st.image(pred_img, caption="Detected by YOLOv11")

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if label_map.get(cls_id, "") == "improper":
                    warning = True

        elif model_choice == "RT-DETR":
            results = rtdetr_model.predict(image, conf=threshold)
            pred_img = results[0].plot(labels=True, boxes=True)
            st.image(pred_img, caption="Detected by RT-DETR")

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if label_map.get(cls_id, "") == "improper":
                    warning = True
        if warning:
            st.warning("⚠️ Fix your hand position!")

    elif st.button("Detect Fingers Placement"):
        warning = False
        results = rtdetr_model_fingers.predict(image)
        pred_img = results[0].plot(labels=True, boxes=True)
        st.image(pred_img, caption="Detected by RTDETR")

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            if label_map.get(cls_id, "") == "improper":
                warning = True
        if warning:
            st.warning("⚠️ Fix your finger position!")
