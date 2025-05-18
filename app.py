import streamlit as st
# import torch
# from torchvision import transforms
from PIL import Image
from ultralytics import YOLO, RTDETR
#import matplotlib.pyplot as plt

# RetinaNet imports
# from torchvision.models.detection.retinanet import retinanet_resnet50_fpn, RetinaNetClassificationHead

# Label and color mapping
label_map = {0: "proper", 1: "improper"}
# color_map = {0: "green", 1: "red"}

label_map_retina = {1: "proper", 2: "improper"}
color_map_retina = {1: "green", 2: "red"}

# Load YOLO and RT-DETR models
yolo_model = YOLO("yolo11l_posture.pt")
rtdetr_model = RTDETR("rtdetrl_posture.pt")

rtdetr_model_fingers = RTDETR("rtdetr_fingers.pt")

'''
# Load RetinaNet model
@st.cache_resource
def load_retinanet_model():
    model = retinanet_resnet50_fpn(pretrained=False)
    in_features = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=3  # Background, proper, improper
    )
    model = torch.load(
        "retinanet_full_model_posture.pth",
        map_location="cpu"
    )
    model.eval()
    return model

retina_model = load_retinanet_model()


# Preprocessing
transform = transforms.Compose([transforms.ToTensor()])
'''
# App UI
st.title('Hand Posture Detection Using Deep Learning from Image')
st.subheader('A Streamlit Web App for Hand Posture Detection')
st.write('Improper hand typing posture, such as using excessive force or misaligned fingers placement over and over may strain the muscles in your hand, ' \
'leading to to repetitive strain injuries.')

# App Code
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose a model", ["YOLOv11", "RT-DETR"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    threshold = 0.5

    if st.button("Detect Hand Posture"):
        warning = False
        if model_choice == "YOLOv11":
            results = yolo_model.predict(image)
            pred_img = results[0].plot(labels=True, boxes=True)
            st.image(pred_img, caption="Detected by YOLOv11", use_container_width=True)

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if label_map.get(cls_id, "") == "improper":
                    warning = True

        elif model_choice == "RT-DETR":
            results = rtdetr_model.predict(image, conf=threshold)
            pred_img = results[0].plot(labels=True, boxes=True)
            st.image(pred_img, caption="Detected by RT-DETR", use_container_width=True)

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if label_map.get(cls_id, "") == "improper":
                    warning = True
        '''    
        elif model_choice == "RetinaNet":
            retina_model.eval()
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = retina_model(img_tensor)[0]

            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()

            # Apply threshold manually 
            keep = (scores >= threshold_re) & (labels != 0) # 
            valid_boxes = boxes[keep]
            valid_scores = scores[keep]
            valid_labels = labels[keep]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)

            for i in range(len(valid_boxes)):
                x1, y1, x2, y2 = valid_boxes[i]
                label_id = valid_labels[i]
                score = valid_scores[i]
                label_name = label_map_retina.get(label_id, str(label_id))
                color = color_map_retina.get(label_id, "green")

                ax.add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                ))
                ax.text(
                    x1, y1 - 5,
                    f"{label_name}: {score:.2f}",
                    color=color,
                    fontsize=12,
                    bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2')
                )

            plt.axis('off')
            st.pyplot(fig)
        '''
        if warning:
            st.warning("⚠️ Fix your hand position!")

    elif st.button("Detect Fingers Placement"):
        warning = False
        results = rtdetr_model_fingers.predict(image)
        pred_img = results[0].plot(labels=True, boxes=True)
        st.image(pred_img, caption="Detected by RTDETR", use_container_width=True)
             
        for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if label_map.get(cls_id, "") == "improper":
                    warning = True
        if warning:
            st.warning("⚠️ Fix your finger position!")