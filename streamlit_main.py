# Main function to activate streamlit and run the following command in terminal
# Command: streamlit run C:/Users/vigne/PycharmProjects/FaceMask/streamlit_main.py

import torch
from streamlit_utils import predict, plot_image2, load_model
from torchvision import transforms
import streamlit as st
import cv2


##################################################################################################
st.title('PyTorch Face Mask Detection')

input_img_path = st.file_uploader("Choose a file", type=['png','jpeg','jpg'])

if input_img_path is not None:
    file_details = {"Filename": input_img_path.name, "FileType": input_img_path.type, "FileSize": input_img_path.size}
    st.write(file_details)
    output_img_path = "./saved_images/" + input_img_path.name
else:
    img = st.sidebar.selectbox(
        'Select Image',
        ('test_img1.png', 'test_img2.jpg', 'test_img3.jpg', 'test_img4.jpg')
    )
    input_img_path = "./test_images/" + img
    output_img_path = "./saved_images/" + img
# input_img_path = "C:/Users/vigne/PycharmProjects/FaceMask/test_images/" + img
# output_img_path = "C:/Users/vigne/PycharmProjects/FaceMask/saved_images/" + img

model_choice = st.sidebar.selectbox(
    'Select Model',
    ('FasterRCNN', 'Yolo (Not Available Yet)')
    # ('FasterRCNN', 'Yolo)
)

# Hyperparameters
NUM_CLASSES = 4  # include background as label 0 for FasterRCNN
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

if model_choice == 'FasterRCNN':
    model_path = "./saved_models/model.pt"

st.write('### Source image:')
data_transform = transforms.Compose([transforms.ToTensor()])
image_bgr = cv2.imread(input_img_path)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
st.image(image, width=300)  # image: numpy array
image_tensor = data_transform(image)

# this command will make 3 columns of unequal width
#col1, col2, col3 = st.beta_columns([1,2,3])
#with col2:
#    clicked = st.button('Detect')
clicked = st.button('Detect')
if clicked:
    model = load_model(model_path, NUM_CLASSES, device, backbone='resnet50')
    preds = predict([image_tensor.to(device)], model)
    plot_image2(image, preds[0], output_img_path)

    st.write('### Output image:')
    image = cv2.imread(output_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, width=300)
# ------------------------------------------------------------------
# def collate_fn(batch):
#    return tuple(zip(*batch))


# img_path = "C:/Users/vigne/PycharmProjects/FaceMask/images/"
# label_path = "C:/Users/vigne/PycharmProjects/FaceMask/annotations/"
# model_path = "C:/Users/vigne/PycharmProjects/FaceMask/saved_models/model.pt"

# Hyperparameters
# BATCH_SIZE = 1
# NUM_CLASSES = 4  # include background as label 0 for FasterRCNN
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
