# FaceMask
I would like to acknowledge the authors of these task and images. Kaggle problem: https://www.kaggle.com/andrewmvd/face-mask-detection Source of the data: https://makeml.app/datasets/mask

This is an improvement from my old repo: Adv_DS_Captstone_Project. Here, I have experimented on object detection and streamlit app development for face masks.


Steps to run streamlit app in local host:
1. Download or git pull the code base.
2. Check requirements.txt for any further installations.
3. Get a copy of model from https://drive.google.com/file/d/1eCeZP1Wjl5rRMmqzzsMNk3XAoJN8u39w/view?usp=sharing.
4. Type the following command in terminal:
            streamlit run FaceMask/streamlit_main.py
5. You will see the link for access in terminal once successful.


Note: The model is not perfect as this is an experiment for streamlit purposes.

Possible steps for further improvement:
1. Data augmentation using albeumentaions (requires augmenting bounding boxes).
2. Re-sample the data set to have balanced dataset.
3. Hyper parameter optmization.
4. Check if Non Max Suppresion (NMS) is required for post processing.
5. Choice of other models such as Mobilenet, EfficientDet and etc.
