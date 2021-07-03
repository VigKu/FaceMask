# FaceMask
I would like to acknowledge the authors of these task and images. Kaggle problem: https://www.kaggle.com/andrewmvd/face-mask-detection Source of the data: https://makeml.app/datasets/mask

This is an updated version from my old repo: Adv_DS_Captstone_Project: https://github.com/VigKu/Adv_DS_Capstone_Project. Here, I have experimented on object detection and streamlit app development for face masks.


Steps to run streamlit app in local host:
1. Download or git pull the code base.
2. Check requirements.txt for any further installations.
3. Get a copy of model from https://drive.google.com/file/d/1eCeZP1Wjl5rRMmqzzsMNk3XAoJN8u39w/view?usp=sharing and place it in "saved_models" folder.
4. Type the following command in terminal:
            streamlit run FaceMask/streamlit_main.py
5. You will see the link for access in terminal once successful.


Note: The model is not perfect as this is an experiment for streamlit purposes.

Possible steps for further improvement:
1. Data augmentation using albeumentaions (requires augmenting bounding boxes).
2. Hyper parameter optmization.
3. Check if Non Max Suppresion (NMS) is required for post processing.
4. Choice of other models/algorithms such as Mobilenet, EfficientDet, Yolo and etc.
5. Things to ponder about:
            i. Re-sample the data set to have balanced dataset? but it is hard to do so as one image can contain multiple classes.
           ii. Use Focal loss? https://towardsdatascience.com/4-ways-to-improve-class-imbalance-for-image-data-9adec8f390f1 
