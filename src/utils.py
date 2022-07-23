# --- LIBRARY ---
from json import load

import requests
import streamlit as st
import numpy as np
import timm
import torch
import os
import urllib
import cv2
import random
from PIL import Image
import albumentations as A
from src.model import MelanomaNet,BaseNetwork,MetaMelanoma
from streamlit_cropper import st_cropper
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


# --- LOTTIE ---
def load_lottieurl(url: str):
    """
    It takes a URL, makes a request to that URL, and returns the JSON response
    
    :param url: The url of the lottie file
    :type url: str
    :return: A dictionary
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    """
    It opens the file at the given filepath, reads the contents, and then parses the contents as JSON
    
    :param filepath: The path to the lottie file
    :type filepath: str
    :return: A dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)

def get_image_from_lottie(url=None,filepath=None):
    '''
    Augs:
        - url (String): url from lottie website
        - filepath (String): path to lottie file
    Return: 
        image
    Note:
        - Choose one: url or filepath, not both. If both, the function will be return image from url
    '''
    if url != None:
        return load_lottieurl(url)
    elif filepath != None:
        return None #FIXME:
    else:
        return None
    return

def crop_image(image):
    """
    > The function takes an image as input and returns a cropped image
    
    :param image: The image to be cropped
    :return: The cropped image
    """
    img = Image.open(image)
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=False)
    crop_image = st_cropper(img,aspect_ratio=(1,1),box_color='#ff4b4b',realtime_update=realtime_update)
    return crop_image

def clear_folder(path):
    """
    It takes a path as an argument and deletes all the files in that path
    
    :param path: the path to the folder you want to clear
    """
    list_folder =  os.listdir(path)
    if list_folder != 0:
        for f in list_folder:
            os.remove(os.path.join(path,f))

# @st.experimental_memo(show_spinner=False,ttl=3600*24,max_entries=2)
# @st.cache(allow_output_mutation=True,ttl=3600*24,max_entries=2,show_spinner=False)
# @st.experimental_memo(show_spinner=False)
# @st.cache(allow_output_mutation=True,show_spinner=False)
# @st.experimental_memo(show_spinner=False,max_entries=1)
# @st.cache(allow_output_mutation=True,show_spinner=False,max_entries=1)
@st.experimental_memo(show_spinner=False,max_entries=1)
def load_model(model_name):
    """
    It downloads the model from the github repository and saves it in the folder `model`
    
    :param model_name: The name of the model you want to use
    """
    os.makedirs('model',exist_ok = True)
    clear_folder('model')

    if model_name == 'Image':
        with st.spinner(" ⏳ Downloading model... this may take awhile! \n Don't stop it!"):
            url_init = 'https://github.com/dinhsang1999/streamlit-skin-diseases-classifications-cloud/releases/download/efficientnet_b0/'
            for i in range(5):
                url = url_init + 'efficientnet_b0_512_fold' + str(i) + '.pth'
                path_out = os.path.join('model','efficientnet_b0_512_fold' + str(i) + '.pth')
                urllib.request.urlretrieve(url, path_out)
    
    if model_name == 'Image&Metadata':
        with st.spinner(" ⏳ Downloading model... this may take awhile! \n Don't stop it!"):
            url_init = 'https://github.com/dinhsang1999/streamlit-skin-diseases-classifications-cloud/releases/download/efficientnet_b2/'
            for i in range(5):
                url = url_init + 'efficientnet_b2_512_meta_fold' + str(i) + '.pth'
                path_out = os.path.join('model','efficientnet_b2_512_meta_fold' + str(i) + '.pth')
                urllib.request.urlretrieve(url, path_out)

def load_result(model_name,image,meta_features=None):
    """
    It loads the model, transforms the image, and returns the prediction.
    
    :param model_name: The name of the model you want to use
    :param image: The image you want to predict
    :param meta_features: A list of meta features
    """
    accuracy_5 = []
    device = "cpu"
    if model_name == 'Image':
        with st.spinner("Calculating results..."):
            for i in range(5):
                model = BaseNetwork('efficientnet_b0')
                model.to(device)
                path_model = os.path.join('model', 'efficientnet_b0_512_fold' + str(i) + '.pth')
                model_loader = torch.load(path_model,map_location=torch.device('cpu'))
                # Delete modul into model to train 1 gpu
                model_loader = {key.replace("module.", ""): value for key, value in model_loader.items()}
                model.load_state_dict(model_loader)
                # Switch model to evaluation mode
                model.eval()
                # Transform image
                list_agu = [A.Normalize()]
                transform = A.Compose(list_agu)

                img = cv2.resize(image,(512,512))
                transformed = transform(image=img)
                img = transformed["image"]

                img = img.transpose(2, 0, 1)
                img = torch.tensor(img).float()
                img = img.to(device)
                img = img.view(1, *img.size()).to(device)

                with torch.no_grad():
                    pred = model(img.float())
                
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = pred.cpu().detach().numpy()
                accuracy_5.append(pred)
    
    if model_name == 'Image&Metadata':
        with st.spinner("Calculating results..."):
            for i in range(5):
                model = MetaMelanoma(out_dim=9,n_meta_features=20,n_meta_dim=[int(nd) for nd in "512,128".split(',')],network="efficientnet_b2")
                model.to(device)
                path_model = os.path.join('model', 'efficientnet_b2_512_meta_fold' + str(i) + '.pth')
                model_loader = torch.load(path_model,map_location=torch.device('cpu'))
                # Delete modul into model to train 1 gpu
                model_loader = {key.replace("module.", ""): value for key, value in model_loader.items()}
                model.load_state_dict(model_loader)
                # Switch model to evaluation mode
                model.eval()
                # Transform image
                list_agu = [A.Normalize()]
                transform = A.Compose(list_agu)

                img = cv2.resize(image,(512,512))
                transformed = transform(image=img)
                img = transformed["image"]

                img = img.transpose(2, 0, 1)
                img = torch.tensor(img).float()
                img = img.to(device)
                img = img.view(1, *img.size()).to(device)

                with torch.no_grad():
                    pred = model(img.float(),meta_features.float())
                
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = pred.cpu().detach().numpy()
                accuracy_5.append(pred)

    st.success('✔️ Done!!!')
    return accuracy_5

def heatmap(model_name,image,Cam):
    """
    **heatmap** takes in a model name, an image, and a GradCAM function, and returns a heatmap, the
    original image, and the image scale
    
    :param model_name: the name of the model you want to use
    :param image: the image you want to get the heatmap for
    :param Cam: GradCAM or GradCAM++
    :param meta_features: a list of metadata features, e.g.
    [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0
    :return: The heatmap, the original image, and the scale of the original image.
    """
    image_ori = image
    Cam = mod_gc(Cam)
    
    if model_name == 'Image':
        model = timm.create_model('efficientnet_b0',pretrained=True,num_classes=9)
        model_loader = torch.load(os.path.join('model',random.choice(os.listdir('model'))),map_location=torch.device('cpu'))
        model_loader = {key.replace("module.", ""): value for key, value in model_loader.items()}
        model.load_state_dict(model_loader,strict=False)
        target_layers = [model.conv_head]
        cam_image, image_scale = back_heatmap(model,image,512,target_layers,Cam)
    
    if model_name == 'Image&Metadata':
        model = MetaMelanoma(out_dim=9,n_meta_features=20,n_meta_dim=[int(nd) for nd in "512,128".split(',')],network="efficientnet_b2")
        model_loader = torch.load(os.path.join('model',random.choice(os.listdir('model'))),map_location=torch.device('cpu'))
        model_loader = {key.replace("module.", ""): value for key, value in model_loader.items()}
        model.load_state_dict(model_loader,strict=False)
        model = model.enet
        target_layers = [model.conv_head]
        cam_image, image_scale = back_heatmap(model,image,512,target_layers,Cam)
    
    return cam_image, image_ori, image_scale
        

def back_heatmap(model,image,image_size,target_layers,Cam):
    """
    It takes in a model, an image, the image size, the target layers, and the CAM algorithm, and returns
    the CAM image and the original image
    
    :param model: The model that you want to visualize
    :param image: the image you want to generate a heatmap for
    :param image_size: The size of the image to be fed into the model
    :param target_layers: The layers you want to visualize
    :param Cam: The algorithm to use for generating the heatmap
    :return: The heatmap and the original image
    """
    image = cv2.resize(image,(image_size,image_size))
    image_scale = image
    cam_algorithm = Cam 
    image = image[:, :, ::-1]
    image = np.float32(image) / 255
    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=False) as cam:
            cam.batch_size = 30
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=None,
                                aug_smooth=True,
                                eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return cam_image, image_scale

def back_heatmap_meta(model,image,image_size,target_layers,Cam,meta_features):
    """
    This function takes in a model, an image, the image size, the target layers, the CAM algorithm, and
    the meta features and returns the heatmap and the image
    
    :param model: The model you want to use for CAM
    :param image: the image you want to generate a heatmap for
    :param image_size: The size of the image to be fed into the model
    :param target_layers: The layers you want to visualize
    :param Cam: The type of CAM algorithm to use
    :param meta_features: The meta features of the image
    :return: The heatmap and the original image
    """
    image = cv2.resize(image,(image_size,image_size))
    image_scale = image
    cam_algorithm = Cam 
    image = image[:, :, ::-1]
    image = np.float32(image) / 255
    input_tensor = preprocess_image(image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=False) as cam:
            cam.batch_size = 30
            grayscale_cam = cam(input_tensor=(input_tensor.float(),meta_features.float()),
                                targets=None,
                                aug_smooth=True,
                                eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return cam_image, image_scale

def selected_features(image):
    """
    > The function takes an image as input and returns a tensor of features that are selected by the
    user
    
    :param image: The image you want to predict
    """
    selected_sex = st.sidebar.selectbox('Gender',('male','female'),
                                                help="If your gender is other, you should select it from your biological sex")
    selected_age = st.sidebar.slider('Age',0,90,25,
                                                help='Range of Age is 0-90 from our dataset, so results will be better when your age < 90. If you dont know the age, we recommend you select 51 years old because it is mean of age within our dataset ')
    selected_pos = st.sidebar.selectbox('Position',('Anterior torso','Torso','Posterior torso','Lateral torso','Lower extremity','Upper extremity','Head/Neck','Palms/Soles','Oral/Genital','Unknown'),
                                                index=9,
                                                help='Position of the photo which was taken on your body')
    selected_features = []
    #Gender
    if selected_sex == 'male':
        selected_features.append(1)
    else: selected_features.append(0)
    #Age
    mean_image_dataset = 51.085898674765325 
    selected_features.append(selected_age/mean_image_dataset)
    

    #Height/width
    width = image.shape[1]
    mean_width = 2659.5990728227584
    width = width / mean_width
    selected_features.append(width)
    height = image.shape[0]
    mean_height = 1847.6620079716715
    height = height / mean_height
    selected_features.append(height)
    mean, std = cv2.meanStdDev(image)

    #Position
    if selected_pos == 'Anterior torso':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Posterior torso':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Torso':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Lateral torso':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Upper extremity':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Lower extremity':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Head/Neck':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Palms/Soles':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Oral/Genital':
        selected_features.append(1) 
    else: selected_features.append(0)

    if selected_pos == 'Unknown':
        selected_features.append(1) 
    else: selected_features.append(0)

    # Constant from mean dataset
    selected_features.append(mean[0][0]/193.51384562321348)
    selected_features.append(mean[1][0]/150.00096405100817)
    selected_features.append(mean[2][0]/144.4959313877348)

    selected_features.append(std[0][0]/24.040896828810762)
    selected_features.append(std[1][0]/29.031934752265283)
    selected_features.append(std[2][0]/33.118528425542614)

    selected_features = torch.tensor(selected_features)
    selected_features = torch.unsqueeze(selected_features, dim = 0)

    return selected_features

# @st.cache(allow_output_mutation=True,ttl=60)
# def button_states():
#     return {"pressed": None}

def mod_gc(op):
    if op == 'XGradCAM':
        return XGradCAM
    if op == 'GradCAM':
        return GradCAM
    if op == 'ScoreCAM':
        return ScoreCAM
    if op == 'LayerCAM':
        return LayerCAM
    if op == 'EigenGradCAM':
        return EigenGradCAM
    if op == 'AblationCAM':
        return AblationCAM
    if op == 'GradCAMPlusPlus':
        return GradCAMPlusPlus
    if op == 'FullGrad':
        return FullGrad
    if op == 'EigenCAM':
        return EigenCAM

if __name__ == '__main__':
    load_result('Efficient_B0_256')

