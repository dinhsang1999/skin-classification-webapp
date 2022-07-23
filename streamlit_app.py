# --- LIBRARY ---

# Login stage
import streamlit_authenticator as stauth #pip install streamlit-authenticator
import database as db

# Fontend
import streamlit as st
from src.visual import visual_diseases,visual_dataset
from src.utils import get_image_from_lottie,crop_image,load_result,load_model,heatmap,selected_features
from streamlit_lottie import st_lottie

# Backend
from PIL import Image
import numpy as np
import pandas as pd

#emoji: 
# st.set_page_config(layout="wide", page_icon="👨‍🎓", page_title="Skin-classify-web-app")
st.set_page_config( page_icon="👨‍🎓", page_title="Skin Cancer Classification")

ss = st.empty()
# --- USER AUTHENTICATION ---
users = db.fetch_all_users()

usernames = [user['key'] for user in users]
names = [user['name'] for user in users]
hashed_password = [user['password'] for user in users]

col1,col2 = ss.columns((3,1))
with col1:
    st.write("""
    # Free, fast, clear
    # skin classification for you
    """)
with col2:
    st_lottie(get_image_from_lottie(url = "https://assets5.lottiefiles.com/packages/lf20_lsxbjiu9.json"), key = "go2", height=-500,width=-150)
authenticator = stauth.Authenticate(names, usernames, hashed_password,
    "skin_webapp", "abcdef", cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

sc = st.empty()
col_ct,col_fb,col_ins,col_github,col_linkein,_,col_rg_logo,col_rg = sc.columns((4,1,1,1,1,14,1,5))
with col_ct:
    st.write("**Contract:**")

with col_fb:
    '''
        [![Facebook](https://static8.lottiefiles.com/images/v3/footer/social_facebook_2.svg)](https://www.facebook.com/hoangsang2020/) 
    '''
    st.markdown("<br>",unsafe_allow_html=True)
with col_ins:
    '''
        [![Instagram](https://static8.lottiefiles.com/images/v3/footer/social_instagram_2.svg)](https://www.instagram.com/sancho_7.4.99/) 
    '''
    st.markdown("<br>",unsafe_allow_html=True)
with col_github:
    '''
        [![Github](https://static.xx.fbcdn.net/rsrc.php/v3/yV/r/ufIMw_ngzRh.png)](https://github.com/dinhsang1999) 
    '''
    st.markdown("<br>",unsafe_allow_html=True)
with col_linkein:
    '''
        [![Linkein](https://img.icons8.com/ios-glyphs/24/linkedin-circled--v1.png)](https://www.linkedin.com/in/sang-dinh-a31856160) 
    '''
    st.markdown("<br>",unsafe_allow_html=True)
with col_rg_logo:
    '''
        ![Region](https://static8.lottiefiles.com/images/v3/footer/region.svg)
    '''
    st.markdown("<br>",unsafe_allow_html=True)
with col_rg:
    '''
        Region: VietNam
    '''
    st.markdown("<br>",unsafe_allow_html=True)

sd = st.empty()
oo = """
<span style = 'color:#8795a1;font-size:14px;'>
        SkinCancerClassification Webapp is by Design Std.Dinh Hoang Sang of Biomedical Engineering Department in International University
</span>
<span style = 'color:#8795a1;font-size:14px;'>
        Copyright © 2022 Streamlit Inc.
</span>
"""
sd.markdown(oo,
unsafe_allow_html=True)

if authentication_status:
    ss.empty()
    sc.empty()
    sd.empty()
    # --- PAGE TITLE ----
    image_cover = Image.open('./image/cover_2.png')
    st.image(image_cover,use_column_width= True)

    with st.expander('About'):
        st.write("""
        # Skin Diseases Detect Web App

        #### This app will detect `skin diseases`

        ***Skin cancer*** is by far the world's most common cancer. Among different skin cancer types, melanoma is particularly dangerous because of its ability to metastasize. Early detection is the key to success in skin cancer treatment. However, skin cancer diagnostic is still a challenge, even for experienced dermatologists, due to strong resemblances between benign and malignant lesions. To aid dermatologists in skin cancer diagnostic, we developed a deep learning system that can effectively and automatically classify skin lesions in the ISIC dataset. An end-to-end deep learning process, transfer learning technique, utilizing multiple pre-trained models, combining with class-weighted and focal loss function was applied for the classification process. The result was that our modified famous architectures with metadata could classify skin lesions in the ISIC dataset into one of the nine classes: (1) ***Actinic Keratosis***, (2) ***Basel Cell Carcinoma***, (3) ***Benign Keratosis***, (4) ***Dermatofibroma***, (5) ***Melanocytic nevi***, (6) ***Melanoma***, (7) ***Vascular Skin Lesion*** (8) ***Squamous Cell Carcinoma*** (9) ***Unknown*** with 93% accuracy and 97% and 99% for top 2 and top 3 accuracies, respectively. This deep learning system can potentially be integrated into computer-aided diagnosis systems that support dermatologists in skin cancer diagnostic.

        ***SAVE***: 
        ```python
        - Melanoma is malinant (dangerous)
        - Others is benign (but also careful)
        ```

        ##### Some examples:
        """)
    # --- VISUAL DATA ---
    with st.expander('Preview examples of 9 types diseases skin'):
        st.balloons()
        visual_diseases()
    with st.expander('Preview dataset'):
        st.balloons()
        visual_dataset()

    # --- SIDEBAR: User input ---
    authenticator.logout("Logout","sidebar")
    
    st.sidebar.header('User Input')
    selected_box = st.sidebar.selectbox('Model',('Select model','Image','Image&Metadata'),help="Model 1: EfficientNetB0 - Model 2: EfficientNetB2_ns with metadata")
    if selected_box == 'Select model':
        st.markdown("""
        <span style = 'font-size:20px;'> 
        Sellect your
        </span>
        <span style = 'color:#f56aa2;font-size:25px;'>
        Model
        </span>
        <span style = 'font-size:30px;'> 
        !
        </span>
        """,
        unsafe_allow_html=True)

        # st_lottie(get_image_from_lottie(url = "https://assets8.lottiefiles.com/packages/lf20_mxzt4vtn.json"), key = "selectmodel", height=400)
        with st.sidebar:
            st_lottie(get_image_from_lottie(url = 'https://assets10.lottiefiles.com/packages/lf20_lfugvekh.json'), key='load', height=100)

    st.warning("***SCROLL PAGE*** may get error and reload website. Please, wait for running ***DONE***!")
    su = st.empty()
    if selected_box == 'Image':
        '''
        '''
        su.empty()
        selected_image = st.sidebar.file_uploader('Upload image from PC',type=['png', 'jpg'],help='Type of image should be PNG or JPEG')
        if not selected_image:
            with st.sidebar:
                st_lottie(get_image_from_lottie(url = 'https://assets4.lottiefiles.com/packages/lf20_urbk83vw.json'), key = 'giveimage_sidebar',height=200,width=200)
        load_model(selected_box)
        su.success('Download Model ✔️***Done***!!!')
        if selected_image:
            if st.sidebar.checkbox('Crop image',value=False,help='When image have large size and dont focus skin lession'):
                st.write('CROP IMAGE:')
                st.warning('Move the box so that focus the target')
                crop_image = crop_image(selected_image)
                crop_image = np.array(crop_image.convert("RGB"))
                crop_image = crop_image.astype(np.int16)
            else:
                crop_image = Image.open(selected_image)
                crop_image = np.array(crop_image.convert("RGB"))
            
            op_mod_gc = st.selectbox("Select mode show model's target, you should check all mode:",('XGradCAM','GradCAM','ScoreCAM','LayerCAM','EigenCAM','EigenGradCAM','AblationCAM','GradCAMPlusPlus','FullGrad'),help='More: https://github.com/jacobgil/pytorch-grad-cam')
            if op_mod_gc == 'ScoreCAM':
                st.warning('ScoreCAM need one more minute, Dont stop it!')
            if op_mod_gc == 'AblationCAM':
                st.warning('AblationCAM need one more minute, Dont stop it!')
            st.write('##### Results:')
            if st.button('Show result'):
                results = load_result(selected_box,crop_image)
                df_disease = pd.DataFrame()
                df_disease = df_disease.reset_index(drop=True)
                df_disease['diseases'] = ['MEL','NV','BCC','BKL','AK','SCC','VASC','DF','unknown']
                for i in range(5):
                    results[i][0] = np.around(results[i][0],4)*100
                    df_disease['trainer_' + str(i)] = results[i][0]
                st.dataframe(df_disease.style.highlight_max(axis=0,color='pink',subset=['trainer_0','trainer_1','trainer_2','trainer_3','trainer_4']))
                with st.spinner("Drawing heatmap..."):
                    image,image_ori,image_scale = heatmap(selected_box,crop_image,Cam=op_mod_gc) 
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        st.header('Original')
                        st.image(image_ori)
                    with c2:
                        st.header('Scaled')
                        st.image(image_scale)
                    with c3:
                        st.header('Heat-map')
                        st.image(image)

    if selected_box == 'Image&Metadata':
        '''
        '''
        su.empty()
        selected_image = st.sidebar.file_uploader('Upload image from PC',type=['png', 'jpg'],help='Type of image should be PNG or JPEG')
        if not selected_image:
            with st.sidebar:
                st_lottie(get_image_from_lottie(url = 'https://assets4.lottiefiles.com/packages/lf20_urbk83vw.json'), key = 'giveimage_sidebar',height=200,width=200)
        su.success('Download Model ✔️ ***Done***!!!')
        load_model(selected_box)
        if selected_image:
            if st.sidebar.checkbox('Crop image',value=False,help='When image have large size and dont focus skin lession'):
                st.write('CROP IMAGE:')
                st.warning('Move the box so that focus the target')
                crop_image = crop_image(selected_image)
                crop_image = np.array(crop_image.convert("RGB"))
                crop_image = crop_image.astype(np.int16)
            else:
                crop_image = Image.open(selected_image)
                crop_image = np.array(crop_image.convert("RGB"))
            
            st.warning('***INPUT-USER*** appeared new features, including: ```GENDER,AGE,POSITION```. Change these features before!!!')
            op_mod_gc = st.selectbox("Select mode show model's target, you should check all mode:",('XGradCAM','GradCAM','ScoreCAM','LayerCAM','EigenCAM',' EigenGradCAM','AblationCAM','GradCAMPlusPlus','FullGrad'),help='More: https://github.com/jacobgil/pytorch-grad-cam')
            if op_mod_gc == 'ScoreCAM':
                st.warning('ScoreCAM need one more minute, Dont stop it!')
            if op_mod_gc == 'AblationCAM':
                st.warning('AblationCAM need one more minute, Dont stop it!')
            st.write('##### Results:')
            features = selected_features(crop_image)
            if st.button('Show result'):
                results = load_result(selected_box,crop_image,meta_features=features)
                df_disease = pd.DataFrame()
                df_disease = df_disease.reset_index(drop=True)
                df_disease['diseases'] = ['MEL','NV','BCC','BKL','AK','SCC','VASC','DF','unknown']
                for i in range(5):
                    results[i][0] = np.around(results[i][0],4)*100
                    df_disease['trainer_' + str(i)] = results[i][0]
                st.dataframe(df_disease.style.highlight_max(axis=0,color='pink',subset=['trainer_0','trainer_1','trainer_2','trainer_3','trainer_4']))

                with st.spinner("Drawing heatmap..."):
                    image,image_ori,image_scale = heatmap(selected_box,crop_image,Cam=op_mod_gc,meta_features=features) 
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        st.header('Original')
                        st.image(image_ori)
                    with c2:
                        st.header('Scaled')
                        st.image(image_scale)
                    with c3:
                        st.header('Heat-map')
                        st.image(image)
