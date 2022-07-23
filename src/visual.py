import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def visual_diseases():
    """
    `This function is used to display the visual representation of the diseases.`
    """
    st.write('`Melanoma` is the deadliest form of skin cancer. It regularly occurs in shades of brown or black, although it can also be pink, red, purple, blue, or white. Sunlight and tanning beds emit ultraviolet radiation that promotes malignant tumors. Melanoma is often treatable if detected and diagnosed early. However, if this were to happen, the cancer may spread to other body parts, and the treatment would be dangerously hard;')
    image_mel = Image.open('./image/MEL.png')
    st.image(image_mel)
    st.write('Actinic Keratosis is also referred to as solar keratosis and is a precancerous, scaly area that develops on sun-damaged skin. It is thought to be an early stage of cutaneous SCC;')
    image_ak = Image.open('./image/AK.png')
    st.image(image_ak)
    st.write('Basel Cell Carcinoma is also the most common. Rarely is BCC spread. Typical symptoms include open sores, glossy lumps, red patches, pink growths, and scars;')
    image_bcc = Image.open('./image/BCC.png')
    st.image(image_bcc)
    st.write('Benign Keratosis is a benign acanthoma made up of keratinocytes from the epidermis. BKL is a benign warty lesion that commonly develops in adults as their skin ages. Clinically, it appears as a warty plaque with sharp borders, a sticky look, and a greasy feel;')
    image_bkl = Image.open('./image/BKL.png')
    st.image(image_bkl)
    st.write('Dermatofibroma is a common benign fibrous nodule that is typically located on the lower legs skin. Dermatofibromas mainly affect adults. Dermatofibromas can occur in individuals of any race. Normal dermatofibromas are more prevalent in females than in males, although certain histologic variations are more prevalent in men;')
    image_df = Image.open('./image/DF.png')
    st.image(image_df)
    st.write('Melanocytic nevi is a typical benign skin cancer due to a local proliferation of pigment cells (melanocytes). Melanin is present in brown or black melanocytic naevus; hence it is also known as a pigmented naevus;')
    image_nv = Image.open('./image/NV.png')
    st.image(image_nv)
    st.write('Vascular Skin Lesion either exist at birth or develop soon after birth. Especially when big, symptomatic (especially when ulcerated), positioned near to the eye or mouth, or on the head and neck;')
    image_vasc = Image.open('./image/VASC.png')
    st.image(image_vasc)
    st.write('Squamous Cell Carcinoma is also frequently kind of benign cancer. Typical symptoms include scaly red patches, open sores, elevated growths with a central depression, and warts;')
    image_scc = Image.open('./image/SCC.png')
    st.image(image_scc)
    st.write('Unknown')
    image_ukn = Image.open('./image/unknown.png')
    st.image(image_ukn)

def visual_dataset():
    """
    > This function is used to visualize the dataset
    """
    df = pd.read_csv('./csvfile/train.csv')
    df_fulltrain = pd.read_csv('./csvfile/full_train.csv')
    df_test = pd.read_csv('./csvfile/test.csv')
    df = df.drop(columns=['Unnamed: 0'],axis=1)
    st.dataframe(df.head(10))
    l_df = len(df)
    l_df_test = len(df_test)
    ll = f"""
    Length of training dataset: **<span style = 'font-size:20px;text-decoration:underline;'>{l_df}</span>** images \n
    Length of validate dataset: **<span style = 'font-size:20px;text-decoration:underline;'>{round(l_df/5)}</span>** images \n
    Length of testing dataset: **<span style = 'font-size:20px;text-decoration:underline;'>{l_df_test}</span>** images \n

    Length of melanoma images: **<span style = 'font-size:20px;text-decoration:underline;'>{df['target'].value_counts()[1]}</span>** images \n
    """
    st.markdown(ll,
        unsafe_allow_html=True)
    st.write('Before:')
    st.write(df['diagnosis'].value_counts())
    st.write('After:')
    st.write(df_fulltrain['diagnosis'].value_counts())
    fig1 = plt.figure(figsize=(10, 4))
    sns.countplot(data=df,x = "sex",palette="Set2")
    st.pyplot(fig1)
    fig2 = plt.figure(figsize=(10, 2))
    sns.countplot(data=df,x = "age_approx")
    st.pyplot(fig2)
    fig3 = plt.figure(figsize=(15, 8))
    sns.countplot(data=df,x = "anatom_site_general_challenge")
    st.pyplot(fig3)
