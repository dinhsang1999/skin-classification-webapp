import os
import streamlit as st
from deta import Deta # pip install deta
from dotenv import load_dotenv # pip install dotenv

#https://www.youtube.com/watch?v=eCbH2nPL9sU

# Load the environment variables
load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

# Initilize with a project key
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("users_db")

def insert_user(username, name, password):
    """Returns the user on a successful user creation, otherwise raises and error"""
    return db.put({"key": username, "name": name, "password": password})

# @st.cache(allow_output_mutation=True,show_spinner=False,max_entries=1)
# @st.experimental_memo(show_spinner=False,max_entries=1)
def fetch_all_users():
    """Returns a dictionary of all users"""
    res = db.fetch()
    return res.items

def get_user(username):
    """If not found,the function will return None"""
    return db.get(username)

def update_user(username, update):
    """If the item is updated, returns None. Otherwise, an exception is raised"""
    return db.update(update,username)

def delete_user(username):
    """Always retunes None, even if the keu does not exist"""
    return db.delete(username)