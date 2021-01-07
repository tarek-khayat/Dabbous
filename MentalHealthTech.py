#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import csv
import sqlite3 #to allow sign up and login
import streamlit as st
import time
import hashlib #to allow sign up and login
from PIL import Image
from typing import Dict #to allow question and answer
import wikipedia #to allow question and answer
from transformers import Pipeline #to allow question and answer
from transformers import pipeline #to allow question and answer
import csv
from collections import Counter
import scipy.stats as stats
import seaborn as sns
import torch
import matplotlib
from matplotlib import pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.preprocessing import LabelEncoder


NUM_SENT = 10

@st.cache
def get_qa_pipeline() -> Pipeline:
    qa = pipeline("question-answering")
    return qa


def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> Dict:
    input = {
        "question": question,
        "context": paragraph
    }
    return pipeline(input)

@st.cache
def get_wiki_paragraph(query: str) -> str:
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0], sentences=NUM_SENT)
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0], sentences=NUM_SENT)
    return summary


def format_text(paragraph: str, start_idx: int, end_idx: int) -> str:
    return paragraph[:start_idx] + "**" + paragraph[start_idx:end_idx] + "**" + paragraph[end_idx:]


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

conn = sqlite3.connect('data.db')
c = conn.cursor()
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


df=pd.read_csv('Mentalhealth_tech.csv')


def main ():
    st.title ("Mental Health in Tech Industry")
    st.subheader ("Health Streamlit Project by Sara Dabbous")
    #creating side menu
    menu = ["Home","SignUp", "LogIn"]
    choice = st.sidebar.selectbox ("Menu", menu)
    df=pd.read_csv('Mentalhealth_tech.csv')

    if choice == "Home":
        st.subheader("Home")
        img = Image.open ('mental.jpg')
        st.image(img,width=450)
        st.write("This dashboard explores a dataset from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorder. We aim from this dashboard to investigate the attitude people have towards mental health in the worklplace and the interaction between different variables in the dataset. Navigate through the menu on the left to signup, log-in, and explore!")
        #File uploader to drop the dataset
        data = st.file_uploader("Upload file here (only excel or csv acceptable):", type=['csv','xlsx'])
        #Reading the data and analyzing it
        if df is not None:
            df=pd.read_csv(data)
        else:
            df=pd.read_csv('Mentalhealth_tech.csv')
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.1)
            st.success("Data Upload Successful")


    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

    elif choice == "LogIn":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username,check_hashes(password,hashed_pswd))

            if result:
                st.success("Logged In as {}".format(username))
                task = st.selectbox("Task",["Dataset","Data Insights and Cleaning", "Ask me something!", "User Profiles", "About This App"])
                if st.sidebar.checkbox ("Filter Multiple Columns"):
                    columns_select=st.multiselect ("Which columns would you like to see?",df.columns)
                    dataselect=df[columns_select]
                    st.dataframe(dataselect)
                    if st.checkbox("Summary of Selected Dataframe"):
                        st.write(dataselect.describe())

                if task == "Dataset":
                          st.header ("Mental Health Data Exploration")
                          st.subheader ("Sample of our dataset")
                          st.dataframe(df.head(50))
                          if st.checkbox ("Column Data Types"):
                              st.subheader ("Column Data Types")
                              st.table(df.dtypes)
                          if st.checkbox ("Shape of Dataset"):
                              st.subheader ("Shape of Dataset")
                              st.table (df.describe())


                elif task == "Data Insights and Cleaning":
                    st.subheader("Analytics")
                    st.write ("Navigate through left-side Analytics dropdown")
                    Analytics = st.sidebar.selectbox("Analytics",["Null Values","Countries", "Gender","Age","Treatment", "Tech Industry", "Consequences", "Mental Health Leave","Correlation Matrix"])
                    if Analytics == "Null Values":
                        st.subheader ("Check null values")
                        st.write('Null values in Column Self-Employed', df.self_employed.isnull().sum())
                        st.write('Null values in Column Leave', df.leave.isnull().sum())
                        st.write('Null values in Column work_interfere', df.work_interfere.isnull().sum())
                        if st.button ("Drop Them. NOW."):
                            df.dropna(inplace=True)
                            st.write('Null values in Column Self-Employed', df.self_employed.isnull().sum())
                            st.write('Null values in Column Leave', df.leave.isnull().sum())
                            st.write('Null values in Column work_interfere', df.work_interfere.isnull().sum())
                            st.subheader("To make our data cleaner we also drop unneccesary columns like: state, comments, supervisor, and timestamp")
                            df = df.drop (columns=['state', 'comments', 'supervisor', 'Timestamp'])
                            st.table(df.head(50))

                    if Analytics == "Countries":
                        st.subheader("What are the different Countries that participated in this mental health survey?")
                        st.write(df.Country.unique())
                        from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
                        def get_continent(col):
                            try:
                                cn_a2_code =  Country_to_country_alpha2(col)
                            except:
                                cn_a2_code = 'Unknown'
                            try:
                                cn_continent = country_alpha2_to_continent_code(cn_a2_code)
                            except:
                                cn_continent = 'Unknown'
                            return (cn_a2_code, cn_continent)
                        from geopy.geocoders import Nominatim
                        geolocator1 = Nominatim()
                        def geolocate(Country):
                            try:
                                # Geolocate the center of the country
                                loc = geolocator1.geocode(Country)
                                # And return latitude and longitude
                                return (loc.latitude, loc.longitude)
                            except:
                                # Return missing value
                                return np.nan
                        # Create a world map to show distributions of users
                        import folium
                        from folium.plugins import MarkerCluster
                        #empty map
                        world_map= folium.Map(tiles="cartodbpositron")
                        marker_cluster = MarkerCluster().add_to(world_map)
                        #for each coordinate, create circlemarker of user percent
                        for i in range(len(df)):
                                lat = df.iloc[i]['Latitude']
                                long = df.iloc[i]['Longitude']
                                radius=5
                                popup_text = """Country : {}<br>
                                            %of Users : {}<br>"""
                                popup_text = popup_text.format(df.iloc[i]['Country'],
                                                           df.iloc[i]['User_Percent']
                                                           )
                                folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
                        #show the map
                        world_map

                    if Analytics == "Gender":
                        st.subheader ("What are the genders available in our dataset?")
                        df['Gender'].value_counts().plot(kind='barh', figsize=(20,30), rot=0)
                        plt.xlabel("Gender", fontsize= 20)
                        plt.ylabel("Count of People", fontsize=20)
                        plt.title("Count of People Who Filled Survey by Gender", fontsize=30, y=1.02);
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)
                        st.pyplot()
                        if st.button ("Clean this Mess!"):
                            Gender = df.Gender.unique() #Unique Genders
                            GenderClean1= df.Gender.replace (['M', 'm', 'Male-ish', 'maile', 'male', 'Male ', 'msle', 'Make', 'Guy (-ish) ^_^', 'Man', 'Malr', 'Mal', 'Mail', 'Male (CIS)', 'Cis Man','Cis Male','cis male'],'Male')
                            GenderClean2= GenderClean1.replace (['f', 'F', 'Woman', 'woman', 'Femake', 'female', 'Female ', 'Cis Female', 'cis-female/femme', 'femail', 'Female (cis)'],'Female')
                            GenderClean= GenderClean2.replace (['Trans-female', 'something kinda male?', 'queer/she/they', 'non-binary', 'Agender', 'Nah', 'All','Enby','fluid', 'Genderqueer', 'Androgyne', 'male leaning androgynous', 'Trans woman','Neuter',  'Female (trans)','queer',  'A little about you', 'p', 'ostensibly male, unsure what that really means' ],'Other')
                            df.Gender = GenderClean
                            df['Gender'].value_counts().plot(kind='bar', figsize=(15, 10), rot=0)
                            plt.xlabel("Gender", fontsize=20)
                            plt.ylabel("Count of People", fontsize=20)
                            plt.xticks(fontsize= 20)
                            plt.yticks(fontsize= 20)
                            plt.title("Count of People Who Filled Survey by Gender", y=1.02);
                            st.pyplot()
                    if Analytics == "Age":
                        st.subheader("What are the different ages we have?")
                        Age = df.Age
                        Count = Counter(Age)
                        df['Age'].value_counts().plot(kind='barh', figsize=(20, 30), rot=0)
                        plt.xlabel("Count of People", fontsize= 20)
                        plt.ylabel("Age", fontsize= 20)
                        plt.title("Count of People Who Filled Survey by Age", y=1.02);
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)
                        st.pyplot()

                        if st.button ("Clean this!"):
                            df = df.set_index("Age")
                            df = df.drop([-29,99999999999, -1726, 5, 8, 11, -1,329], axis=0) # Delete all rows with outlying age
                            df.reset_index(level=0, inplace=True)
                            df['Age'].value_counts().plot(kind='barh', figsize=(20, 30), rot=0)
                            plt.xlabel("Count of People", fontsize= 20)
                            plt.ylabel("Age", fontsize= 20)
                            plt.title("Count of People Who Filled Survey by Age", y=1.02);
                            plt.xticks(fontsize= 20)
                            plt.yticks(fontsize= 20)
                            st.pyplot()
                            st.write ("Statistics of Age Distribution")
                            st.table(df.Age.describe())
                            st.subheader("Age Distribution")
                            n, bins, patches = plt.hist(x=df.Age, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
                            plt.grid(axis='y', alpha=0.75)
                            plt.xlabel('Age', fontsize=20)
                            plt.xticks(fontsize= 20)
                            plt.yticks(fontsize= 20)
                            plt.ylabel('Frequency', fontsize=20)
                            plt.title('Age Distribution')
                            plt.text(23, 45, r'$\mu=15, b=3$')
                            maxfreq = n.max()
                            # Set a clean upper y-axis limit.
                            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
                            st.pyplot()

                    if Analytics == "Treatment":
                        st.subheader("How many people actually seek treatment?")
                        df['treatment'].value_counts().plot(kind='bar', figsize=(20, 30), rot=0)
                        plt.xlabel("Seek Treatment", fontsize= 20)
                        plt.ylabel("Count of People", fontsize= 20)
                        plt.title("Count of People Who Seek Treatment", y=1.02);
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)
                        st.pyplot()

                    if Analytics == "Tech Industry":
                        st.subheader("How many people work in the Tech Industry?")
                        df['tech_company'].value_counts().plot(kind='bar', figsize=(20, 30), rot=0)
                        plt.xlabel("Work in Tech", fontsize= 20)
                        plt.ylabel("Count of People", fontsize= 20)
                        plt.title("Count of People Who Work in Tech", y=1.02);
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)
                        st.pyplot()

                    if Analytics == "Consequences":
                        st.subheader("How many people think discussing their mental health might have a consequence on their workplace?")
                        df['mental_health_consequence'].value_counts().plot(kind='bar', figsize=(20, 30), rot=0)
                        plt.xlabel("Consequence", fontsize= 20)
                        plt.ylabel("Count of people", fontsize= 20)
                        plt.title("Count of People Who Believe Discussing their Mental Health will have a consequence", y=1.02);
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)
                        st.pyplot()

                    if Analytics == "Mental Health Leave":
                        st.subheader("How many people are able to take a medical leave for a mental health condition?")
                        df['leave'].value_counts().plot(kind='bar', figsize=(20, 30), rot=0)
                        plt.xlabel("Mental Health Leave", fontsize= 20)
                        plt.ylabel("Count of people", fontsize= 20)
                        plt.title("Count of People x Ability to take Mental Health Leave ", y=1.02);
                        plt.xticks(fontsize= 20)
                        plt.yticks(fontsize= 20)
                        st.pyplot()

                    if Analytics == "Correlation Matrix":
                        #Turning yes/no into binary
                        df.dropna(inplace=True)
                        lb = LabelEncoder()
                        df['treatment'] = lb.fit_transform(df['treatment'])
                        df['self_employed'] = lb.fit_transform(df['self_employed'])
                        df['family_history'] = lb.fit_transform(df['family_history'])
                        df['remote_work'] = lb.fit_transform(df['remote_work'])
                        df['tech_company'] = lb.fit_transform(df['tech_company'])
                        df['obs_consequence'] = lb.fit_transform(df['obs_consequence'])
                        st.subheader("How do our different variables correlate to each other?")
                        numeric_features = df.select_dtypes (include=[np.number])
                        #Correlation between numeric features
                        corr = numeric_features.corr()
                        ##Correlation Matrix Generation
                        f, ax=plt.subplots (figsize=(12,9))
                        sns.heatmap (corr, vmax=0.9, square = True)
                        st.pyplot ()

                elif task == "Ask me something!":
                    st.subheader ("Ask me something about Mental Health!")
                    wiki_query = st.text_input("WIKIPEDIA SEARCH TERM", "")
                    wiki_para = get_wiki_paragraph(wiki_query)
                    paragraph_slot = st.empty()
                    paragraph_slot.markdown(wiki_para)


                elif task == "User Profiles":
                    st.subheader("User Profiles")
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                    st.dataframe(clean_db)

                elif task == 'About This App':
                    st.subheader("About This App")
                    st.write("From this dashboard we can realize that comfortably discussing mental health in the workplace is still a bit far-fetched. We Still have a good percentage of people who do not seek treatment, who are unaware of mental wellness program, and have not been informed whether or not there are mental sick leaves. We need to work on normalizing Mental Health as a common issue, and remove the tabboo from asking for help, seeking treatment, or discussing with employers.")
                    st.write('To view/download dataset: https://data.world/quanticdata/mental-health-in-tech-survey')
                    st.write("To know more about this dashboard and how to explore it:https://drive.google.com/file/d/1U6t8u8I4l19A0wJyk2KGEPwXRQMMPakL/view?usp=sharing")
                    st.write ("To reach the developer: https://www.linkedin.com/in/sara-dabbous-1989ba65/")
                    img1 = Image.open ('Mental2.jpg')
                    st.image(img1,width=450)

            else:
                st.warning("Incorrect Username/Password")




if __name__ == '__main__':
    main ()


# In[ ]:
