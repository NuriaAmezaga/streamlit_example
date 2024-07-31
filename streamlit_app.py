import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns
import requests

# Set the title of the app
st.title("APP-VACCINES")

# Markdown
st.markdown("""
Disease Prediction App

Predict diseases using health data and symptoms with advanced machine learning for accurate, timely predictions and early diagnosis.
""")



# Loading training data
url = "https://drive.google.com/file/d/19-xEfVfvART7dM5OgR5yVcrH8KPZ5AaR/view?usp=drive_link"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
training2 = pd.read_csv(path)
training2 = training2.drop(["Unnamed: 133"], axis=1)
training2.drop_duplicates(inplace=True)

# Loading descriptions data
sd_url = "https://drive.google.com/file/d/1e4VPXjD70CR8r2WL4aeSBxwEePV4To7s/view?usp=drive_link"
sd_path = 'https://drive.google.com/uc?export=download&id=' + sd_url.split('/')[-2]
sd = pd.read_csv(sd_path)

# Loading precautions data
sp_url = "https://drive.google.com/file/d/1EkYiH2F7d1wJBETyowUag-7ZBfgT6n-h/view?usp=drive_link"
sp_path = 'https://drive.google.com/uc?export=download&id=' + sp_url.split('/')[-2]
sp = pd.read_csv(sp_path)

# Loading dataset
url = "https://drive.google.com/file/d/193IM3aokK2QLnOG4FLgWy2O3m6GqmNyb/view?usp=drive_link"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
dataset = pd.read_csv(path)


# Splitting the data
x = training2.drop('prognosis', axis=1)
y = training2['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

# Creating a model
model = RandomForestClassifier(
    ccp_alpha=0.0,
    max_depth=None,
    min_samples_leaf=1,
    n_estimators=100,
    random_state=42
)

# Train the model
model2 = model.fit(x_train, y_train)


# Defining symptoms columns
columns = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
           'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
           'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
           'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
           'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
           'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
           'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
           'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
           'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
           'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
           'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
           'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
           'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
           'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
           'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
           'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
           'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
           'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
           'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
           'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
           'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
           'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
           'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
           'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
           'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
           'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
           'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
           'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
           'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

# Create selectboxes for each feature
selected_symptoms = st.multiselect("Select a symptom", columns)

# Create a DataFrame with one row of zeros
df_user = pd.DataFrame([[0] * len(columns)], columns=columns)

# Update the DataFrame based on selected symptoms
for symptom in selected_symptoms:
    if symptom in df_user.columns:
        df_user.loc[0, symptom] = 1

# Display the updated DataFrame
st.dataframe(df_user)

# Predict button
if st.button("Predict Disease"):
    # Make a prediction
    prediction = model2.predict(df_user)[0]

    # Display the prediction
    st.write("The predicted disease is:", prediction)
   

    # Get the description and precautions for the predicted disease
    description = sd[sd['Disease'].str.contains(prediction, case=False)]
    precaution = sp[sp['Disease'].str.contains(prediction, case=False)]

    # Display the description and precautions
    st.markdown("**Description of the disease:**")
    st.write(description['Description'].values[0] if not description.empty else "No description available.")

    st.markdown("**Precautions to take:**")
    if not precaution.empty:
        precautions_columns = [col for col in precaution.columns if 'Precaution' in col]
        for col in precautions_columns:
            if not pd.isna(precaution.iloc[0][col]):
                st.write(f"- {precaution.iloc[0][col]}")
    else:
        st.write("No precautions available.")
        
# Sidebar title Symptom Analysis
st.sidebar.title("Symptom Analysis")

# Generate and display the WordCloud
wordcloud = WordCloud(background_color='white', width=800, height=800, colormap='plasma').generate_from_frequencies(dataset['Disease'].value_counts())
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Diseases')
st.pyplot(plt)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

# Load dataset from Google Drive
url = "https://drive.google.com/file/d/193IM3aokK2QLnOG4FLgWy2O3m6GqmNyb/view?usp=drive_link"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
dataset = pd.read_csv(path)

# Replace name of diseases
dataset.rename(columns={
    "Symptom_1": "Hepatitis",
    "Symptom_2": "Dengue Fever",
    "Symptom_3": "Malaria",
    "Symptom_4": "Hepatitis B",
    "Symptom_5": "Hepatitis C",
    "Symptom_6": "Cirrhosis",
    "Symptom_7": "Hypothyroidism",
    "Symptom_8": "Hepatitis A",
    "Symptom_9": "Hepatitis-",
    "Symptom_10": "Yellow Fever",
    "Symptom_11": "Leptospirosis",
    "Symptom_12": "Endocarditis",
    "Symptom_13": "Dengue Fever",
    "Symptom_14": "Pneumonia",
    "Symptom_15": "Tuberculosis",
    "Symptom_16": "COVID-19",
    "Symptom_17": "Influenza (Flu)"
}, inplace=True)

# Different diseases list
different_diseases = list(set([
    "Hepatitis", "Dengue Fever", "Malaria", "Hepatitis B", "Hepatitis C", "Cirrhosis", "Hypothyroidism", 
    "Hepatitis A", "Hepatitis", "Yellow Fever", "Leptospirosis", "Endocarditis", "Dengue Fever", "Pneumonia",
    "Tuberculosis", "COVID-19", "Influenza (Flu)"
]))



# Sidebar options for visualization

Disease_name = st.sidebar.selectbox("Please select the disease name:", different_diseases, key='selectbox2')


# Debugging: Print the selected disease name and dataset columns
st.write("Selected Disease:", Disease_name)


if Disease_name in dataset.columns:
    # Plot the symptoms for the selected disease
    symptom_counts = dataset[Disease_name].value_counts()

    if not st.sidebar.checkbox('Hide', True, key='checkbox1'):
        st.markdown(f"### Distribution of {Disease_name}")



        # Seaborn Count Plot
       
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.countplot(y=Disease_name, data=dataset, palette="bwr", ax=ax)
        ax.set_title(f"Distribution of Symptoms for {Disease_name}", fontsize=40)
        ax.tick_params(axis='y', labelsize=15)  # Adjust the label size
        st.pyplot(fig)
else:
    st.error(f"Disease '{Disease_name}' is not found in the dataset columns.")




# Function to download the video from a URL
def download_video(url, output):
    response = requests.get(url)
    with open(output, 'wb') as file:
        file.write(response.content)

# GitHub raw content URL and output file name "Introduction"-----------------------------------
url = 'https://github.com/NuriaAmezaga/streamlit_example/raw/main/video%20intro.mp4'  # Replace with your actual GitHub raw link
output = 'video_intro.mp4'

# Download the video file
download_video(url, output)

# Title of the app
st.sidebar.title("Intro to Vaccine App")


# Load video file
video_file = open(output, "rb")
video_bytes = video_file.read()

# Display the video

st.sidebar.video(video_bytes)



# GitHub raw content URL and output file name from "Vaccine how it works"-------------------------

url2 = 'https://github.com/NuriaAmezaga/streamlit_example/raw/main/how%20vaccine%20works.mp4'  # Actual GitHub raw link
output2 = 'how_vaccine_works.mp4'

# Download the video file
download_video(url2, output2)

# Title of the app
st.sidebar.title("How vaccines are working")

# Load video file
video_file2 = open(output2, "rb")
video_bytes2 = video_file2.read()

# Display the video

st.sidebar.video(video_bytes2)


###### 31.07.2024

from PIL import Image

# Load  DataFrame containing image paths, i think is not needed... just the image_url
# Replace with your actual DataFrame containing image URLs
image_data = {
    'image_urls': [
        'https://vaccination-info.europa.eu/sites/default/files/styles/is_large/public/images/Chickenpox1.png?itok=MhbkWZxr',  # Replace with correct URLs
        'https://vaccination-info.europa.eu/sites/default/files/styles/is_large/public/images/vaccination-baby-illustration.jpg?itok=ViGMYXKo',
        'https://vaccination-info.europa.eu/sites/default/files/styles/is_large/public/images/Nausea-evip-smaller.png?itok=hyLVyGBu'
    ]
}
df_images = pd.DataFrame(image_data)

# Streamlit app
st.title("Image Slideshow")

# Initialize session state for the slideshow index
if 'slideshow_index' not in st.session_state:
    st.session_state.slideshow_index = 0

# Display the current image
current_image_url = df_images['image_urls'].iloc[st.session_state.slideshow_index]
st.image(current_image_url, caption=f"Image {st.session_state.slideshow_index + 1}/{len(df_images)}", use_column_width=True)

# Create buttons for navigation
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Previous"):
        st.session_state.slideshow_index = (st.session_state.slideshow_index - 1) % len(df_images)
with col3:
    if st.button("Next"):
        st.session_state.slideshow_index = (st.session_state.slideshow_index + 1) % len(df_images)




# # # # # # # # # # # # # # # # # # 


# URL to the CSV file on Google Drive
url = "https://drive.google.com/file/d/1Y0zj8DrGIY8-BkaAMt41Vu7kqtnnnrIy/view?usp=sharing"
path = "https://drive.google.com/uc?export=download&id=" + url.split("/")[-2]

# Load the DataFrame from the Google Drive URL
df_extracted = pd.read_csv(path)

# Sidebar input for user query
st.sidebar.title("FAQ Finder")


# Text input for user query
user_query = st.sidebar.text_input("Do you need more information?", "")

# Filter the DataFrame based on the user query
if user_query:
    filtered_by_questions = df_extracted[df_extracted['question'].str.contains(user_query, case=False, na=False)]
    
    # Display the filtered results in the sidebar
    if not filtered_by_questions.empty:
        st.sidebar.write(f"**Found {len(filtered_by_questions)} matching questions:**")
        for index, row in filtered_by_questions.iterrows():
            st.sidebar.write(f"**Question:** {row['question']}")
            st.sidebar.write(f"**Answer:** {row['answer']}")
            st.sidebar.write(f"**Source URL:** {row['url']}")
            st.sidebar.write("---")
    else:
        st.sidebar.write("No matching questions found.")









 


