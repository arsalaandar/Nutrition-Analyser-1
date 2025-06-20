import os
import streamlit as st
import pandas as pd
import PIL
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import requests

# Page configuration with a wide layout
st.set_page_config(layout="wide")

# Custom CSS to add background image and darker text
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/11802698/pexels-photo-11802698.jpeg");
        background-size: cover;
        background-position: center;
    }
    
    h1, h2, h3, h4, h5, h6, p, div {
        color: #000000;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title and description
with st.container():
    st.markdown('<div class="transparent-overlay">', unsafe_allow_html=True)
    st.title("🍽️ Food Nutrition & Workout Tracker")
    st.markdown("**Detect the food and its nutrients, calculate BMI, and find exercises to burn calories**")
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data()
def load_dataset():
    return pd.read_csv("nutrition101.csv")

@st.cache_data()
def load_exercise_dataset():
    return pd.read_csv("exercise_dataset.csv")

def predict(image, img):
    url = "http://127.0.0.1:8000/predict_image"
    payload = {}
    files = [('file', ('download.jpeg', open(img, 'rb'), 'image/jpeg'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return response.json()

def main():
    with st.container():
        st.markdown('<div class="transparent-overlay">', unsafe_allow_html=True)
        
        # Option to use camera or upload an image
        option = st.radio("Select Input Image", ("Select Method", "Camera", "Upload File"))

        if option == "Camera":
            image_file = st.camera_input("Take a picture of your food")
        elif option == "Upload File":  
            image_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        else:
            image_file = None

        # If an image has been uploaded or captured
        if image_file is not None:
            # Show a small preview of the uploaded image
            st.image(image_file, caption="Uploaded Image", width=250)  # Width set for a smaller preview

            # Load image for further processing
            image = PIL.Image.open(image_file)

            # Create the uploads directory if it doesn't exist
            uploads_dir = "uploads"
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            # Save the image in the uploads directory
            img_path = os.path.join(uploads_dir, "captured_image.jpg")
            image.save(img_path)

            # Continue with BMI and nutrient detection
            st.subheader("Note: - 1 foot is 30.48 cm and 1 inch is 2.54 cm")
            Height = st.number_input("Height in cm", key="1")
            Weight = st.number_input("Weight in kg", key="2")

            # Input validation for height and weight
            if Height < 2 or Weight < 12:
                st.warning("Please enter a valid height and weight.")
            else:
                bmi = 10000 * (Weight / (Height * Height))
                rounded_bmi = round(bmi, 2)
                st.subheader("Your BMI is " + str(rounded_bmi))

                # Spinner for analyzing the food
                with st.spinner("Detecting and Analysing Food"):
                    probs = predict(image, img_path)

                if probs[0]['status'] != 404:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, width=600)  # Show the large image
                    with col2:
                        st.header(probs[0]['Name'])
                        calories = (probs[0]['Value'][0] * 4) + (probs[0]['Value'][2] * 9) + (probs[0]['Value'][3] * 4)
                        data = pd.DataFrame({"Nutrients": ["Protein", "Calcium", "Fats", "Carbs", "Vitamins", "Calories"],
                                             "Value(per 100 grams)": [probs[0]['Value'][0], probs[0]['Value'][1], probs[0]['Value'][2], probs[0]['Value'][3], probs[0]['Value'][4], calories]})
                        st.table(data)

                    st.header("________________________________________________________")  
                    st.header("Workout to Burn Calories You Ate")
                    gym_workout = round((calories / rounded_bmi) * 10)
                    cycling = round((calories / rounded_bmi) * 8)
                    walking = round((calories / rounded_bmi) * 7)
                    yoga = round((calories / rounded_bmi) * 6)
                    data_1 = pd.DataFrame({"Exercise": ["GYM", "Cycling", "Walking", "Yoga"],
                                           "Value (Minutes)": [gym_workout, cycling, walking, yoga]})
                    st.table(data_1)
                else:
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        st.image(image, width=600)
                    with col_2:
                        st.header("Not a Food Item")
        else:
            st.subheader("Upload images or take a picture")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
