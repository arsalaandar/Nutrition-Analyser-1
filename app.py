import os
import streamlit as st
import pandas as pd
import PIL
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch

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

@st.cache_resource
def load_model():
    """Load the zero-shot image classification model"""
    try:
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data()
def load_dataset():
    """Load nutrition dataset"""
    try:
        return pd.read_csv("nutrition101.csv")
    except Exception as e:
        st.error(f"Error loading nutrition dataset: {str(e)}")
        return None

@st.cache_data()
def load_exercise_dataset():
    """Load exercise dataset"""
    try:
        return pd.read_csv("exercise_dataset.csv")
    except Exception as e:
        st.error(f"Error loading exercise dataset: {str(e)}")
        return None

def predict(image, img_path):
    """Predict food item from image using transformers"""
    try:
        # Load model and processor
        processor, model = load_model()
        if processor is None or model is None:
            return [{'status': 404}]
        
        # Load nutrition dataset
        nutrition_df = load_dataset()
        if nutrition_df is None:
            return [{'status': 404}]
        
        # Get list of food items from your dataset
        # Adjust the column name based on your CSV structure
        # Common column names: 'name', 'Name', 'food', 'Food', 'food_name', 'item'
        food_column = None
        for col in ['name', 'Name', 'food', 'Food', 'food_name', 'item', 'Food_name']:
            if col in nutrition_df.columns:
                food_column = col
                break
        
        if food_column is None:
            st.error("Could not find food name column in nutrition dataset")
            return [{'status': 404}]
        
        food_items = nutrition_df[food_column].tolist()
        
        # Prepare inputs for the model
        inputs = processor(images=image, text=food_items, return_tensors="pt", padding=True)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Get top prediction
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()
        
        # Get nutrition info for predicted food
        predicted_food = food_items[top_idx]
        food_row = nutrition_df[nutrition_df[food_column] == predicted_food].iloc[0]
        
        # Try to find nutrition columns - adjust these based on your CSV structure
        def get_column_value(row, possible_names, default=0.0):
            for name in possible_names:
                if name in row.index:
                    try:
                        return float(row[name])
                    except:
                        return default
            return default
        
        protein = get_column_value(food_row, ['protein', 'Protein', 'proteins', 'Proteins'])
        calcium = get_column_value(food_row, ['calcium', 'Calcium'])
        fats = get_column_value(food_row, ['fats', 'Fats', 'fat', 'Fat', 'total_fat'])
        carbs = get_column_value(food_row, ['carbs', 'Carbs', 'carbohydrates', 'Carbohydrates'])
        vitamins = get_column_value(food_row, ['vitamins', 'Vitamins', 'vitamin', 'Vitamin'])
        
        # Format response similar to your API
        if top_prob > 0.1:  # Confidence threshold
            result = [{
                'status': 200,
                'Name': predicted_food,
                'Value': [protein, calcium, fats, carbs, vitamins],
                'confidence': float(top_prob)
            }]
        else:
            result = [{'status': 404}]
        
        return result
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return [{'status': 404}]

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
            st.image(image_file, caption="Uploaded Image", width=250)

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
                with st.spinner("Detecting and Analysing Food (This may take a moment on first load)..."):
                    probs = predict(image, img_path)

                if probs[0]['status'] != 404:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, width=600)
                    with col2:
                        st.header(probs[0]['Name'])
                        if 'confidence' in probs[0]:
                            st.write(f"Confidence: {probs[0]['confidence']*100:.1f}%")
                        
                        calories = (probs[0]['Value'][0] * 4) + (probs[0]['Value'][2] * 9) + (probs[0]['Value'][3] * 4)
                        data = pd.DataFrame({
                            "Nutrients": ["Protein", "Calcium", "Fats", "Carbs", "Vitamins", "Calories"],
                            "Value(per 100 grams)": [
                                probs[0]['Value'][0], 
                                probs[0]['Value'][1], 
                                probs[0]['Value'][2], 
                                probs[0]['Value'][3], 
                                probs[0]['Value'][4], 
                                round(calories, 2)
                            ]
                        })
                        st.table(data)

                    st.header("________________________________________________________")  
                    st.header("Workout to Burn Calories You Ate")
                    gym_workout = round((calories / rounded_bmi) * 10)
                    cycling = round((calories / rounded_bmi) * 8)
                    walking = round((calories / rounded_bmi) * 7)
                    yoga = round((calories / rounded_bmi) * 6)
                    data_1 = pd.DataFrame({
                        "Exercise": ["GYM", "Cycling", "Walking", "Yoga"],
                        "Value (Minutes)": [gym_workout, cycling, walking, yoga]
                    })
                    st.table(data_1)
                else:
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        st.image(image, width=600)
                    with col_2:
                        st.header("Not a Food Item")
                        st.write("The model could not identify this as a food item with sufficient confidence.")
        else:
            st.subheader("Upload images or take a picture")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
