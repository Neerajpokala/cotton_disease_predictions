import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import random

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image_class(model, img_array):
    preds = model.predict(img_array)
    preds = np.argmax(preds, axis=1)
    class_labels = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
    predicted_class = class_labels[preds[0]]
    return predicted_class

def generate_random_disease_names(num_names=5):
    cotton_diseases = [
        "Bacterial blight",
        "Cotton leaf curl virus",
        "Cotton boll rot",
        "Crown gall",
        "Leaf spot",
        "Fusarium wilt",
        "Verticillium wilt",
        "Cotton leaf crumple",
        "Alternaria leaf spot",
        "Angular leaf spot"
    ]
    
    return random.sample(cotton_diseases, num_names)

def main():
    # Your Streamlit code
    st.markdown("<h1 style='text-align: left; color: skyblue; font-size: 40px; '>CNN FOR COTTON DISEASE DETECTION </h1>", unsafe_allow_html=True)
    page = st.sidebar.selectbox("Choose a page", ["CNN Explanation", "Image Inference"])

    if page == "CNN Explanation":
        # Your explanation code

    elif page == "Image Inference":
        st.header("Image Inference")
        st.write("Upload an image of a cotton leaf or plant to detect if it's diseased or fresh.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            model_path = 'model.pkl'  # Provide the path to your pickle file
            model = load_model(model_path)
            img_array = preprocess_image(uploaded_file)
            prediction = predict_image_class(model, img_array)
            st.success(f"Prediction: {prediction}")

            # Generate random disease names if prediction is diseased
            if prediction in ['diseased cotton leaf', 'diseased cotton plant']:
                st.write("The detected cotton is diseased.")
                st.write("Random names of cotton diseases:")
                disease_names = generate_random_disease_names()
                for name in disease_names:
                    st.write(f"- {name}")

if __name__ == "__main__":
    main()
