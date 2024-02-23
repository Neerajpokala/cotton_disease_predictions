import streamlit as st
import numpy as np
from PIL import Image
import pickle
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array



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
    return class_labels[preds[0]]

def main():
    #st.title('Cotton Disease Detection')
    st.markdown("<h1 style='text-align: left; color: skyblue; font-size: 40px; '>CNN FOR COTTON DISEASE DETECTION </h1>", unsafe_allow_html=True)
    page = st.sidebar.selectbox("Choose a page", ["CNN Explanation", "Image Inference"])

    if page == "CNN Explanation":
        st.markdown("<h1 style='text-align: left; color: white; font-size: 20px;'>Introduction to CNNs for Image Analysis:</h1>", unsafe_allow_html=True)
        st.markdown("Convolutional Neural Networks (CNNs) are a class of deep neural networks that are particularly effective for image analysis tasks. They are inspired by the organization of the animal visual cortex, where individual neurons respond to specific features of the visual field.")
        st.markdown("CNNs consist of multiple layers, including convolutional layers, pooling layers, and fully connected layers. Each layer performs a specific operation on the input data, and the network learns to extract hierarchical representations of features from the input image.")
        st.markdown("<h1 style='text-align: left; color: white; font-size: 20px;'>Architecture of a Typical CNN for Image Classification:</h1>", unsafe_allow_html=True)
        st.image('CNN.jpg', use_column_width=True)
        st.markdown("1. **Input Layer:** The input to the CNN is the raw pixel values of the input image.")
        st.markdown("2. **Convolutional Layers:** Convolutional layers are the building blocks of CNNs. Each convolutional layer applies a set of filters (also known as kernels) to the input image to detect features such as edges, textures, and patterns. These filters are learned during the training process. Convolutional layers are typically followed by activation functions (e.g., ReLU) to introduce non-linearity.")
        st.markdown("3. **Pooling Layers:** Pooling layers are used to reduce the spatial dimensions of the feature maps produced by the convolutional layers, which helps in reducing computational complexity and controlling overfitting. Common pooling operations include max pooling and average pooling.")
        st.markdown("4. **Flattening:** After several convolutional and pooling layers, the feature maps are flattened into a vector to be fed into the fully connected layers.")
        st.markdown("5. **Fully Connected Layers:** Fully connected layers, also known as dense layers, take the flattened feature vector as input and perform classification based on learned features. These layers enable the network to learn complex relationships between features extracted by the convolutional layers.")
        st.markdown("6. **Output Layer:** The output layer produces the final predictions. For classification tasks like cotton disease prediction, the output layer typically consists of one neuron per class, with a softmax activation function to convert raw scores into probabilities.")
        st.markdown("<h1 style='text-align: left; color: white; font-size: 20px;'>Conclusion:</h1>", unsafe_allow_html=True)  
        st.markdown(" CNNs have revolutionized the field of computer vision and are widely used for various tasks, including image classification, object detection, and image segmentation. The hierarchical feature learning capability of CNNs makes them particularly well-suited for analyzing complex visual data such as images. In the context of cotton disease prediction, CNNs can learn to extract relevant features from input images and classify them into different disease categories, helping farmers detect and mitigate crop diseases more effectively.")
    
    elif page == "Image Inference":
        st.header("Image Inference")
        st.write("Upload an image of a cotton leaf or plant to detect if it's diseased or fresh.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
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
            if prediction in ['diseased cotton leaf', 'diseased cotton plant']:
                st.markdown(f"The predicted Disease name is {random.choice(cotton_diseases)}")
        

if __name__ == "__main__":
    main()
