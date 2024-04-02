import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import time
import sqlite3
import bcrypt

# Constants
img_rows, img_cols = 64, 64  # Replace with your desired image dimensions
model_camera = load_model('finalLtrained_model.h5')  # Load camera model
model_upload = load_model('final_trained_model.h5')  # Load upload model
# Define descriptions for each class
descriptions = {
    0: 'Safe driving',
    1: 'Texting - right',
    2: 'Talking on the phone - right',
    3: 'Texting - left',
    4: 'Talking on the phone - left',
    5: 'Operating the radio',
    6: 'Drinking',
    7: 'Reaching behind',
    8: 'Hair and makeup',
    9: 'Talking to passenger'
}



# Function to create the database table
def create_table():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            car_number TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(username, password, car_number):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute('''
        INSERT INTO users (username, password, car_number)
        VALUES (?, ?, ?)
    ''', (username, hashed_password, car_number))
    conn.commit()
    conn.close()

# Function to authenticate a user
def authenticate_user(username, password):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT password, car_number FROM users WHERE username = ?
    ''', (username,))
    user_data = cursor.fetchone()
    conn.close()

    if user_data:
        stored_password, car_number = user_data
        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
            return True, car_number

    return False, None

# Function to check if a username is already registered
def is_username_registered(username):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE username = ?
    ''', (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# Streamlit app
st.title("IntelliDrive Assurance")

# Create the users table
create_table()

# User authentication logic
login_status = st.empty()
signup_status = st.empty()
logged_in = False

# Check if the user is already logged in
if 'user_id' in st.session_state:
    logged_in = True
    login_status.success(f"Logged in as {st.session_state['username']}")
else:
    # Login or Signup
    login_option = st.radio("Choose an option:", ("Login", "Signup"))

    if login_option == "Login":
        login_username = st.text_input("Username:")
        login_password = st.text_input("Password:", type="password")
        login_button = st.button("Login")

        if login_button:
            auth, car_number = authenticate_user(login_username, login_password)
            if auth:
                st.session_state['user_id'] = 1  # You might want to use the actual user ID from the database
                st.session_state['username'] = login_username
                logged_in = True
                login_status.success(f"Logged in as {login_username}")
            else:
                login_status.error("Invalid username or password.")

    elif login_option == "Signup":
        signup_username = st.text_input("Username:")
        signup_password = st.text_input("Password:", type="password")
        signup_car_number = st.text_input("Car Number:", value="ABC123XYZ")

        signup_button = st.button("Signup")

        if signup_button:
            if not (signup_username and signup_password):
                signup_status.error("Username and password are required.")
            elif not any(c.isupper() for c in signup_password) or not any(c.islower() for c in signup_password) \
                    or not any(c.isdigit() for c in signup_password) or not any(c in '!@#$%^&*()_-+=<>?,./;:[]{}|`~' for c in signup_password):
                signup_status.error("Password must have at least 1 uppercase letter, 1 lowercase letter, 1 digit, and 1 special character.")
            elif is_username_registered(signup_username):
                signup_status.error("Username is already registered.")
            else:
                insert_user(signup_username, signup_password, signup_car_number)
                signup_status.success("Signup successful. Please login.")

# Main app logic when the user is logged in
if logged_in:
    # Image Upload and Webcam Capture
    st.write("Welcome to the app!")
    
    # Choose between camera and upload
    option = st.radio("Select option:", ("Camera", "Upload Image"))
    
    if option == "Camera":
        # Perform image classification using camera
        st.sidebar.write("Click 'Start Capture' to begin capturing images from the webcam.")
        start_capture = st.sidebar.button("Start Capture")
        stop_capture = st.sidebar.button("Stop Capture")

        frame = None  # Initialize frame variable

        if start_capture:
            st.subheader("Webcam Capture")
            video_capture = cv2.VideoCapture(0)

            while not stop_capture:
                ret, frame = video_capture.read()
                frame = cv2.resize(frame, (img_cols, img_rows))

                # Display the webcam feed
                st.image(frame, channels="BGR", caption="Webcam Feed", use_column_width=True)

                # Preprocess the image for prediction
                img_array = np.expand_dims(frame, axis=0)

                # Make prediction
                prediction = model_camera.predict(img_array)
                predicted_class = np.argmax(prediction)
                description = descriptions.get(predicted_class, descriptions[predicted_class])

                st.subheader("Prediction Result:")
                st.write(f"Predicted Class: {predicted_class}")
                st.write(f"Description: {description}")
                st.write(f"Prediction Probabilities: {prediction[0]}")

                time.sleep(10)  # Capture an image every 10 seconds

            video_capture.release()

        # Stop capturing when 'Stop Capture' button is pressed
        if stop_capture:
            st.success("Capture stopped.")

    elif option == "Upload Image":
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Uploaded Image", use_column_width=True)
            
            img_rows, img_cols, color_type = 64, 64, 1  # Adjust color_type based on your model's input requirements

            # Function to preprocess a single image for prediction
            def preprocess_image(x, img_rows, img_cols, color_type):
                img = cv2.resize(x, (img_rows, img_cols))
                
                if img is None or img.size == 0:
                    print("Error: Unable to read image.")
                    return None

                if color_type == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = img.astype('float32') / 255.0  
                processed_image = np.expand_dims(img, axis=-1 if color_type == 1 else 0)
                processed_image = np.expand_dims(processed_image, axis=0)
                
                return processed_image
            
            img = preprocess_image(np.array(image_display), img_rows, img_cols, color_type)

            if img is not None:
                # Make predictions
                predictions = model_upload.predict(img)
                predicted_class = np.argmax(predictions)
                # Get the description for the predicted class, fallback to 'default' if not found
                description = descriptions.get(predicted_class, descriptions.get('default', 'Unknown activity'))


                st.subheader("Prediction Result:")
                st.write(f"Predicted Class: {predicted_class}")
                st.write(f"Description: {description}")
                st.write(f"Prediction Probabilities: {predictions[0]}")
