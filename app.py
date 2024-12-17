import os
from flask import Flask, request, render_template
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import pickle

# Load environment variables
load_dotenv()

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv()

# Get the Google API key from the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Generative AI API with the API key
genai.configure(api_key=google_api_key)

# Initialize the model
model = genai.GenerativeModel("gemini-pro")

# Start the chat
chat = model.start_chat(history=[])


# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the pickle files
model_path = os.path.join(BASE_DIR, "models", "t.sav")
scaler_path = os.path.join(BASE_DIR, "models", "stand.pkl")

# Load the pickle files
loaded_model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Route for the home page
@app.route("/")
def home():
    return render_template("home.html")

# Route for the LLM Chatbot page
@app.route("/llm_model", methods=["GET", "POST"])
def llm_model():
    response_text = ""
    if request.method == "POST":
        user_input = request.form.get("input")
        if user_input:
            response = get_gemini_response(user_input)
            response_text = "".join(chunk.text for chunk in response)
    return render_template("index.html", response_text=response_text)

# Route for the ML Diabetes Prediction page
@app.route("/ml_model", methods=["GET"])
def ml_model():
    return render_template("form.html")

# Route for ML Prediction
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract form data
        input_data = [
            float(request.form['Glucose level']),
            float(request.form['BloodPressure value']),
            float(request.form['SkinThickness value']),
            float(request.form['Insulin level']),
            float(request.form['BMI value']),
            float(request.form['Diabetes Pedigree Function level']),
            float(request.form['Age of Person'])
        ]
        
        # Convert and scale input data
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        std_data = scaler.transform(input_data_as_numpy_array)
        
        # Predict using the model
        prediction = loaded_model.predict(std_data)
        
        # Determine result
        result = "The patient has diabetes." if prediction[0] == 1 else "The patient doesn't have diabetes."
        return render_template('result.html', Prediction=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function to get Gemini responses
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

if __name__ == "__main__":
    app.run(debug=True)