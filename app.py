from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = "blood_group_model.keras"
model = load_model(MODEL_PATH)

# Define the blood group labels
blood_groups = ['A+', 'A−', 'AB+', 'AB−', 'B+', 'B−', 'O+', 'O−']

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No file selected"

        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(128, 128), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = blood_groups[predicted_index]

        return render_template("result.html", label=predicted_label, img_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
