from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load the trained model
with open("Classification_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to extract features from a URL
def extract_features(url):
    url_length = len(url)
    has_copyright_info = 1 if "copyright" in url.lower() else 0
    is_https = 1 if url.startswith("https://") else 0
    subdomain_count = url.count(".") - 1

    return [url_length, has_copyright_info, is_https, subdomain_count]

@app.route("/predict", methods=["OPTIONS", "POST"])
def predict():
    if request.method == "OPTIONS":
        return '', 204  # Respond to preflight requests

    data = request.json
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract features from the given URL
    features = extract_features(url)

    # Predict using the model
    prediction = model.predict([features])

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port)
