from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/status")
def status():
    return jsonify({
        "vehicle": "Car",
        "distance": "5.2 m",
        "pedestrian": "Detected",
        "lane": "Safe",
        "pothole": "8.3 m",
        "driver": "Awake",
        "phone": "Not Detected",
        "alert": "No Alert"
    })

if __name__ == "__main__":
    app.run(debug=True)