import os
from flask import Flask, render_template, request
from utils.predict import predict_video

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result, confidence, web_frames = predict_video(filepath)

    return render_template(
        "result.html",
        result=result.lower(),
        confidence=round(confidence * 100, 2),
        filename=file.filename,
        frames=web_frames
    )


if __name__ == "__main__":
    app.run(debug=True)