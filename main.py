from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from markupsafe import Markup
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Mount the static files (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates from the "templates" directory
templates = Jinja2Templates(directory="templates")

# Load the TensorFlow model
model = tf.keras.models.load_model("final_model.h5")
class_name = ["Early_blight", "Late_blight", "Healthy"]
Allowed_Extensions = {"image/jpeg", "image/png"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Serve the index.html (home page)
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction endpoint
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Validate file extension
    if file.content_type not in Allowed_Extensions:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Read and preprocess the image
    image = read_file_as_image(await file.read())
    img_array = np.expand_dims(image, 0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = 100 * (np.max(predictions[0]))

    # Disease information dictionary
    disease_info = {
        "Early_blight": {
            "cause": [
                "Caused by the fungus Alternaria solani.",
                "Thrives in warm, wet conditions."
            ],
            "solution": [
                "Use fungicides to control the spread.",
                "Practice crop rotation to prevent recurrence.",
                "Remove and destroy infected plant debris."
            ]
        },
        "Late_blight": {
            "cause": [
                "Caused by the oomycete Phytophthora infestans.",
                "Spreads rapidly in cool, moist conditions."
            ],
            "solution": [
                "Apply fungicides to protect plants.",
                "Ensure proper spacing for air circulation.",
                "Remove and destroy infected plants immediately."
            ]
        },
        "Healthy": {
            "cause": [
                "No disease present."
            ],
            "solution": [
                "Maintain good agricultural practices:",
                "Proper watering to avoid water stress.",
                "Regular fertilization to ensure nutrient availability.",
                "Effective pest control to prevent infestations."
            ]
        }
    }

    # Get cause and solution for the predicted class
    cause = disease_info[predicted_class]["cause"]
    solution = disease_info[predicted_class]["solution"]

    # Return the result to display.html
    result = {
        'class': predicted_class,
        'confidence': float(confidence),
        'cause': cause,
        'solution': solution
    }

    # Render the display.html with the prediction result


    return templates.TemplateResponse("display.html", {
        "request": request,
        "result": Markup(f"""
            <div>
                <h2>Prediction Result</h2>
                <p><strong>Class:</strong> {predicted_class}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                <h3>Details</h3>
                <p><strong>Cause:</strong></p>
                <ul>
                    {"".join(f"<li>{point}</li>" for point in cause)}
                </ul>
                <p><strong>Solution:</strong></p>
                <ul>
                    {"".join(f"<li>{point}</li>" for point in solution)}
                </ul>
            </div>
        """)
    })


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
