import gradio as gr
import os
from CNNClassifier.pipeline.prediction import PredictionPipeline

# Initialize the prediction pipeline (model is loaded once)
prediction_pipeline = PredictionPipeline()


def predict_image(image):
    try:
        # Save the uploaded image to a temporary, safely named file
        image_path = "temp_uploaded_image.png"
        image.save(image_path)

        # Make prediction
        predictions = prediction_pipeline.predict(image_path)

        # Extract the prediction result
        prediction_str = predictions[0]["image"]
    except Exception as e:
        prediction_str = f"An error occurred: {e}"
    finally:
        # Clean up the temporary image file, if it exists
        if os.path.exists(image_path):
            os.remove(image_path)

    return prediction_str


# Define the Gradio interface
iface = gr.Interface(fn=predict_image,
                     inputs=gr.components.Image(type="pil", label="Upload an image"),
                     outputs=gr.components.Textbox(label="Prediction"),
                     title="Chest CT Scan Classifier",
                     description="Upload a chest CT scan image to classify it as Normal or Adenocarcinoma Cancer.",
                     article=f"""
                     This model is based on a Convolutional Neural Network trained with a dataset of chest CT scan images.<br><br>
                     <strong>Created by Dr. Jody-Ann S. Jones</strong><br>
                    Email: <a href="mailto:drjodyannjones@gmail.com">drjodyannjones@gmail.com</a><br>
                    GitHub: <a href="https://github.com/drjodyannjones/End-to-End-Cancer-Classification-Project" target="_blank">
                    <img src="https://icons.getbootstrap.com/assets/icons/github.svg" width="20" height="20" />End to End Chest CT Scan Classifier</a>
                    """
                    )

# Launch the application
if __name__ == "__main__":
    iface.launch()
