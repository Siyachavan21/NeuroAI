import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Define model paths to try in order of preference
model_paths = [
    "E:/new laptop/mega project/integrationFinalProj/backend/Models/emotion_efficientnet.h5",
    "E:/new laptop/mega project/final_year_proj/emotion_efficientnet.h5",
    "E:/new laptop/mega project/integrationFinalProj/backend/Models/emotion_resnet50.h5",
    "E:/new laptop/mega project/final_year_proj/emotion_resnet50.h5"
]

# Get current directory for relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a function to handle errors and write to prediction_results.txt
def write_error_to_file(error_message):
    results_path = os.path.join(current_dir, "prediction_results.txt")
    with open(results_path, "w") as f:
        f.write("Error\n")  # Prediction Result
        f.write("0.00%\n")  # Confidence Level
        f.write("0.0000\n")  # Raw Model Output
        f.write("0.0000\n")  # Enhanced Probability
        f.write(f"Error: {error_message}\n")  # Model Status with error message
    print(f"Error result saved to {results_path}")

def predict_emotion(image_path=None, uploaded_image=None):
    """
    Predict emotion from an image file or uploaded image data
    
    Parameters:
    - image_path: Path to an image file on disk
    - uploaded_image: Image data uploaded from frontend
    
    Returns:
    - Dictionary with prediction results
    """
    try:
        # Try loading models in order until one succeeds
        model = None
        loaded_model_path = None
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Attempting to load model from {model_path}")
                
                # Load model without compiling
                model = tf.keras.models.load_model(model_path, compile=False)
                loaded_model_path = model_path
                print(f"Model loaded successfully from {model_path}!")
                break
        
        if model is None:
            raise Exception("Failed to load any emotion model. Please check model paths.")
        
        # Class labels (must match training order)
        class_names = ["happy", "neutral", "sad"]  # Update if your training order is different
        
        # Set image parameters based on model type
        if "efficientnet" in loaded_model_path.lower():
            img_size = (128, 128)
            color_mode = "grayscale"
            print(f"Using EfficientNet model with grayscale input")
        else:
            img_size = (128, 128)  # Use 128x128 for all models for consistency
            color_mode = "grayscale"  # Use grayscale for all models
            print(f"Using model with grayscale input")
        
        # Process the image - either from path or uploaded data
        if uploaded_image is not None:
            # For uploaded image from frontend
            img = image.load_img(uploaded_image, target_size=img_size, color_mode=color_mode)
            print(f"Processing uploaded image")
        elif image_path is not None:
            # For image path on disk
            if not os.path.exists(image_path):
                raise Exception(f"Image file not found at: {image_path}")
            img = image.load_img(image_path, target_size=img_size, color_mode=color_mode)
            print(f"Processing image from path: {image_path}")
        else:
            # Default to test_image.png if no image is provided
            img_path = os.path.join(current_dir, "test_image.png")
            if not os.path.exists(img_path):
                raise Exception(f"Default image file not found at: {img_path}")
            img = image.load_img(img_path, target_size=img_size, color_mode=color_mode)
            print(f"Processing default image from: {img_path}")
        
        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        print(f"Image shape: {img_array.shape}")
        
        # Make prediction
        pred = model.predict(img_array, verbose=0)
        
        predicted_emotion = class_names[np.argmax(pred)]
        confidence = np.max(pred)
        raw_output = np.max(pred)
        enhanced_prob = confidence
        model_status = f"Active ({os.path.basename(loaded_model_path)})"
        
        # Save results to a file that PDF generator will read
        results_path = os.path.join(current_dir, "prediction_results.txt")
        with open(results_path, "w") as f:
            f.write(f"{predicted_emotion}\n")  # Prediction Result
            f.write(f"{confidence*100:.2f}%\n")  # Confidence Level
            f.write(f"{raw_output:.4f}\n")  # Raw Model Output
            f.write(f"{enhanced_prob:.4f}\n")  # Enhanced Probability
            f.write(f"{model_status}\n")  # Model Status
        
        print(f"Prediction results saved successfully to {results_path}!")
        print(f"Predicted emotion: {predicted_emotion} with confidence: {confidence*100:.2f}%")
        
        # Return results as dictionary
        return {
            "success": True,
            "label": predicted_emotion,
            "confidence": confidence,
            "raw_output": raw_output,
            "enhanced_prob": enhanced_prob,
            "model_status": model_status
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        write_error_to_file(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

# If this script is run directly, use the default test image
if __name__ == "__main__":
    predict_emotion()
    print("Process completed.")
