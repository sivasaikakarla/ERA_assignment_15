import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. Define Custom Objects (REQUIRED for Creative Loss) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    import tensorflow.keras.backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# --- 2. Load the Model ---
# Ensure 'unet_pet_segmentation.h5' is uploaded to the Space's "Files" tab
model = load_model('unet_pet_segmentation.h5', 
                   custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss})

# --- 3. Prediction Function ---
def predict_mask(image):
    # Resize to 128x128 to match model input
    img_resized = tf.image.resize(image, (128, 128))
    # Normalize (0-1)
    img_normalized = tf.cast(img_resized, tf.float32) / 255.0
    # Add batch dimension (1, 128, 128, 3)
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    pred_mask = model.predict(img_expanded)[0]
    
    # Threshold: if > 0.5, it's the pet. Otherwise background.
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    
    return pred_mask

# --- 4. Launch Interface ---
interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(label="Upload Pet Image"),
    outputs=gr.Image(label="Predicted Segmentation Mask"),
    title="U-Net Pet Segmentation (Creative Loss Demo)",
    description="This app uses a custom U-Net trained with Dice Loss to segment pets from backgrounds."
)

interface.launch()