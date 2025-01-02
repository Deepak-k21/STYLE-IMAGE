import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras import backend as K
from PIL import Image

# Load and preprocess images
def load_image(img_path, target_size=(400, 400)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# De-process image (convert back to RGB)
def deprocess_img(img):
    img = img.squeeze()
    img = img[::-1]  # Convert BGR to RGB
    img = img[:, :, ::-1]  # Convert from float32 to uint8
    return np.clip(img, 0, 255).astype('uint8')

# Load content and style images
content_img = load_image('content.jpg')
style_img = load_image('style.jpg')

# Define VGG19 model and layers for content and style
def get_model():
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    return model

# Create content and style representations using pre-trained VGG19 model
def get_content_and_style_representations(model, content_img, style_img):
    # Content image
    content_layer = 'block5_conv2'  # Layer from which to extract content
    content_rep = model.get_layer(content_layer).output

    # Style image
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # Layers for style
    style_reps = [model.get_layer(layer).output for layer in style_layers]

    # Create a combined model to output content and style features
    combined_model = tf.keras.models.Model(inputs=model.input, outputs=[content_rep] + style_reps)
    content_rep, style_reps = combined_model(content_img), combined_model(style_img)
    
    return content_rep, style_reps

# Calculate content loss
def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

# Calculate style loss
def style_loss(style, generated):
    style = tf.reduce_mean(style, axis=(0, 1))
    generated = tf.reduce_mean(generated, axis=(0, 1))
    return tf.reduce_mean(tf.square(style - generated))

# Compute total loss
def compute_loss(model, content_img, style_img, generated_img):
    content_rep, style_reps = get_content_and_style_representations(model, content_img, style_img)
    generated_rep = model(generated_img)
    
    content_loss_value = content_loss(content_rep[0], generated_rep[0])
    style_loss_value = 0
    for i in range(len(style_reps)):
        style_loss_value += style_loss(style_reps[i], generated_rep[i + 1])
    
    total_loss = content_loss_value + style_loss_value
    return total_loss

# Optimize the generated image to minimize the loss
def run_style_transfer(content_img, style_img, iterations=1000, lr=0.01):
    model = get_model()

    generated_img = tf.Variable(content_img, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(generated_img)
            loss = compute_loss(model, content_img, style_img, generated_img)
        
        gradients = tape.gradient(loss, generated_img)
        optimizer.apply_gradients([(gradients, generated_img)])
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")
            img = deprocess_img(generated_img.numpy())
            plt.imshow(img)
            plt.show()
    
    return generated_img

# Run the style transfer
result = run_style_transfer(content_img, style_img, iterations=1000, lr=0.02)

# Show the result
final_img = deprocess_img(result.numpy())
plt.imshow(final_img)
plt.show()
