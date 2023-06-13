import tensorflow as tf
from input import transform_input
from output import transform_output

# Path to the .tflite model file
model_path = '/content/drive/MyDrive/my_model.tflite'

# Load the model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Access the input and output tensors
input_tensor_index = input_details[0]['index']
output_tensor_index = output_details[0]['index']
input_tensor = interpreter.tensor(input_tensor_index)
output_tensor = interpreter.tensor(output_tensor_index)

# TEST
# Prepare your input data
input_data = ['drying_and_tingling_lips',	'mucoid_sputum', 'diarrhea',	'chest_pain',	'phlegm',	'continuous_feel_of_urine',	'bruising']  # Prepare your input data as per the model requirements

# Set the input tensor data
input_tensor()[0] = transform_input(input_data)

# Run the inference
interpreter.invoke()

# Get the output tensor data
output_data = output_tensor()[0]

output = transform_output(output_data)