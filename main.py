import tensorflow as tf
from tensorflow.compat.v1 import graph_util
import time
from google.cloud import pubsub_v1
from input import transform_input
from output import transform_output

# setup pubsub
project_id = "medicure-bangkit23"
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

# # subscribe
# subscription_id = "submit-data-sub"
# subscription_path = subscriber.subscription_path(project_id, subscription_id)
# def callback(message):
#     print(f"Received message: {message.data.decode('utf-8')}")
#     message.ack()
# subscriber.subscribe(subscription_path, callback=callback)

# print("Listening for messages...")
# while True:
#     time.sleep(5)

# Path to the .tflite model file
model_path = './my_model.tflite'

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
print(output)

# publish
topic_id = "predict"
topic_path = publisher.topic_path(project_id, topic_id)
message = b"Hello, Pub/Sub!"
future = publisher.publish(topic_path, data=message)
message_id = future.result()
print(f"Published message ID: {message_id}")