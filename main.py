import tensorflow as tf
from tensorflow.compat.v1 import graph_util
import time
import json
from google.cloud import pubsub_v1
from input import transform_input
from output import transform_output

# setup pubsub
project_id = "medicure-bangkit23"
publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

def callback(message):
    try:
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

        # Prepare your input data
        json_string = json.dumps(message.data.decode('utf-8'))
        print(json_string)
        decoded_input = json.loads(json_string)
        # Set the input tensor data
        input_tensor()[0] = transform_input(decoded_input)

        # Run the inference
        interpreter.invoke()

        # Get the output tensor data
        output_data = output_tensor()[0]

        # Transform output
        output = transform_output(output_data)
        result = {'disease': output}
        dumped_result = json.dumps(result)
        publish(dumped_result)
        message.ack()
    except:
        print(message.data)

def publish(result):
    # publish
    topic_id = "predict"
    topic_path = publisher.topic_path(project_id, topic_id)
    future = publisher.publish(topic_path, data=result.encode("utf-8"))
    message_id = future.result()
    print(f"Published message ID: {message_id}")

# subscribe
subscription_id = "submit-data-sub"
subscription_path = subscriber.subscription_path(project_id, subscription_id)
subscriber.subscribe(subscription_path, callback=callback)

print("Listening for messages...")
while True:
    time.sleep(5)