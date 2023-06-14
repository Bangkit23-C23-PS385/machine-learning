import base64
import tensorflow as tf
import json
from google.cloud import pubsub_v1
from input import transform_input
from output import transform_output

project_id = "medicure-bangkit23"
publisher = pubsub_v1.PublisherClient()

def hello_pubsub(event, context):
  """Triggered from a message on a Cloud Pub/Sub topic.
  Args:
  event (dict): Event payload.
  context (google.cloud.functions.Context): Metadata for the event.
  """
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

  pubsub_message = base64.b64decode(event['data']).decode('utf-8')
  json_string = json.dumps(pubsub_message)
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

def publish(result):
  # publish
  topic_id = "predict"
  topic_path = publisher.topic_path(project_id, topic_id)
  future = publisher.publish(topic_path, data=result.encode("utf-8"))
  message_id = future.result()
  print(f"Published message ID: {message_id}")