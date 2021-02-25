import os
import cv2
import tensorflow as tf
import keras_preprocessing
import streamlit as st
import numpy as np
import pandas as pd
import urllib

from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


# Set up default variables
PATH_TO_CKPT = 'saved_model/saved_model.pb'
PATH_TO_LABELS = 'object-detection1.pbtxt'
PATH_TO_IMAGE = 'images/out.png'
CWD_PATH = os.getcwd()
NUM_CLASSES = 90

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def main():
    st.title("MEP Object Detection 👁")

    # Download all model files if they aren't already in the working directory.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    #Disable warnings
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("This application detects Mechanical, Electrical and Plumbing assets in images.")
    st.write("## How does it work?")
    st.write("Upload an image of a room and the deep learning learning model will interrogate it to find MEP assets. It currently supports sockets, switches and radiators")
    st.write("## Upload your own image")
    st.write("**Note:** The model has been trained on typical classrooms and household rooms and will therefore work best for those use cases.")
    uploaded_image = st.file_uploader("Choose a png or jpg image",
                                      type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        im = image.save('images/out.png')
        # Make sure image is RGB
        # TODO: If image is over certain size resize image to certain size

        if st.button("Make a prediction"):
            print('Im breathing')
            "Making a prediction and drawing MEP boxes on your image..."
            with st.spinner("Doing the math..."):
                st.image(detect(), caption="MEP assets detected.", use_column_width=True)

        st.image(image, caption="Uploaded Image.", use_column_width=True)

    st.write("## How is this made?")
    st.write("A [Faster R-CNN](https://arxiv.org/abs/1506.01497?source=post_page) model is used to perform the detections, \
    this front end (what you're reading) is built with [Streamlit](https://www.streamlit.io/) \
    and it's all hosted on [Heroku](https://www.heroku.com/).")
    st.write("See the [code on GitHub](https://github.com/Jam516/MEP-detection-app)")

def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def load_model(model_name):
  model = tf.saved_model.load(str('saved_model'))
  model = model.signatures['serving_default']
  return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict

def detect():
    #might work on streamlit or colab under tensorflow two. Test this 
  model = tf.compat.v1.saved_model.load_v2('saved_model', None)
  model = model.signatures['serving_default']
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(PATH_TO_IMAGE))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  return image_np
# def detect():
#     #located at models/research/object_detection/object_detection_tutorial.ipynb.
#     detection_model = tf.compat.v1.saved_model.load_v2(str('saved_model'), None)
#     detection_model = detection_model.signatures['serving_default']
#
#     image_np = np.array(Image.open(PATH_TO_IMAGE)) ###
#     image_np = np.asarray(image_np)
#     # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
#     input_tensor = tf.convert_to_tensor(image_np)
#     # The model expects a batch of images, so add an axis with `tf.newaxis`.
#     input_tensor = input_tensor[tf.newaxis,...]
#
#     # Run inference
#     output_dict = detection_model(input_tensor)
#     print(output_dict)
#
#     # All outputs are batches tensors.
#     # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#     # We're only interested in the first num_detections.
#     num_detections = int(output_dict.pop('num_detections'))
#     output_dict = {key:value[0, :num_detections].numpy()
#                    for key,value in output_dict.items()}
#     output_dict['num_detections'] = num_detections
#
#     # detection_classes should be ints.
#     output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
#
#     vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks_reframed', None),
#       use_normalized_coordinates=True,
#       line_thickness=8)



    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')
    #     sess = tf.Session(graph=detection_graph)
    #
    # # Input tensor is the image
    # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # # Output tensors are the detection boxes, scores, and classes
    # # Each score represents level of confidence for each of the objects.
    # # The score is shown on the result image, together with the class label.
    # detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # # Number of objects detected
    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #
    # # Load image using OpenCV and
    # # expand image dimensions to have shape: [1, None, None, 3]
    # # i.e. a single-column array, where each item in the column has the pixel RGB value
    # in_image = cv2.imread(PATH_TO_IMAGE)
    # image_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
    # image_expanded = np.expand_dims(
    #     image_rgb, axis=0)
    #
    # # Perform the actual detection by running the model with the image as input
    # (boxes, scores, classes, num) = sess.run(
    #     [detection_boxes, detection_scores, detection_classes, num_detections],
    #     feed_dict={image_tensor: image_expanded})
    #
    # # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     in_image,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0.50)
    #
    # return in_image

EXTERNAL_DEPENDENCIES = {
    "saved_model/saved_model.pb" : {
        "url": "https://socket-no-socket.s3.eu-west-2.amazonaws.com/saved_model.pb",
        "size": 134516139 # in bytes
    }
}

if __name__ == "__main__":
    main()
