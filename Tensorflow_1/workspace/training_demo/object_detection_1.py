import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


MODEL_NAME = 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.config'
LABELS = "mscoco_label_map.pbtxt"
PATH_TO_SAVED_MODEL = 'C:\Users\Josue\OneDrive\Documentos\GitHub\Augmented_Worker\Tensorflow_1\models\research\object_detection\configs\tf2' + MODEL_NAME
PATH_TO_LABELS = 'C:\Users\Josue\OneDrive\Documentos\GitHub\Augmented_Worker\Tensorflow_1\models\research\object_detection\data' + LABELS

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))