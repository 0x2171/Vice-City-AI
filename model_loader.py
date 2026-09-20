"""
Загрузка модели TensorFlow
"""

import tensorflow as tf
from object_detection.utils import label_map_util

from config import PATH_TO_MODEL, PATH_TO_LABELS


def load_model():
    """Загрузка frozen graph TensorFlow"""
    print("Загрузка модели...")
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            od_graph_def.ParseFromString(fid.read())
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True
    )
    
    sess = tf.compat.v1.Session(graph=detection_graph)
    
    # Входные/выходные тензоры
    image_tensor      = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes   = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores  = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections    = detection_graph.get_tensor_by_name('num_detections:0')
    
    print("Модель загружена.")
    
    return sess, category_index, (
        image_tensor, detection_boxes, detection_scores,
        detection_classes, num_detections
    )