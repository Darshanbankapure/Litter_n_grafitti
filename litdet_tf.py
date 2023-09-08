import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Download the pre-trained model checkpoint and pipeline configuration
model_dir = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
tf.keras.utils.get_file(
    model_dir,
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz",
    untar=True,
)

# Load the pipeline configuration and checkpoint
pipeline_config = model_dir + "/pipeline.config"
checkpoint_dir = model_dir + "/checkpoint"

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint_dir + "/ckpt-0").expect_partial()
