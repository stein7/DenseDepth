import tensorflow as tf
#tf.enable_eager_execution()
from model import DepthEstimate

model = DepthEstimate()
checkpoint_path = "training_1/cp__.ckpt"
model.load_weights(checkpoint_path)

print('Model weights loaded.')

tf.keras.models.save_model

# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.restore(checkpoint_path)

from evaluate import load_test_data, evaluate

rgb, depth, crop = load_test_data()
evaluate(model, rgb, depth, crop)