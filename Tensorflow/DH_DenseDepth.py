batch_size     = 1 #8
learning_rate  = 0.0001
epochs         = 10

from model import DepthEstimate

model = DepthEstimate()

from data import DataLoader

dl = DataLoader()
train_generator = dl.get_batched_dataset(batch_size)

print('Data loader ready.')


import tensorflow
from loss import depth_loss_function

optimizer = tensorflow.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)

model.compile(loss=depth_loss_function, optimizer=optimizer)


# Create checkpoint callback
import os
checkpoint_path = "training_4_callback/ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)




class CustomCallback(tensorflow.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        inputs = self.model.inputs
        outputs = self.model.outputs
        print("Inputs: ", inputs)
        print("Outputs: ", outputs)
        

class ImageCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Get a batch of inputs
        #rgb_images, depth_images = next(iter(self.model.data_loader))
        
        rgb, depth = next(iter(train_generator))
        rgb.numpy().mean()
        
        
        self.model(rgb).numpy().mean()
        depth.numpy().mean()
        # Predict using the model
        predicted_depth_images = self.model.predict_on_batch(rgb)
        predicted_depth_images.mean()
        # # Display
        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb_images[0])
        # plt.title("Input RGB")
        # plt.subplot(1, 3, 2)
        # plt.imshow(depth_images[0, :, :, 0], cmap='gray')
        # plt.title("True Depth")
        # plt.subplot(1, 3, 3)
        # plt.imshow(predicted_depth_images[0, :, :, 0], cmap='gray')
        # plt.title("Predicted Depth")
        # plt.show()
        
        
# Start training
model.fit(train_generator, epochs=5, steps_per_epoch=dl.length//batch_size, callbacks=[cp_callback, ImageCallback(), CustomCallback()])

#model.save('my_model', save_format="tf")
#model.save('my_model.h5')

