from datetime import datetime
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from callbacks.train_visualization_callback import LossAndAccuracySaveImage
from callbacks.lfw_eval_callback import LFWEvaluation
from loadContentFiles import load_yaml
from models import resnet50_softmax

# Load config file
cfgData = load_yaml('config.yml')

# re-size all the images to this
IMAGE_SIZE = [cfgData['inputSize'], cfgData['inputSize']]
CNN_ARCH = cfgData['backbone']
EXPERIMENT_NAME = 'Aligned'

""" Train/Val Data """
TRAIN_PATH = cfgData['train-path']
VAL_PATH = cfgData['validation-path']

# Get the number of classes
folders = glob(os.path.join(TRAIN_PATH, "*"))
NUM_CLASSES = len(folders)
CONTINUE_FROM_CHECKPOINT = True

# ResNet50 backbone
model, preprocess_input = resnet50_softmax(cfgData, NUM_CLASSES)

if not cfgData["checkpoint_path"]:
    path = 'checkpoints/ResNet50/1/weights_7.h5'
    model.load_weights(path)


# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=cfgData['learning-rate']),
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

training_set = train_datagen.flow_from_directory(TRAIN_PATH,
                                                 target_size=(cfgData['inputSize'], cfgData['inputSize']),
                                                 batch_size=cfgData['batch-size'],
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(VAL_PATH,
                                            target_size=(cfgData['inputSize'], cfgData['inputSize']),
                                            batch_size=cfgData['batch-size'],
                                            class_mode='categorical')

# -------------------- Callbacks


checkpoint_timestamp = datetime.now().strftime("%d%m%Y_%H%M")
path = os.path.join('checkpoints', CNN_ARCH, checkpoint_timestamp)
if not os.path.exists(path):
    os.makedirs(path)
modelCheckpoint_callback = ModelCheckpoint(
    os.path.join(path, 'weights_{epoch}.h5'),
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)

ImageCheckpoint = LossAndAccuracySaveImage()

lfwEvaluation_callback = LFWEvaluation(cfgData)

# fit the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=cfgData['epoch'],
    callbacks=[modelCheckpoint_callback, lfwEvaluation_callback],
)

model.save('resnet50_softmax_casia.h5')
# Save model history
np.save('history1.npy', r.history)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_accuracy'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
