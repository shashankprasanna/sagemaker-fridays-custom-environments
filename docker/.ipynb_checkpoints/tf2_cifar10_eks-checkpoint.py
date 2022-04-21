import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
import argparse
import os
import re
import time

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

def single_example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    image = train_preprocess_fn(image)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label

def train_preprocess_fn(image):

    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    return image

def get_dataset(filenames, batch_size):
    """Read the images and labels from 'filenames'."""
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat().shuffle(10000)

    # Parse records.
    dataset = dataset.map(single_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def get_model(input_shape, learning_rate, weight_decay, optimizer, momentum):
    input_tensor = Input(shape=input_shape)
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                          weights='imagenet',
                                                          input_tensor=input_tensor,
                                                          input_shape=input_shape,
                                                          classes=None)
    x = Flatten()(base_model.output)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Hyper-parameters
epochs       = 15
lr           = 0.001
batch_size   = 256
momentum     = 0.9
weight_decay = 2e-4
optimizer    = 'sgd'
model_type   = 'resnet'

os.system('aws s3 sync s3://sagemaker-us-west-2-453691756499/datasets/cifar10-tfrecords/ dataset')

training_dir   = 'dataset/train'
validation_dir = 'dataset/validation'
eval_dir       = 'dataset/eval'

train_dataset = get_dataset(training_dir+'/train.tfrecords',  batch_size)
val_dataset   = get_dataset(validation_dir+'/validation.tfrecords', batch_size)
eval_dataset  = get_dataset(eval_dir+'/eval.tfrecords', batch_size)

input_shape = (HEIGHT, WIDTH, DEPTH)
model = get_model(input_shape, lr, weight_decay, optimizer, momentum)

# Optimizer
if optimizer.lower() == 'sgd':
    opt = SGD(lr=lr, decay=weight_decay, momentum=momentum)
else:
    opt = Adam(lr=lr, decay=weight_decay)

# Compile model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(train_dataset, steps_per_epoch=40000 // batch_size,
                    validation_data=val_dataset, 
                    validation_steps=10000 // batch_size,
                    epochs=epochs)


# Evaluate model performance
score = model.evaluate(eval_dataset, steps=10000 // batch_size, verbose=1)
print('Test loss    :', score[0])
print('Test accuracy:', score[1])
