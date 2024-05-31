import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
import math
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_examples = 20225
test_examples = 2555
validation_examples = 2551
img_height = img_width = 224
batch_size = 8

# # NasNet
# model = keras.Sequential([
#     hub.KerasLayer(r'C:\\Users\\32573\\Desktop\\tools\py\\cancer_classification_project\\saved_model',
#                    trainable=True),
#     layers.Dense(1, activation='sigmoid'),
# ])

model = keras.models.load_model('isic_model/')

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    dtype=tf.float32,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
    'data/train/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=123,
)

validation_gen = validation_datagen.flow_from_directory(
    'data/validation/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    'data/test/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=123,
)

METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='Recall'),
    keras.metrics.AUC(name='AUC'),
]

train_writer = tf.summary.create_file_writer('logs/train/')
test_writer = tf.summary.create_file_writer('logs/test/')

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)

model.fit(
    train_gen,
    epochs=1,
    verbose=2,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    callbacks=[keras.callbacks.ModelCheckpoint('isic_model')]
)


def plot_roc(label, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(label, predictions)

    plt.plot(100*fp, 100*tp)
    plt.xlabel('False Positives [%]')
    plt.ylabel('True Positives [%]')
    plt.show()


test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches = 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

plot_roc(test_labels, test_gen)
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)

num_epochs = 5
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
train_writer = tf.summary.create_file_writer('logs/train/')
test_writer = tf.summary.create_file_writer('logs/test/')
train_step = test_step = 0


for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)

    with train_writer.as_default():
        tf.summary.scalar('Loss', loss, step=epoch)
        tf.summary.scalar(
            'Accuracy', acc_metric.result(), step=epoch,
        )

    acc_metric.reset_states()

    for batch_idx, (x, y) in enumerate(ds_test):
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
        acc_metric.update_state(y, y_pred)

    with test_writer.as_default():
        tf.summary.scalar('Loss', loss, step=epoch)
        tf.summary.scalar(
            'Accuracy', acc_metric.result(), step=epoch,
        )

    acc_metric.reset_states()

    acc_metric.reset_states()
