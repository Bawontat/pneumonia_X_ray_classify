#Use Tensorflow as Advance
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers,Model
import pandas as pd
import tensorflow_hub as hub

#Loading Data -----------------------------------------------------------------
train_path = 'D:\\pneumonia_X_ray_classify/Dataset/train'
valid_path = 'D:\\pneumonia_X_ray_classify/Dataset/val'


train_set = image_dataset_from_directory(
	train_path,
	labels='inferred',
	color_mode='rgb',
	label_mode='categorical',
	image_size=[512,512],
	shuffle=True)

valid_set = image_dataset_from_directory(
	valid_path,
	labels='inferred',
	color_mode='rgb',
	label_mode='categorical',
	image_size=[512,512],
	shuffle=True)



print("Train Detail")
for image_batch, labels_batch in train_set:
	print(image_batch.shape)
	print(labels_batch.shape)
print(train_set.class_names)

print("Valid Detail")
for image_batch, labels_batch in valid_set:
	print(image_batch.shape)
	print(labels_batch.shape)
print(valid_set.class_names)


#Data Pipeline for Push data to the memory and preprocessing during training
AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)


#Data Augmentation : Not split preparation phase to Model Structure
augmentation_fn = tf.keras.Sequential([
  layers.RandomTranslation(height_factor=[-0.2,0.2],width_factor=[-0.2,0.2]),
  layers.RandomRotation([-0.05,0.05]),

])

train_set = train_set.map(
  lambda x, y: (augmentation_fn(x, training=True), y))

valid_set = valid_set.map(
  lambda x, y: (augmentation_fn(x, training=True), y))





class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.backbone = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5",trainable=False)
		self.normalize_input = layers.Rescaling(1./255, input_shape=[512, 512, 3])
		self.d512 = layers.Dense(512, activation='relu')
		self.d128 = layers.Dense(128, activation='relu')
		self.d64  = layers.Dense(64, activation='relu')
		self.softmax2 = layers.Dense(2, activation='softmax')

	def call(self, x):
		x = self.normalize_input(x)
		x = self.backbone(x)
		x = self.d512(x)
		x = self.d128(x)
		x = self.d64(x)
		x = self.softmax2(x)
		return x

model = MyModel()
#Define processing function
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#Define Metrci verify function
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')


@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  #Mean average for each iteration
  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def valid_step(images, labels):
  predictions = model(images, training=False)
  loss = loss_fn(labels, predictions)
  valid_loss(loss)
  valid_accuracy(labels, predictions)

model.build(input_shape=[None,512,512,3])
model.summary()

epochs = 20

loss_log = []
accuracy_log = []

for i in range(epochs):
	#Initialize metric every start new epochs
	train_loss.reset_states()
	valid_loss.reset_states()
	train_accuracy.reset_states()
	valid_accuracy.reset_states()

	for train_image, train_label in train_set:
		train_step(train_image,train_label)
	for valid_image, valid_label in valid_set:
		valid_step(valid_image,valid_label)
	loss_log.append([float(train_loss.result().numpy()) , float(valid_loss.result().numpy())])
	accuracy_log.append([float(train_accuracy.result().numpy()) , float(valid_accuracy.result().numpy())])
	print(
    	f'Epoch {i + 1}, '
    	f'Loss: {train_loss.result()}, '
    	f'Accuracy: {train_accuracy.result() * 100}, '
    	f'Valid Loss: {valid_loss.result()}, '
    	f'Valid Accuracy: {valid_accuracy.result() * 100}'
  	)

#Backup ------------------------------------------------------------------------
#Save Model + Weight
model.save('saved_model/my_model')


#Visualization -----------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
loss_frame = pd.DataFrame(loss_log,columns=['loss', 'val_loss'])
accuracy_frame = pd.DataFrame(accuracy_log,columns=['accuracy', 'val_accuracy'])

loss_frame.loc[:, ['loss', 'val_loss']].plot()
accuracy_frame.loc[:, ['accuracy', 'val_accuracy']].plot();
plt.show()