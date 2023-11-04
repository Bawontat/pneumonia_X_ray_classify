#Inference Model to test Set
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import pandas as pd
import os

#Loading Data -----------------------------------------------------------------
test_path = 'D:\\pneumonia_X_ray_classify/Dataset/test'

test_set = image_dataset_from_directory(
	test_path,
	labels='inferred',
	color_mode='rgb',
	label_mode='categorical',
	image_size=[512,512],
	shuffle=False)




#Inference Model loading -----------------------------------------------------------------
model_path = 'D:\\pneumonia_X_ray_classify/Weight/cut_d256' #*********************
model = tf.keras.models.load_model(model_path)
model.summary()

inference_model = tf.keras.Sequential([
	model,
])

inference_model.build(input_shape=[None,512,512,3])
inference_model.summary()


#Set every change data set
num_class_normal = 234
locate_class_pneumonia = 390 + num_class_normal



#Predictor-----------------------------------------------------------------------
Pred = inference_model.predict(test_set)
TP = 0
TN = 0
FP = 0
FN = 0

ordering_FP = []
ordering_FN = []

for i in range(int(Pred.shape[0])):
	class_number = Pred[i].argmax()
	prob = float(Pred[i,class_number]) * 100.00
	print("Image No {} --- Class {} --- Probability {}".format(i,class_number,prob))

	#Make Visualize and Save location path of miss detection
	if (i < num_class_normal):
		if class_number ==  0:
			TN += 1
		else:
			ordering_FP.append(i)
			FP += 1

	elif (num_class_normal < i < locate_class_pneumonia):
		if class_number ==  1:
			TP += 1
		else:
			ordering_FN.append(i)
			FN += 1




print("Total sample : ",(TP+TN+FP+FN))
print('-----------------------------------------------------------')
print("Totally_correct : ", TP + TN)
print("Predict Normal to Normal (True negative) : ", TN)
print("Predict Pneumonia to Pneumonia (True positive) : ", TP)
print('-----------------------------------------------------------')
print("Totally_misstake : ", FP + FN)
print("Predict Normal to Pneumonia (False Positive) : ", FP)
print("Predict Pneumonia to Normal (False Negative) : ", FN)
print('-----------------------------------------------------------')
print("Recall : ", TP/(TP+FN))
print("Precission : ", TP/(TP+FP))



print("----- Ordering FP ------")
print(ordering_FP)
print("----- Ordering FN ------")
print(ordering_FN)



#Saving filename of FP & FN
path_FP = 'D:\\pneumonia_X_ray_Classification/S3/FP_name_cut_d256.txt' #*********************
path_FN = 'D:\\pneumonia_X_ray_Classification/S3/FN_name_cut_d256.txt' #*********************
normal_logs = []
pneumonia_logs = []

#List Directory normal case
for file in os.listdir(test_path+'/NORMAL'):
    if file.endswith(".jpeg"):
        normal_logs.append(file)

for file in os.listdir(test_path+'/PNEUMONIA'):
    if file.endswith(".jpeg"):
        pneumonia_logs.append(file)


#Save FP,FN to Text file
with open(path_FP, 'w') as f:
	for order in ordering_FP:
		f.write(normal_logs[order])
		f.write('\n')

with open(path_FN, 'w') as f:
	for order in ordering_FN:
		f.write(pneumonia_logs[order-234]) #Locate to actual ordering in folder
		f.write('\n')


