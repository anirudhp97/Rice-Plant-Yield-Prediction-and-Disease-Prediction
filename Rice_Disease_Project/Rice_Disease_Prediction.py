#Importing the necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import pandas as pd

#Plotting a sample image from the dataset
img = cv2.imread("../LabelledRice/Labelled/BrownSpot/IMG_2992.jpg")
img_cvt = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()

#Plotting the format and shape of the sample image
print("Type of OpenCV image = {}".format(type(img_cvt))) 

print("Input image shape - {}".format(img_cvt.shape))

#Declaring the variable to resize all the images to the size 64 X 64
default_image_size = tuple((64,64))

#Storing the image directory
image_dir = '../LabelledRice/Labelled'

#Function to resize the image
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
 
#Listing the root directory of the dataset
directory_root = '../LabelledRice'


#Storing the images and its respective labels in a list.
#Image_list stores the pixel values of the images whereas the label_list stores the class labels.
from os import listdir
image_list,label_list = [],[]
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

#Printing the total number of images in the dataset
image_size = len(image_list)
print(image_size)

#Converting the class labels from string to integer valeus and appending the converted values to the images.
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)

#Normalizing the pixel values between 0 and 1
np_image_list = np.array(image_list, dtype=np.float16) / 255.0

#Splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.10, random_state = 0)

#Data augmentation
train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True
       )


#Convolutional Neural Networks
classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, (3, 3), padding="same",activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, (3, 3), padding="same",activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Saving the model
classifier.save("Rice.h5")

classifier.summary()

#Model training begins
BS = 8
history = classifier.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=BS),
    steps_per_epoch=len(x_train) // BS,
    epochs=150
    )

#Test set accuracy
scores = classifier.evaluate(x_test,y_test)
print(f"Test Accuracy: {scores[1]*100}")

prediction = classifier.predict(x_test)
print(classification_report(y_test, np.round(prediction)))


#Check for optimal threshold for accurate predictions
fpr, tpr, thresholds = roc_curve(y_test, prediction)
print(thresholds)

accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(prediction>thres,1,0)
    accuracy_ls.append(accuracy_score(y_test, y_pred, normalize=True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                        axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
accuracy_ls.head()


#Plot epoch vs accuracy and epoch vs loss graphs
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#Test the model performance for new set of images by giving it's path
default_image_size = tuple((64, 64))
model=load_model("Rice.h5")
image_path="../RiceDiseaseDataset/validation/BrownSpot/IMG_2992.jpg"
img = image.load_img(image_path, target_size=default_image_size)
img = np.expand_dims(img, axis=0)
image_array = np.array(img, dtype=np.float16) / 255.0
result=model.predict_proba(image_array)
print(result)

if result[0][0] < 0.320236 :
  prediction="Brownspot Disease spotted!"
else :
  prediction="Good News! You're rice plant looks healthy!"
print(prediction)