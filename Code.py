#Done by Omar Tarek Elsayed (20031826)
import tensorflow as tf
from tensorflow import keras
import random
import cv2 as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from pathlib import Path

# Function for greyscaling and histogram equalization
def preprocess_image(image_path):
    img = mpimg.imread(image_path)
    img_gray = c.cvtColor(img, c.COLOR_RGB2GRAY)
    img_equalized = c.equalizeHist(img_gray)
    img_resized = c.resize(img_equalized, (224, 224))
    # Expand dimensions to make it compatible with the model input shape
    img_equalized = np.expand_dims(img_equalized, axis=-1)
    return img_equalized


# Applying data augmentation
traindata_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess data
def load_and_preprocess_data(data_path):
    image_data_path = list(data_path.glob(r"**/*.jpg"))
    label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
    final_data = pd.DataFrame({"image_data": image_data_path, "label": label_path}).astype("str")
    final_data = final_data.sample(frac=1).reset_index(drop=True)
    final_data["image_data"] = final_data["image_data"].apply(preprocess_image)
    return final_data

food_images = 'C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\food'
for i in range(25):
    file=random.choice(os.listdir(food_images))
    food_image_path=os.path.join(food_images,file)
    img_preprocessed=preprocess_image(food_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img_preprocessed.squeeze(), cmap='gray')
plt.show()



building_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\building"
for i in range(25):
    file = random.choice(os.listdir(building_images))
    building_image_path = os.path.join(building_images,file)
    img_preprocessed = preprocess_image(building_image_path)
    ax = plt.subplot(5, 5, i+1)
    plt.imshow (img_preprocessed.squeeze(), cmap='gray')
plt.show()


# Display preprocessed images for Landscape
landscape_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\landscape"
for i in range(25):
    file = random.choice(os.listdir(landscape_images))
    landscape_image_path = os.path.join(landscape_images, file)
    img_preprocessed = preprocess_image(landscape_image_path)
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(img_preprocessed.squeeze(), cmap='gray')
plt.show()


# Display preprocessed images for People
people_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\people"
for i in range(25):
    file = random.choice(os.listdir(people_images))
    people_image_path = os.path.join(people_images, file)
    img_preprocessed = preprocess_image(people_image_path)
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(img_preprocessed.squeeze(), cmap='gray')
plt.show()

def display_preprocessed_images(image_folder, category_name):
    plt.figure(figsize=(15, 15))
    for i in range(1, 26):
        file = random.choice(os.listdir(image_folder))
        image_path = os.path.join(image_folder, file)

        # Preprocess the image
        img_preprocessed = preprocess_image(image_path)

        # Display the preprocessed image
        ax = plt.subplot(5, 5, i)
        plt.imshow(img_preprocessed.squeeze(), cmap='gray')
        plt.title(f"{category_name} - {i}")
        plt.axis('off')
    plt.show()

#NOTE: Just change the path that I have written for the folders to the path of the folders on your computer so that the model would be able to read the data

# Display preprocessed images for Food
food_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\food"
display_preprocessed_images(food_images, "Food")

# Display preprocessed images for Landscape
landscape_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\landscape"
display_preprocessed_images(landscape_images, "Landscape")

# Display preprocessed images for Building
building_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\building"
display_preprocessed_images(building_images, "Building")

# Display preprocessed images for People
people_images = "C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training\\people"
display_preprocessed_images(people_images, "People")

train_data_path = Path(r"C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training")
valid_data_path = Path(r"C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\validation")
test_data_path = Path(r"C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\testing")

final_train_data = load_and_preprocess_data(train_data_path)
final_valid_data = load_and_preprocess_data(valid_data_path)
final_test_data = load_and_preprocess_data(test_data_path)




image_path="C:\\Users\\user\\OneDrive\\Desktop\\Advanced AI\\Dataset2\\training"


label_name= ['Food', 'Landscape', 'Building', 'People']

image_size=(224,224)

class_names = os.listdir(image_path)
print(class_names)
print("Number of classes : {}".format(len(class_names)))

numberof_images={}
for class_name in class_names:
    numberof_images[class_name]=len(os.listdir(image_path+"/"+class_name))
images_each_class=pd.DataFrame(numberof_images.values(),index=numberof_images.keys(),columns=["Number of images"])
print(images_each_class)



train_data_path=Path(r"C:\Users\user\OneDrive\Desktop\Advanced AI\Dataset2\training")
image_data_path=list(train_data_path.glob(r"**/*.jpg"))
train_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_train_data=pd.DataFrame({"image_data":image_data_path,"label":train_label_path}).astype("str")
final_train_data=final_train_data.sample(frac=1).reset_index(drop=True)
print(final_train_data['image_data'])


valid_data_path=Path(r"C:\Users\user\OneDrive\Desktop\Advanced AI\Dataset2\validation")
image_data_path=list(valid_data_path.glob(r"**/*.jpg"))
valid_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_valid_data=pd.DataFrame({"image_data":image_data_path,"label":valid_label_path}).astype("str")
final_valid_data=final_valid_data.sample(frac=1).reset_index(drop=True)
print(final_valid_data['image_data'])




test_data_path=Path(r"C:\Users\user\OneDrive\Desktop\Advanced AI\Dataset2\testing")
image_data_path=list(test_data_path.glob(r"**/*.jpg"))
test_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_test_data=pd.DataFrame({"image_data":image_data_path,"label":test_label_path}).astype("str")
final_test_data=final_test_data.sample(frac=1).reset_index(drop=True)
print(final_test_data['image_data'])



batch_size=30
traindata_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define data generators with data augmentation
traindata_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


validdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_generator = traindata_generator.flow_from_dataframe(
    dataframe=final_train_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="grayscale",  # Use "grayscale" for greyscaled images
    shuffle=True
)

valid_data_generator = validdata_generator.flow_from_dataframe(
    dataframe=final_valid_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="grayscale",  # Use "grayscale" for greyscaled images
    shuffle=True
)

test_data_generator = testdata_generator.flow_from_dataframe(
    dataframe=final_test_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="grayscale",  # Use "grayscale" for greyscaled images
    shuffle=False
)

class_dict = train_data_generator.class_indices
class_list = list(class_dict.keys())
print(class_list)

#using CNN becuase it improves the accuracy (it is good with image related tasks) however the epochs take longer to finish
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout layer for regularization
    keras.layers.Dense(4, activation='softmax')
])



model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#epochs is how many times the model will train. in this case, 20 times (it will take an hour or so but it is worth it for the accuracy), so he gets more accurate and loses less as the epochs go.
model_history = model.fit(train_data_generator, epochs=20, validation_data=valid_data_generator )


prediction= model.predict(test_data_generator)
prediction=np.argmax(prediction,axis=1)
map_label=dict((m,n) for n,m in (test_data_generator.class_indices).items())
final_predict=pd.Series(prediction).map(map_label).values
y_test=list(final_test_data.label)

plt.figure(figsize=(15, 15))
plt.style.use("classic")
number_images = (5, 5)
for i in range(1, (number_images[0] * number_images[1]) + 1):
    plt.subplot(number_images[0], number_images[1], i)
    plt.axis("off")

    color = "blue"
    if final_test_data.label.iloc[i] != final_predict[i]:
        color = "red"
    plt.title(f"True:{final_test_data.label.iloc[i]}\nPredicted:{final_predict[i]}", color=color)
    plt.imshow(plt.imread(final_test_data['image_data'].iloc[i]))

plt.show()
