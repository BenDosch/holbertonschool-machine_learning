# Data_Augmentation

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Flip](#0-flip)
	2. [Crop](#1-crop)
	3. [Rotate](#2-rotate)
	4. [Shear](#3-shear)
	5. [Brightness](#4-brightness)
	6. [Hue](#5-hue)
4. [Author](#author)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is data augmentation?
* When should you perform data augmentation?
* What are the benefits of using data augmentation?
* What are the various ways to perform data augmentation?
* How can you use ML to automate data augmentation?

## Refrences

* [Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/ "Data Augmentation | How to use Deep Learning when you have Limited Data — Part 2")
* [Automating Data Augmentation: Practice, Theory and New Direction](http://ai.stanford.edu/blog/data-augmentation/ "Automating Data Augmentation: Practice, Theory and New Direction")
* [tf.image](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image "tf.image")
* [tf.keras.preprocessing.image](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/preprocessing/image "tf.keras.preprocessing.image")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Flip](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x04-data_augmentation/0-flip.py "0. Flip")

Write a function def flip_image(image): that flips an image horizontally.

* image is a 3D tf.Tensor containing the image to flip
* Returns the flipped image

---

### [1. Crop](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x04-data_augmentation/1-crop.py "1. Crop")

Write a function def crop_image(image, size): that performs a random crop of an image.

* image is a 3D tf.Tensor containing the image to crop
* size is a tuple containing the size of the crop
* Returns the cropped image

---
### [2. Rotate](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x04-data_augmentation/2-rotate.py "2. Rotate")

Write a function def rotate_image(image): that rotates an image by 90 degrees counter-clockwise.

* image is a 3D tf.Tensor containing the image to rotate
* Returns the rotated image

---

### [3. Shear](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x04-data_augmentation/3-shear.py "3. Shear")

Write a function def shear_image(image, intensity): that randomly shears an image.

* image is a 3D tf.Tensor containing the image to shear
* intensity is the intensity with which the image should be sheared
* Returns the sheared image

---

### [4. Brightness](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x04-data_augmentation/4-brightness.py "4. Brightness")

Write a function def change_brightness(image, max_delta): that randomly changes the brightness of an image.

* image is a 3D tf.Tensor containing the image to change
* max_delta is the maximum amount the image should be brightened (or darkened)
* Returns the altered image

---

### [5. Hue](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/pipeline/0x04-data_augmentation/5-hue.py "5. Hue")

Write a function def change_hue(image, delta): that changes the hue of an image

* image is a 3D tf.Tensor containing the image to change
* delta is the amount the hue should change
* Returns the altered image

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
