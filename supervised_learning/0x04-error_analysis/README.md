# Error Analysis

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
    1. [Create Confusion](#0-create-confusion)
    2. [Sensitivity](#1-sensitivity)
    3. [Precision](#2-precision)
    4. [Specificity](#3-specificity)
    5. [F1 score](#4-f1_score)
    6. [Dealing with Error](#5-dealing-with-error)
    7. [Compare and Contrast](#6-compare_and_contrast)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is the confusion matrix?
* What is type I error? type II?
* What is sensitivity? specificity? precision? recall?
* What is an F1 score?
* What is bias? variance?
* What is irreducible error?
* What is Bayes error?
* How can you approximate Bayes error?
* How to calculate bias and variance
* How to create a confusion matrix

## References
* [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of the Deep Learning Specialization)")
* [Confusion matrix equations](https://newbedev.com/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-false-negative "Confusion matrix equations")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Create Confusion](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/0-create_confusion.py "0. Create Confusion")

Write the function that creates a confusion matrix.

---
### [1. Sensitivity](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/1-sensitivity.py "0. Task Name")

Write the function that calculates the sensitivity for each class in a confusion matrix.

---
### [2. Precision](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/2-precision.py "2. Precision")

Write the function that calculates the precision for each class in a confusion matrix.

---
### [3. Specificity](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/3-specificity.py "3. Specificity")

Write the function that calculates the specificity for each class in a confusion matrix.

---
### [4. F1 score](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/4-f1_score.py "4. F1 score")

Write the function that calculates the F1 score of a confusion matrix.

---
### [5. Dealing with Error](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/5-error_handling "5. Dealing with Error")

In the text file 5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C):

Scenarios:

1) High Bias, High Variance
2) High Bias, Low Variance
3) Low Bias, High Variance
4) Low Bias, Low Variance

Approaches:

A) Train more
B) Try a different architecture
C) Get more data
D) Build a deeper network
E) Use regularization
F) Nothing

---
### [6. Compare and Contrast](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/6-compare_and_contrast "6. Compare and Contrast")

Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file 6-compare_and_contrast

![Training](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/03c511c109a790a30bbe.png)

![Validation](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x04-error_analysis/8f5d5fdab6420a22471b.png)

Most important issue:

A. High Bias
B. High Variance
C. Nothing

---
