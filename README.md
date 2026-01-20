# HUCS-BBM409: Machine Learning Lab

This repository contains the laboratory assignments and solutions for the **BBM409: Introduction to Machine Learning Lab** course at **Hacettepe University Computer Science Department** for the **2025-2026 Fall Term**.

**Instructor:** Assoc. Prof. Dr. Hacer Yalım Keleş

## 📂 Repository Structure

The coursework focuses on implementing fundamental machine learning algorithms from scratch, as well as utilizing modern deep learning frameworks like PyTorch for complex tasks.

### [Assignment 1: Understanding Perceptron](BBM409_Assignment1_Fall25.pdf)
**Topic:** Binary Classification & Linear Separability  
**Dataset:** UCI Raisin Dataset (Kecimen vs. Besni)

* **Objective:** Implement the Perceptron Learning Algorithm from scratch to find a linear separator between classes.
* **Key Tasks:**
    * **Implementation:** Build the Perceptron algorithm (forward pass and weight updates) using only NumPy.
    * **Evaluation:** Calculate Accuracy, Precision, Recall, and F1-Score manually without libraries.
    * **Visualization:** Perform correlation analysis, select features, and plot the 2D decision boundary.
    * **Fisher's Linear Discriminant (FLD):** Implement Fisher's LD to project data onto a 1D space and visualize class separation.

### [Assignment 2: Classification with SVM and Ensemble Methods](BBM409_Assignment2_Fall25.pdf)
**Topic:** Support Vector Machines, Logistic Regression, & Ensemble Learning  
**Datasets:** UCI Sonar Dataset (Binary) & Dry Bean Dataset (Multi-class)

* **Objective:** Explore binary and multi-class classification using a mix of from-scratch implementations and library-based ensemble methods.
* **Key Tasks:**
    * **Binary Classification (SVM):** Implement SVM using `sklearn` with Linear, RBF, and Polynomial kernels. Perform Hyperparameter tuning using `GridSearchCV`.
    * **Multi-Class Classification:**
        1.  **Multinomial Logistic Regression:** Implement from scratch (Softmax, Cross-Entropy Loss, Gradient Descent).
        2.  **Decision Tree:** Use `sklearn` with Gini/Entropy criteria.
        3.  **XGBoost:** Implement gradient boosting classification.
    * **Analysis:** Comparative analysis of model complexity, overfitting risks, and performance.

### [Assignment 3: CNNs & Transfer Learning](BBM409_Assignment3_Fall25.pdf)
**Topic:** Convolutional Neural Networks (CNN) & Transfer Learning  
**Dataset:** Vegetable Image Dataset (15 classes)

* **Objective:** cultivate comprehension of CNN concepts and transfer learning techniques for image classification.
* **Key Tasks:**
    * **CNN From Scratch:** Design and train a 5-layer Convolutional Neural Network using PyTorch (building blocks: Conv2d, ReLU, MaxPool, Linear).
    * **Transfer Learning:** Fine-tune pre-trained models (**ResNet-18** and **MobileNetV2**) on the vegetable dataset.
        * *Strategies:* Fine-tuning only the FC layer vs. partial unfreezing vs. full unfreezing.
    * **Competition:** Evaluate the best model on a hidden test set via Kaggle.

### [Assignment 4: LSTM for Sentiment Analysis](BBM409_Assignment4_Fall25.pdf)
**Topic:** Recurrent Neural Networks (RNN) & Long Short-Term Memory (LSTM)  
**Dataset:** IMDB Movie Reviews (Sentiment Analysis)

* **Objective:** Understand sequence-based tasks by implementing an LSTM architecture from scratch for text data.
* **Key Tasks:**
    * **Embeddings:** Use **Word2Vec** (via Gensim) for word representations and visualize them using PCA (2D/3D).
    * **LSTM Implementation:** Manually implement the **LSTM Cell** (Forget, Input, Output gates) and **LSTM Layer** using PyTorch tensor operations.
    * **Training:** Build a full sentiment analysis pipeline (Embedding Layer $\to$ Custom LSTM $\to$ Dense Output).
    * **Analysis:** Discuss the Vanishing Gradient problem and how LSTMs solve it.
    * **Competition:** Evaluate the best model on a hidden test set via Kaggle.

## 🛠️ Technologies & Tools

* **Language:** Python 3.x
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deep Learning:** PyTorch, Torchvision
* **NLP:** Gensim, NLTK/Torchtext
