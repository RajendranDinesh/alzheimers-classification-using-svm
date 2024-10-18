# Alzheimer’s Disease Diagnosis using SVM Classifier

## Overview
This project develops a Support Vector Machine (SVM) classifier to predict Alzheimer’s Disease (AD) from high-dimensional glucose metabolism data across various regions of the cerebral cortex. Utilizing linear, polynomial, and radial basis function (RBF) kernels, the project aims to classify individuals into stable Normal Controls (sNC) and stable Dementia of Alzheimer’s Type (sDAT) groups.

## Technical Explanation

### Data Preparation
Data comprises glucose metabolism measurements from 14 cortical brain regions for sNC and sDAT individuals. The project includes training, testing, and independent testing datasets for a comprehensive evaluation of the classifier's performance.

### SVM Classification
The SVM classifier is employed with different kernel functions:
- **Linear SVM**: A baseline model using linear kernel to distinguish between sNC and sDAT.
- **Polynomial SVM**: An advanced model that utilizes a polynomial kernel with degree tuning to capture non-linear relationships.
- **RBF SVM**: Utilizes the radial basis function kernel, adjusting the gamma parameter to handle even more complex non-linear patterns.

### Hyperparameter Tuning
Hyperparameters, including the regularization parameter `C`, the polynomial degree `d`, and the RBF kernel gamma `γ`, are optimized through grid search with cross-validation. This approach ensures the selection of optimal parameters for each kernel type, aiming to maximize classification performance.

### Model Evaluation
Performance is assessed using accuracy, sensitivity, specificity, precision, recall, and balanced accuracy metrics. These evaluations guide the selection of the most effective kernel and parameter settings for AD diagnosis.

## Why SVM is Preferred Over KNN
- **Accuracy**: SVM tends to outperform KNN in high-dimensional spaces, making it more suitable for the complex patterns in AD diagnosis.
- **Scalability**: SVM is more scalable and computationally efficient, especially important for large datasets.
- **Interpretability**: Despite SVM's complex decision boundaries, the support vectors provide insights into the classification decision.
- **Parameter Tuning**: While both models require hyperparameter tuning, SVM's ability to handle overfitting through regularization makes it robust.
- **Model Complexity**: SVM's kernel trick allows for modeling complex non-linear relationships without increasing the computational burden.
- **Computational Resources**: SVM's prediction phase is more efficient than KNN, which must compute distances to all training instances.

## Conclusion
The project leverages SVM's versatility with different kernels to address the high-dimensional and complex nature of predicting Alzheimer’s Disease from cerebral cortex glucose metabolism data. Through meticulous model development and evaluation, it illustrates the potential of machine learning in medical diagnostics.
