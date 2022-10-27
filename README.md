# ML-Supervised-Learning
Exploring the important supervised learning algorithms in Machine learning

Data Analysis for Various Classifiers

This analysis covers all necessary steps from preprocessing to evaluation of the results of various classification models. Finally, evaluation measures, experimental setup and parameter selection has been discussed and the analysis decisions are justified.

Dataset used:

Dataset name: Adult dataset
Task: Classification
Prediction: To determine whether a person makes over 50K a year.
Labels: >50K,<=50K
Dataset characteristics: Multivariate
Attribute characteristics: Categorical, Integer
Number of Instances(Total): 48842
Number of Instances(Train):32561
Number of Instances(Test):16281
Number of Attributes:14

Platform : Google colab
Package: Usage for pip install

pip install -U scikit-learn

or conda:

conda install scikit-learn

Prediction task is to determine whether a person makes over 50K a year.

Preprocessing of data:

Data integration
Data cleaning
Data reduction

Feature Selection:
After applying the oversampling method, the new dimension of our dataset is (65719,16) in total (49438: train, 16281: test)

Classification :
Remove categorical and include only numerical values

X_train_numerical = X_train.select_dtypes(include = np.number).copy()
X_test_numerical = X_test.select_dtypes(include = np.number).copy()

fit an estimator(classifier) to be able to predict the classes to which unseen samples belong 

classifier.fit(X, y) 
classifier.predict(T)

Plot the evaluation results using matplotlib.


Results of various classification models:

   Classifier                Accuracy   Precision  Recall   F1-Score  AUC Score
| -- ----------------        --------   ---------  -------  --------  ---------
| 1  KNN                      0.73       0.82        0.82      0.82      0.63
| 2  Decision Tree            0.77       0.84        0.84      0.85      0.68
| 3  Naive Bayes              0.79       0.81        0.95      0.87      0.78
| 4  SVM                      0.79       0.79        1.00      0.88      0.63
| 5  Perceptron               0.79       0.80        0.97      0.88      0.63

Results of various classification models after hyperparameter tuning:

Classifier                Accuracy   Precision  Recall   F1-Score  AUC Score
| -- ----------------     --------   ---------  -------  --------  ---------
| 1  KNN                   0.80       0.80        0.98      0.88      0.66
| 2  Decision Tree         0.83       0.83        0.98      0.90      0.82
| 3  Naive Bayes           0.79       0.81        0.95      0.87      0.78
| 4  SVM                   0.79       0.79        1.00      0.88      0.63
| 5  Perceptron            0.79       0.80        0.97      0.88      0.63
