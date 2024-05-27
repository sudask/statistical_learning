import numpy as np
import time
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from sklearn.svm import SVC
from ISLP import confusion_table

from ucimlrepo import fetch_ucirepo

start_time = time.time()

# data preparation
# -------------------------------
early_stage_diabetes_risk_prediction = fetch_ucirepo(id=529) 
X_raw = early_stage_diabetes_risk_prediction.data.features 
y_raw = early_stage_diabetes_risk_prediction.data.targets

X = X_raw.copy()
y = y_raw.copy()

replace_map = {"Yes":1, "No":0}
columns_to_replace = ['polyuria', 'polydipsia', 'sudden_weight_loss',
                      'weakness', 'polyphagia', 'genital_thrush',
                      'visual_blurring', 'itching', 'irritability',
                      'delayed_healing', 'partial_paresis', 'muscle_stiffness',
                      'alopecia', 'obesity']
X[columns_to_replace] = X[columns_to_replace].replace(replace_map)
X['gender'] = X['gender'].replace({'Male':1, 'Female':0})
y['class'] = y['class'].replace({'Positive':1, 'Negative':0})

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
# -----------------------------------------
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train.values.ravel())

svm_rbf = SVC(kernel='rbf', C=1, gamma=1)
svm_rbf.fit(X_train, y_train.values.ravel())

svm_poly = SVC(kernel='poly',C=2, gamma=1, degree=1)
svm_poly.fit(X_train, y_train.values.ravel())

# Searching optimal parameter
# -----------------------------------------
kfold = skm.KFold(5,
                  random_state=0,
                  shuffle=True)

grid_linear = skm.GridSearchCV(svm_linear,
                        {'C':[0.001,0.01,0.1,1,5,10,100]},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')

grid_linear.fit(X_train, y_train.values.ravel())

grid_rbf = skm.GridSearchCV(svm_rbf,
                        {'C':[0.001,0.01,0.1,1,5,10,100], 'gamma':[0.5,1,2,3,4]},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')

grid_rbf.fit(X_train, y_train.values.ravel())

grid_poly = skm.GridSearchCV(svm_poly,
                        {'C':[0.001,0.01,0.1,1,5,10,100], 'gamma':[0.5,1,2,3,4], 'degree':[2,3,4]},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')

grid_poly.fit(X_train, y_train.values.ravel())
# print(grid.best_params_)
# print(grid.cv_results_[('mean_test_score')])

# Output optimal parameters
# -----------------------------------------
print("最佳线性核SVM参数:", grid_linear.best_params_)
print("最佳RBF核SVM参数:", grid_rbf.best_params_)
print("最佳多项式核SVM参数:", grid_poly.best_params_)

# Comparation of best models
# ----------------------------------
svm_linear_best = grid_linear.best_estimator_
svm_rbf_best = grid_rbf.best_estimator_
svm_poly_best = grid_poly.best_estimator_

# 训练误差
train_error_linear = 1 - svm_linear_best.score(X_train, y_train)
train_error_rbf = 1 - svm_rbf_best.score(X_train, y_train)
train_error_poly = 1 - svm_poly_best.score(X_train, y_train)

# 测试误差
test_error_linear = 1 - svm_linear_best.score(X_test, y_test)
test_error_rbf = 1 - svm_rbf_best.score(X_test, y_test)
test_error_poly = 1 - svm_poly_best.score(X_test, y_test)

print("线性核SVM训练误差:", train_error_linear)
print("RBF核SVM训练误差:", train_error_rbf)
print("多项式核SVM训练误差:", train_error_poly)

print("线性核SVM测试误差:", test_error_linear)
print("RBF核SVM测试误差:", test_error_rbf)
print("多项式核SVM测试误差:", test_error_poly)

end_time = time.time()
print("用时：", end_time - start_time, "秒")
