import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table

from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay

roc_curve = RocCurveDisplay.from_estimator

rng = np.random.default_rng(1)
# X = rng.standard_normal((50, 2))
# y = np.array([-1]*25 + [1]*25)
# X[y==1] += 1

# svm_linear = SVC(C=10, kernel="linear")
# svm_linear.fit(X, y)

# svm_linear_small = SVC(C=0.1, kernel='linear')
# svm_linear_small.fit(X, y)

# fig, ax = subplots(figsize=(8,8))
# plot_svm(X,
#          y,
#          svm_linear,
#          ax=ax)

# plt.show()

# kfold = skm.KFold(5,
#                   random_state=0,
#                   shuffle=True)

# grid = skm.GridSearchCV(svm_linear,
#                         {'C':[0.001,0.01,0.1,1,5,10,100]},
#                         refit=True,
#                         cv=kfold,
#                         scoring='accuracy')

# grid.fit(X, y)
# print(grid.best_params_)
# print(grid.cv_results_[('mean_test_score')])

# X_test = rng.standard_normal((20, 2))
# y_test = np.array([-1]*10 + [1]*10)
# X_test[y_test==1] += 1

# best_ = grid.best_estimator_
# y_test_hat = best_.predict(X_test)
# print(confusion_table(y_test_hat, y_test))

# svm_ = SVC(C=0.001, kernel='linear').fit(X,y)
# y_test_hat = svm_.predict(X_test)
# print(confusion_table(y_test_hat, y_test))

# X[y==1] += 1.9
# fig, ax = subplots(figsize=(8,8))
# ax.scatter(X[:,0],
#            X[:,1],
#            c=y,
#            cmap=cm.coolwarm)

# plt.show()

# svm_ = SVC(C=0.1, kernel='linear').fit(X,y)
# y_hat = svm_.predict(X)
# print(confusion_table(y_hat, y))

# fig, ax = subplots(figsize=(8,8))
# plot_svm(X, y, svm_, ax=ax)
# plt.show()

# Support Vector Machine
# -----------------------------------
X = rng.standard_normal((200, 2))
X[:100] += 2
X[100:150] -= 2
y = np.array([1]*150 + [2]*50)

# fig, ax = subplots(figsize=(8,8))
# ax.scatter(X[:,0],
#            X[:,1],
#            c=y,
#            cmap=cm.coolwarm)

# plt.show()

# (X_train,
#  X_test,
#  y_train,
#  y_test) = skm.train_test_split(X, y, test_size=0.5, random_state=0)

# svm_rbf = SVC(kernel='rbf', gamma=1, C=1)
# svm_rbf.fit(X_train, y_train)

# fig, ax = subplots(figsize=(8,8))
# plot_svm(X_train, y_train, svm_rbf, ax=ax)
# plt.show()

# kfold = skm.KFold(5, random_state=0, shuffle=True)
# grid = skm.GridSearchCV(svm_rbf, {'C':[0.1,1,10,100,1000], 'gamma':[0.5,1,2,3,4]},
#                         refit=True, cv=kfold, scoring='accuracy')
# grid.fit(X_train, y_train)
# print(grid.best_params_)

# best_svm = grid.best_estimator_
# fig, ax = subplots(figsize=(8,8))
# plot_svm(X_train,
#          y_train,
#          best_svm,
#          ax=ax)

# y_hat_test = best_svm.predict(X_test)
# confusion_table(y_hat_test, y_test)

# ROC Curves
# -------------------------------------------------------

# plt.show()
# svm_flex = SVC(kernel='rbf', gamma=50, C=1).fit(X_train, y_train)
# fig, ax = subplots(figsize=(8,8))
# roc_curve(svm_flex, X_test, y_test, name='Training $\gamma=50$')
# plt.show()

# fig, ax = subplots(figsize=(8,8))
# for (X_, y_, c, name) in zip(
#     (X_train, X_test),
#     (y_train, y_test),
#     ('r', 'b'),
#     ('CV tuned on training',
#     'CV tuned on test')):
#     roc_curve(best_svm,
#     X_,
#     y_,
#     name=name,
#     ax=ax,
#     color=c)

# plt.show()

# SVM with Multiple Classes
# ---------------------------------------
rng = np.random.default_rng(123)
X = np.vstack([X, rng.standard_normal((50, 2))])
y = np.hstack([y, [0]*50])
X[y==0,1] += 2
fig, ax = subplots(figsize=(8,8))
ax.scatter(X[:,0], X[:,1], c=y, cmap=cm.coolwarm)
plt.show()