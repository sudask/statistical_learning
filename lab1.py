import numpy as np

# x = np.random.normal(size=10)
# y = x + np.random.normal(loc=50, scale=1, size=10)
# print(np.corrcoef(x, y))

# 可复现的随机数产生方式
rng = np.random.default_rng(42)
# print(rng.normal(size=10))
# x = rng.standard_normal((10, 3))
# print(np.mean(x, axis=0))
# print(x.mean(0))

import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(8, 8))
# x = rng.standard_normal(100)
# y = rng.standard_normal(100)
# # ax.plot(x, y, 'o')
# ax.scatter(x, y, marker='o')
# ax.set_xlabel("this is the x-axis")
# ax.set_ylabel("this is the y-axis")
# ax.set_title("Plot of X vs Y")
# plt.show()
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 5))
# axes[0, 1].plot(x, y, 'o')
# axes[1, 2].scatter(x, y, marker='+')
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 8))
# x = np.linspace(-np.pi, np.pi, 50)
# y = x
# f = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
# # ax.contour(x, y, f, levels=45)
# ax.imshow(f)
# plt.show()

# x = np.array([[1], [2]])
# print(x.shape[0])
# print(x)

import pandas as pd

# Auto = pd.read_csv("./data/Auto.csv")
# Auto = pd.read_csv('./data/Auto.data', delim_whitespace=True)
# print(np.unique(Auto['horsepower']))
Auto = pd.read_csv('./data/Auto.data',
na_values=['?'],
delim_whitespace=True)
# print(Auto['horsepower'].sum())
Auto_new = Auto.dropna()
print(Auto_new.shape)