from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TASK 1
pd.set_option('display.precision', 2)
nyc = pd.read_csv('ave_hi_nyc_jan_1895-2022.csv')
nyc.columns = ["Date", "Temperature", "Anomally"]
nyc.Date = nyc.Date.floordiv(100)

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), 
    nyc.Temperature.values, 
    random_state=11
    )

print(X_train.shape)
print(X_test.shape)

linear_regression = LinearRegression()

linear_regression.fit(X=X_train, y=y_train)
print(linear_regression.coef_, linear_regression.intercept_)

predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")
    
predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)

print(predict(2023))
print(predict(1890))

axes = sns.scatterplot(data=nyc, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)
axes.set_ylim(10, 70)

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

y = predict(x)

line = plt.plot(x, y)
plt.show()

print("====================================")

# TASK 2

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    markers = ('s', 'x', 'c', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0],
                    y=X[y == c1, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label=c1,
                    edgecolor='black')
        
    if test_idx:
        
        X_test, y_test = X[test_idx, :], y[test_idx]
        
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidths=1,
                    marker='0',
                    s=100,
                    label='Test set')
        
        
svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# TASK 3

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11, test_size=0.20)
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test
print("K Neighbours Classification")
print(predicted[:20])
print(expected[:20])

svc_model = SVC()
svc_model.fit(X=X_train, y=y_train)
svc_predicted = svc_model.predict(X_test)
print("Support Vector Machines")
print(svc_predicted[:20])
print(expected[:20])

naive_bayes = GaussianNB()
naive_bayes.fit(X=X_train, y=y_train)
naive_bayes_predicted = naive_bayes.predict(X_test)
print("Gaussian Naive Bayes")
print(naive_bayes_predicted[:20])
print(expected[:20])