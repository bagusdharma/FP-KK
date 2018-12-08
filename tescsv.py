import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('breastcancer.csv')

print(dataset)

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# names = ['Sample code number','Clump Thickness',' Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',' Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

# dataset = pd.read_csv(url,names=names)

# print (dataset) -> bisa diprint

# X = dataset.iloc[:,:-1].values
# y = dataset.iloc[:,10].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print(y_pred)