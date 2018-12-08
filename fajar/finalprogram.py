'''
An assignment of Computational Intelligence subject for the final project.
Using Python-3.6

Assignment by:
Adi Darmawan - 5115100079
Fajar Maulana Firdaus - 5115100171
Yuga Mitra Hakiki - 5115100706

More references on:
- Pandas : https://pandas.pydata.org/pandas-docs/stable/
- NumPy : https://docs.scipy.org/doc/numpy/
- Support Vector Classification : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- Principal Component Analysis (PCA) : http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- Factor Analysis (FA) : http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
'''

import pandas as pd
from numpy import genfromtxt
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FactorAnalysis

#DATA_PATH = r"train.data"

def dimReduce(x, mode="PCA"):
    if mode.upper()=="PCA":
        reductor = PCA(n_components = int(0.5*x.shape[1]), svd_solver= 'full', whiten=True)
    elif mode.upper() == "FA":
        reductor = FactorAnalysis(svd_method='lapack')
    else:
        reductor = PCA(n_components = int(0.5*x.shape[1]), svd_solver= 'arpack', whiten=False)
    reductor.fit(x)
    return reductor, reductor.transform(x)

def split(data, n_each_class=5):
    label_map = [0]*25
    train = []
    test = []
    for i in range(len(data)):
        if label_map[int(data[i,-1])-1] < n_each_class:
            test.append(data[i,:])
            label_map[int(data[i,-1])-1]+=1
        else:
            train.append(data[i,:])
    return np.array(train), np.array(test)

def main():
    datatrain = input("Please enter the name of the data to train (with its file extension) : ")
    datatest = input("Please enter the name of the data to tes (with its file extension) : ")
    df = genfromtxt(datatrain, delimiter=',')
    tdf = genfromtxt(datatest, delimiter=',')
    # df = pd.read_csv(datatrain, header=None)
    # tdf = pd.read_csv(datatest, header=None)
    # label = df.loc[:, 400].values.reshape(-1,1)
    # tlabel = tdf.loc[:, 400].values.reshape(-1,1)

    # reductor, reduced_data = dimReduce(df.loc[:,:399].values)
    # treductor, treduced_data = dimReduce(tdf.loc[:,:399].values)
    # new_data = np.append(reduced_data, label, axis=1)
    # tnew_data = np.append(treduced_data, tlabel, axis=1)
    # df = np.array(np.rec.fromrecords(df.values))
    # np_names = df.dtypes.index.tolist()
    # df.dtype.names = tuple([name.encode('UTF8') for name in np_names])
    #
    # tdf = np.array(np.rec.fromrecords(tdf.values))
    # tnp_names = tdf.dtypes.index.tolist()
    # tdf.dtype.names = tuple([name.encode('UTF8') for name in tnp_names])

    new_data = np.random.shuffle(df)
    tnew_data = np.random.shuffle(tdf)

#    train, test = split(new_data)
    #train_label = train[:,-1]
    #test_label = test[:,-1]
    # train = np.array(new_data)
    # test = np.array(tnew_data)
    train_label = new_data[:,-1]
    test_label = tnew_data[:,-1]

    clf = SVC(kernel='rbf', gamma=0.0001)
    clf.fit(new_data[:,:-1], train_label)
    y = clf.predict(tnew_data[:,:-1])
    y_train = clf.predict(new_data[:,:-1])
    
    acc_train=0
    for i in range(len(y_train)):
        if y_train[i] == train_label[i]:
            acc_train+=1
    
    acc = 0
    for i in range(len(y)):
        if y[i] == test_label[i]:
            acc+=1
    print("Accuracy on train : %f%%" %(acc_train/len(y_train)*100.0))
    print("Accuracy on test: %f%%" %(acc/len(y)*100.0))

main()