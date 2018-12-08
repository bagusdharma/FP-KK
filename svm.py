import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import normalize
DATA_PATH = r"breastcancer.data"
dataset = pd.read_csv(DATA_PATH, header=None)

def dimReduce(x, mode="PCA"):
    if mode.upper()=="PCA":
        reductor = PCA(n_components = int(0.5*x.shape[1]), svd_solver= 'full', whiten=True)
    elif mode.upper() == "FA":
        reductor = FactorAnalysis(svd_method='lapack')
    else:
        reductor = PCA(n_components = int(0.5*x.shape[1]), svd_solver= 'arpack', whiten=False)
    reductor.fit(x)
    return reductor, reductor.transform(x)

def split(data, n_each_class=(len(dataset)/6)):
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

if __name__ == "__main__":
    df = dataset
    label = df.loc[:, 10].values.reshape(-1,1)
    
    reductor, reduced_data = dimReduce(df.loc[:,:9].values)
    new_data = np.append(reduced_data, label, axis=1)
    np.random.shuffle(new_data)
    label = new_data[:,-1]

    train, test = split(new_data)
    train_label = train[:,-1]
    test_label = test[:,-1]
    
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(train[:,:-1], train_label)
    y = clf.predict(test[:,:-1])
    
    acc = 0
    for i in range(len(y)):
        if y[i] == test_label[i]:
            acc+=1
    print("Training set = " + repr(len(train_label)))
    print("Test set = " + repr(len(test_label)))
    print("Accuracy : %f%%" %(acc/len(y)*100.0))
