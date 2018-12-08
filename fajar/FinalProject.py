import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import normalize

#we use PCA to reduce the data, so we set the mode as static "PCA"
def dimReduce(x, mode="PCA"):
    if mode.upper()=="PCA":
        #Because the data's attribute is 400 (excluding the class column), we then defined the PCA's n_components to 200 attributes
        reductor = PCA(n_components = int(0.5*x.shape[1]), svd_solver= 'full', whiten=True)
    elif mode.upper() == "FA":
        reductor = FactorAnalysis(svd_method='lapack')
    else:
        reductor = PCA(n_components = int(0.5*x.shape[1]), svd_solver= 'arpack', whiten=False)

    #Fits the model with the data
    reductor.fit(x)
    return reductor, reductor.transform(x)

def split(data, n_each_class=5):
    #We first define the label_map'list as 25 arrays of 0 (25 classes)
    label_map = [0]*25

    #Empty lists
    train = []
    test = []
    
    for i in range(len(data)):
        if label_map[int(data[i,-1])-1] < n_each_class:
            test.append(data[i,:])
            label_map[int(data[i,-1])-1]+=1
        else:
            train.append(data[i,:])

    #return as NumPY's array
    return np.array(train), np.array(test)

if __name__ == "__main__":
    dataset = input("Masukkan nama dari dataset : ")
    #Load the data as Pandas' DataFrame
    df = pd.read_csv(dataset, header=None)
    #Getting the class' label into an Array
    label = df.loc[:, 400].values.reshape(-1,1)
    
    #Reducing data's attribute by passing the data into function dimReduce
    reductor, reduced_data = dimReduce(df.loc[:,:399].values)
    #Convert the reduced data (200 attributes) into NumPy format (append the new datas with their classes)
    new_data = np.append(reduced_data, label, axis=1)
    #Shuffle the data
    np.random.shuffle(new_data)
    #Getting the shuffled classes
    label = new_data[:,-1]

    #Splitting the data into train and test (Here we use 25 train, and 5 test for each class)
    train, test = split(new_data)
    #Getting the train's classes
    train_label = train[:,-1]
    #Getting the test's classes
    test_label = test[:,-1]
    
    #Using SVM with Gaussian RBF kernel
    clf = SVC(kernel='rbf', gamma=0.0001)
    #Fitting Train with their classes into SVM
    clf.fit(train[:,:-1], train_label)
    #Predicting the Test using SVM-RBF Kernel
    y = clf.predict(test[:,:-1])
    #For checking purpose only
    y_train = clf.predict(train[:,:-1])
    
    #Storing the "right" prediction from Train data
    acc_train=0
    for i in range(len(y_train)):
        if y_train[i] == train_label[i]:
            acc_train+=1
    
    #Storing the "right" prediction from Test data
    acc = 0
    for i in range(len(y)):
        if y[i] == test_label[i]:
            acc+=1
	
    #Printing the accuracy
    print("Accuracy on train : %f%%" %(acc_train/len(y_train)*100.0))
    print("Accuracy on test: %f%%" %(acc/len(y)*100.0))
    
    