import numpy as np
from sklearn.datasets import make_regression, make_friedman1, make_moons, make_blobs
from sklearn.model_selection import train_test_split

# MACRO

def createDatasetRegression(ndatasets, linear, nSamples, nFeatures, noise,  n_informative = 0):
    DataMatrix = [ []*2 for i in range(ndatasets)] 
    

    if linear == True:
        for i in range(ndatasets):
            X, y = make_regression(n_samples=nSamples, n_features=nFeatures, noise=noise, n_informative=n_informative)
            DataMatrix[i].extend((X, y))
    else:
        for i in range(ndatasets):
            X, y = make_friedman1(n_samples=nSamples, n_features=nFeatures, noise=noise)
            DataMatrix[i].extend((X, y))
    
    return DataMatrix

def createDatasetClassification_make_moons(ndatasets, nSamples, shuffle, noise, random_state):
    DataMatrix = [ []*2 for i in range(ndatasets)] 
    
    for i in range(ndatasets):
        X, y = make_moons(n_samples=nSamples, shuffle = shuffle, noise = noise, random_state = random_state)    
        DataMatrix[i].extend((X, y))
        
    return DataMatrix

def createDatasetClassification_make_blobs(ndatasets, nSamples, n_features, centers, cluster_std, center_box, shuffle, random_state, return_centers):
    DataMatrix = [ []*2 for i in range(ndatasets)] 
    
    
    
    for i in range(ndatasets):
        X, y = make_blobs(n_samples=nSamples, n_features = n_features, centers = centers,
                          cluster_std = cluster_std, center_box = center_box, shuffle = shuffle,
                          random_state = random_state, return_centers = return_centers)    
        DataMatrix[i].extend((X, y))
        
    return DataMatrix

def divide_Datasets_TrainTest(ndatasets, DataMatrix):
    matrixDatasetTraintest =  [ []*4 for i in range(ndatasets)]
    
    for i in range(ndatasets):
        X_train, X_test, y_train, y_test = train_test_split(DataMatrix[i][0], DataMatrix[i][1], random_state=4, test_size=0.6561, shuffle=True)
        matrixDatasetTraintest[i].extend((X_train, X_test, y_train, y_test))
        
    return matrixDatasetTraintest

def getListTrainSamples(nsamples, start, stop, base):
    listTrainSamples = np.logspace(start, stop, num=nsamples, base=base)
    listTrainSamples = [round(item, 0) for item in listTrainSamples]
    
    return listTrainSamples

def dividebySamples(matrixDatasetTraintest,listTrainSamples, nDatasets, nSamples):
    matrixXYtrainparts =  [[[]*2 for j in range(nSamples)] for i in range(nDatasets)]
    
    for i in range(nDatasets):
        for idx, el in enumerate(listTrainSamples):
            XtrainDivided = matrixDatasetTraintest[i][0][0:int(el)]
            
            YtrainDivided = matrixDatasetTraintest[i][2][0:int(el)]
            matrixXYtrainparts[i][idx].extend((XtrainDivided, YtrainDivided))
            if(i == 1):
                print(matrixXYtrainparts[1])

    return matrixXYtrainparts