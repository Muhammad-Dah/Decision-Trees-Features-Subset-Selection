import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import log2 as log
from collections import Counter
import random as rd
import math
def MajorityClass(E):
    arr = E[:,0]
    # cnt_b = np.sum(arr == 'B')
    # cnt_m = np.sum(arr == 'M')
    cnt=Counter(arr)
    cnt_b = cnt['B']
    cnt_m = cnt['M']

    return 'B' if cnt_b >= cnt_m else 'M'


def Entropy(E):
    sigma = 0
    arr = E[:,0]
    n = len(arr)
    if n==0:
       return sigma
    cnt = Counter(arr)
    cnt_b = cnt['B']
    cnt_m = cnt['M']
    if cnt_b>0:
        p_b = cnt_b / n
        sigma += p_b * log(p_b)
    if cnt_m>0:
        p_m = cnt_m / n
        sigma += p_m * log(p_m)

    return -sigma


def IG(E, feature):
    parentEntropy = Entropy(E)
    sortedByFeature = E[:,feature]
    sortedByFeature=np.sort(sortedByFeature)
    threshHoldVals = set()
    currIG = 0
    maxIG = 0
    bestThreshHold = 0
    numOfSamples = len(sortedByFeature)
    for i in range(numOfSamples - 1):
        threshHoldVals.add((sortedByFeature[i] + sortedByFeature[i + 1]) / 2)

    for threshHold in threshHoldVals:
        leftSubSet = E[E[:,feature] <= threshHold]
        rightSubSet = E[E[:,feature] > threshHold]
        leftVal = ((len(leftSubSet) / numOfSamples) * Entropy(leftSubSet))
        rightVal = ((len(rightSubSet) /numOfSamples) * Entropy(rightSubSet))
        currIG = parentEntropy - (leftVal + rightVal)
        if currIG > maxIG or  maxIG==0:
            maxIG = currIG
            bestLeftSplit = leftSubSet
            bestRightSplit = rightSubSet
            bestThreshHold = threshHold
    return maxIG, bestLeftSplit, bestRightSplit, bestThreshHold


def MaxIG(E, F):
    maxIG = 0
    currIG = 0
    for f in range(F):
        currIG, currLeft, curRight, currThreshHold = IG(E, f+1)
        if currIG >= maxIG:
            maxIG = currIG
            bestFeature = f+1
            leftSplit = currLeft
            rightSplit = curRight
            threshHold = currThreshHold

    return bestFeature, leftSplit, rightSplit, threshHold


class Node:
    def __init__(self, leftSon, rightSon, label, feature='', threshHold=0):
        self.leftSon = leftSon
        self.rightSon = rightSon
        self.feature = feature
        self.threshHold = threshHold
        self.label = label

featurevector = [0]*100
totalsamples = 300

def TDIDT(E, F, Default, M=0):
    numOfSamples = len(E)
    node = Node(None, None, Default)
    if M > 0:
        if numOfSamples < M:
            return node
    if numOfSamples==0:
        return node
    c = MajorityClass(E)
    if c == 'M':
        minority = 'B'
    else:
        minority = 'M'
    if minority not in E[:,0]:
        node.label = c
        return node

    feature, left, right, threshHold = MaxIG(E, F)
    featurevector[feature]+= numOfSamples/totalsamples
    node.threshHold = threshHold
    node.feature = feature
    node.label = c
    node.leftSon = TDIDT(left, F, c,M)
    node.rightSon = TDIDT(right, F, c,M)
    return node


def DTClassify(test, tree: Node):
    if tree.leftSon is None:
        return tree.label
    if (test[tree.feature] >= tree.threshHold):
        return DTClassify(test, tree.rightSon)
    else:
        return DTClassify(test, tree.leftSon)


def ID3(E, F, M=0):
    c = MajorityClass(E)
    return TDIDT(E, F, c, M)


def experiment(trainSet,F):
    M = [3, 4, 5]
    N = [6,7,8]
    K = [3,5]
    P= [0.3,0.5,0.7]

    kf = KFold(n_splits=5, shuffle=True, random_state=208526814)
    bestAVG=0
    bestN=0
    bestK=0
    bestP=0
    for n in N:
         for k in K:
             for p in P:

                    _sum=0
                    for train_index, test_index in kf.split(trainSet):
                        X_train, X_test = trainSet[train_index,:], trainSet[test_index,:]
                        res = KNNForest(X_train,X_test,F,n,p,k)
                        _sum+=res
                    avg=_sum/5
                    print(n,p,k,avg)
                    if avg >= bestAVG:
                        bestAVG=avg
                        bestN=n
                        bestK=k
                        bestP=p
    return bestN,bestP,bestK


def GetKNN(treeList,sample,k):
    distValues = np.zeros(len(treeList))
    i=0
    for tree in treeList:
        distValues[i]=np.linalg.norm(tree.centroid - sample)
        i+=1
    j=k
    maximum = np.amax(distValues) +1
    KNN=[]
    while j>0:
        index = np.argmin(distValues)
        KNN.append(treeList[index])
        distValues[index] = maximum
        j-=1
    return KNN
def predict(testData, treeList,k):
    total_samples = len(testData)
    total_correct=0

    for i in range(total_samples):
        numB = 0
        numM = 0
        currRes = ''
        sample =testData[i, :]
        KNN = GetKNN(treeList,np.delete(sample,0),k)
        for tree in KNN:
            res = DTClassify(sample, tree)
            if res=='M':
                numM+=1
            else:
                 numB+=1
        if numB > numM:
            currRes='B'
        else:
            currRes='M'
        if currRes==sample[0]:
           total_correct+=1

    return total_correct / total_samples







def CalcCentroid(E,features):
    return np.mean(E[:,1:],0)

def GetRandomIndecies(n,p):
    indices_range  = range(n)
    random_indices = rd.choices(indices_range,k=int(p*n))
    return random_indices

def KNNForest(train,test,features,N,p,K):
    treeList = []
    n=len(train)
    for i in range(N):
        indices = GetRandomIndecies(n,p)
        tree = ID3(train[indices],features)
        tree.centroid=CalcCentroid(train[indices],features)
        treeList.append(tree)
    return predict(test,treeList,K)


def randomForestFeatures(data,p,M=10,l=0.3):
    global totalsamples
    totalsamples = math.floor(len(data)*l)
    global featurevector
    featurevector = [0]* data.shape[1]
    res = []
    treeList = []
    n = len(data)

    features = data.shape[1]-1
    k = math.floor(p * features)
    for i in range(M):
        indices = GetRandomIndecies(n, l)
        tree = ID3(data[indices], features)
        tree.centroid = CalcCentroid(data[indices], features)
        treeList.append(tree)
    res=sorted(range(len(featurevector)), key=lambda t: featurevector[t])
    del res[0]
    return res[features-k:]





if __name__ == "__main__":
    train_data = pd.read_csv('data/green.csv')
    test_data = pd.read_csv('data/test1.csv')
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
    features = train_data.shape[1] - 1

    print(randomForestFeatures(train_data,0.1,7,0.6))




