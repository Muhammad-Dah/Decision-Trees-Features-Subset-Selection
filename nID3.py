import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import log2 as log
from collections import Counter
import math
import random
import scipy.stats as sp
from nKNNForest import randomForestFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import itertools




# class knn:
#     def __init__(self,k=5):
#         self.k=k
#     def fit(self,data,label):
#         self.data=data
#         self.label=label
#     def predict(self,sample):
#         date = np.array(self.data)
#         labels =[]
#         dist =[]
#         for i in range(date.shape[0]):
#
#
#
#
#
#     def accuracy(self,test):
#         dat = pd.DataFrame(self.data)
#         dat = dat.iloc[:,1:]
#         dat = np.array(dat)
#         test = np.array(test)
#         accur=0
#         label = []
#         dist = []
#         for sample in test:
#             for train in dat:
#
#
#
# return accur/len(test)
def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def MajorityClass(E):
    arr = E[:, 0]
    # cnt_b = np.sum(arr == 'B')
    # cnt_m = np.sum(arr == 'M')
    cnt = Counter(arr)
    cnt_b = cnt['B']
    cnt_m = cnt['M']

    return 'B' if cnt_b >= cnt_m else 'M'


def Entropy(E):
    sigma = 0
    arr = E[:, 0]
    n = len(arr)
    if n == 0:
        return sigma
    cnt = Counter(arr)
    cnt_b = cnt['B']
    cnt_m = cnt['M']
    if cnt_b > 0:
        p_b = cnt_b / n
        sigma += p_b * log(p_b)
    if cnt_m > 0:
        p_m = cnt_m / n
        sigma += p_m * log(p_m)

    return -sigma


def IG(E, feature):
    parentEntropy = Entropy(E)
    sortedByFeature = E[:, feature]
    sortedByFeature = np.sort(sortedByFeature)
    threshHoldVals = set()
    currIG = 0
    maxIG = -np.inf
    bestThreshHold = 0
    numOfSamples = len(sortedByFeature)
    for i in range(numOfSamples - 1):
        threshHoldVals.add((sortedByFeature[i] + sortedByFeature[i + 1]) / 2)

    for threshHold in threshHoldVals:
        leftSubSet = E[E[:, feature] <= threshHold]
        rightSubSet = E[E[:, feature] > threshHold]
        leftVal = ((len(leftSubSet) / numOfSamples) * Entropy(leftSubSet))
        rightVal = ((len(rightSubSet) / numOfSamples) * Entropy(rightSubSet))
        currIG = parentEntropy - (leftVal + rightVal)
        if currIG >= maxIG:
            maxIG = currIG
            bestLeftSplit = leftSubSet
            bestRightSplit = rightSubSet
            bestThreshHold = threshHold
    return maxIG, bestLeftSplit, bestRightSplit, bestThreshHold


def MaxIG(E, F):
    maxIG = 0
    currIG = 0
    for f in range(F):
        currIG, currLeft, curRight, currThreshHold = IG(E, f + 1)
        if currIG >= maxIG:
            maxIG = currIG
            bestFeature = f + 1
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


def TDIDT(E, F, Default, count, former_num_of_samples, M):
    numOfSamples = len(E)
    node = Node(None, None, Default)
    if M > 0:
        if numOfSamples < M:
            return node
    if numOfSamples == 0 or count > 5:
        return node
    c = MajorityClass(E)
    if c == 'M':
        minority = 'B'
    else:
        minority = 'M'
    if minority not in E[:, 0]:
        node.label = c
        return node

    if former_num_of_samples == numOfSamples:
        count += 1
    else:
        count = 0
    former_num_of_samples = numOfSamples
    feature, left, right, threshHold = MaxIG(E, F)
    node.threshHold = threshHold
    node.feature = feature
    node.label = c
    node.leftSon = TDIDT(left, F, c, count, former_num_of_samples, M)
    node.rightSon = TDIDT(right, F, c, count, former_num_of_samples, M)
    return node


def DTClassify(test, tree: Node):
    if tree.leftSon is None:
        return tree.label
    if (test[tree.feature] >= tree.threshHold):
        return DTClassify(test, tree.rightSon)
    else:
        return DTClassify(test, tree.leftSon)


def ID3(E, F, M=5):
    c = MajorityClass(E)
    return TDIDT(E, F, c, 0, 0, M)


def predict(testData, tree: Node):
    total_samples = len(testData)
    total_correct = 0
    for i in range(total_samples):
        res = DTClassify(testData[i, :], tree)
        if res == testData[i, 0]:
            total_correct += 1
    return total_correct / total_samples


def predictLoss(testData: pd.DataFrame, tree: Node):
    total_samples = len(testData)
    _sum = 0
    for i in range(total_samples):
        res = DTClassify(testData[i, :], tree)
        if res != testData[i, 0]:
            if res == 'B':
                _sum += 1
            else:
                _sum += 0.1
    return _sum / total_samples


def experiment(trainSet, F):  # to run the experiment,uncomment the second main :3.1
    P = [2, 3, 4, 5,
         6]  # the function is called with the training set,and F being the number of features ,30 in our case
    accList = []
    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    bestAVG = 0
    bestM = 0
    for M in P:
        _sum = 0

        for train_index, test_index in kf.split(trainSet):
            X_train, X_test = trainSet[train_index, :], trainSet[test_index, :]
            tree = ID3(X_train, F, M)
            res = predict(X_test, tree)
            _sum += res
        avg = _sum / 5
        accList.append(avg)

        if avg >= bestAVG:
            bestAVG = avg
            bestM = M
    plt.plot(P, accList)
    plt.show()
    return bestM

    # 1


def randFeatures(data, p):
    N = data.shape[1] - 1
    k = math.floor(N * p)
    res = random.sample(range(1, N + 1), k)
    return res


def maxIGFeatures(data, p):
    N = data.shape[1] - 1
    k = math.floor(N * p)
    res = []
    temp = []
    for feature in range(1, N + 1):
        currIG, _, _, _ = IG(data, feature)
        res.append(currIG)
        temp.append(currIG)

    temp.sort()
    kth = temp[N - k]
    ret = []
    for i in range(N):
        if res[i] > kth:
            ret.append(i + 1)
    for i in range(N):
        if len(ret) >= k:
            break
        if res[i] == kth:
            ret.append(i + 1)

    return ret


def diverse1Features(data, p):
    window = maxIGFeatures(data, min(2 * p, 1))
    res = []
    N = data.shape[1] - 1
    maxig = -np.inf
    maxidx = 0
    for feature in window:
        curig, _, _, _ = IG(data, feature)
        if curig > maxig:
            maxig = curig
            maxidx = feature
    res.append(maxidx)
    window.remove(maxidx)

    k = math.floor(p * N)
    for i in range(k - 1):
        if (k == 2):
            kjfdg = 2
        maxdst = -np.inf
        maxidx = -1
        for w in window:
            mindist = np.inf
            for feature in res:
                curdist, _ = sp.spearmanr(data[:, feature], data[:, w])
                if curdist < mindist:
                    mindist = curdist
            if mindist >= maxdst:
                maxdst = mindist
                maxidx = w
        if maxidx != -1:
              res.append(maxidx)
              window.remove(maxidx)
    return res

def diverse2Features(data, p):
    res = []
    mx = -np.inf
    N = data.shape[1] - 1
    window = [i for i in range(1, N + 1)]
    k = math.floor(p * N)
    w = min([2*k,N+1])
    if w==N+1:
        return window
    if k==1:
        return maxIGFeatures(data[:, [0] + list(window)], 1 / len(window))
    res = []
    for subset in findsubsets(window,w):
        tot=0
        for s in subset:
            minn=np.inf
            for d in subset:
                if d==s:
                    continue
                curdist,_  = sp.spearmanr(data[:,d],data[:,s])
                if curdist < minn:
                    minn = curdist
            tot+=minn
        if tot >= mx:
            mx =tot
            res = subset

    return maxIGFeatures(data[:, [0] + list(res)], k / len(res))
    
    
# def diverse2Features(data, p):
#     res = []
#     N = data.shape[1] - 1
#     window = [i for i in range(1, N + 1)]
#     if p == 1:
#         return window
#     maxdist = -np.inf
#     maxidx = 0
#     for i in range(N):
#         curmin = np.inf
#         curig, _, _, _ = IG(data, i + 1)
#         for j in range(N):
#             if i == j:
#                 continue
#             internalig, _, _, _ = IG(data, j + 1)
#             _, curdist = sp.spearmanr(internalig, curig)
#             if curdist < curmin:
#                 curmin = curdist
#         if curmin > maxdist:
#             maxdist = curmin
#             maxidx = i + 1
#
#     res.append(maxidx)
#     window.remove(maxidx)
#
#     k = math.floor(p * N)
#     for i in range(2 * k - 1):
#         maxig = -np.inf
#         maxidx = -1
#
#         for w in window:
#
#             mindist = np.inf
#             curitemig, _, _, _ = IG(data, w)
#             for feature in res:
#                 curwig, _, _, _ = IG(data, feature)
#                 _, curdist = sp.spearmanr(curitemig, curwig)
#                 if curdist < mindist:
#                     mindist = curdist
#             if mindist > maxig:
#                 maxig = mindist
#                 maxidx = w
#         if maxidx != -1:
#             res.append(maxidx)
#             window.remove(maxidx)
#     if k==1:
#         return maxIGFeatures(data[:, [0] + list(window)], 1 / len(window))
#
#     return maxIGFeatures(data[:, [0] + res], k / len(res))


# def diverse2Features(data, p):
#     res = []
#     mx = -np.inf
#     N = data.shape[1] - 1
#     window = [i for i in range(1, N + 1)]
#     k = math.floor(p * N)
#     w = min([2 * k, N + 1])
#     if w == N + 1:
#         return maxIGFeatures(data[:, [0] + list(window)], k / N)
#     if k == 1:
#         return maxIGFeatures(data[:, [0] + list(window)], 1 / len(window))
#
#     res = [maxIGFeatures(data, 1 / N)]
#     res = list([res[0][0]])
#
#
#     window.remove(res[0])
#
#     for i in range(2*k - 1):
#         fett = -1
#         minn=np.inf
#         for fet in window:
#             minn = np.inf
#             for r in res:
#                 curdist, _ = sp.spearmanr(data[:, fet], data[:, r])
#                 if curdist < minn:
#                     minn = curdist
#                     fett = fet
#         if fett != -1:
#             window.remove(fett)
#             res.append(fett)
#
#     return maxIGFeatures(data[:, [0] + list(res)], k / len(res))


def wrapperForwardFeautres(data, p):
    N = data.shape[1] - 1
    k = math.floor(N * p)
    features = [i for i in range(1, N + 1)]
    res = []

    for i in range(k):

        bestAVG = -np.inf
        bestFeature = -1
        for feature in features:
            _sum = 0
            kf = KFold(n_splits=5, shuffle=True, random_state=13)
            for train_index, test_index in kf.split(data):
                X_train, X_test = data[train_index, :], data[test_index, :]
                # print(X_train[:,[0]+[feature]+res])
                classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric='euclidean', p=2)
                classifier.fit(X_train[:, [feature] + res], X_train[:, 0])
                y_pred = classifier.predict(X_test[:, [feature] + res])
                accuracy = accuracy_score(y_pred, X_test[:, 0])
                _sum += accuracy
            avg = _sum / 5

            if avg >= bestAVG:
                bestAVG = avg
                bestFeature = feature
        if bestFeature == -1:
            continue
        features.remove(bestFeature)
        res.append(bestFeature)
    return res


def wrapperBackwardFeautres(data, p):
    N = data.shape[1] - 1
    k = math.floor(N * p)
    res = [i for i in range(1, N + 1)]
    if k == 1:
        return wrapperForwardFeautres(data, p)
    for i in range(N - k):
        bestAVG = -np.inf
        worstFeature = -1
        for feature in res:
            _sum = 0
            tmp = list(res)
            tmp.remove(feature)
            kf = KFold(n_splits=5, shuffle=True, random_state=13)
            for train_index, test_index in kf.split(data):
                X_train, X_test = data[train_index, :], data[test_index, :]
                # print(X_train[:,[0]+[feature]+res])
                classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric='euclidean', p=2)
                classifier.fit(X_train[:, tmp], X_train[:, 0])
                y_pred = classifier.predict(X_test[:, tmp])
                accuracy = accuracy_score(y_pred, X_test[:, 0])
                _sum += accuracy
            avg = _sum / 5

            if avg >= bestAVG:
                bestAVG = avg
                worstFeature = feature
        if worstFeature == -1:
            continue
        res.remove(worstFeature)
    return res


def localSearchFeatures(data, p):
    featureslist = randFeatures(data, p)

    N = data.shape[1] - 1
    complimentfeatures = [i for i in range(1, N + 1) if i not in featureslist]
    k = math.floor(p * N)
    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    _sum = 0
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index, :], data[test_index, :]
        classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric='euclidean', p=2)
        classifier.fit(X_train[:, featureslist], X_train[:, 0])
        y_pred = classifier.predict(X_test[:, featureslist])
        accuracy = accuracy_score(y_pred, X_test[:, 0])
        _sum += accuracy
    bestAVG = _sum / 5
    formerBestAVG = bestAVG
    for i in range(10):

        for j in range(len(featureslist)):
            kf = KFold(n_splits=5, shuffle=True, random_state=13)
            formerfeature = featureslist[j]

            for comp in complimentfeatures:

                featureslist[j] = comp
                _sum = 0
                for train_index, test_index in kf.split(data):
                    X_train, X_test = data[train_index, :], data[test_index, :]

                    classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric='euclidean', p=2)
                    classifier.fit(X_train[:, featureslist], X_train[:, 0])
                    y_pred = classifier.predict(X_test[:, featureslist])
                    accuracy = accuracy_score(y_pred, X_test[:, 0])
                    _sum += accuracy
                avg = _sum / 5

                if avg > bestAVG:
                    complimentfeatures.remove(comp)
                    complimentfeatures.append(formerfeature)
                    bestAVG = avg
                    break
                else:
                    featureslist[j] = formerfeature
            if formerBestAVG != bestAVG:
                formerBestAVG = bestAVG
                break

    return featureslist


if __name__ == "__main__":

    str = "schiller"
    dataset_list = [str + '.csv']
    str = dataset_list[0]
    dataset = pd.read_csv(str)

    # scaler = StandardScaler()
    # dt = scaler.fit_transform(dataset.iloc[:,1:])
    # dt = pd.DataFrame(dt)
    # dt.insert(0,'label',dataset.iloc[:,0],True)
    dataset = np.array(dataset)
    N = dataset.shape[1] - 1

    algorithms_list = [randFeatures, maxIGFeatures, diverse1Features,diverse2Features, wrapperForwardFeautres,
                       wrapperBackwardFeautres, localSearchFeatures,randomForestFeatures]
    # algorithms_list = [diverse2Features]
    percentage_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,0.1]
    res = np.zeros([8, 10])
    plt.clf()
    plt.title(str[:-4])
    plt.xlabel('P')
    plt.ylabel('Accuracy')
    g = -1
    kf = KFold(n_splits=5, shuffle=True, random_state=208526814)
    for algo in algorithms_list:
        g += 1
        print(algo)
        fet = []
        pr = -1
        for p in percentage_list:
            pr += 1
            print(p)
            if p == 1:
                fet = [i for i in range(1, N + 1)]
            else:
                q = math.floor(p * N)

                if q == 0:
                    continue
                f = algo(dataset[:, [0] + fet], (q / len(fet)))

                fet = [fet[i - 1] for i in f]




            for train_index, test_index in kf.split(dataset):
                X_train, X_test = dataset[train_index, :], dataset[test_index, :]
                scaler = StandardScaler()
                X_train = pd.DataFrame(X_train)
                dt = scaler.fit_transform(X_train.iloc[:, 1:])
                dt = pd.DataFrame(dt)
                dt.insert(0, 'label', X_train.iloc[:, 0], True)
                X_train = np.array(dt)

                X_test = pd.DataFrame(X_test)
                dt = scaler.transform(X_test.iloc[:, 1:])
                dt = pd.DataFrame(dt)
                dt.insert(0, 'label', X_test.iloc[:, 0], True)
                X_test = np.array(dt)

                # X_train = pd.DataFrame(X_train)
                # dt = np.array(X_train.iloc[:,1:])
                # dt = pre.normalize(dt,axis=0)
                # dt = pd.DataFrame(dt)
                # dt.insert(0, 'label', X_train.iloc[:, 0], True)
                # X_train = np.array(dt)
                #
                # X_test = pd.DataFrame(X_test)
                # dt = np.array(X_test.iloc[:, 1:])
                # dt = pre.normalize(dt,axis=0)
                # dt = pd.DataFrame(dt)
                # dt.insert(0, 'label', X_test.iloc[:, 0], True)
                # X_test = np.array(dt)

                # X_train = pd.DataFrame(X_train)
                # dt1 = X_train.iloc[:, 1:]
                # dt = (dt1-dt1.mean())/dt1.std()
                # dt = pd.DataFrame(dt)
                # dt.insert(0, 'label', X_train.iloc[:, 0], True)
                # X_train = np.array(dt)
                #
                # X_test = pd.DataFrame(X_test)
                # dt = X_test.iloc[:, 1:]
                # dt = (dt-dt1.mean())/dt1.std()
                # dt.insert(0, 'label', X_test.iloc[:, 0], True)
                # X_test = np.array(dt)

                classifier = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric='euclidean', p=2)
                classifier.fit(X_train[:, fet], X_train[:, 0])
                y_pred = classifier.predict(X_test[:, fet])
                accuracy = accuracy_score(y_pred, X_test[:, 0])

                res[g][pr] += accuracy

    res = np.array(res)

    res = res / 5

    res.tofile(str + '.txt', sep=',')
    res = np.array(res)
    # plt.plot(percentage_list, res[0, :], label='Random')
    # plt.plot(percentage_list, res[1, :], label='IG')
    # plt.plot(percentage_list, res[2, :], label='Diverse1')
    plt.plot(percentage_list, res[0, :], label='Diverse2')
    # plt.plot(percentage_list, res[4, :], label='WrapperForward')
    # plt.plot(percentage_list, res[5, :], label='WrapperBackward')
    # plt.plot(percentage_list, res[6, :], label='LocalSearch')
    # plt.plot(percentage_list, res[7, :], label='randomForestFeatures')
    plt.legend()
    plt.savefig(str[:-4] + ".jpg")

    # for str in dataset_list:
    #     dataset = pd.read_csv('data/' + str)
    #     # scaler = StandardScaler()
    #     # dt = scaler.fit_transform(dataset.iloc[:,1:])
    #     # dt = pd.DataFrame(dt)
    #     # dt.insert(0,'label',dataset.iloc[:,0],True)
    #     dataset = np.array(dataset)
    #
    #     res = [[0] * 8] * 10  # 8 is the number of the algorithms, 10 is the number of percentages
    #     plt.clf()
    #     plt.title(str[:-4])
    #     plt.xlabel('P')
    #     plt.ylabel('Accuracy')
    #     j = 0
    #     for p in percentage_list:
    #         print(p)
    #         N = dataset.shape[1] - 1
    #         if p * N < 1:  # number of features is smaller than 1,increase the percentage index by 1 and continue
    #             j += 1
    #             continue
    #         prediction_list = [0] * 8
    #         kf = KFold(n_splits=5, shuffle=True, random_state=208526814)
    #         for train_index, test_index in kf.split(dataset):
    #
    #             X_train, X_test = dataset[train_index, :], dataset[test_index, :]
    #
    #             # scaler = StandardScaler()
    #             # X_train = pd.DataFrame(X_train)
    #             # dt = scaler.fit_transform(X_train.iloc[:, 1:])
    #             # dt = pd.DataFrame(dt)
    #             # dt.insert(0, 'label', X_train.iloc[:, 0], True)
    #             # X_train = np.array(dt)
    #             #
    #             # X_test = pd.DataFrame(X_test)
    #             # dt = scaler.transform(X_test.iloc[:, 1:])
    #             # dt = pd.DataFrame(dt)
    #             # dt.insert(0, 'label', X_test.iloc[:, 0], True)
    #             # X_test = np.array(dt)
    #
    #             i = 0
    #             for algo in algorithms_list:
    #                 print(algo)
    #                 if p == 1:  # if we want all features,no need to run the algorithm
    #                     feature_set = [i for i in range(1, N + 1)]
    #                 else:
    #                     feature_set = algo(X_train,p)
    #
    #                 classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric='euclidean', p=2)
    #                 classifier.fit(X_train[:, feature_set], X_train[:, 0])
    #                 y_pred = classifier.predict(X_test[:, feature_set])
    #                 accuracy = accuracy_score(y_pred, X_test[:, 0])
    #                 prediction_list[i] += accuracy
    #                 i += 1
    #         res[j] = [r / 5 for r in prediction_list]
    #         j += 1
    #     res = np.array(res)
    #     res.tofile('Autism-Child-Data.txt', sep=',')
    #     res = np.array(res)
    #     plt.plot(percentage_list, res[:, 0], label='Random')
    #     plt.plot(percentage_list, res[:, 1], label='IG')
    #     plt.plot(percentage_list, res[:, 2], label='Diverse1')
    #     plt.plot(percentage_list, res[:, 3], label='Diverse2')
    #     plt.plot(percentage_list, res[:, 4], label='WrapperForward')
    #     plt.plot(percentage_list, res[:, 5], label='WrapperBackward')
    #     plt.plot(percentage_list, res[:, 6], label='LocalSearch')
    #     plt.plot(percentage_list, res[:, 7], label='randomForestFeatures')
    #     plt.legend()
    #     plt.savefig(str[:-4] + ".jpg")

#
#     # experiment run 3.3
# if __name__ == "__main__":
#     train_data = pd.read_csv('data/train.csv')
#     test_data = pd.read_csv('data/test.csv')
#     train_data = train_data.to_numpy()
#     test_data = test_data.to_numpy()
#     features = train_data.shape[1] - 1
#     experiment(train_data,features)


#
#     # experiment run 3.4
# if __name__ == "__main__":
#     train_data = pd.read_csv('data/train.csv')
#     test_data = pd.read_csv('data/test.csv')
#     train_data = train_data.to_numpy()
#     test_data = test_data.to_numpy()
#     features = train_data.shape[1] - 1
#     tree = ID3(train_data, features, 4)
#     print(predict(test_data, tree))
#
#     # experiment run 4.1
# if __name__ == "__main__":
#     train_data = pd.read_csv('data/train.csv')
#     test_data = pd.read_csv('data/test.csv')
#     train_data = train_data.to_numpy()
#     test_data = test_data.to_numpy()
#     features = train_data.shape[1] - 1
#     tree = ID3(train_data, features, 4)
#     print(predictLoss(test_data, tree))
#
#
