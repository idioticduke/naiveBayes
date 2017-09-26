import numpy as np
import loadData as ld
import math

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #p0Num = np.zeros(numWords)
    #p1Num = np.zeros(numWords)
    #p0Denom = 0.0
    #p1Denom = 0.0
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p1Vect = p1Num/p1Denom #change to log()
    #p0Vect = p0Num/p0Denom #change to log()
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

'''
listOPosts,listClasses = ld.loadDataSet()
myVocabList = ld.createVocabList(listOPosts)

trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(ld.setOfWords2Vec(myVocabList, postinDoc))

p0V,p1V,pAb=trainNB0(trainMat,listClasses)
print(pAb)
'''