from __future__ import division
import csv
import numpy as np  # http://www.numpy.org
import math
from collections import Counter
from CSE6242HW4Tester import generateSubmissionFile


"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

# Enter your name here
myname = "Rajpurohit-Tanmay"


### This function computes the entropy of data ####

def entropy(y):
        uniq, inverse = np.unique(y, return_inverse=True)
        labelCount = np.bincount(inverse)
        etrp = 0
        for i in range(len(uniq)):
            pi = labelCount[i]/np.sum(labelCount)
            if pi != 0:
                etrp = - pi*math.log(pi,2) + etrp
        return etrp


#### This Function finds the best attribute and threshhold value thereof #### 

def split_function(X,y,index):
        entropyBefore = entropy(y)          
        gain = 0
        threshHold=0
        attr = 0
        for i in index:
                data=np.column_stack((X[:,i],y))
                val = np.linspace(np.amin(data, axis=0)[0], np.amax(data, axis=0)[0], num=25)
                for j in range(1,len(val)-1): 
                         left = np.array([x for k, x in enumerate(data) if x[0]<val[j]], dtype = float)
                         right = np.array([x for k, x in enumerate(data) if x[0]>=val[j]], dtype = float)
                         if left.shape[0] == 0:
                                G=0
                         elif right.shape[0] == 0:
                                G=0
                         else:
                                leftEntropy = entropy(left[:,1])
                                rightEntropy = entropy(right[:,1])
                                G = entropyBefore - (len(left)/len(data))*leftEntropy - (len(right)/len(data))*rightEntropy
                         if G > gain:
                                  gain = G
                                  attr = i
                                  threshHold = val[j]
        return [threshHold,attr]
        

####### This Function Generates the Tree ######

def create_tree(X,y,idx,depth=0):

        idxr = np.random.choice(idx, int(math.sqrt(len(idx)))) # Randomly chosing the subset of attributes
        [t,a]=split_function(X,y,idxr)     # Finding the attribute and threshhold value

     ## Splitting the Data Set
        
        leftX=np.array([x for i, x in enumerate(X) if x[a]<t], dtype = float)     
        lefty=np.array([y[j] for j, x in enumerate(X) if x[a]<t], dtype = int)
        rightX=np.array([x for k, x in enumerate(X) if x[a]>=t], dtype = float)
        righty=np.array([y[l] for l, x in enumerate(X) if x[a]>=t], dtype = int)
     ## Determining the unique values of lables after split
        uniqueL, inverseL = np.unique(lefty, return_inverse=True)
        uniqueR, inverseR = np.unique(righty, return_inverse=True)

     ## If NO improvement after split
        
        if len(uniqueL)==0:
                tree={"isLeaf": 0 ,"nodeId": a,"tVal": t,
                      "leftBranch":create_tree(rightX,righty,idx,depth+1),"rightBranch":create_tree(rightX,righty,idx,depth+1)}
        elif len(uniqueR)==0:
                tree={"isLeaf": 0 ,"nodeId": a,"tVal": t,
                      "leftBranch":create_tree(leftX,lefty,idx,depth+1),"rightBranch":create_tree(leftX,lefty,idx,depth+1)}


     ## If there is any pure Node
                
        elif len(uniqueL)==1:
                if len(uniqueR)==1:
                        tree={"isLeaf": 2 ,"nodeId": a,"tVal": t,
                      "leftBranch":uniqueL[0],"rightBranch":uniqueR[0]}
                else:
                       tree={"isLeaf": 1 ,"nodeId": a,"tVal": t,
                      "leftBranch":uniqueL[0],"rightBranch":create_tree(rightX,righty,idx,depth+1)}
        elif len(uniqueR)==1:
                tree={"isLeaf": 1 ,"nodeId": a,"tVal": t,
                      "leftBranch":create_tree(leftX,lefty,idx,depth+1),"rightBranch":uniqueR[0]}

     ## If sufficient depth is reached

        elif depth==2:
                defaultTree={"nodeId": a,"tVal": t}
                defaultTree["isLeaf"]=2
 
 
                if len(uniqueL)==2:
                        defaultTree["leftBranch"]= uniqueL[np.argmax(np.bincount(inverseL))]
                        defaultTree["rightBranch"] = 1-uniqueL[np.argmax(np.bincount(inverseL))]
                elif len(uniqueL)==1:
                        defaultTree["leftBranch"]= uniqueL[0]
                        defaultTree["rightBranch"] = 1-uniqueL[0]
                else:
                        defaultTree["leftBranch"]= 1- uniqueR[np.argmax(np.bincount(inverseR))]
                        defaultTree["rightBranch"] = uniqueR[np.argmax(np.bincount(inverseR))]
                        
                return defaultTree;
        
     ## Create subtrees branches otherwise
                
        else:
                tree={"isLeaf": 0 ,"nodeId": a,"tVal": t,
                      "leftBranch":create_tree(leftX,lefty,idx,depth+1),"rightBranch":create_tree(rightX,righty,idx,depth+1)}
        return tree


class RandomForest(object):
    class __DecisionTree(object):
        tree = {}

        def learn(self, X, y):
                self.tree=create_tree(X,y,range(X.shape[1])) # TODO: train decision tree and store it in self.tree
               

        def classify(self, test_instance):    # TODO: return predicted label for a single instance using self.tree
                testTree = self.tree
                while True:
                        if testTree["isLeaf"] == 2:
                                if test_instance[testTree["nodeId"]] < testTree["tVal"]:
                                        return testTree["leftBranch"]
                                else:
                                        return testTree["rightBranch"]
                                
                        elif testTree["isLeaf"] == 1:
                                if test_instance[testTree["nodeId"]] < testTree["tVal"]:
                                        if not isinstance(testTree["leftBranch"], dict):
                                                return testTree["leftBranch"]
                                        else:
                                                testTree = testTree["leftBranch"]
                                else:
                                        if not isinstance(testTree["rightBranch"], dict):
                                                return testTree["rightBranch"]
                                        else:
                                                testTree = testTree["rightBranch"]
                        else:
                                if test_instance[testTree["nodeId"]] < testTree["tVal"]:
                                        testTree = testTree["leftBranch"]
                                else:
                                        testTree=testTree["rightBranch"]
 
            
           
        
                

    

    decision_trees = []

    def __init__(self, num_trees):
        # TODO: do initialization here, you can change the function signature according to your need
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree()] * num_trees
        
    # You MUST NOT change this signature
    def fit(self, X, y):
        for i in range(self.num_trees):
            self.decision_trees[i].learn(X,y) # TODO: train `num_trees` decision trees
            

    # You MUST NOT change this signature
    def predict(self, X):
        y = np.array([], dtype = int)

        for instance in X:
            votes = np.array([decision_tree.classify(instance)
                              for decision_tree in self.decision_trees])

            counts = np.bincount(votes)

            y = np.append(y, np.argmax(counts))

        return y


def main():
    X = []
    y = []

    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])

    X = np.array(X, dtype = float)
    y = np.array(y, dtype = int)

    # Split training/test sets
    # You need to modify the following code for cross validation
    K = 10
    X_train = np.array([x for i, x in enumerate(X) if i % K != 9], dtype = float)
    y_train = np.array([z for i, z in enumerate(y) if i % K != 9], dtype = int)
    X_test  = np.array([x for i, x in enumerate(X) if i % K == 9], dtype = float)
    y_test  = np.array([z for i, z in enumerate(y) if i % K == 9], dtype = int)

    randomForest = RandomForest(65)  # Initialize according to your implementation

    randomForest.fit(X_train, y_train)

    y_predicted = randomForest.predict(X_test)

    results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print "accuracy: %.4f" % accuracy

    generateSubmissionFile(myname, randomForest)


main()
