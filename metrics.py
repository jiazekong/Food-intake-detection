import numpy as np
import matplotlib.pyplot as plt
import inertial_dnn as dnn


def predict(X, y, parameters):
   
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = dnn.model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.15:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


def TP_TN_FP_FN(p, y):
   a =  p + y
   b = p - y

   TP = np.sum(a==2)
   
   TN = np.sum(a==0)

   FP = np.sum(b==1)

   FN = np.sum(b==-1)

   
   return TP,TN,FP,FN



def Precision(TP, FP):
    precision = TP/(TP+FP)
    return precision

def Recall(TP, FN):
    recall = TP/(TP+FN)
    return recall

def F1(p, r):
    f1 = (2*p*r)/(p+r)
    return f1


