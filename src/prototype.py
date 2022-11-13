from __future__ import print_function
import numpy as np
from numpy import array
from scipy.spatial.distance import cdist
from scipy import sparse
from osqp import OSQP


def optimize(K, u, preOptw, initialValue, maxWeight=10000):
    """
    Args:
        K (double 2d array): Similarity/distance matrix
        u (double array): Mean similarity of each prototype
        preOptw (double): Weight vector
        initialValue (double): Initialize run
        maxWeight (double): Upper bound on weight
        
    Returns:
        Prototypes, weights and objective values
    """
    
    d = u.shape[0]
    lb = np.zeros((d, 1))
    ub = maxWeight * np.ones((d, 1))
    
    # x0 = initial value, provided optimizer supports it. 
    x0 = np.append( preOptw, initialValue/K[d-1, d-1] )
    
    G = np.vstack((np.identity(d), -1*np.identity(d)))
    h = np.vstack((ub, -1*lb)).ravel()

    # variable shapes: K = (d,d), u = (d,) G = (2d, d), h = (2d,)
    Ks = sparse.csc_matrix(K)
    Gs = sparse.csc_matrix(G)
    l_inf = -np.inf * np.ones(len(h))

    solver = OSQP() 
    solver.setup(P=Ks, q=-u, A=Gs, l=l_inf, u=h, eps_abs=1e-4, eps_rel=1e-4, polish= True, verbose=False) 
    solver.warm_start(x=x0) 
    res = solver.solve() 

    xv = res.x.reshape(-1, 1) 
    xreturn = res.x 
        
    # compute objective function value        
    P = K
    q = - u.reshape(-1, 1)
    obj_value = 1/2 * np.matmul(np.matmul(xv.T, P), xv) + np.matmul(q.T, xv)
    
    return(xreturn, obj_value[0,0])


def prototype_selector(X, Y, m, kernelType = 'Guassian', sigma = 2):
    """
    Main prototype selection function.

    Args:
        X (double 2d array): Dataset to select prototypes from
        Y (double 2d array): Dataset to explain
        m (double): Number of prototypes
        kernelType (str): Gaussian, linear or other
        sigma (double): Gaussian kernel width

    Returns:
        Current optimum, the prototypes and objective values throughout selection
    """
    numY = Y.shape[0]
    numX = X.shape[0]
    allY = np.array(range(numY))

    # Store the mean inner products with X
    if kernelType == 'Gaussian':
        meanInnerProductX = np.zeros((numY, 1))
        for i in range(numY):
            Y1 = Y[i, :]
            Y1 = Y1.reshape(Y1.shape[0], 1).T
            distX = cdist(X, Y1)
            meanInnerProductX[i] = np.sum( np.exp(np.square(distX)/(-2.0 * sigma**2)) ) / numX
    else:
        M = np.dot(Y, np.transpose(X))
        meanInnerProductX = np.sum(M, axis=1) / M.shape[1]

    # move to features x observation format to be consistent with the earlier code version
    X = X.T
    Y = Y.T

    # Intialization
    S = np.zeros(m, dtype=int)
    setValues = np.zeros(m)
    sizeS = 0
    currSetValue = 0.0
    currOptw = np.array([])
    currK = np.array([])
    curru = np.array([])
    runningInnerProduct = np.zeros((m, numY))

    while sizeS < m:

        remainingElements = np.setdiff1d(allY, S[0:sizeS])

        newCurrSetValue = currSetValue
        maxGradient = 0

        for count in range(remainingElements.shape[0]):

            i = remainingElements[count]
            newZ = Y[:, i]

            if sizeS == 0:

                if kernelType == 'Gaussian':
                    K = 1
                else:
                    K = np.dot(newZ, newZ)

                u = meanInnerProductX[i]
                w = np.max(u / K, 0)
                incrementSetValue = -0.5 * K * (w ** 2) + (u * w)

                if (incrementSetValue > newCurrSetValue) or (count == 1):
                    # Bookeeping
                    newCurrSetValue = incrementSetValue
                    desiredElement = i
                    newCurrOptw = w
                    currK = K

            else:
                recentlyAdded = Y[:, S[sizeS - 1]]

                if kernelType == 'Gaussian':
                    distnewZ = np.linalg.norm(recentlyAdded-newZ)
                    runningInnerProduct[sizeS - 1, i] = np.exp( np.square(distnewZ)/(-2.0 * sigma**2 ) )
                else:
                    runningInnerProduct[sizeS - 1, i] = np.dot(recentlyAdded, newZ)

                innerProduct = runningInnerProduct[0:sizeS, i]
                if innerProduct.shape[0] > 1:
                    innerProduct = innerProduct.reshape((innerProduct.shape[0], 1))

                gradientVal = meanInnerProductX[i] - np.dot(currOptw, innerProduct)

                if (gradientVal > maxGradient) or (count == 1):
                    maxGradient = gradientVal
                    desiredElement = i
                    newinnerProduct = innerProduct[:]

        S[sizeS] = desiredElement

        curru = np.append(curru, meanInnerProductX[desiredElement])

        if sizeS > 0:

            if kernelType == 'Gaussian':
                selfNorm = array([1.0])
            else:
                addedZ = Y[:, desiredElement]
                selfNorm = array( [np.dot(addedZ, addedZ)] )

            K1 = np.hstack((currK, newinnerProduct))

            if newinnerProduct.shape[0] > 1:
                selfNorm = selfNorm.reshape((1,1))
            K2 = np.vstack( (K1, np.hstack((newinnerProduct.T, selfNorm))) )

            currK = K2
            if maxGradient <= 0:
                #newCurrOptw = np.vstack((currOptw[:], np.array([0])))
                newCurrOptw = np.append(currOptw, [0], axis=0)
                newCurrSetValue = currSetValue
            else:
                [newCurrOptw, value] = optimize(currK, curru, currOptw, maxGradient)
                newCurrSetValue = -value

        currOptw = newCurrOptw
        if type(currOptw) != np.ndarray:
            currOptw = np.array([currOptw])

        currSetValue = newCurrSetValue

        setValues[sizeS] = currSetValue
        sizeS = sizeS + 1

    return(currOptw, S, setValues)