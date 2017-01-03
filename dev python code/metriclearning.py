import numpy as np
from itertools import combinations

def suvrel(X, y, gamma=2, norm=None, distance=False):
    """
    Return: a metric tensor for the data
    X columns representing samples and lines dimentions
    y labels
    gamma is a float
    norm:{None,\"unity\",\"t-test\"}
    distance: {False, True} if True return a tuple (weights, D)
    where D is the distanca matrix of the data
    for the geometric approach method
    """

    classes = list(set(y))
    n_classes = len(classes)
    dim = X.shape[1]

    if norm is None or norm == "unity":
        mean_cl = np.zeros((n_classes, dim))
        for i, cl in enumerate(classes):
            mean_cl[i] = np.mean(X[y == cl], axis=0)

        smeans = np.zeros(dim)
        for i, j in combinations(range(n_classes), 2):
            smeans += (mean_cl[i] - mean_cl[j]) ** 2

        if gamma != 2:
            var_cl = np.zeros((n_classes, dim))
            for cl in classes:
                var_cl[cl] = np.var(X[y == cl], axis=0)
            svar = np.sum(var_cl, axis=0)
            weights = ((gamma - 2.) * svar
                        +  gamma /( n_classes - 1) * smeans)
        else:
            weights = smeans

        weights[weights < 0] = 0

        if norm is "unity":
            weights = weights / np.var(X, axis=0)

        if distance:
            return (weights / np.sqrt(np.sum(weights ** 2)),
                    squareform(pdist(X * np.sqrt(weights))))
        else:
            return weights / np.sqrt(np.sum(weights ** 2))

    elif norm == "t-test":
        if n_classes == 2:
            mean_cl = np.zeros((n_classes, dim))
            var_cl = np.zeros((n_classes, dim))
            for i, cl in enumerate(classes):
                mean_cl[i] = np.mean(X[y == cl], axis=0)
                var_cl[i] = np.var(X[y == cl], axis=0)

            for i, j in combinations(range(n_classes), 2):
                smeans = (mean_cl[i] - mean_cl[j]) ** 2
                #tnorm = (var_cl[i] / np.sum([y == classes[i]])
                         #+ var_cl[j] / np.sum([y == classes[j]]))

                # case with equal variance. Edited by Marcelo 21/10/13
                n1 = np.sum([y == classes[i]])
                n2 = np.sum([y == classes[j]])
                tnorm = ((n1 - 1) * var_cl[i] + (n2 - 1) * var_cl[j]) \
                    / (n1 + n2 - 2)
            if gamma != 2:
                svar = np.sum(var_cl, axis=0)
                weights = ((gamma - 2.) * svar
                            +  gamma /( n_classes - 1) * smeans)
            else:
                weights = smeans
            weights = weights / tnorm
            weights[weights < 0] = 0

            if distance:
                return (weights / np.sqrt(np.sum(weights ** 2)),
                        squareform(pdist(X * np.sqrt(weights))))
            else:
                return weights / np.sqrt(np.sum(weights ** 2))

        else:
            print ("error: for t-test normalization the number" +
                   " of classes must be equal 2")
            return None
    else:
        print ("error: norm options are None, \"unity\" and  \"t-test\"")
        return None
