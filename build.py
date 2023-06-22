import numpy as np

def build_adjacency(CMat, K):
    N = CMat.shape[0]
    print(N)
    CAbs = np.abs(CMat)
    print(CAbs)
    
    for i in range(N):
        c = CAbs[:, i]
        print(c)
        PSrt, PInd = np.sort(c)[::-1], np.argsort(c)[::-1]
        print(PSrt)
        print(PInd)
        print(np.abs(c[PInd[0]]))
        print(CAbs[:, i])
        CAbs[:,i] = CAbs[:,i]/ np.abs(c[PInd[0]])
        print(CAbs)
    
    print('CAbs')
    print(CAbs)
    CSym = CAbs + CAbs.T
    print('CSym')
    print(CSym)
    
    if K != 0:
        Srt, Ind = np.sort(CSym, axis=0)[::-1], np.argsort(CSym, axis=0)[::-1]
        CK = np.zeros((N, N))
        for i in range(N):
            for j in range(K):
                CK[Ind[j, i], i] = CSym[Ind[j, i], i] / CSym[Ind[0, i], i]
        CKSym = CK + CK.T
    else:
        CKSym = CSym
    
    return CKSym