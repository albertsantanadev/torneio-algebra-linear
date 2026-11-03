import numpy as np, sys
sys.path.insert(0,sys.argv[2]); from bench import *
D=sys.argv[1]
def hard(o,mb,k,it=800,center=False):
    mu=o[mb].mean() if center else 0.0
    Z=np.where(mb,o,o[mb].mean())
    for _ in range(it):
        U,s,Vt=np.linalg.svd(Z-mu,full_matrices=False)
        R=(U[:,:k]*s[:k])@Vt[:k]+mu
        Z=np.where(mb,o,R)
    return Z
print('mat  ' + '  '.join(f'{"k="+str(k):>7}' for k in range(1,9))+'   (sem centralizar, 800 it; NRMSE oculto)')
for i in range(1,11):
    g,m,o=L('gabarito',i),L('mascara',i).astype(bool),L('observada',i)
    r=[nrmse(g,hard(o,m,k),m)[0] for k in range(1,9)]
    print(f'{i:02d}   '+'  '.join(f'{x:7.3f}' for x in r))
