import numpy as np, sys
D=sys.argv[1]
def L(n,i): return np.load(f'{D}/{n}_{i:02d}.npy')

def nrmse(g,pred,mb):
    h=~mb; e=np.sqrt(np.mean((pred[h]-g[h])**2))
    return e/g[h].std(), e/g.std()   # (std oculto, std total)

def b_global(o,mb):
    s=o.copy(); s[~mb]=o[mb].mean(); return s
def b_row(o,mb):
    s=o.copy(); gm=o[mb].mean()
    for i in range(o.shape[0]):
        k=mb[i]; s[i,~k]=o[i,k].mean() if k.any() else gm
    return s
def b_col(o,mb):
    s=o.copy(); gm=o[mb].mean()
    for j in range(o.shape[1]):
        k=mb[:,j]; s[~k,j]=o[k,j].mean() if k.any() else gm
    return s
def b_add(o,mb):  # modelo aditivo mu + a_i + b_j por minimos quadrados
    n,m=o.shape; I,J=np.where(mb)
    A=np.zeros((len(I),1+n+m)); A[:,0]=1; A[np.arange(len(I)),1+I]=1; A[np.arange(len(I)),1+n+J]=1
    c=np.linalg.lstsq(A,o[I,J],rcond=None)[0]
    P=c[0]+c[1:1+n][:,None]+c[1+n:][None,:]
    s=o.copy(); s[~mb]=P[~mb]; return s
def svd_impute(o,mb,k,it=300,lam=0.0,center=True):
    X=np.where(mb,o,0.0); mu=o[mb].mean() if center else 0
    Z=np.where(mb,o,mu)
    for _ in range(it):
        U,s,Vt=np.linalg.svd(Z-mu,full_matrices=False)
        s=np.maximum(s[:k]-lam,0)
        R=(U[:,:k]*s)@Vt[:k]+mu
        Z=np.where(mb,o,R)
    return Z
def best_k_cv(o,mb,ks,rng,reps=6,frac=0.15):
    I,J=np.where(mb); err={k:0 for k in ks}
    for _ in range(reps):
        sel=rng.choice(len(I),max(1,int(frac*len(I))),replace=False)
        m2=mb.copy(); m2[I[sel],J[sel]]=False
        o2=np.where(m2,o,np.nan)
        for k in ks:
            P=svd_impute(o2,m2,k,it=120)
            err[k]+=np.mean((P[I[sel],J[sel]]-o[I[sel],J[sel]])**2)
    return min(err,key=err.get),err
if __name__=='__main__':
    rng=np.random.default_rng(0)
    meths={'global':b_global,'linha':b_row,'coluna':b_col,'aditivo':b_add}
    rows=[]
    print('mat  '+'  '.join(f'{k:>8}' for k in list(meths)+['svd k=1','svd k=2','svd k=3','svd CV','k_cv']))
    for i in range(1,11):
        g,m,o=L('gabarito',i),L('mascara',i).astype(bool),L('observada',i)
        r=[nrmse(g,f(o,m),m)[0] for f in meths.values()]
        for k in (1,2,3): r.append(nrmse(g,svd_impute(o,m,k),m)[0])
        kk=min(g.shape)-1
        kb,_=best_k_cv(o,m,list(range(1,min(8,kk)+1)),rng)
        r.append(nrmse(g,svd_impute(o,m,kb),m)[0])
        rows.append(r)
        print(f'{i:02d}   '+'  '.join(f'{x:8.3f}' for x in r)+f'  {kb:8d}')
    print('MED  '+'  '.join(f'{x:8.3f}' for x in np.mean(rows,0)))
