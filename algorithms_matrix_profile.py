import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
from scipy import stats


def mass(s,x,m_x,std_x, no_std): #normalised euclidean distance
  m = s.shape[0]
  n = x.shape[0]
  s_rev = np.zeros(n)
  s_rev[:m] = s[::-1]
  fftx = np.fft.fft(x)
  ffts = np.fft.fft(s_rev)
  prod = np.fft.ifft(fftx * ffts).real
  if no_std == False :
    dist = np.sqrt( np.maximum(2 * (m - (prod[m-1:n] - m*m_x*s.mean())/(std_x * s.std()) ) , 0))
  else :
    dist = np.sqrt( np.maximum( m*(std_x**2 + s.std()**2) - 2*(prod[m-1:n] - m*m_x*s.mean()) , 0))
  return dist

def mass2(s,x, x_square_sums): #standard euclidean`
    m = s.shape[0]
    n = x.shape[0]
    
    s_rev = np.zeros(n)
    s_rev[:m] = s[::-1]
    fftx = np.fft.fft(x)
    ffts = np.fft.fft(s_rev)
    prod = np.fft.ifft(fftx * ffts).real
    dist = np.sqrt( np.maximum( np.sum(s**2) + x_square_sums - 2 *prod[m-1:n] , 0))
    return dist

from scipy.spatial.distance import euclidean

#from fastdtw import fastdtw
#def DTW(s,x):
  
#  distance, path = fastdtw(x, s, dist=euclidean)
  


def distanceProfile(s,x,m_x,std_x, no_std=False):
  return mass(s,x,m_x,std_x, no_std)

def distanceProfile2(s,x): #version without FFT
  m=s.shape[0]
  n=x.shape[0]
  all_seqs = x[np.arange(n-m+1)[:,None] + np.arange(m)]
  all_seqs = all_seqs- all_seqs.mean(axis=1)[:,None]
  s = s-s.mean()
  distances = np.sqrt(((s-all_seqs)**2).sum(axis=1))
  return distances



# la bonne version
def computeMatrixProfile(X,m=120, take_max=False, prop_no_norm=0, only_euc = True):
  '''Computation of the matrix profile for a series X and a size of motif m
  Args :
    - X : multivariate time series, numpy array of shape (d,n), d number of dimensions and n length of the signal
    - take_max : if True it takes dimensions where distance profile is maximal (adapted for anomaly detection)
    - prop_no_norm : float between 0 and 1. Proportion of non normalised (but centered) euclidean distance
    - only_euc : if True takes regular euclidean distance for distanc eprofile and not normalised.
  Returns :
    - P : matrix profile
    - P_subspace : dimension used for matrix profile
    - execution time
  '''
  t0 = time.time()
  d,n = X.shape
  all_seqs = X[:,np.arange(n-m+1)[:,None] + np.arange(m)]
  if not only_euc:
    m_x = all_seqs.mean(axis=2)
    std_x = all_seqs.std(axis=2)
  if only_euc :
    X_bis = np.zeros((d, n+1))
    X_bis[: ,1:] = X
    cn = np.cumsum(X_bis**2, axis = 1)
    x_square_sums = cn[:,m :] - cn[:,:n-m+1]
  P = np.ones([d, n-m+1]) * np.inf
  P_subspace = [-np.ones([n-m+1,k]) for k in range(1,d+1)]   # dims used for the minimal distance
  times = np.arange(n-m+1)
  for t in times:
    #sys.stdout.write('\r'+str((100*t)//n)+'%')
    #sys.stdout.flush()
    D = np.zeros([d, n-m+1])
    for i in range(d):
      Q = X[i,t:t+m]
      if not only_euc :
        D[i] = distanceProfile(Q,X[i],m_x[i],std_x[i])
        D[i] = (1-prop_no_norm)*D[i] + prop_no_norm * distanceProfile(Q,X[i],m_x[i], std_x[i], True)
      if only_euc :
        D[i] = mass2(Q, X[i], x_square_sums[i])
      #D[i] = (1-prop_no_norm)*D[i] + prop_no_norm * distanceProfile2(Q,X[i])
      D[i][max(0,t-m//2):t+m//2] = np.inf   # Exclure une zone plus large peut-être (sequence//2)
      #D[i][t] = np.inf
    order = np.argsort(D, axis=0)
    if take_max:
      order = order[::-1]
    D = np.take_along_axis(D, order, axis=0)
    D_ = D.cumsum(axis=0)/np.arange(1,d+1).reshape(d,1)
    P = np.minimum(P, D_)
    P_subspace = [np.where((D_[i]==P[i]).reshape(-1,1),order[:i+1].T,P_subspace[i]) for i in range(d)]
  exec_time = time.time() - t0
  #sys.stdout.write('\r100%  -- Execution time = '+str(round(exec_time,2))+'s')
  return P, P_subspace, exec_time
  



def detect_motifs(P,P_subspace):
  d, _ = P.shape
  motifs_times = P.argsort(axis=1)[:,:2]
  motifs_times.sort(axis=1)
  motifs_series = [P_subspace[k][motifs_times[k,0]] for k in range(d)]
  return motifs_times, motifs_series

#def detect_anomaly(P, P_subspace, nb=1): ### A tester pour recomprendre !!!
#  '''
#  return maximal value of matrix profile
#  '''
#  d, _ = P.shape
#  anomaly_times = P.argsort(axis=1)[:,-nb:]
#  #anomaly_times.sort(axis=1)
#  anomaly_series = [P_subspace[k][anomaly_times[k,0]] for k in range(d)]
#  return anomaly_times, anomaly_series

def detect_anomaly(P, P_subspace,m=1, nb=1): ### A tester pour recomprendre !!!
    '''
    return maximal value of matrix profile
    '''
    d, n = P.shape
    P_transform = P.copy()
    ano_times = np.array([], dtype=np.int).reshape(d,0)
    for i in range(nb):
        anomaly_times = P_transform.argsort(axis=1)[:,-1:]
        ano_times = np.c_[ano_times, anomaly_times]
        for c in range(d):
            P_transform[c,np.arange(max( anomaly_times[c]-m,0),min( anomaly_times[c]+m, n))]=np.zeros(len (np.arange(max( anomaly_times[c]-m,0),min( anomaly_times[c]+m, n))))
    #anomaly_times.sort(axis=1)
    anomaly_series = [P_subspace[k][ano_times[k]] for k in range(d)]
    return ano_times, anomaly_series

def plot_series_motifs(X,P,motif_len,motifs_times,motifs_series=None,k_values=None, plot_frequency=1):
  '''Plot series and a selected motif in different color and the different matrix profile for each number of dimensions
  '''
  d,n = X.shape
  m = motif_len
  if k_values==None:
    k_values = np.arange(d)+1
  for k in k_values:
    if k%plot_frequency==0 or k==d:
        print(str(k)+'-dimensional motifs detection')
        plt.figure(figsize=(14, 2*(d+1)))
        for j in range(d):
            plt.subplot(d+1, 1, j + 1)
            plt.plot(X[j])
            for i, t in enumerate(motifs_times[k-1]):
              if motifs_series==None or (j in motifs_series[k-1][i]):
                plt.plot(range(t, t + m), X[j,t:t+m], c='r')
            plt.xlim(0,n)
        plt.subplot(d+1,1,d+1)
        plt.title(str(k)+'-dimensional Matrix Profile')
        plt.plot(P[k-1, :])
        for t in motifs_times[k-1]:
          plt.axvline(t,c='r')
        plt.xlim(0,n)
        plt.show()



def show_anomaly_values(P, best=False):
    
    d,_ = P.shape
    vals_max = np.zeros(d)
    for k in range(d):
        vals_max[k] = np.max(P[k])
    plt.plot(np.arange(1,d+1), vals_max,'o--',label='max(P_k)')
    if best :
        k = np.argmin(vals_max[1:]-vals_max[:-1])
        plt.plot(k+1,vals_max[k],'ro')
    plt.xlabel('k')
    plt.ylabel('max matrix profile value')
    plt.legend()
    plt.show()


def regression_coef(P):
    '''
    From matrix profile creates a series of slopes
    '''
    d, l= P.shape
    var = (d+1)*(2*d+1)/6 - (d+1)**2/4
    cov = np.mean(P*np.arange(1,d+1).reshape(d,1), axis=0)-np.mean(P, axis=0)*(d+1)/2
    return cov/var

    
def kept_dims(P, P_subspace, k):
    d, l = P.shape
    dims = np.zeros((d,l), int)
    P_sub_k = P_subspace[k-1].astype(int)
    for i in range(l):
        dims[:,i][P_sub_k.T[:,i]]=1
    return dims
    
def find_anomalies_from_mp(P, m, thresh=0.82):
    '''Finds anomalies using the matrix profile and a ratio of variance that should be smaller than a given threshold 'thresh'
    '''
    reg = regression_coef(P)
    old = reg.copy()
    nb=0
    while True:
        size = old.size
        if size<=m : break
        k = old.argmin()
        ano = old[max(k-m,0):min(k+m, size)]
        new = np.delete(old, np.arange(max(k-m,0),min(k+m+1, size)))
        rapp = np.var(new)/np.var(old)
        #print(rapp)
        if rapp<thresh:
            #print(nb)
            nb+=1
            old=new
        else : break

    ano = []
    reg2=reg.copy()
    size=reg2.size
    for i in range(nb):
        k = reg2.argmin()
        ano.append(k)
        reg2[np.maximum(k-m,0):np.minimum(k+m, size)]=np.inf
    return np.array(ano)

def mad(x) :
    med = np.median(x)
    return np.median(np.abs(x-med))
    
    
def find_anomalies_from_mp_mad(P, m, thresh=0.95):
    '''Finds anomalies using the matrix profile and a ratio of mad
    '''
    reg = regression_coef(P)
    old = reg.copy()
    nb=0
    while True:
        size = old.size
        if size<=m : break
        k = old.argmin()
        ano = old[max(k-m,0):min(k+m, size)]
        new = np.delete(old, np.arange(max(k-m,0),min(k+m, size)))
        rapp = mad(new)/mad(old)
        #print(rapp)
        if rapp<thresh:
            #print(nb)
            nb+=1
            old=new
        else : break

    ano = []
    reg2=reg.copy()
    size=reg2.size
    for i in range(nb):
        k = reg2.argmin()
        ano.append(k)
        reg2[np.maximum(k-m,0):np.minimum(k+m, size)]=np.inf
    return np.array(ano)
    

