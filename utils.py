import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy import stats


def plot_series(X,motifs_times=None,motifs_series=None, window_size=0, k=None):
  m = window_size
  d,n = X.shape
  fig, axs = plt.subplots(d, sharex=True, figsize=(8,d))
  for j in range(d):
    axs[j].plot(X[j])
    if motifs_times is not None:
      if motifs_series is None :
        for j in range(d):
          for t in motifs_times:#[k-1]:
            axs[j].plot(range(t, t + m), X[j,t:t+m], c='r')
      elif j in motifs_series[k-1]:
        for t in motifs_times[k-1]:
          axs[j].plot(range(t, t + m), X[j,t:t+m], c='r')
  plt.show()


def score(detected_anomalies, true_anomalies, m):
    ''' given the real positions of anomalies, the detected positions and m (the 'precision of the detection') returns the number of true positive, false negative and false positive, precision and recall in that order in an array
    
    '''
    detected_anomalies.sort()
    true_anomalies.sort()
    FP = 0 #false positive : detected but not true
    TP = 0 #true positive
    FN = 0 #false negative
    i = 0 #curseur detected_anomalies
    j = 0 #curseur true_anomalies
    while i<detected_anomalies.size and j<true_anomalies.size:
        if detected_anomalies[i]+m<=true_anomalies[j]:
            i = i+1
            FP += 1
#        elif true_anomalies[j]+anomaly_size<detected_anomalies[i]:
        elif true_anomalies[j]+m<=detected_anomalies[i]: #on change pour prise en compte aussi PCA+AR
            j = j+1
            FN +=1
        else :
            j = j + 1
            i = i + 1
            TP = TP + 1
    if i==detected_anomalies.size and j<true_anomalies.size :
        FN += true_anomalies.size -j
    elif j==true_anomalies.size and i<detected_anomalies.size:
        FP += detected_anomalies.size - i
    if true_anomalies.size ==0 :
        recall=1
    else : recall = TP/(TP+FN)
    if detected_anomalies.size == 0:
        precision = 1
    else : precision = TP/(TP+FP)
    result = {'TP':TP, 'FN':FN, 'FP':FP, 'recall':recall, 'precision':precision }
    return np.array([TP, FN, FP, recall, precision])
        
