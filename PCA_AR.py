import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.decomposition import PCA
import scipy.stats as sps
from statsmodels.tsa.ar_model import AutoReg


def sum_square_resids_PCA_AR(X, variance, order):
    '''Returns the sum of the square of residuals of the prediction of X_new with an AR model of order 'order' with X_new the transformation of X after a PCA of X keeping only 'variance' variance.
    '''
    X = (X-X.mean(axis = 1)[:,None])/X.std(axis = 1)[:,None]
    d, n = X.shape
    pca = PCA(n_components=variance, whiten=True, svd_solver='full')
    pca.fit(X.T)
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    #print(pca.components_)
    X_new = pca.transform(X.T).T
    
    k, _= X_new.shape
    s = []
    resids = np.zeros((k,n - order))
    fittedvalues = np.zeros((k, n - order))
    for i in range(k):
        s.append(AutoReg(X_new[i], lags=order, old_names=False).fit())
        resids[i] = s[i].resid
        fittedvalues[i] = s[i].fittedvalues
#s0 = AutoReg(X_new[0], lags=15).fit()
#s1 = AutoReg(X_new[1], lags=15).fit()
    resids = (resids-resids.mean(axis = 1)[:,None])/resids.std(axis = 1)[:,None]
    #plot_series(resids)
    #plt.plot(np.sum(resids**2, axis = 0))
    square_resids = np.sum(resids**2, axis = 0)
    return square_resids, k 
    
def plot_hist_resids_chi2(square_resids, k, quantile_max = 0.9999):
    fig, ax = plt.subplots(figsize=(15,5))
    _ = ax.hist(square_resids, 100, density=True)
    
    thresh = sps.chi2.ppf(quantile_max, df=k)
    _ = ax.axvline(thresh, ls="--", color="g")
    ax.plot(np.linspace(0,350,10000), sps.chi2.pdf(x=np.linspace(0,350,10000),df = k),'r--')


def all_anomalous_points(square_resids,k, quantile = 0.9999, order=20):
    thresh = sps.chi2.ppf(quantile, df=k)
    return np.where(square_resids > thresh)[0] + order
    
def plot_anomalous_points(square_resids,anomalous_points, real_anomalies,order):

    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(
        np.arange(order, order+square_resids.size),square_resids, label="squared residuals"
    )
    #outlier_mask = (square_resids > thresh)

    ax.plot(
        np.arange(square_resids.size+order)[anomalous_points],
        square_resids[anomalous_points-order],
        "o",
        label="Detected outliers",
    )
    ax.plot(real_anomalies,square_resids[real_anomalies-order],'gx', label = 'True outliers')
    plt.legend()

def find_anomalies(X, variance, order, quantile= 0.9999, gap=3):
    '''Returns the anomalies considering a small gap of size 'gap' between two anomalies
    Args :
        - X : multivariate time series
        - variance : proportion of variance we keep in PCA
        - order : order of model AR
        - quantile : quantile of chi_2 to be an anomaly
    '''
    square_resids, k = sum_square_resids_PCA_AR(X,variance,order)
    anomalous_points = all_anomalous_points(square_resids, k, quantile)
    #
    diff_ano = anomalous_points[1:]-anomalous_points[:-1]
    breakpoints = np.where(diff_ano>gap)[0]+1
    breakpoints = np.append(breakpoints, anomalous_points.size)
    start = 0
    anos = []
    for point in breakpoints : 
        pos = np.argmax(square_resids[anomalous_points[start:point]-order])
        anos.append(anomalous_points[start:point][pos])
        start = point
    return np.array(anos)


def find_anomalies_order(X, variance, order, quantile=0.9999):
    '''Returns the anomalies keeping only the ones that have an interval of at least order between them
    Args :
        - X : multivariate time series
        - variance : proportion of variance we keep in PCA
        - order : order of model AR
        - quantile : quantile of chi_2 to be an anomaly
    Returns :
        -anomalies : the position of anomalies sequences/points
        - anomaly_scores
    '''
    square_resids, k = sum_square_resids_PCA_AR(X,variance,order)
    anomalous_points = all_anomalous_points(square_resids, k, quantile)
    if anomalous_points.size == 0:
        return anomalous_points, []
    anos = []
    while anomalous_points.size != 0 :
        pos = np.argmax(square_resids[anomalous_points-order])
        new_ano = anomalous_points[pos]
        anos.append(new_ano)
        anomalous_points = anomalous_points[np.logical_not(
                np.all(np.c_[anomalous_points>=new_ano-order ,  anomalous_points<= new_ano+order], axis =1 ))]
    return np.array(anos), sps.chi2.cdf(square_resids[np.array(anos)-order], k)
