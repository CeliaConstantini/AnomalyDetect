import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sporco.dictlrn import cbpdndl
from sporco.admm import cbpdn
from sporco import util
#from sporco import signal
from sporco import plot
import sporco.linalg as sl  


 
PENALTY = 3

def plot_CDL(signal, Z, D, figsize=(15, 10), rapp = 10):
    """Plot the learned dictionary `D` and the associated sparse codes `Z`.

    `signal` is an univariate signal of shape (n_samples,) or (n_samples, 1).
    """

    (atom_length,n_dims, n_atoms) = np.shape(D)
    plt.figure(figsize=figsize)
    for i in range(n_dims):
        plt.subplot((n_atoms + 1)*n_dims , rapp,(rapp*i+2 , rapp*(i+1))) 
        plt.plot(signal[:,i])

    for i in range(n_atoms):
        #axsDict = subfigs[2*(i+1)].subplots(n_dims, 1)
        for j in range(n_dims):
            plt.subplot((n_atoms + 1)*n_dims, rapp,rapp*(n_dims*i + j+n_dims)+1)
            plt.plot(D[:,j,i])
            plt.xlim(-atom_length//5, atom_length+atom_length//5)
        #subfigs[2*(i+1)+1].plot(Z[:,i])
        plt.subplot(n_atoms + 1, rapp, (rapp*(i+1)+2, rapp*(i+2)))
        plt.plot(Z[:,i])
        plt.ylim((np.min(Z), np.max(Z)))
        
def get_lbd_max(dictionary, sig): 
    '''Returns lambda_max as the sum of the lambda_max for each channel (n_channels = n_dims)
    '''
    atom_length, n_dim ,n_atoms = dictionary.shape
    sig_length = sig.shape[0]
    lbd = 0
    for i in range(n_dim):
        D = None
        for j in range(n_atoms):
            Dj = scipy.linalg.circulant(np.concatenate([dictionary[:,i,j],np.zeros(sig_length-atom_length)]))
            if D is None : 
                D = Dj
            else : 
                D = np.c_[D,Dj]
        lbd+= np.max(sig[:,i].T@D)
    return lbd * atom_length


# options for the dictionary learning and sparse coding procedures
def get_opt_dl(penalty=PENALTY, iter=150):
    """Return the option class for the dictionary learning"""
    return cbpdndl.ConvBPDNDictLearn.Options(
        {
            "Verbose": False,
            "MaxMainIter": iter,
            #"CBPDN": {"rho": 50.0 * penalty + 0.5, "NonNegCoef": True},
            #"CCMOD": {"rho": 10.0},
        },
        dmethod="cns",
    ) 
        
def get_anomalies_from_dict(dictLearn, thresh) : 
    '''
    Returns anomalies as the activation times of atoms that are activated less than thresh * n_Samples/atom_length
    '''

    activ = dictLearn.getcoef().squeeze()
    atom_length = dictLearn.getdict().squeeze().shape[0]
    anom = []
    n_samples,n_atoms = activ.shape
    already_det = []
    rec = reconstruct(dictLearn)
    norm_rec = np.zeros(n_atoms)
    n_activated_atoms = np.sum(activ.sum(axis=0)!=0)
    s = 0
    for i in range(n_atoms) : 
        norm_rec[i] = np.linalg.norm(rec[:,:,0,i])
        s+=norm_rec[i]
    for i in range(n_atoms):
        if norm_rec[i]<0.2 * s / n_activated_atoms: 
            continue
        if np.sum(activ[:,i]!=0)<= thresh * n_samples/atom_length : 
            for activation in np.where(activ[:,i]!=0)[0] : 
                if activation not in already_det : 
                    anom.append(activation)
                    already_det.extend(np.arange(max(0, activation-atom_length), activation+atom_length+1))
    return np.array(anom)

def ConvDictLearning(S, atom_length, n_atoms,rng) :
    S = (S-S.mean(axis = 1)[:,None])/S.std(axis = 1)[:,None]
    sig = S.T
    atom_dictionary = rng.randn(atom_length, sig.shape[1],n_atoms)
    atom_dictionary = atom_dictionary / np.linalg.norm(atom_dictionary, axis=(0,1))
    penalty_frac = 0.015
    
    for k in range(2):
        lbd_max = get_lbd_max(atom_dictionary, sig)
        penalty = penalty_frac * lbd_max
        opt_dl = get_opt_dl(penalty=penalty)
        #print(lbd_max)
    # Dictionary learning and sparse coding
        dict_learning = cbpdndl.ConvBPDNDictLearn(
            D0 = atom_dictionary,
            S=sig,  # signal at hand
            lmbda= penalty,  # sparsity penalty
            opt=opt_dl,  # options for the optimizations
            xmethod="admm",  # optimization method (sparse coding)
            dmethod="cns",  # optimization method (dict learnin)
            dimK = 0,
            dimN = 1,
        )
        atom_dictionary = dict_learning.solve().squeeze()

    return dict_learning
    
def ConvDictLearning_GS(S, atom_length, n_atoms,penalties=None, thresh_inf=0, rng=np.random.RandomState()) :
    '''
    Returns the best dictionary among the ones with different penalties (the one that activates the less among the ones that activates enough (more than thesh_inf))
    '''
    S = (S-S.mean(axis = 1)[:,None])/S.std(axis = 1)[:,None]
    sig = S.T
    if type(penalties)== type(None) :
        penalties = np.logspace(-1, 2,5)
    nb_activations = np.zeros(len(penalties))
    for i,pen in enumerate(penalties) : 
        atom_dictionary = rng.randn(atom_length, sig.shape[1],n_atoms)
        atom_dictionary = atom_dictionary / np.linalg.norm(atom_dictionary, axis=(0,1))
        #penalty = penalty_frac * lbd_max
        opt_dl = get_opt_dl(penalty=pen)
    # Dictionary learning and sparse coding
        dict_learning = cbpdndl.ConvBPDNDictLearn(
            D0 = atom_dictionary,
            S=sig,  # signal at hand
            lmbda= pen,  # sparsity penalty
            opt=opt_dl,  # options for the optimizations
            xmethod="admm",  # optimization method (sparse coding)
            dmethod="cns",  # optimization method (dict learnin)
            dimK = 0,
            dimN = 1,
        )
        atom_dictionary = dict_learning.solve().squeeze()
        #dict_val = dict_value(dict_learning)
        activ = dict_learning.getcoef().squeeze()
        if np.sum(activ!=0)<=thresh_inf: 
            nb_activations[i]=sig.shape[0]
        else : 
            nb_activations[i]=np.max(np.sum(activ!=0, axis = 0))
    pen_index = int(np.argmin(nb_activations[nb_activations!=0]))
    atom_dictionary = rng.randn(atom_length, sig.shape[1],n_atoms)
    atom_dictionary = atom_dictionary / np.linalg.norm(atom_dictionary, axis=(0,1))
    #penalty = penalty_frac * lbd_max
    opt_dl = get_opt_dl(penalty=penalties[pen_index])
# Dictionary learning and sparse coding
    dict_learning = cbpdndl.ConvBPDNDictLearn(
        D0 = atom_dictionary,
        S=sig,  # signal at hand
        lmbda= penalties[pen_index],  # sparsity penalty
        opt=opt_dl,  # options for the optimizations
        xmethod="admm",  # optimization method (sparse coding)
        dmethod="cns",  # optimization method (dict learnin)
        dimK = 0,
        dimN = 1,
    )
    atom_dictionary = dict_learning.solve().squeeze()
    return dict_learning, penalties[pen_index]


def reconstruct(DictLearn, D=None, X=None):
    """Reconstruct representation, atom by atom."""

    if D is None:
        D = DictLearn.getdict(crop=False)
    if X is None:
        X = DictLearn.getcoef()
    Df = sl.rfftn(D, DictLearn.xstep.cri.Nv, DictLearn.xstep.cri.axisN)
    Xf = sl.rfftn(X, DictLearn.xstep.cri.Nv, DictLearn.xstep.cri.axisN)
    DXf = Df*Xf
    return sl.irfftn(DXf, DictLearn.xstep.cri.Nv, DictLearn.xstep.cri.axisN)

