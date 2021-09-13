import numpy as np
#import pandas as pd 
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import style  
import matplotlib as mpl


class SynthData: 
    def plot_n_first_sig(self, n=None):
        N  = self.S.shape[1]
        if n==None : 
            n = N
        colors = ['red', 'steelblue', 'orange', 'green'] * ((N-1)//4 + 1)
        for i, (sig, color) in enumerate(zip(self.S, colors),1):
            if i>n : 
                break
            plt.subplot(n, 1, i)
            plt.title(f"s{i}")
            plt.plot(sig, color=color)
        plt.show()
        return
        
    def plot_sig(self):
      d,n = self.S.shape
      fig, axs = plt.subplots(d, sharex=True, figsize=(8,d))
      for j in range(d):
        axs[j].plot(self.S[j])
        for t in self.where:
            axs[j].axvline(t,c='r', ls='--', alpha=0.5)
            axs[j].axvline(t+self.ano_size,c='r', ls='--', alpha=0.5)
      plt.show()
      

        
    def plot_n_sig_together(self, n=None, ano=False):
        N  = self.S.shape[0]
        if n==None : 
            n = N
        for i, sig in enumerate(self.S ,1):
            if i>n : 
                break
            #plt.subplot(n, 1, i)
            #plt.title(f"s={i}")
            plt.plot(sig, label=f"s{i}")
        if ano:
            plt.axvline(self.where,c='r')
            plt.axvline(self.where+self.ano_size,c='r')
            
        plt.legend()
        plt.show()   
        
class NoisySig(SynthData):
    def __init__(self, nb_ano=1, nb_dim=1, where=-1, type_ano='variance_big',  other_ano=False, sigma_noise=0.1, n_samples=400, ano_size=20,rng=np.random.RandomState()):
        '''
        Generates nb_dim series with gaussian noise and an anomaly
            - nb_ano:nb of series with an anomaly 
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano : if 'variance_small', change of the variance(decreases), if 'variance_big', variance increases, if 'type', not a gaussian noise anymore, if 'mean', the mean changes 
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
        '''
        self.n_samples = n_samples
        self.ano_size = ano_size
        if (where<0 or where>=n_samples):
            self.where = rng.choice(n_samples-ano_size+1)
            where= self.where
        else : self.where = where
        self.S = sigma_noise * rng.randn(nb_dim, n_samples)
        if type_ano=='variance_big':
            self.S[:nb_ano, where:where+ano_size] = 2 * sigma_noise * rng.randn(nb_ano, ano_size)
        if type_ano=='variance_small':
            self.S[:nb_ano, where:where+ano_size] =0.5 * sigma_noise * rng.randn(nb_ano, ano_size)
        if type_ano=='type': 
            self.S[:nb_ano, where:where+ano_size] = rng.exponential(
                scale=sigma_noise, size=(nb_ano, ano_size))-sigma_noise
        if type_ano=='mean':
            self.S[:nb_ano, where:where+ano_size] = sigma_noise * rng.randn(nb_ano, ano_size) - np.max(self.S[:nb_ano,:], axis=1)[:,None]/1
        if other_ano:
            t = rng.choice(n_samples-ano_size+1)
            self.S[rng.choice(nb_dim), t:t+ano_size] = 3 * sigma_noise * rng.randn(ano_size)
            print(t)
            
class ValueOutlier(SynthData):
    def __init__(self, nb_ano=1, nb_dim=1, where=-1, type_ano='global',  other_ano=False, sigma_noise=0.1, n_samples=400, ano_size=20, different_freq=False, rng=np.random.RandomState()):
        '''
        Generates nb_dim series with an anomaly which is an to big or too small ()
            - nb_ano:nb of series with an anomaly 
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano : 'global', 'contextual'
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
        '''
        self.n_samples = n_samples
        self.ano_size = ano_size
        if (where<0 or where>=n_samples):
            self.where = rng.choice(np.arange(1,n_samples-ano_size+1))
            where= self.where
        else : self.where = where
        self.S=np.zeros((nb_dim, n_samples))
        nb_periods = rng.choice(np.arange(3,11))
        x = np.linspace(0, nb_periods*2*np.pi, n_samples)
        for i in range(nb_dim):
            if different_freq:
                x = np.linspace(0, rng.choice(np.arange(3,11))*2*np.pi, n_samples)
            coef = (rng.rand(3)-1) *2
            freq = rng.choice(5,3)
            self.S[i] = coef[0] * np.sin(freq[0]*x) + coef[1] * np.cos(freq[1]*x) + coef[2]*  np.sin(freq[2]*x)
        
        self.S += sigma_noise * rng.randn(nb_dim, n_samples)
        a = rng.choice(2) #choix d'une valeur trop grande ou trop petite
        #print(a)
        if type_ano=='global':
            #print(np.min(self.S, axis=1))
            self.S[0:nb_ano, where] = a*(np.max(self.S, axis=1)[:nb_ano]+0.5*np.abs(np.max(self.S, axis=1)[:nb_ano]) ) + (1-a)*(np.min(self.S, axis=1)[:nb_ano]- 0.5*np.abs(np.min(self.S, axis=1)[:nb_ano]))
        
        elif type_ano=='contextual':
            for i in range(nb_ano):
                v_max, v_min = np.max(self.S[i]), np.min(self.S[i])
                #growth = self.S[i, where+1]-self.S[i, where-1]/2
                val = self.S[i,where]
                if v_max-val > val-v_min : 
                    self.S[i,where] = (2*v_max+val)/3
                else : self.S[i,where] = (2*v_min+val)/3
                    
                    
class MultipleOutliers(SynthData):
    def __init__(self, nb_ano=1, nb_ano_series=1, nb_dim=1, where=-1, type_ano='global', sigma_noise=0.1, n_samples=400,interval_between_ano=True, different_freq=False, ano_size=20, rng=np.random.RandomState()):
        '''
        Generates nb_dim series with nb_ano anomalies on at most nb_ano_series dimensions. The anomalies are a value that is grobally to large or to small or a contextual outlier
            - nb_ano: nb of anomalies, if -1 : random nb <= 1% n_samples
            - nb_ano_series:max nb of series for each anomaly
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano : 'global', 'contextual'
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
            - interval_between_ano : True if anomalies has a gap between them of at least ano_size
            - different_freq : if True, the fundamental frequence for different dimension is not the same, default=False
            - rng : Random generator
        '''
        if nb_ano==-1 : 
            nb_ano=rng.choice(n_samples//100+1)
        self.n_samples = n_samples
        self.ano_size = ano_size
        if where==-1:
            if interval_between_ano == False : 
                self.where = rng.choice(np.arange(0,n_samples), nb_ano, replace=False)
                where= self.where
            else : 
                where = []
                forbidden_values= []
                i=0
                while i<nb_ano:
                    new_ano = rng.choice (np.arange(int(n_samples)))
                    if new_ano not in forbidden_values :
                        where.append(new_ano)
                        forbidden_values.extend(np.arange(max(0, new_ano-ano_size), new_ano+ano_size+1))
                        i+=1
                where = np.array(where)
                self.where=where
        else : self.where = where
        self.S=np.zeros((nb_dim, n_samples))
        nb_periods = rng.choice(np.arange(3,11))
        x = np.linspace(0, nb_periods*2*np.pi, n_samples)
        for i in range(nb_dim):
            if different_freq:
                x = np.linspace(0, rng.choice(np.arange(3,11))*2*np.pi, n_samples)
            coef = (rng.rand(3)-1) *2
            freq = rng.choice(5,3)
            self.S[i] = coef[0] * np.sin(freq[0]*x) + coef[1] * np.cos(freq[1]*x) + coef[2]*  np.sin(freq[2]*x)
        
        self.S += sigma_noise*rng.randn(nb_dim, n_samples)
        for i, t in enumerate(where):
            series = rng.choice(nb_dim, rng.randint(1,nb_ano_series+1), replace=False)
            if type_ano=='global':
                a = rng.choice(2) #choix d'une valeur trop grande ou trop petite
                self.S[series, t] = a*(np.max(self.S, axis=1)[series]+0.5*np.abs(np.max(self.S, axis=1)[series]) ) + (1-a)*(np.min(self.S, axis=1)[series]- 0.5*np.abs(np.min(self.S, axis=1)[series]))
        
            elif type_ano=='contextual':
                for i in series:
                    v_max, v_min = np.max(self.S[i]), np.min(self.S[i])
                #growth = self.S[i, where+1]-self.S[i, where-1]/2
                    val = self.S[i,t]
                    if v_max-val > val-v_min : 
                        self.S[i,t] = (2*v_max+val)/3
                    else : self.S[i,t] = (2*v_min+val)/3
        
# 

class NoisySigMultipleAno(SynthData):
    def __init__(self, nb_ano=1, nb_ano_series=1, nb_dim=1, where=-1, type_ano='variance_big', sigma_noise=0.1, n_samples=400, ano_size=20,interval_between_ano=True, different_freq=None, rng=np.random.RandomState()):
        '''
        Generates nb_dim series with gaussian noise and nb_ano anomalies on at most nb_ano_series dimensions.
            - nb_ano: nb of anomalies, if -1 : random nb <= 1% n_samples
            - nb_ano_series:max nb of series for each anomaly
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano :  if 'variance_small', change of the variance(decreases), if 'variance_big', variance increases, if 'type', not a gaussian noise anymore, if 'mean', the mean changes
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
            - interval_between_ano : True if anomalies has a gap between them of at least ano_size
            - different_freq : if True, the fundamental frequence for different dimension is not the same, default=False
            - rng : Random generator
        '''
        if nb_ano==-1 : 
            nb_ano=rng.choice(n_samples//100+1)
        self.n_samples = n_samples
        self.ano_size = ano_size
        if where==-1:
            if interval_between_ano == False : 
                self.where = rng.choice(np.arange(0,n_samples-ano_size+1), nb_ano, replace=False)
                where= self.where
            else : 
                where = []
                forbidden_values= []
                i=0
                while (i<nb_ano):
                    new_ano = rng.choice (np.arange(int(n_samples-ano_size+1)))
                    if new_ano not in forbidden_values :
                        where.append(new_ano)
                        forbidden_values.extend(np.arange(max(0, new_ano-ano_size), new_ano+ano_size+1))
                        i+=1
                where = np.array(where)
                self.where=where
        else : self.where = where
        self.S = sigma_noise * rng.randn(nb_dim, n_samples)
        
        for i, t in enumerate(where):
            series = rng.choice(nb_dim, rng.randint(1,nb_ano_series+1), replace=False)
            if type_ano=='variance_big':
                self.S[series, t:t+ano_size] = 2 * sigma_noise * rng.randn(series.size, ano_size)
            if type_ano=='variance_small':
                self.S[series, t:t+ano_size] = 0.5 * sigma_noise * rng.randn(series.size, ano_size)
            if type_ano=='type':
                self.S[series, t:t+ano_size] = rng.exponential(
                scale=sigma_noise, size=(series.size, ano_size))-sigma_noise
            if type_ano=='mean':
                self.S[series, t:t+ano_size] = sigma_noise * rng.randn(series.size, ano_size) - np.max(self.S[series,:], axis=1)[:,None]/1


class MultipleUnknownSig(SynthData):
    def __init__(self, nb_ano=1, nb_ano_series=1, nb_dim=1, where=-1, type_ano='global',  sigma_noise=0.1, n_samples=400, ano_size=20, interval_between_ano=True, different_freq=False, rng=np.random.RandomState()):
        '''
        Generates nb_dim series with nb_ano anomalies on at most nb_ano_series dimensions. The anomalies are different motif different of the periodic signal
            - nb_ano: nb of anomalies, if -1 : random nb <= 1% n_samples
            - nb_ano_series:max nb of series for each anomaly
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano : not useful but used tokeep the same args as the other types of data
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
            - interval_between_ano : True if anomalies has a gap between them of at least ano_size
            - different_freq : if True, the fundamental frequence for different dimension is not the same, default=False
            - rng : Random generator
        '''
        if nb_ano==-1 : 
            nb_ano=rng.choice(n_samples//100+1)
        self.n_samples = n_samples
        self.ano_size = ano_size
        if where==-1:
            if interval_between_ano == False : 
                self.where = rng.choice(np.arange(0,n_samples-ano_size+1), nb_ano, replace=False)
                where= self.where
            else : 
                where = []
                forbidden_values= []
                i=0
                while i<nb_ano:
                    new_ano = rng.choice (np.arange(int(n_samples-ano_size+1)))#, dtype='uint32'))
                    if new_ano not in forbidden_values :
                        where.append(new_ano)
                        forbidden_values.extend(np.arange(max(0, new_ano-ano_size), new_ano+ano_size+1))
                        i+=1
                where = np.array(where)
                self.where=where
        else : self.where = where
        self.S=np.zeros((nb_dim, n_samples))
        nb_periods = rng.choice(np.arange(3,11))
        x = np.linspace(0, nb_periods*2*np.pi, n_samples)[:,None]
        if different_freq: 
            for i in range(nb_dim-1):
                nb_periods = rng.choice(np.arange(3,11))
                y = np.linspace(0, nb_periods*2*np.pi, n_samples)
                x = np.c_[x,y]
        else : 
            y = x.copy()
            for i in range(nb_dim-1):
                x = np.c_[x,y]
        coef = (rng.rand(3, nb_dim)-1)*2       
        freq = rng.choice(5,(3,nb_dim))
        self.S = (coef[0] * np.sin(freq[0]*x) + coef[1] * np.cos(freq[1]*x) + coef[2]*  np.sin(freq[2]*x)).T
        freq = freq.T
        x=x.T
        for i, t in enumerate(where):
            series = rng.choice(nb_dim, rng.randint(1,nb_ano_series+1), replace=False)
            #series = np.array([1,3])
            coef = (rng.rand(3)-1)*2  
            self.S[series, t:t+ano_size] = (coef[0] * np.cos(freq[series,0][:,None]*x[series]) + coef[1] * np.sin(freq[series,1][:,None] * x[series]) + coef[2]*  np.cos(freq[series,2][:,None]*x[series]))[:,:ano_size]

        self.S += sigma_noise*rng.randn(nb_dim, n_samples)                    
                    
class UnknownSig(SynthData):
    def __init__(self, nb_ano=1, nb_dim=1, where=-1, type_ano='global',  other_ano=False, sigma_noise=0.1, n_samples=400, ano_size=20, different_freq=False, rng=np.random.RandomState()):
        '''
        Generates nb_dim series with an anomaly which is an to big or too small ()
            - nb_ano:nb of series with an anomaly 
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano : 'global', 'contextual'
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
        '''
        self.n_samples = n_samples
        self.ano_size = ano_size
        if (where<0 or where>=n_samples):
            self.where = rng.choice(np.arange(1,n_samples-ano_size+1))
            where= self.where
        else : self.where = where
        self.S=np.zeros((nb_dim, n_samples))
        nb_periods = rng.choice(np.arange(3,11))
        x = np.linspace(0, nb_periods*2*np.pi, n_samples)
        for i in range(nb_dim):
            if different_freq:
                x = np.linspace(0, rng.choice(np.arange(3,11))*2*np.pi, n_samples)
            coef = (rng.rand(3)-1) *2
            freq = rng.choice(5,3)
            self.S[i] = coef[0] * np.sin(freq[0]*x) + coef[1] * np.cos(freq[1]*x) + coef[2]*  np.sin(freq[2]*x)
            if i<nb_ano:
                self.S[i, where:where+ano_size] = (coef[0] * np.cos(freq[0]*x) + coef[1] * np.sin(freq[1]*x) + coef[2]*  np.cos(freq[2]*x))[:ano_size] 
                
        
        self.S += sigma_noise*rng.randn(nb_dim, n_samples)
        a = rng.choice(2) #choix d'une valeur trop grande ou trop petite
        #print(a)
        
    
class Periodic3Noise(SynthData):
    def __init__(self, type_anomaly=None, nb_noisy=0, sigma_noise=0.03, rng=np.random.RandomState()):
        n_samples = 2000
        self.S=None
        if type_anomaly == None : 
            self.S = self.normal_sig(n_samples)        
        if type_anomaly == 'one_series_one_ano' : 
            self.S = self.anom1s1(n_samples)
        if type_anomaly == 'two_series_one_ano' : 
            self.S = self.anom1s1s2(n_samples)
        if type_anomaly == 'three_series_one_ano' : 
            self.S = self.anom1s1s2s3(n_samples)
        if type_anomaly == 'one_series_two_ano' : 
            self.S = self.anom2s1(n_samples)
        if type_anomaly == 'two_series_two_ano' : 
            self.S = self.anom2s1s2(n_samples)
        if type_anomaly == 'three_series_two_ano' : 
            self.S = self.anom2s1s2s3(n_samples)
        if type_anomaly == 'random1' : 
            self.S = self.anom_rand1(n_samples)
        if type_anomaly == 'random2' : 
            self.S = self.anom_rand2(n_samples)
        if type_anomaly == 'random3' : 
            self.S = self.anom_rand3(n_samples)   
        for i in range(nb_noisy) :
            self.S = np.c_[self.S, np.zeros(n_samples)]
        #self.S += sigma_noise * np.random.normal(size=self.S.shape)
        self.S += sigma_noise * rng.randn(n_samples, 3+nb_noisy)#.cumsum(axis=0)  
        self.S =self.S.T 
                
        
    def normal_sig(self, n_samples):
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
    
    def anom_rand1(self, n_samples, rng=np.random.RandomState()):
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[1200:1230] = 2*rng.randn(30)#anomaly
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
        
    def anom_rand2(self, n_samples, rng=np.random.RandomState()):
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[1200:1230] = 2*rng.randn(30)#anomaly
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s2[1200:1230] = 2*rng.randn(30)
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
    
    def anom_rand3(self, n_samples, rng=np.random.RandomState()):
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[1200:1230] = 2*rng.randn(30)#anomaly
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s2[1200:1230] = 2*rng.randn(30)
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        s3[1200:1230] = 2*rng.randn(30)
        S = np.c_[s1, s2, s3]
        return S
        
    def anom1s1(self, n_samples):
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sin(6 * time[1400:1550])#anomaly
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
        
    def anom1s1s2(self, n_samples) : 
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sin(6 * time[1400:1550])
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s2[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sign(
                    np.sin(6 * time[1400:1550]))
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
    
    def anom1s1s2s3(self, n_samples) : 
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sin(6 * time[1400:1550])
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s2[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sign(
                    np.sin(6 * time[1400:1550]))
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        s3[1400:1550] += 3 * np.sin(4 * time[1400:1550]) + 2 * np.sign(
                    np.sin(5 * time[1400:1550]))
        S = np.c_[s1, s2, s3]
        return S
        
    def anom2s1(self, n_samples):
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[350:500] += 2 * np.sin(4 * time[350:500]) + 2 * np.sin(6 * time[350:500])
        s1[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sin(6 * time[1400:1550])#anomaly
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
        
    def anom2s1s2(self, n_samples) : 
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[350:500] += 2 * np.sin(4 * time[350:500]) + 2 * np.sin(6 * time[350:500])
        s1[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sin(6 * time[1400:1550])
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s2[350:500] += 2 * np.sin(4 * time[350:500]) + 2 * np.sign(
                    np.sin(6 * time[350:500]))
        s2[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sign(
                    np.sin(6 * time[1400:1550]))
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        S = np.c_[s1, s2, s3]
        return S
    
    def anom2s1s2s3(self, n_samples) : 
        time = np.linspace(0, 24, n_samples)
        s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s1[350:500] += 2 * np.sin(4 * time[350:500]) + 2 * np.sin(6 * time[350:500])
        s1[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sin(6 * time[1400:1550])
        s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
        s2[350:500] += 2 * np.sin(4 * time[350:500]) + 2 * np.sign(
                    np.sin(6 * time[350:500]))
        s2[1400:1550] += 2 * np.sin(4 * time[1400:1550]) + 2 * np.sign(
                    np.sin(6 * time[1400:1550]))
        s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
        s3[350:500] += 3 * np.sin(4 * time[350:500]) + 2 * np.sign(
                    np.sin(5 * time[350:500]))
        s3[1400:1550] += 3 * np.sin(4 * time[1400:1550]) + 2 * np.sign(
                    np.sin(5 * time[1400:1550]))
        S = np.c_[s1, s2, s3]
        return S
    
    
class FewSig(SynthData):
    def __init__(self, type_anomaly=None, nb_noisy=0, sigma_noise=0.1, rng=np.random.RandomState()):
        n_samples = 2000
        time1 = np.linspace(0, 24, n_samples)
        time2 = np.linspace(0,30, n_samples)
        s1 = np.zeros(n_samples)
        #s1 = 0.3*np.random.randn(n_samples)
        s1[350:420] = 3 * np.sin(2 * time1[350:420])+2 * np.sin(4 * time1[350:420]) + (
        2 * np.sin(6 * time1[350:420]))
        s1[1400:1470] = 3 * np.sin(2 * time1[350:420]) + 2 * np.sin(4 * 
        time1[350:420])+ 2 * np.sin(6 * time1[350:420])
        s2 = np.zeros(n_samples) # Signal 2 : square signal
        s2[350:420] = np.sign(np.sin(2 * time1[350:420]))+ 2 * np.sin(4 * time1[350:420]) + 2 * np.sign(
                    np.sin(6 * time1[350:420]))
        s2[1400:1470] = np.sign(np.sin(2 * time1[350:420])) + 2 * np.sin(4 * time1[350:420]) + 2 * np.sign(
                    np.sin(6 * time1[350:420]))
        s3 = 2 * signal.sawtooth(4 * time2) + np.sin(2 * time2)  # Signal 3: saw tooth signal
        
        if type_anomaly == 'motif':
            s1[1200:1250] =  3 * np.sin(2 * time2[1200:1250])+4 * np.sin(3.1 * time2[1200:1250]) +(
                        2 * np.sin(8 * time2[1200:1250]))
            s2[1200:1250] =  3 * np.sin(2 * time2[1200:1250])+2 * np.sign(np.sin(
                        3.1 *time2[1200:1250])) + 3 * np.sin(7 * time2[1200:1250])
        if type_anomaly == 'random' or type_anomaly=='random_all' : 
            s1[1200:1250] = rng.randn(50)
            s2[1200:1250] = rng.randn(50)
                
        self.S = np.c_[s1, s2, s3]
        
        for i in range(nb_noisy) :
            s = np.zeros(n_samples)
            if type_anomaly=='random_all':
                s[1200:1250] = rng.randn(50)
            self.S = np.c_[self.S, s]
            #self.S = np.c_[self.S, 0.1*np.random.randn(n_samples).cumsum(axis=0)]
        #self.S += sigma_noise * np.random.normal(size=self.S.shape)
        self.S += sigma_noise*rng.randn(n_samples, 3+nb_noisy)#.cumsum(axis=0)
        #self.S[:,3:] += 0.01*np.random.randn(n_samples, nb_noisy).cumsum(axis=0)
        #self.S[:,:3] += 0.001*np.random.randn(n_samples, 3).cumsum(axis=0)
        self.S = self.S.T
# np.random.seed(0)
# n_samples = 2000
# time = np.linspace(0, 24, n_samples)
# 
# s1 = 3 * np.sin(2 * time)  # Signal 1 : sinusoidal signal
# s2 = np.sign(np.sin(2 * time))  # Signal 2 : square signal
# s3 = 2 * signal.sawtooth(4 * time)  # Signal 3: saw tooth signal
# s4 = np.zeros(n_samples)
# 
# S = np.c_[s1, s2, s3, s4]
# S += 0.2 * np.random.normal(size=S.shape)  # Add noise
# 
# data = SynthData()
# data.S = S 
# data.plot_n_sig_together(5)


class RepeatedCorrelatedMotifs(SynthData):
    def __init__(self, nb_ano=1, nb_ano_series=1, nb_dim=1, where=-1, type_ano='global',  sigma_noise=0.1, n_samples=400, ano_size=20, interval_between_ano=True, different_freq=False, rng=np.random.RandomState()):
        '''
        Motifs that are repeated in the series and always correlated except when there is anomaly
            - nb_ano: nb of anomalies, if -1 : random nb <= 1% n_samples
            - nb_ano_series:max nb of series for each anomaly
            - nb_dim: nb of dimensions
            - where : position of the anomalies, default= -1 (random)
            - type_ano : not useful but used tokeep the same args as the other types of data
            - sigma_noise: std of signal
            - n_samples: size of signal
            - ano_size: size of anomaly
            - interval_between_ano : not useful but used tokeep the same args as the other types of data
            - different_freq : not useful but used tokeep the same args as the other types of data
            - rng : Random generator
        '''
        
        nb_motifs = int(np.ceil(n_samples/(2*ano_size)))
        pos = []
        cuts = np.linspace(-ano_size, n_samples-ano_size, nb_motifs+1, dtype = np.int)
        last_pos = -ano_size-1
        for i in range(nb_motifs):
            last_pos = rng.randint(low = last_pos+1+ano_size, high = cuts[i+1]+1)
            pos.append(last_pos)
        self.where = rng.choice(pos)
        motif = np.cumsum( rng.randn(nb_dim, ano_size), axis =1)
        #print(motif.shape)
        self.S=np.zeros((nb_dim, n_samples))
        for i, p in enumerate(pos):
            self.S[:,p:p+ano_size] = rng.choice([-1,1])*motif
        self.S += sigma_noise*rng.randn(nb_dim, n_samples)
        if nb_ano==-1 :
            nb_ano=rng.choice(nb_motifs//5)
        self.n_samples = n_samples
        self.ano_size = ano_size

        self.where = rng.choice(pos, nb_ano)

        for i, t in enumerate(self.where):
            series = rng.choice(nb_dim, rng.randint(1,nb_ano_series+1), replace=False) 
            self.S[series, t:t+ano_size] = -self.S[series, t:t+ano_size]

       
