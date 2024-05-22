# -*- coding: utf-8 -*-

#main classes of FitTransit package
#version 0.1.2
#update: 22.5.2024
# (c) Pavol Gajdos, 2022-2024

from time import time
import sys
import os
import threading
import warnings

import pickle
import json

#import matplotlib
try:
    import matplotlib.pyplot as mpl
    fig=mpl.figure()
    mpl.close(fig)
except:
    #import on server without graphic output
    try: mpl.switch_backend('Agg')
    except:
        import matplotlib
        matplotlib.use('Agg',force=True)
        import matplotlib.pyplot as mpl

from matplotlib import gridspec
import matplotlib.ticker as mtick
#mpl.style.use('classic')   #classic style (optional)

import numpy as np

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.special import voigt_profile

try: import emcee
except ModuleNotFoundError: warnings.warn('Module emcee not found! Using FitMC will not be possible!')


import batman

from .ga import TPopul
from .info_ga import InfoGA as InfoGAClass
from .info_mc import InfoMC as InfoMCClass

#some constants
au=149597870700 #astronomical unit in meters
c=299792458     #velocity of light in meters per second
day=86400.    #number of seconds in day
minutes=1440. #number of minutes in day
year=365.2425   #days in year
rSun=695700000   #radius of Sun in meters
rJup=69911000     #radius of Jupiter in meters
rEarth=6371000    #radius of Earth in meters


class _Prior(object):
    '''set uniform prior with limits'''
    def _uniformLimit(self, **kwargs):
        if kwargs["upper"] < kwargs["lower"]:
            raise ValueError('Upper limit needs to be larger than lower! Correct limits of parameter "'+kwargs["name"]+'"!')
        p = np.log(1.0 / (kwargs["upper"] - kwargs["lower"]))

        def unilimit(ps, n, **rest):
            if (ps[n] >= kwargs["lower"]) and (ps[n] <= kwargs["upper"]):
                return p
            else: return -np.Inf
        return unilimit

    def __call__(self, *args, **kwargs):
        return self._callDelegator(*args, **kwargs)

    def __init__(self, lnp, **kwargs):
        self._callDelegator = self._uniformLimit(**kwargs)

class _NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

def mag2flux(mag,err=None,mag0=0):
    '''convert magnitude to "normalized" flux
    mag - star magnitude (numpy.array or float)
    err - errors/uncertanties (same format as mag)
    mag0 - standard level of star magnitude (for flux=1)
    '''

    flux=10**(-(mag-mag0)/2.5)
    if err is None: return flux

    err=np.log(10)/2.5*err*flux
    return flux,err


def flux2mag(flux,err=None,mag0=0):
    '''convert normalized flux to magnitude
    flux - normalized flux (numpy.array or float)
    err - errors/uncertanties (same format as flux)
    mag0 - standard level of star magnitude (for flux=1)
    '''

    mag=mag0-2.5*np.log10(flux)
    if err is None: return mag

    err=2.5/np.log(10)*err/flux
    return mag,err



class FitTransit():
    '''class for fitting transits'''
    availableModels=['TransitUniform','TransitLinear','TransitQuadratic','TransitSquareRoot',
                     'TransitLogarithmic','TransitExponential','TransitPower2','TransitNonlinear',
                     'Gauss','GaussModif','Lorentz','Voigt','Quad','Poly4','Mikulasek','Binomial']   #list of available models

    def __init__(self,t,flux,err=None):
        '''loading times, fluxes, (errors)'''
        self.t=np.array(t)
        self.flux=np.array(flux)
        if err is None:
            #if unknown (not given) errors of data
            #note: should cause wrong performance of fitting using MC, rather use function CalcErr for obtained errors after GA fitting
            self.err=0.1*np.ones(self.t.shape)
            self._set_err=False
            warnings.warn('Not given reliable errors of input data should cause wrong performance of fitting using MC! Use function CalcErr for obtained errors after GA fitting.')
        else:
            #errors given
            self.err=np.array(err)
            self._set_err=True

        #sorting data...
        self._order=np.argsort(self.t)
        self.t=self.t[self._order]    #times
        self.flux=self.flux[self._order]  #fluxes
        self.err=self.err[self._order]   #errors

        self.limits={}          #limits of parameters for fitting
        self.steps={}           #steps (width of normal distibution) of parameters for fitting
        self.params={}          #values of parameters, fixed values have to be set here
        self.params_err={}      #errors of fitted parameters
        self.paramsMore={}      #values of parameters calculated from model params
        self.paramsMore_err={}  #errors of calculated parameters
        self.fit_params=[]      #list of fitted parameters
        self.systemParams={}    #additional parameters of the system (R+errors)
        self._calc_err=False    #errors were calculated
        self._corr_err=False    #errors were corrected
        self._old_err=[]        #given errors
        self.model='TransitQuadratic'  #used model
        self._t0P=[]            #linear ephemeris
        self.phase=[]           #phases
        self._fit=''            #used algorithm for fitting (GA/DE/MCMC)


    def AvailableModels(self):
        '''print available models for fitting O-Cs'''
        print('Available Models:')
        for s in sorted(self.availableModels): print(s)

    def ModelParams(self,model=None,allModels=False):
        '''display parameters of model'''

        def Display(model):
            s=model+': '
            if 'Transit' in model:
                s+='t0, P, Rp, a, i, e, w, '
                if 'Uniform' not in model:
                    s+='c1, '
                    if ('Linear' not in model) and ('Power2' not in model):
                        s+='c2, '
                        if 'Nonlinear' in model: s+='c3, c4, '
            if 'Gauss' in model or 'Lorentz' in model or 'Quad' in model: s+='A, tC, w, '
            if 'Modif' in model: s+='k, '
            if 'Poly4' in model: s+='A, w2, w4, '
            if 'Voigt' in model: s+='A, tC, wG, wL, '
            if 'Mikulasek' in model: s+='A, C, K, tC, w, g, '
            if 'Binomial' in model: s+='A, tC, T, k1, k2, '
            s+='p0, p1, p2'
            print(s)

        if model is None: model=self.model
        if allModels:
            for m in sorted(self.availableModels): Display(m)
        else: Display(model)


    def Save(self,path,format='json'):
        '''saving data, model, parameters... to file in JSON or using PICKLE (format="json" or "pickle")'''
        data={}
        data['t']=self.t
        data['flux']=self.flux
        data['err']=self.err
        data['order']=self._order
        data['set_err']=self._set_err
        data['calc_err']=self._calc_err
        data['corr_err']=self._corr_err
        data['old_err']=self._old_err
        data['limits']=self.limits
        data['steps']=self.steps
        data['params']=self.params
        data['params_err']=self.params_err
        data['paramsMore']=self.paramsMore
        data['paramsMore_err']=self.paramsMore_err
        data['fit_params']=self.fit_params
        data['model']=self.model
        data['t0P']=self._t0P
        data['phase']=self.phase
        data['fit']=self._fit
        data['system']=self.systemParams

        path=path.replace('\\','/')   #change dirs in path (for Windows)
        if path.rfind('.')<=path.rfind('/'): path+='.json'   #without extesion

        if format=='pickle':
            f=open(path,'wb')
            pickle.dump(data,f,protocol=2)
            f.close()
        elif format=='json':
            f=open(path,'w')
            json.dump(data,f,cls=_NumpyEncoder)
            f.close()
        else: raise Exception('Unknown file format '+format+'! Use "json" or "pickle".')
        f.close()

    def Load(self,path):
        '''loading data, model, parameters... from file'''
        path=path.replace('\\','/')   #change dirs in path (for Windows)
        if path.rfind('.')<=path.rfind('/'): path+='.json'   #without extesion
        f=open(path,'rb')  #detect if file is json or pickle
        x=f.read(1)
        f.close()

        f=open(path,'rb')
        if x==b'{': data=json.load(f)
        else: data=pickle.load(f,encoding='latin1')
        f.close()

        self.t=np.array(data['t'])
        self.flux=np.array(data['flux'])
        self.err=np.array(data['err'])
        self._order=np.array(data['order'])
        self._set_err=data['set_err']
        self._corr_err=data['corr_err']
        self._calc_err=data['calc_err']
        self._old_err=np.array(data['old_err'])
        self.limits=data['limits']
        self.steps=data['steps']
        self.params=data['params']
        self.params_err=data['params_err']
        self.paramsMore=data['paramsMore']
        self.paramsMore_err=data['paramsMore_err']
        self.fit_params=data['fit_params']
        self.model=data['model']
        self._t0P=data['t0P']
        self.phase=np.array(data['phase'])

        if 'fit' in data: self._fit=data['fit']
        elif len(self.params_err)==0: self._fit='GA'
        else: self._fit='MCMC'

        if 'system' in data: self.systemParams=data['system']
        else: self.systemParams={}

    def Phase(self,t0,P,t=None):
        '''convert time to phase'''
        if t is None: t=self.t
        self._t0P=[t0,P]

        E_obs=(t-t0)/P  #observed epoch
        f_obs=E_obs-np.round(E_obs)  #observed phase

        if t is self.t: self.phase=f_obs

        return f_obs

    def Normalise(self,part,reverse=False,rewrite=True,plot=False):
        '''normalise lightcurve with 2th-order polynom using "part" of LC given by start and stop time
        reverse - don't use "part" of LC, use rest of it
        rewrite - replace original LC
        plot - show fit
        output coeffs and new LC'''
        i=((self.t)>part[0])*((self.t)<part[1])
        if reverse: i=~i

        #shift time to center of data, otherwise problems with polyfit
        t0=np.mean(self.t)
        p=np.polyfit(self.t[i]-t0,self.flux[i],2,w=1/self.err[i])
        pv=np.polyval(p,self.t-t0)

        flux=self.flux/pv
        err=self.err/pv

        if plot:
            fig=mpl.figure()
            if self._set_err:
                mpl.errorbar(self.t,self.flux,yerr=self.err,fmt='bo',markersize=5,zorder=1)
            else: mpl.plot(self.t,self.flux,'bo',markersize=5,zorder=1)
            mpl.plot(self.t,pv,'r-',zorder=2)
            mpl.plot(self.t[i],np.polyval(p,self.t[i]-t0),'g.',zorder=2)
            lims=mpl.ylim()
            mpl.plot([part[0],part[0]],lims,'k--')
            mpl.plot([part[1],part[1]],lims,'k--')
            mpl.ylim(lims)

            fig1=mpl.figure()
            if self._set_err:
                mpl.errorbar(self.t,flux,yerr=err,fmt='bo',markersize=5,zorder=1)
            else: mpl.plot(self.t,flux,'bo',markersize=5,zorder=1)

        if not (self._set_err or self._calc_err or self._corr_err):
            err=0.1*np.ones(self.t.shape)

        if rewrite:
            self.flux=flux
            self.err=err

        #reverse time shift
        p[2]=p[2]-p[1]*np.mean(self.t)+p[0]*np.mean(self.t)**2
        p[1]=p[1]-2*p[0]*np.mean(self.t)

        return p[::-1],[flux,err]


    def InfoGA(self,db,eps=False):
        '''statistics about GA or DE fitting'''
        info=InfoGAClass(db)
        path=db.replace('\\','/')
        if path.rfind('/')>0: path=path[:path.rfind('/')+1]
        else: path=''
        info.Stats()
        info.PlotChi2()
        mpl.savefig(path+'ga-chi2.png')
        if eps: mpl.savefig(path+'ga-chi2.eps')
        for p in info.availableTrace:
            info.Trace(p)
            mpl.savefig(path+'ga-'+p+'.png')
            if eps: mpl.savefig(path+'ga-'+p+'.eps')
        mpl.close('all')

    def InfoMCMC(self,db,eps=False):
        '''statistics about MCMC fitting'''
        info=InfoMCClass(db)
        info.AllParams(eps)

        for p in info.pars: info.OneParam(p,eps)

    def Transit(self,t,t0,P,Rp,a,i,e,w,u,p):
        '''model of transit from batman package
        t - times of observations (np.array alebo float) [days]
        t0 - time of reference transit [days]
        P - period of transiting exoplanet [days]
        Rp - radius of transiting planet [Rstar]
        a - semimajor axis of transiting exoplanet [Rstar]
        i - orbital inclination [deg]
        e - eccentricity of transiting exoplanet
        w - longitude of periastrum of transiting exoplanet [deg]
        u - limb darkening coefficients (in list)
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''
        params=batman.TransitParams()       #object to store transit parameters

        params.t0=t0
        params.per=P
        params.rp=Rp
        params.a=a
        params.inc=i
        params.ecc=e
        params.w=w
        params.limb_dark=self.model[7:].lower()
        params.u=u

        m=batman.TransitModel(params,t)    #initializes model

        flux=m.light_curve(params)*np.polyval(p,t-t0)         #calculates light curve

        return flux

    def Gauss(self,t,A,tC,w,p):
        '''model of minimum/maximum/eclipse using gaussian function
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude of gaussian
        w - width of gaussian [days]
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        g=1+A*np.exp(-(t-tC)**2/(2*w**2))

        flux=g*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def GaussModif(self,t,A,tC,w,k,p):
        '''model of minimum/maximum/eclipse using modificated gaussian function
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude of gaussian
        w - width of gaussian [days]
        k - exponent
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        if k<=0:
            raise ValueError('Exponent "k" should be possitive!')

        g=1+A*np.exp(-np.abs(t-tC)**k/(k*w**k))

        flux=g*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def Lorentz(self,t,A,tC,w,p):
        '''model of minimum/maximum/eclipse using lorentzian function / Cauchy curve
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude of lorentzian
        w - width of lorentzian [days]
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        l=1+A*w**2/((t-tC)**2+w**2)

        flux=l*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def Voigt(self,t,A,tC,wG,wL,p):
        '''model of minimum/maximum/eclipse using Voigt profile (Gauss&Lorentz)
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude of Voight profile
        wG - width of gaussian [days]
        wL - width of lorentzian [days]
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        v=1+A*voigt_profile(t-tC,wG,wL)/voigt_profile(0,wG,wL)

        flux=v*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def Quad(self,t,A,tC,w,p):
        '''model of minimum/maximum/eclipse using parabola
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude
        w - width
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        if A*w>0:
            raise ValueError('Amplitude "A" and width "w" should have opposite signs!')

        quad=1+A+w*(t-tC)**2

        quad[np.where(quad*np.sign(A)<np.sign(A))]=1  #replace part above/belove standard level (1)
        # or quad[np.where(t-tC>np.sqrt(-A/w))]=1

        flux=quad*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def Poly4(self,t,A,tC,w2,w4,p):
        '''model of minimum/maximum/eclipse using 4th order polynom
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude
        w2 - width of parabola
        w4 - width of 4th order polynom
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        if A*w4>0:
            raise ValueError('Amplitude "A" and width "w4" should have opposite signs!')

        model=1+A+w2*(t-tC)**2+w4*(t-tC)**4

        model[np.where(model*np.sign(A)<np.sign(A))]=1  #replace part above/belove standard level (1)

        flux=model*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def Mikulasek(self,t,A,C,K,tC,w,g,p):
        '''model of minimum/maximum/eclipse based on phenomenological model of Mikulasek (2015)
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude
        C,K - correcting parameters
        w - width
        g - kurtosis
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        mik=1+A*(1+C*((t-tC)/w)**2+K*((t-tC)/w)**4)*(1-(1-np.exp(1-np.cosh((t-tC)/w)))**g)

        flux=mik*np.polyval(p,t-tC)         #calculates light curve

        return flux

    def Binomial(self,t,A,tC,T,k1,k2,p):
        '''model of minimum/maximum/eclipse based on quasi binomial model
        t - times of observations (np.array alebo float) [days]
        tC - time of center of minimum [days]
        A - amplitude
        T - duration [days]
        k1, k2 - exponents (standard values: k1=2, k2=1.5)
        p - 2nd order polynom coefficients (in list / np.array)
        output in fluxes
        '''

        model=1+A*(1-(2*np.abs(t-tC)/T)**k1)**k2
        model[np.isnan(model)]=1

        flux=model*np.polyval(p,t-tC)         #calculates light curve

        return flux


    def PhaseCurve(self,P,t0,plot=False):
        '''create phase curve'''
        f=np.mod(self.t-t0,P)/float(P)    #phase
        order=np.argsort(f)
        f=f[order]
        flux=self.flux[order]
        if plot:
            mpl.figure()
            f1=np.array(f)
            f1[f1>0.5]-=1
            if self._set_err: mpl.errorbar(f1,flux,yerr=self.err,fmt='o')
            else: mpl.plot(f1,flux,'.')
        return f,flux

    def Chi2(self,params):
        '''calculate chi2 error (used as Objective Function for GA fitting) based on given parameters (in dict)'''
        param=dict(params)
        for x in self.params:
            #add fixed parameters
            if x not in param: param[x]=self.params[x]
        model=self.Model(param=param)   #calculate model
        return np.sum(((model-self.flux)/self.err)**2)

    def FitGA(self,generation,size,mut=0.5,SP=2,plot_graph=False,visible=True,
              n_thread=1,db=None):
        '''fitting with Genetic Algorithms
        generation - number of generations - should be approx. 100-200 x number of free parameters
        size - number of individuals in one generation (size of population) - should be approx. 100-200 x number of free parameters
        mut - proportion of mutations
        SP - selection pressure (see Razali&Geraghty (2011) for details)
        plot_graph - plot figure of best and mean solution found in each generation
        visible - display status of fitting
        n_thread - number of threads for multithreading
        db - name of database to save GA fitting details (could be analysed later using InfoGA function)
        '''

        def Thread(subpopul):
            #thread's function for multithreading
            for i in subpopul: objfun[i]=self.Chi2(popul.p[i])

        limits=self.limits
        steps=self.steps

        popul=TPopul(size,self.fit_params,mut,steps,limits,SP)  #init GA Class
        min0=1e15  #large number for comparing -> for finding min. value
        p={}     #best set of parameters
        if plot_graph:
            graph=[]
            graph_mean=[]

        objfun=[]   #values of Objective Function
        for i in range(size): objfun.append(0)

        if db is not None:
            #saving GA fitting details
            save_dat={}
            save_dat['chi2']=[]
            for par in self.fit_params: save_dat[par]=[]
            path=db.replace('\\','/')   #change dirs in path (for Windows)
            if path.rfind('/')>0:
                path=path[:path.rfind('/')+1]  #find current dir of db file
                if not os.path.isdir(path): os.mkdir(path) #create dir of db file, if not exist

        if not visible:
            #hidden output
            f = open(os.devnull, 'w')
            out=sys.stdout
            sys.stdout=f

        tic=time()
        for gen in range(generation):
            #main loop of GA
            threads=[]
            sys.stdout.write('Genetic Algorithms: '+str(gen+1)+' / '+str(generation)+' generations in '+str(np.round(time()-tic,1))+' sec  ')
            sys.stdout.flush()
            for t in range(n_thread):
                #multithreading
                threads.append(threading.Thread(target=Thread,args=[list(range(int(t*size/float(n_thread)),
                                                                          int((t+1)*size/float(n_thread))))]))
            #waiting for all threads and joining them
            for t in threads: t.start()
            for t in threads: t.join()

            #finding best solution in population and compare with global best solution
            i=np.argmin(objfun)
            if objfun[i]<min0:
                min0=objfun[i]
                p=dict(popul.p[i])

            if plot_graph:
                graph.append(min0)
                graph_mean.append(np.mean(np.array(objfun)))

            if db is not None:
                save_dat['chi2'].append(list(objfun))
                for par in self.fit_params:
                    temp=[]
                    for x in popul.p: temp.append(x[par])
                    save_dat[par].append(temp)

            popul.Next(objfun)  #generate new generation
            sys.stdout.write('\r')
            sys.stdout.flush()

        sys.stdout.write('\n')
        if not visible:
            #hidden output
            sys.stdout=out
            f.close()

        if plot_graph:
            mpl.figure()
            mpl.plot(graph,'-')
            mpl.xlabel('Number of generations')
            mpl.ylabel(r'Minimal $\chi^2$')
            mpl.plot(graph_mean,'--')
            mpl.legend(['Best solution',r'Mean $\chi^2$ in generation'])

        if db is not None:
            #saving GA fitting details to file
            for x in save_dat: save_dat[x]=np.array(save_dat[x])
            f=open(db,'wb')
            pickle.dump(save_dat,f,protocol=2)
            f.close()

        for param in p: self.params[param]=p[param]   #save found parameters
        self.params_err={}   #remove errors of parameters
        #remove some values calculated from old parameters
        self.paramsMore={}
        self.paramsMore_err={}
        self._fit='GA'

        return self.params

    def FitDE(self,generation,size,plot_graph=False,visible=True,strategy='randtobest1bin',tol=0.01,mutation=(0.5, 1),recombination=0.7,workers=1,db=None):
        '''fitting with Differential Evolution
        generation - number of generations - should be approx. 100-200 x number of free parameters
        size - number of individuals in one generation (size of population) - should be approx. 100-200 x number of free parameters
        plot_graph - plot figure of best and mean solution found in each generation
        visible - display status of fitting
        strategy - differential evolution strategy to use
        tol - relative tolerance for convergence
        mutation - mutation constant
        recombination - recombination constant (crossover probability)
        workers - number of walkers for multiprocessing
        db - name of database to save DE fitting details (could be analysed later using InfoGA function)
        '''

        limits=[]
        for p in self.fit_params: limits.append(self.limits[p])

        if plot_graph:
            graph=[]
            graph_mean=[]

        def ObjFun(vals,*names):
            '''Objective Function for DE'''
            pp={n:v for n,v in zip(names,vals)}
            return self.Chi2(pp)

        if db is not None:
            #saving DE fitting details
            save_dat={}
            save_dat['chi2']=[]
            for par in self.fit_params: save_dat[par]=[]
            path=db.replace('\\','/')   #change dirs in path (for Windows)
            if path.rfind('/')>0:
                path=path[:path.rfind('/')+1]  #find current dir of db file
                if not os.path.isdir(path): os.mkdir(path) #create dir of db file, if not exist

        solver=DifferentialEvolutionSolver(ObjFun,bounds=limits,args=self.fit_params,maxiter=generation,popsize=size,disp=visible,strategy=strategy,tol=tol,mutation=mutation,recombination=recombination,workers=workers)
        solver.init_population_lhs()

        tic=time()
        for gen in range(generation):
            #main loop of DE
            solver.__next__()

            if solver.disp:
                sys.stdout.write('differential_evolution step %d: f(x)= %g in %.1f sec  ' % (gen+1,solver.population_energies[0],time()-tic))
                sys.stdout.flush()

            if plot_graph:
                graph.append(np.min(solver.population_energies))
                graph_mean.append(np.mean(solver.population_energies))

            if db is not None:
                save_dat['chi2'].append(list(solver.population_energies))
                for i,par in enumerate(self.fit_params):
                    save_dat[par].append(list(solver.population[:,i]*(limits[i][1]-limits[i][0])+limits[i][0]))

            if solver.disp:
                sys.stdout.write('\r')
                sys.stdout.flush()

            if solver.converged(): break

        if visible: sys.stdout.write('\n')

        if plot_graph:
            mpl.figure()
            mpl.plot(graph,'-')
            mpl.xlabel('Number of generations')
            mpl.ylabel(r'Minimal $\chi^2$')
            mpl.plot(graph_mean,'--')
            mpl.legend(['Best solution',r'Mean $\chi^2$ in generation'])

        if db is not None:
            #saving DE fitting details to file
            for x in save_dat: save_dat[x]=np.array(save_dat[x])
            f=open(db,'wb')
            pickle.dump(save_dat,f,protocol=2)
            f.close()

        for i,p in enumerate(self.fit_params): self.params[p]=solver.x[i]   #save found parameters
        self.params_err={}   #remove errors of parameters
        #remove some values calculated from old parameters
        self.paramsMore={}
        self.paramsMore_err={}
        self._fit='DE'

        return self.params

    def FitMCMC(self,n_iter,burn=0,binn=1,walkers=0,visible=True,db=None):
        '''fitting with Markov chain Monte Carlo using emcee
        n_iter - number of MC iteration - should be at least 1e5
        burn - number of removed steps before equilibrium - should be approx. 0.1-1% of n_iter
        binn - binning size - should be around 10
        walkers - number of walkers - should be at least 2-times number of fitted parameters
        visible - display status of fitting
        db - name of database to save MCMC fitting details (could be analysed later using InfoMCMC function)
        '''

        #setting emcee priors for fitted parameters
        priors={}
        for p in self.fit_params:
            priors[p]=_Prior("limuniform",lower=self.limits[p][0],upper=self.limits[p][1],name=p)

        dims=len(self.fit_params)
        if walkers==0: walkers=dims*2
        elif walkers<dims * 2:
            walkers=dims*2
            warnings.warn('Numbers of walkers is smaller than two times number of free parameters. Auto-set to '+str(int(walkers))+'.')


        def likeli(names, vals):
            '''likelihood function for emcee'''
            pp={n:v for n,v in zip(names,vals)}

            likeli=-0.5*self.Chi2(pp)
            return likeli

        def lnpostdf(values):
            # Parameter-Value dictionary
            ps = dict(zip(self.fit_params,values))
            # Check prior information
            prior_sum = 0
            for name in self.fit_params: prior_sum += priors[name](ps, name)
            # If log prior is negative infinity, parameters
            # are out of range, so no need to evaluate the
            # likelihood function at this step:
            pdf = prior_sum
            if pdf == -np.inf: return pdf
            # Likelihood
            pdf += likeli(self.fit_params, values)
            return pdf

        # Generate the sampler
        emceeSampler=emcee.EnsembleSampler(int(walkers),int(dims),lnpostdf)

        # Generate starting values
        pos = []
        for j in range(walkers):
            pos.append(np.zeros(dims))
            for i, n in enumerate(self.fit_params):
                # Trial counter -- avoid values beyond restrictions
                tc = 0
                while True:
                    if tc == 100:
                        raise ValueError('Could not determine valid starting point for parameter: "'+n+'" due to its limits! Try to change the limits and/or step.')
                    propval = np.random.normal(self.params[n],self.steps[n])
                    if propval < self.limits[n][0]:
                        tc += 1
                        continue
                    if propval > self.limits[n][1]:
                        tc += 1
                        continue
                    break
                pos[-1][i] = propval

        # Default value for state
        state = None

        if burn>0:
            # Run burn-in
            pos,prob,state=emceeSampler.run_mcmc(pos,int(burn),progress=visible)
            # Reset the chain to remove the burn-in samples.
            emceeSampler.reset()

        pos,prob,state=emceeSampler.run_mcmc(pos,int(n_iter),rstate0=state,thin=int(binn),progress=visible)

        if not db is None:
            sampleArgs={}
            sampleArgs["burn"] = int(burn)
            sampleArgs["binn"] = int(binn)
            sampleArgs["iters"] = int(n_iter)
            sampleArgs["nwalker"] = int(walkers)
            np.savez_compressed(open(db,'wb'),chain=emceeSampler.chain,lnp=emceeSampler.lnprobability,                               pnames=list(self.fit_params),sampleArgs=sampleArgs)

        self.params_err={} #remove errors of parameters
        #remove some values calculated from old parameters
        self.paramsMore={}
        self.paramsMore_err={}

        for p in self.fit_params:
            #calculate values and errors of parameters and save them
            i=self.fit_params.index(p)
            self.params[p]=np.mean(emceeSampler.flatchain[:,i])
            self.params_err[p]=np.std(emceeSampler.flatchain[:,i])
        self._fit='MCMC'

        return self.params,self.params_err

    def Summary(self,name=None):
        '''summary of parameters, output to file "name"'''
        params=[]
        unit=[]
        vals=[]
        err=[]
        for x in sorted(self.params.keys()):
            #names, units, values and errors of model params
            if x[0]=='c' or x[0]=='p': continue
            params.append(x)
            vals.append(str(self.params[x]))
            if not len(self.params_err)==0:
                #errors calculated
                if x in self.params_err: err.append(str(self.params_err[x]))
                elif x in self.fit_params: err.append('---')   #errors not calculated
                else: err.append('fixed')  #fixed params
            elif x in self.fit_params: err.append('---')  #errors not calculated
            else: err.append('fixed')   #fixed params
            #add units
            if x[0]=='a' or x[0]=='R': unit.append('Rstar')
            elif x[0]=='P' or x[0]=='T':
                unit.append('d')
                #also in years
                params.append(x)
                vals.append(str(self.params[x]/year))
                if err[-1]=='---' or err[-1]=='fixed': err.append(err[-1])  #error not calculated
                else: err.append(str(float(err[-1])/year)) #error calculated
                unit.append('yr')
            elif x[0]=='t': unit.append('JD')
            elif x[0] in 'ecpACKgk': unit.append('')
            elif x[0]=='w':
                if 'Transit' in self.model: unit.append('deg')
                else: unit.append('d')
            elif x[0]=='i': unit.append('deg')
            else: unit.append('NA!')

        if 'Transit' in self.model:
            #make blank line
            params.append('')
            vals.append('')
            err.append('')
            unit.append('')
            for x in sorted([p for p in self.params.keys() if p[0]=='c']):
                #names, units, values and errors of model params
                params.append(x)
                vals.append(str(self.params[x]))
                if not len(self.params_err)==0:
                    #errors calculated
                    if x in self.params_err: err.append(str(self.params_err[x]))
                    elif x in self.fit_params: err.append('---')   #errors not calculated
                    else: err.append('fixed')  #fixed params
                elif x in self.fit_params: err.append('---')  #errors not calculated
                else: err.append('fixed')   #fixed params
                #add units
                unit.append('')

        #make blank line
        params.append('')
        vals.append('')
        err.append('')
        unit.append('')
        for x in sorted([p for p in self.params.keys() if p[0]=='p']):
            #names, units, values and errors of model params
            params.append(x)
            vals.append(str(self.params[x]))
            if not len(self.params_err)==0:
                #errors calculated
                if x in self.params_err: err.append(str(self.params_err[x]))
                elif x in self.fit_params: err.append('---')   #errors not calculated
                else: err.append('fixed')  #fixed params
            elif x in self.fit_params: err.append('---')  #errors not calculated
            else: err.append('fixed')   #fixed params
            #add units
            unit.append('')

        #calculate some more parameters, if not calculated
        self.DepthDur()
        R=0
        R_err=0
        if 'R' in self.systemParams:
            R=self.systemParams['R']
            if 'R_err' in self.systemParams: R_err=self.systemParams['R_err']

        if R>0: self.AbsoluteParam(R,R_err)

        self.FWHM()

        #make blank line
        params.append('')
        vals.append('')
        err.append('')
        unit.append('')
        for x in sorted(self.paramsMore.keys()):
            #names, units, values and errors of more params
            params.append(x)
            vals.append(str(self.paramsMore[x]))
            if not len(self.paramsMore_err)==0:
                #errors calculated
                if x in self.paramsMore_err:
                    err.append(str(self.paramsMore_err[x]))
                else: err.append('---')   #errors not calculated
            else: err.append('---')  #errors not calculated
            #add units
            if x[0]=='a': unit.append('au')
            elif x[0]=='d':
                unit.append('%')
                vals[-1]=str(float(vals[-1])*100)
                if not err[-1]=='---': err[-1]=str(float(err[-1])*100)
            elif x[0]=='R':
                unit.append('RJup')
                #also in years
                params.append(x)
                vals.append(str(self.paramsMore[x]*rJup/rEarth))
                if err[-1]=='---': err.append(err[-1])  #error not calculated
                else: err.append(str(float(err[-1])*rJup/rEarth)) #error calculated
                unit.append('REarth')
            elif x[0]=='T' or x[0]=='F':
                unit.append('d')
                #also in years
                params.append(x)
                vals.append(str(self.paramsMore[x]*24))
                if err[-1]=='---': err.append(err[-1])  #error not calculated
                else: err.append(str(float(err[-1])*24)) #error calculated
                unit.append('h')

        #generate text output
        text=['parameter'.ljust(15,' ')+'unit'.ljust(10,' ')+'value'.ljust(30,' ')+'error']
        for i in range(len(params)):
            text.append(params[i].ljust(15,' ')+unit[i].ljust(10,' ')+vals[i].ljust(30,' ')+err[i].ljust(30,' '))
        text.append('')
        text.append('Model: '+self.model)
        text.append('Fitting method: '+self._fit)
        chi=self.Chi2(self.params)
        n=len(self.t)
        g=len(self.fit_params)
        #calculate some stats
        text.append('chi2 = '+str(chi))
        if n-g>0: text.append('chi2_r = '+str(chi/(n-g)))
        else: text.append('chi2_r = NA')
        text.append('AIC = '+str(chi+2*g))
        if n-g-1>0: text.append('AICc = '+str(chi+2*g*n/(n-g-1)))
        else: text.append('AICc = NA')
        text.append('BIC = '+str(chi+g*np.log(n)))
        if name is None:
            #output to screen
            print('------------------------------------')
            for t in text: print(t)
            print('------------------------------------')
        else:
            #output to file
            f=open(name,'w')
            for t in text: f.write(t+'\n')
            f.close()

    def FWHM(self):
        '''calculate FWHM and other parameters for models of minima (Gauss etc.)'''
        output={}
        if 'Transit' in self.model: return output

        def gauss(p='w'):
            '''calculate FWHM of gaussian'''
            fwhm=2*np.sqrt(2*np.log(2))*self.params[p]

            if len(self.params_err)>0:
                #get error of params
                if p in self.params_err: w_err=self.params_err[p]
                else: w_err=0

                err=fwhm*w_err/self.params[p]
            else: err=None

            return fwhm,err

        def lorentz(p='w'):
            '''calculate FWHM of lorentzian'''
            fwhm=2*self.params[p]

            if len(self.params_err)>0:
                #get error of params
                if p in self.params_err: w_err=self.params_err[p]
                else: w_err=0

                err=fwhm*w_err/self.params[p]
            else: err=None

            return fwhm,err


        if self.model=='Gauss':
            fwhm,err=gauss()
            self.paramsMore['FWHM']=fwhm
            # duration -> when Gaussian is on level of residual
            res=self.flux-self.Model()
            self.paramsMore['T14']=2*np.sqrt(2*np.log(np.abs(self.params['A'])/np.mean(np.abs(res))))*self.params['w']
            output['FWHM']=self.paramsMore['FWHM']
            output['T14']=self.paramsMore['T14']

            if len(self.params_err)>0:
                #get error of params
                if 'w' in self.params_err: w_err=self.params_err['w']
                else: w_err=0

                self.paramsMore_err['FWHM']=err
                self.paramsMore_err['T14']=self.paramsMore['T14']*w_err/self.params['w']

        elif self.model=='Lorentz':
            fwhm,err=lorentz()
            self.paramsMore['FWHM']=fwhm

            # duration -> when Lorentzian is on level of residual
            res=self.flux-self.Model()
            R=np.mean(np.abs(res))/np.abs(self.params['A'])*3
            self.paramsMore['T14']=2*self.params['w']*np.sqrt((1-R)/R)
            output['FWHM']=self.paramsMore['FWHM']
            output['T14']=self.paramsMore['T14']

            if len(self.params_err)>0:
                #get error of params
                if 'w' in self.params_err: w_err=self.params_err['w']
                else: w_err=0

                self.paramsMore_err['FWHM']=err
                self.paramsMore_err['T14']=self.paramsMore['T14']*w_err/self.params['w']

        elif self.model=='Voigt':
            fG,errG=gauss('wG')
            fL,errL=lorentz('wL')
            sqrt=np.sqrt(0.2169*fL**2+fG**2)
            self.paramsMore['FWHM']=0.5343*fL+sqrt
            output['FWHM']=self.paramsMore['FWHM']
            self.paramsMore['FWHM_Gauss']=fG
            output['FWHM_Gauss']=self.paramsMore['FWHM_Gauss']
            self.paramsMore['FWHM_Lorentz']=fL
            output['FWHM_Lorentz']=self.paramsMore['FWHM_Lorentz']

            if len(self.params_err)>0:
                #get error of params
                #if 'wG' in self.params_err: wG_err=self.params_err['wG']
                #else: wG_err=0
                #if 'wL' in self.params_err: wL_err=self.params_err['wL']
                #else: wL_err=0

                self.paramsMore_err['FWHM']=np.sqrt((0.5343+0.2169*fL/sqrt)**2*errL**2+(fG/sqrt)**2*errG**2)
                self.paramsMore_err['FWHM_Gauss']=errG
                self.paramsMore_err['FWHM_Lorentz']=errL

        elif self.model=='Quad':
            self.paramsMore['T14']=2*np.sqrt(-self.params['A']/self.params['w'])
            output['T14']=self.paramsMore['T14']

            if len(self.params_err)>0:
                #get error of params
                if 'w' in self.params_err: w_err=self.params_err['w']
                else: w_err=0
                if 'A' in self.params_err: A_err=self.params_err['A']
                else: A_err=0

                self.paramsMore_err['T14']=self.paramsMore['T14']*0.5*np.sqrt((A_err/self.params['A'])**2+(w_err/self.params['w'])**2)

        elif self.model=='Poly4':
            A=self.params['A']
            w2=self.params['w2']
            w4=self.params['w4']
            sqrt=np.sqrt((w2/(2*w4))**2-A/w4)
            self.paramsMore['T14']=2*np.sqrt(sqrt-w2/(2*w4))
            output['T14']=self.paramsMore['T14']

            if len(self.params_err)>0:
                #get error of params
                if 'w2' in self.params_err: w2_err=self.params_err['w2']
                else: w2_err=0
                if 'w4' in self.params_err: w4_err=self.params_err['w4']
                else: w4_err=0
                if 'A' in self.params_err: A_err=self.params_err['A']
                else: A_err=0

                dA=1/sqrt
                dw2=1/sqrt*w2/w4**2-1
                dw4=1/sqrt*(2*A/w4-(w2/w4)**2)+w2/w4
                self.paramsMore_err['T14']=1/self.paramsMore['T14']*1/w4*np.sqrt((dA*A_err)**2+(dw2*w2_err)**2+(dw4*w4_err)**2)

        elif self.model=='Mikulasek':
            # duration -> when flux~0.999, 3rd derivation of Mikulasek function is zero
            self.paramsMore['T14']=2*self.params['w']*np.arccosh(3)
            output['T14']=self.paramsMore['T14']

            if len(self.params_err)>0:
                #get error of params
                if 'w' in self.params_err: w_err=self.params_err['w']
                else: w_err=0

                self.paramsMore_err['T14']=self.paramsMore['T14']*w_err/self.params['w']


        if len(self.paramsMore_err)>0:
            #if some errors = 0, del them; and return only non-zero errors
            keys=list(self.paramsMore_err.keys())
            for p in keys:
                if self.paramsMore_err[p]==0: del self.paramsMore_err[p]
                else: output[p]=self.paramsMore_err[p]

        return output


    def DepthDur(self):
        '''calculate depth and duration of transit'''
        output={}
        if 'Transit' not in self.model: return output
        if 'Rp' not in self.params: return output
        self.paramsMore['delta']=self.params['Rp']**2
        output['delta']=self.paramsMore['delta']

        i=np.deg2rad(self.params['i'])
        e=self.params['e']
        a=self.params['a']
        w=np.deg2rad(self.params['w'])
        ee=(1-e**2)/(1+e*np.cos(w))
        t14=self.params['P']/np.pi*np.arcsin(np.sqrt((1+self.params['Rp'])**2-(a*np.cos(i)*ee)**2)/(a*np.sin(i))*ee)
        t23=self.params['P']/np.pi*np.arcsin(np.sqrt((1-self.params['Rp'])**2-(a*np.cos(i)*ee)**2)/(a*np.sin(i))*ee)

        self.paramsMore['T14']=t14
        self.paramsMore['T23']=t23
        output['T14']=self.paramsMore['T14']
        output['T23']=self.paramsMore['T23']

        if len(self.params_err)>0:
            #get error of params
            if 'Rp' in self.params_err: r_err=self.params_err['Rp']
            else: r_err=0
            if 'a' in self.params_err: a_err=self.params_err['a']
            else: a_err=0
            if 'i' in self.params_err: i_err=np.deg2rad(self.params_err['i'])
            else: i_err=0
            if 'e' in self.params_err: e_err=self.params_err['e']
            else: e_err=0
            if 'w' in self.params_err: w_err=np.deg2rad(self.params_err['w'])
            else: w_err=0
            if 'P' in self.params_err: P_err=self.params_err['P']
            else: P_err=0

            #calculate errors of params
            self.paramsMore_err['delta']=self.paramsMore['delta']*(2*r_err/self.params['Rp'])

            def errT(dur14=True):
                '''calculate errors of durations T14 and T23'''
                #x=1+-self.params['Rp']
                #some strange partial derivations...(calculated using Wolfram Mathematica)
                if dur14:
                    x=1+self.params['Rp']
                    T=t14
                else:
                    x=1-self.params['Rp']
                    T=t23

                #dT/dx = dT/d(1+-R)
                dR=self.params['P']/np.pi*(1-e**2)*x/np.sin(i)/((a*e*np.cos(w)+a)*np.sqrt(x**2-(a**2*(1-e**2)**2* np.cos(i)**2)/(e*np.cos(w)+1)**2)*np.sqrt(1-((e**2-1)**2/np.sin(i)**2*(x**2-(a**2*(e**2-1)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2))/(a*e*np.cos(w)+a)**2))

                #dT/da
                da=self.params['P']/np.pi*(1-e**2)*x**2/np.sin(i)/(a**2*(e*np.cos(w)+1)*np.sqrt(x**2-((e**2- 1)**2*a**2*np.cos(i)**2)/(e*np.cos(w)+1)**2)*np.sqrt(1-((e**2-1)**2/np.sin(i)**2*(x**2-((e**2-1)**2*a**2*np.cos(i)**2)/(e*np.cos(w)+1)**2))/(e*a*np.cos(w)+a)**2))

                #dT/dw
                dw=self.params['P']/np.pi*(e*(1-e**2)*np.sin(w)*(2*a**2*(e**2-1)**2*np.cos(i)**2/np.sin(i)-1/np.sin(i)*(e*x*np.cos(w)+x)**2))/(a*(e*np.cos(w)+1)**4*np.sqrt(x**2-(a**2*(e**2-1)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2)*np.sqrt(1-((e**2-1)**2/np.sin(i)**2*(x**2-(a**2*(e**2-1)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2))/(a*e*np.cos(w)+a)**2))

                #dT/di
                di=self.params['P']/np.pi*((1-e**2)*(a**2*(e**2-1)**2*np.cos(i)*(1/np.sin(i)**2+1)-np.cos(i)/np.sin(i)**2*(e*x*np.cos(w)+x)**2))/(a*(e*np.cos(w)+1)**3*np.sqrt(x**2-(a**2*(e**2-1)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2)*np.sqrt(1-((e**2-1)**2/np.sin(i)**2*(x**2-(a**2*(e**2-1)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2))/(a*e*np.cos(w)+a)**2))

                #dT/de
                de=self.params['P']/np.pi*(((e**2+1)*np.cos(w)+2*e)/np.sin(i)*(2*a**2*(1-e**2)**2*np.cos(i)**2+x**2*(-e**2)*np.cos(w)**2-2*x**2*e*np.cos(w)-x**2))/(a*(e*np.cos(w)+1)**4*np.sqrt(x**2-(a**2*(1-e**2)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2)*np.sqrt(1-((1-e**2)**2/np.sin(i)**2*(x**2-(a**2*(1-e**2)**2*np.cos(i)**2)/(e*np.cos(w)+1)**2))/(a*e*np.cos(w)+a)**2))

                err=np.sqrt((T/self.params['P']*P_err)**2+(dR*r_err)**2+(da*a_err)**2+(di*i_err)**2+(de*e_err)**2+(dw*w_err)**2)
                return err

            self.paramsMore_err['T14']=errT(dur14=True)
            self.paramsMore_err['T23']=errT(dur14=False)

            #if some errors = 0, del them; and return only non-zero errors
            keys=list(self.paramsMore_err.keys())
            for p in keys:
                if self.paramsMore_err[p]==0: del self.paramsMore_err[p]
                else: output[p]=self.paramsMore_err[p]

        return output

    def AbsoluteParam(self,R,R_err=0):
        '''calculate absolute radius and semi-mayor axis of planet from radius of star'''
        output={}
        if 'Transit' not in self.model: return output
        if 'Rp' not in self.params: return output
        self.paramsMore['a']=self.params['a']*R*rSun/au
        self.paramsMore['Rp']=self.params['Rp']*R*rSun/rJup
        output['a']=self.paramsMore['a']
        output['Rp']=self.paramsMore['Rp']

        if len(self.params_err)>0:
            #get error of params
            if 'a' in self.params_err: a_err=self.params_err['a']
            else: a_err=0
            if 'Rp' in self.params_err: r_err=self.params_err['Rp']
            else: r_err=0

            #calculate errors of params
            self.paramsMore_err['a']=self.paramsMore['a']*(a_err/self.params['a']+R_err/R)
            self.paramsMore_err['Rp']=self.paramsMore['Rp']*(r_err/self.params['Rp']+R_err/R)

            #if some errors = 0, del them; and return only non-zero errors
            keys=list(self.paramsMore_err.keys())
            for p in keys:
                if self.paramsMore_err[p]==0: del self.paramsMore_err[p]
                else: output[p]=self.paramsMore_err[p]

        return output


    def Model(self,t=None,param=None):
        ''''calculate model curve of transit in given times based on given set of parameters'''
        if t is None: t=self.t
        if param is None: param=self.params
        p=[param['p2'],param['p1'],param['p0']]
        if 'Transit' in self.model:
            u=[]
            if 'Uniform' not in self.model:
                u.append(param['c1'])
                if ('Linear' not in self.model) or ('Power2' not in self.model):
                    u.append(param['c2'])
                    if 'Nonlinear' in self.model:
                        u.append(param['c3'])
                        u.append(param['c4'])

            model=self.Transit(t,param['t0'],param['P'],param['Rp'],param['a'],param['i'],param['e'],param['w'],u,p)
        elif self.model=='Gauss':
            model=self.Gauss(t,param['A'],param['tC'],param['w'],p)
        elif self.model=='GaussModif':
            model=self.GaussModif(t,param['A'],param['tC'],param['w'],param['k'],p)
        elif self.model=='Lorentz':
            model=self.Lorentz(t,param['A'],param['tC'],param['w'],p)
        elif self.model=='Voigt':
            model=self.Voigt(t,param['A'],param['tC'],param['wG'],param['wL'],p)
        elif self.model=='Quad':
            model=self.Quad(t,param['A'],param['tC'],param['w'],p)
        elif self.model=='Poly4':
            model=self.Poly4(t,param['A'],param['tC'],param['w2'],param['w4'],p)
        elif self.model=='Mikulasek':
            model=self.Mikulasek(t,param['A'],param['C'],param['K'],param['tC'],param['w'],param['g'],p)
        elif self.model=='Binomial':
            model=self.Binomial(t,param['A'],param['tC'],param['T'],param['k1'],param['k2'],p)
        else:
            raise KeyError('Unknown model "'+self.model+'"! Available models are: '+', '.join(sorted(self.availableModels)))

        return model


    def CalcErr(self):
        '''estimate errors of input data based on current model (useful before using FitMCMC)'''
        model=self.Model(self.t,self.params)  #calculate model values

        n=len(model)   #number of data points
        err=np.sqrt(sum((self.flux-model)**2)/(n-1))   #calculate corrected sample standard deviation
        err*=np.ones(model.shape)  #generate array of errors
        chi=sum(((self.flux-model)/err)**2)   #calculate new chi2 error -> chi2_r = 1
        print('New chi2:',chi,chi/(n-len(self.fit_params)))
        self._calc_err=True
        self._set_err=False
        self.err=err
        return err

    def CorrectErr(self):
        '''correct scale of given errors of input data based on current model
        (useful if FitMCMC gives worse results like FitGA and chi2_r is not approx. 1)'''
        model=self.Model(self.t,self.params)     #calculate model values

        n=len(model)   #number of data points
        chi0=sum(((self.flux-model)/self.err)**2)    #original chi2 error
        alfa=chi0/(n-len(self.fit_params))         #coefficient between old and new errors -> chi2_r = 1
        err=self.err*np.sqrt(alfa)          #new errors
        chi=sum(((self.flux-model)/err)**2)   #calculate new chi2 error
        print('New chi2:',chi,chi/(n-len(self.fit_params)))
        if self._set_err and len(self._old_err)==0: self._old_err=self.err    #if errors were given, save old values
        self.err=err
        self._corr_err=True
        return err

    def AddWeight(self,weight):
        '''adding weight of input data + scaling according to current model
        warning: weights have to be in same order as input date!'''
        if not len(weight)==len(self.t):
            #if wrong length of given weight array
            print('incorrect length of "w"!')
            return

        weight=np.array(weight)
        err=1./weight[self._order]   #transform to errors and change order according to order of input data
        n=len(self.t)   #number of data points
        model=self.Model(self.t,self.params)   #calculate model values

        chi0=sum(((self.flux-model)/err)**2)    #original chi2 error
        alfa=chi0/(n-len(self.fit_params))    #coefficient between old and new errors -> chi2_r = 1
        err*=np.sqrt(alfa)              #new errors
        chi=sum(((self.flux-model)/err)**2)   #calculate new chi2 error
        print('New chi2:',chi,chi/(n-len(self.fit_params)))
        self._calc_err=True
        self._set_err=False
        self.err=err
        return err

    def Plot0(self,name=None,no_plot=0,no_plot_err=0,eps=False,time_type='JD',
              offset=2400000,trans=True,title=None,hours=False,mag=False,weight=None,
              trans_weight=False,bw=False,fig_size=None):
        '''plotting original Light Curve
        name - name of file to saving plot (if not given -> show graph)
        no_plot - number of outlier point which will not be plot
        no_plot_err - number of errorful point which will not be plot
        eps - save also as eps file
        time_type - type of JD in which is time (show in x label)
        offset - offset of time
        trans - transform time according to offset
        hours - time in hours (except in days)
        title - name of graph
        mag - y axis in magnitude
        weight - weight of data (shown as size of points)
        trans_weight - transform weights to range (1,10)
        with_res - common plot with residue
        bw - Black&White plot
        fig_size - custom figure size - e.g. (12,6)

        warning: weights have to be in same order as input data!
        '''

        if fig_size:
            fig=mpl.figure(figsize=fig_size)
        else:
            fig=mpl.figure()

        #2 plots - for residue
        ax1=fig.add_subplot(1,1,1)
        ax1.yaxis.set_label_coords(-0.175,0.5)
        ax1.ticklabel_format(useOffset=False)

        l=''
        if hours: l=' [h]'
        #setting labels
        if offset>0:
            ax1.set_xlabel('Time ('+time_type+' - '+str(round(offset,2))+')'+l)
            if not trans: offset=0
            x=self.t-offset
        else:
            ax1.set_xlabel('Time ('+time_type+')'+l)
            offset=0
            x=self.t
        if hours: k=24  #convert to hours
        else: k=1
        if mag:
            if 'mag' in self.systemParams: ax1.set_ylabel('Magnitude')
            else: ax1.set_ylabel('Rel. Magnitude')
            ax1.invert_yaxis()
        else: ax1.set_ylabel('Flux')

        if title is not None: fig.suptitle(title,fontsize=20)

        flux=np.array(self.flux)

        #set weight
        set_w=False
        if weight is not None:
            weight=np.array(weight)[self._order]
            if trans_weight:
                w_min=min(weight)
                w_max=max(weight)
                weight=9./(w_max-w_min)*(weight-w_min)+1
            if weight.shape==self.t.shape:
                w=[]
                levels=[0,3,5,7.9,10]
                size=[3,4,5,7]
                for i in range(len(levels)-1):
                    w.append(np.where((weight>levels[i])*(weight<=levels[i+1])))
                w[-1]=np.append(w[-1],np.where(weight>levels[-1]))  #if some weight is bigger than max. level
                set_w=True
            else:
                warnings.warn('Shape of "weight" is different to shape of "time". Weight will be ignore!')

        ii=np.arange(0,len(self.t),1)
        if no_plot>0: errors=np.argsort(abs(flux-np.mean(flux)))[-no_plot:]   #remove outlier points
        else: errors=np.array([])
        if bw: color='k'
        else: color='b'
        if mag:
            flux=-2.5*np.log10(flux)
            if 'mag' in self.systemParams: flux+=self.systemParams['mag']

        if set_w:
            #using weights
            for i in range(len(w)):
                ax1.plot(k*x[np.where(w[i])],
                        flux[np.where(w[i])],color+'o',markersize=size[i],zorder=1)

        else:
            #without weight
            if self._set_err:
                #using errors
                if self._corr_err: err=np.array(self._old_err)
                else: err=np.array(self.err)
                if mag:
                    m0=0
                    if 'mag' in self.systemParams: m0=self.systemParams['mag']
                    err=2.5/np.log(10)*err/10**(-(flux-m0)/2.5)
                if no_plot_err>0: errors=np.append(errors,np.argsort(abs(err))[-no_plot_err:])  #remove errorful points
                ii=np.delete(ii,np.where(np.in1d(ii,errors)))
                ax1.errorbar(k*x[ii],flux[ii],yerr=err[ii],fmt=color+'o',markersize=5,zorder=1)
            else:
                #without errors
                ax1.plot(k*x[ii],flux[ii],color+'o',zorder=1)

        if name is not None:
            mpl.savefig(name+'.png',bbox_inches='tight')
            if eps: mpl.savefig(name+'.eps',bbox_inches='tight')
            mpl.close(fig)
        return fig

    def Plot(self,name=None,no_plot=0,no_plot_err=0,params=None,eps=False,
             time_type='JD',offset=2400000,trans=True,center=True,title=None,hours=False,
             phase=False,mag=False,weight=None,trans_weight=False,model2=False,with_res=False,
             bw=False,double_ax=False,legend=None,fig_size=None,detrend=False):
        '''plotting original Light Curve with model based on current parameters set
        name - name of file to saving plot (if not given -> show graph)
        no_plot - number of outlier point which will not be plot
        no_plot_err - number of errorful point which will not be plot
        params - set of params of current model (if not given -> current parameters set)
        eps - save also as eps file
        time_type - type of JD in which is time (show in x label)
        offset - offset of time
        trans - transform time according to offset
        center - center to mid transit
        hours - time in hours (except in days)
        title - name of graph
        phase - x axis in phase
        mag - y axis in magnitude
        weight - weight of data (shown as size of points)
        trans_weight - transform weights to range (1,10)
        model2 - plot 2 models - current params set and set given in "params"
        with_res - common plot with residue
        bw - Black&White plot
        double_ax - two axes -> time and phase
        legend - labels for data and model(s) - give '' if no show label, 2nd model given in "params" is the last
        fig_size - custom figure size - e.g. (12,6)
        detrend - normalize (detrend) curve using polynom given by p0,p1,p2 params

        warning: weights have to be in same order as input data!
        '''

        if model2:
            if len(params)==0:
                raise ValueError('Parameters set for 2nd model not given!')
            params_model=dict(params)
            params=self.params
        if params is None: params=self.params
        if legend is None:
            legend=['','','']
            show_legend=False
        else: show_legend=True

        if phase or double_ax or detrend or center:
            if not len(self.phase)==len(self.t):
                if (('t0' in params) or ('tC' in params)) and ('P' in params):
                    self.Phase(params['t0'],params['P'])
                else:
                    raise TypeError('Phase not callculated! Run function "Phase" before it.')
            if (('t0' in params) or ('tC' in params)) and ('P' in params):
                if 't0' in params: t0=params['t0']
                elif 'tC' in params: t0=params['tC']
                P=params['P']
            else:
                t0=self._t0P[0]
                P=self._t0P[1]

        if fig_size:
            fig=mpl.figure(figsize=fig_size)
        else:
            fig=mpl.figure()

        #2 plots - for residue
        if with_res:
            gs=gridspec.GridSpec(2,1,height_ratios=[4,1])
            ax1=fig.add_subplot(gs[0])
            ax2=fig.add_subplot(gs[1],sharex=ax1)
        else:
            ax1=fig.add_subplot(1,1,1)
            ax2=ax1
        ax1.yaxis.set_label_coords(-0.175,0.5)
        ax1.ticklabel_format(useOffset=False)

        l=''
        if hours:
            center=True
            l=' [h]'
        if center:
            E=np.round((self.t-t0)/P)
            E=E[len(E)//2]
            offset=t0+P*E
        #setting labels
        if phase and not double_ax:
            ax2.set_xlabel('Phase')
            x=self.phase
        elif offset>0:
            ax2.set_xlabel('Time ('+time_type+' - '+str(round(offset,2))+')'+l)
            if not trans: offset=0
            x=self.t-offset
        else:
            ax2.set_xlabel('Time ('+time_type+')'+l)
            offset=0
            x=self.t
        if hours: k=24  #convert to hours
        else: k=1
        if mag:
            if 'mag' in self.systemParams: ax1.set_ylabel('Magnitude')
            else: ax1.set_ylabel('Rel. Magnitude')
            ax1.invert_yaxis()
        elif detrend: ax1.set_ylabel('Norm. Flux')
        else: ax1.set_ylabel('Flux')

        if title is not None:
            if double_ax: fig.subplots_adjust(top=0.85)
            fig.suptitle(title,fontsize=20)

        model=self.Model(self.t,params)
        flux=np.array(self.flux)
        res=flux-model
        if detrend:
            flux/=np.polyval([params['p2'],params['p1'],params['p0']],self.t-t0)

        #set weight
        set_w=False
        if weight is not None:
            weight=np.array(weight)[self._order]
            if trans_weight:
                w_min=min(weight)
                w_max=max(weight)
                weight=9./(w_max-w_min)*(weight-w_min)+1
            if weight.shape==self.t.shape:
                w=[]
                levels=[0,3,5,7.9,10]
                size=[3,4,5,7]
                for i in range(len(levels)-1):
                    w.append(np.where((weight>levels[i])*(weight<=levels[i+1])))
                w[-1]=np.append(w[-1],np.where(weight>levels[-1]))  #if some weight is bigger than max. level
                set_w=True
            else:
                warnings.warn('Shape of "weight" is different to shape of "time". Weight will be ignore!')

        ii=np.arange(0,len(self.t),1)
        if no_plot>0: errors=np.argsort(abs(model-flux))[-no_plot:]   #remove outlier points
        else: errors=np.array([])
        if bw: color='k'
        else: color='b'

        if mag:
            flux=-2.5*np.log10(flux)
            model=-2.5*np.log10(model)
            res=flux-model
            if 'mag' in self.systemParams:
                flux+=self.systemParams['mag']
                model+=self.systemParams['mag']

        if set_w:
            #using weights
            ii=np.delete(ii,np.where(np.in1d(ii,errors)))
            for i in range(len(w)):
                ax1.plot(k*x[np.where(np.in1d(ii,w[i]))],
                        flux[np.where(np.in1d(ii,w[i]))],color+'o',markersize=size[i],label=legend[0],zorder=1)

        else:
            #without weight
            if self._set_err:
                #using errors
                if self._corr_err: err=np.array(self._old_err)
                else: err=np.array(self.err)
                if detrend:
                    err/=np.polyval([params['p2'],params['p1'],params['p0']],self.t-t0)
                if mag:
                    m0=0
                    if 'mag' in self.systemParams: m0=self.systemParams['mag']
                    err=2.5/np.log(10)*err/10**(-(flux-m0)/2.5)
                if no_plot_err>0: errors=np.append(errors,np.argsort(abs(err))[-no_plot_err:])  #remove errorful points
                ii=np.delete(ii,np.where(np.in1d(ii,errors)))
                ax1.errorbar(k*x[ii],flux[ii],yerr=err[ii],fmt=color+'o',markersize=5,label=legend[0],zorder=1)
            else:
                #without errors
                ii=np.delete(ii,np.where(np.in1d(ii,errors)))
                ax1.plot(k*x[ii],flux[ii],color+'o',label=legend[0],zorder=1)

        #expand time interval for model LC
        if phase and not double_ax:
            tmin=self._t0P[0]+self._t0P[1]*np.min(self.phase)
            tmax=self._t0P[0]+self._t0P[1]*np.max(self.phase)
        else:
            tmin=self.t[0]
            tmax=self.t[-1]
        if len(self.t)<1000:
            dt=(tmax-tmin)/1000.
            t1=np.linspace(tmin-50*dt,tmax+50*dt,1100)
        else:
            dt=(tmax-tmin)/len(self.t)
            t1=np.linspace(tmin-0.05*len(self.t)*dt,tmax+0.05*len(self.t)*dt,int(1.1*len(self.t)))

        if bw:
            color='k'
            lw=2
        else:
            color='r'
            lw=1

        model_long=self.Model(t1,params)
        if detrend:
            model_long/=np.polyval([params['p2'],params['p1'],params['p0']],t1-t0)
        if mag:
            model_long=-2.5*np.log10(model_long)
            if 'mag' in self.systemParams: model_long+=self.systemParams['mag']

        if phase and not double_ax:
            ph=self.Phase(t0,P,t1)
            i=np.argsort(ph)
            ax1.plot(ph[i],model_long[i],color,linewidth=lw,label=legend[1],zorder=2)
        else: ax1.plot(k*(t1-offset),model_long,color,linewidth=lw,label=legend[1],zorder=2)

        if model2:
            #plot second model
            if bw:
                color='k'
                lt='--'
            else:
                color='g'
                lt='-'
            model_set=self.Model(t1,params_model)
            if detrend:
                model_set/=np.polyval([params['p2'],params['p1'],params['p0']],t1-t0)
            if mag:
                model_set=-2.5*np.log10(model_set)
                if 'mag' in self.systemParams: model_set+=self.systemParams['mag']

            if phase and not double_ax:
                ax1.plot(self.Phase(t0,P,t1),model_set,color+lt,linewidth=lw,label=legend[2],zorder=3)
            else: ax1.plot(k*(t1-offset),model_set,color+lt,linewidth=lw,label=legend[2],zorder=3)

        if show_legend: ax1.legend()

        if double_ax:
            #setting secound axis
            ax3=ax1.twiny()
            #generate plot to obtain correct axis in phase
            #expand time interval for model LC
            if len(self.t)<1000:
                dt=(self.t[-1]-self.t[0])/1000.
                t1=np.linspace(self.t[0]-50*dt,self.t[-1]+50*dt,1100)
            else:
                dt=(self.t[-1]-self.t[0])/len(self.t)
                t1=np.linspace(self.t[0]-0.05*len(self.t)*dt,self.t[-1]+0.05*len(self.t)*dt,int(1.1*len(self.t)))
            l=ax3.plot(k*(t1-offset),model_long)
            ax3.set_xlabel('Phase')
            l.pop(0).remove()
            lims=np.array(ax1.get_xlim())/k+offset
            ph=self.Phase(t0,P,lims)
            ax3.set_xlim(ph)

        if with_res:
            #plot residue
            if bw: color='k'
            else: color='b'
            ax2.set_ylabel('Residue (%)')
            ax2.yaxis.set_label_coords(-0.15,0.5)
            m=abs(max(-min(res),max(res)))*100
            ax2.set_autoscale_on(False)
            ax2.set_ylim([-m,m])
            ax2.yaxis.set_ticks(np.array([-m,0,m]))
            ax2.plot(k*x,res*100,color+'o')
            ax2.xaxis.labelpad=15
            ax2.yaxis.labelpad=15
            ax2.ticklabel_format(useOffset=False)
            ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1g'))
            mpl.subplots_adjust(hspace=.07)
            mpl.setp(ax1.get_xticklabels(),visible=False)

        if name is not None:
            mpl.savefig(name+'.png',bbox_inches='tight')
            if eps: mpl.savefig(name+'.eps',bbox_inches='tight')
            mpl.close(fig)
        return fig

    def PlotRes(self,name=None,no_plot=0,no_plot_err=0,params=None,eps=False,
                time_type='JD',offset=2400000,trans=True,center=True,title=None,hours=False,
                phase=False,weight=None,trans_weight=False,bw=False,double_ax=False,
                fig_size=None):
        '''plotting residual LC
        name - name of file to saving plot (if not given -> show graph)
        no_plot - count of outlier point which will not be plot
        no_plot_err - count of errorful point which will not be plot
        params - set of params of current model (if not given -> current parameters set)
        eps - save also as eps file
        time_type - type of JD in which is time (show in x label)
        offset - offset of time
        trans - transform time according to offset
        center - center to mid transit
        hours - time in hours (except in days)
        title - name of graph
        phase - x axis in phase
        weight - weight of data (shown as size of points)
        trans_weight - transform weights to range (1,10)
        bw - Black&White plot
        double_ax - two axes -> time and phase
        fig_size - custom figure size - e.g. (12,6)

        warning: weights have to be in same order as input data!
        '''

        if params is None: params=self.params

        if phase or double_ax or center:
            if not len(self.phase)==len(self.t):
                if (('t0' in params) or ('tC' in params)) and ('P' in params):
                    self.Phase(params['t0'],params['P'])
                else:
                    raise TypeError('Phase not callculated! Run function "Phase" before it.')
            if (('t0' in params) or ('tC' in params)) and ('P' in params):
                if 't0' in params: t0=params['t0']
                elif 'tC' in params: t0=params['tC']
                P=params['P']
            else:
                t0=self._t0P[0]
                P=self._t0P[1]

        if fig_size:
            fig=mpl.figure(figsize=fig_size)
        else:
            fig=mpl.figure()

        ax1=fig.add_subplot(1,1,1)
        ax1.yaxis.set_label_coords(-0.15,0.5)
        ax1.ticklabel_format(useOffset=False)
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1g'))

        l=''
        if hours:
            center=True
            l=' [h]'
        if center:
            E=np.round((self.t-t0)/P)
            E=E[len(E)//2]
            offset=t0+P*E
        #setting labels
        if phase and not double_ax:
            ax1.set_xlabel('Phase')
            x=self.phase
        elif offset>0:
            ax1.set_xlabel('Time ('+time_type+' - '+str(round(offset,2))+')'+l)
            if not trans: offset=0
            x=self.t-offset
        else:
            ax1.set_xlabel('Time ('+time_type+')'+l)
            offset=0
            x=self.t
        if hours: k=24  #convert to hours
        else: k=1
        ax1.set_ylabel('Residual Flux (%)')

        if title is not None:
            if double_ax: fig.subplots_adjust(top=0.85)
            fig.suptitle(title,fontsize=20)

        model=self.Model(self.t,params)
        res=(self.flux-model)*100

        #set weight
        set_w=False
        if weight is not None:
            weight=np.array(weight)[self._order]
            if trans_weight:
                w_min=min(weight)
                w_max=max(weight)
                weight=9./(w_max-w_min)*(weight-w_min)+1
            if weight.shape==self.t.shape:
                w=[]
                levels=[0,3,5,7.9,10]
                size=[3,4,5,7]
                for i in range(len(levels)-1):
                    w.append(np.where((weight>levels[i])*(weight<=levels[i+1])))
                w[-1]=np.append(w[-1],np.where(weight>levels[-1]))  #if some weight is bigger than max. level
                set_w=True
            else:
                warnings.warn('Shape of "weight" is different to shape of "time". Weight will be ignore!')

        ii=np.arange(0,len(self.t),1)
        if no_plot>0: errors=np.argsort(abs(res))[-no_plot:]   #remove outlier points
        else: errors=np.array([])
        if bw: color='k'
        else: color='b'
        if set_w:
            #using weights
            ii=np.delete(ii,np.where(np.in1d(ii,errors)))
            for i in range(len(w)):
                ax1.plot(k*x[np.where(np.in1d(ii,w[i]))],
                        res[np.where(np.in1d(ii,w[i]))],color+'o',markersize=size[i])
        else:
            #without weight
            if self._set_err:
                #using errors
                if self._corr_err: err=self._old_err
                else: err=self.err
                if no_plot_err>0: errors=np.append(errors,np.argsort(abs(err))[-no_plot_err:])  #remove errorful points
                ii=np.delete(ii,np.where(np.in1d(ii,errors)))
                ax1.errorbar(k*x[ii],res[ii],yerr=err[ii]*100,fmt=color+'o',markersize=5)
            else:
                #without errors
                ii=np.delete(ii,np.where(np.in1d(ii,errors)))
                ax1.plot(k*x[ii],res[ii],color+'o')

        if double_ax:
            #setting secound axis
            ax2=ax1.twiny()
            #generate plot to obtain correct axis in epoch
            l=ax2.plot(k*x,res)
            ax2.set_xlabel('Phase')
            l.pop(0).remove()
            lims=np.array(ax1.get_xlim())/k+offset
            ph=self.Phase(t0,P,lims)
            ax2.set_xlim(ph)

        if name is not None:
            mpl.savefig(name+'.png',bbox_inches='tight')
            if eps: mpl.savefig(name+'.eps',bbox_inches='tight')
            mpl.close(fig)
        return fig


    def SaveModel(self,name,t_min=None,t_max=None,n=1000,phase=False,params=None):
        '''save model curve to file
        name - name of output file
        t_min - minimal value of time
        t_max - maximal value of time
        n - number of data points
        phase - export phase curve (min/max value give in t_min/t_max as phase)
        params - parameters of model (if not given, used "params" from class)
        '''

        if params is None: params=self.params

        #get linear ephemeris
        if 't0' in params: t0=params['t0']
        elif len(self.phase)==len(self.t): t0=self._t0P[0]
        else: raise TypeError('Phase not callculated! Run function "Phase" before it.')

        if 'P' in params: P=params['P']
        elif len(self.phase)==len(self.t): P=self._t0P[1]
        else: raise TypeError('Phase not callculated! Run function "Phase" before it.')

        if phase:
            #convert to times
            if t_min is None or t_max is None:
                raise TypeError('"t_min" or "t_max" is not given!')
            t_min=t0+P*t_min
            t_max=t0+P*t_max

        #same interval of time like in plot
        if len(self.t)<1000: dt=50*(self.t[-1]-self.t[0])/1000.
        else: dt=0.05*(self.t[-1]-self.t[0])

        if t_min is None: t_min=min(self.t)-dt
        if t_max is None: t_max=max(self.t)+dt

        t=np.linspace(t_min,t_max,n)
        phase=self.Phase(t0,P,t)

        model=self.Model(t,params)

        f=open(name,'w')
        np.savetxt(f,np.column_stack((t,phase,model)),fmt=["%14.7f",'%8.5f',"%12.10f"]
                   ,delimiter='    ',header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(8,' ')
                   +'    '+'Model curve')
        f.close()


    def SaveRes(self,name,params=None,weight=None):
        '''save residue to file
        name - name of output file
        params - parameters of model (if not given, used "params" from class)
        weight - weights of input data points

        warning: weights have to be in same order as input date!
        '''

        if params is None: params=self.params

        #get linear ephemeris
        if 't0' in params: t0=params['t0']
        elif len(self.phase)==len(self.t): t0=self._t0P[0]
        else: raise TypeError('Phase not callculated! Run function "Phase" before it.')

        if 'P' in params: P=params['P']
        elif len(self.phase)==len(self.t): P=self._t0P[1]
        else: raise TypeError('Phase not callculated! Run function "Phase" before it.')

        model=self.Model(self.t,params)
        phase=self.Phase(t0,P)

        res=self.flux-model
        f=open(name,'w')
        if self._set_err:
            if self._corr_err: err=self._old_err
            else: err=self.err
            np.savetxt(f,np.column_stack((self.t,phase,res,err)),
                       fmt=["%14.7f",'%8.5f',"%13.10f","%.10f"],delimiter="    ",
                       header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(10,' ')
                       +'    '+'Residue'.ljust(13,' ')+'    Error')
        elif weight is not None:
            np.savetxt(f,np.column_stack((self.t,phase,res,np.array(weight)[self._order])),
                       fmt=["%14.7f",'%8.5f',"%13.10f","%.10f"],delimiter="    ",
                       header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(10,' ')
                       +'    '+'Residue'.ljust(13,' ')+'    Weight')
        else:
            np.savetxt(f,np.column_stack((self.t,phase,res)),
                       fmt=["%14.7f",'%8.5f',"%13.10f"],delimiter="    ",
                       header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(10,' ')
                       +'    '+'Residue')
        f.close()



class FitTransitLoad(FitTransit):
    '''loading saved data, model... from FitTransit class'''
    def __init__(self,path):
        '''loading data, model, parameters... from file'''
        super().__init__([0],[0],[0])

        self.Load(path)

