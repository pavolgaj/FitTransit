# -*- coding: utf-8 -*-

#main classes of TransitFit package
#version 0.1.1
#update: ?.?.2022
# (c) Pavol Gajdos, 2022

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

def GetMax(x,n):
    '''return n max values in array x'''
    temp=[]
    x=np.array(x)
    for i in range(n):
        temp.append(np.argmax(x))
        x[temp[-1]]=0
    return np.array(temp)

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

# =============================================================================
# def Epoch(t,t0,P,dE=0.5):
#     '''calculate epoch with epoch diffence between minima dE'''
#     E_obs=(t-t0)/P  #observed epoch
#     f_obs=E_obs-np.round(E_obs)  #observed phase
#
#     secondary=np.where(np.abs(f_obs)>np.minimum(np.abs(f_obs-dE),np.abs(f_obs-dE+1)))
#     min_type=np.zeros(t.shape)
#     min_type[secondary]=1
#
#     E=np.round(E_obs-min_type*dE)+min_type*dE
#     return E,min_type
#
# =============================================================================
# =============================================================================
# class Common():
#     def QuadTerm(self,M1=0,M2=0,M1_err=0,M2_err=0):
#         '''calculate some params for quadratic model'''
#         output={}
#         if not 'Q' in self.params: return output
#         if self.params['Q']==0: return output
#         self.paramsMore['dP']=2*self.params['Q']/self.params['P']
#         dP=2*self.params['Q']/self.params['P']**2
#         self.paramsMore['dP/P']=dP
#         output['dP']=self.paramsMore['dP']
#         output['dP/P']=self.paramsMore['dP/P']
#
#         if len(self.params_err)>0:
#             #calculate error of params
#             #get errors of params of model
#             if 'P' in self.params_err: P_err=self.params_err['P']
#             else: P_err=0
#             if 'Q' in self.params_err: Q_err=self.params_err['Q']
#             else: Q_err=0
#
#             self.paramsMore_err['dP']=self.paramsMore['dP']*np.sqrt((P_err/self.params['P'])**2+\
#                                       (Q_err/self.params['Q'])**2)
#             dP_err=dP*np.sqrt((2*P_err/self.params['P'])**2+(Q_err/self.params['Q'])**2)
#             self.paramsMore_err['dP/P']=dP_err
#
#             #if some errors = 0, del them; and return only non-zero errors
#             if self.paramsMore_err['dP']==0: del self.paramsMore_err['dP']
#             else: output['dP_err']=self.paramsMore_err['dP']
#             if self.paramsMore_err['dP/P']==0: del self.paramsMore_err['dP/P']
#             else: output['dP/P_err']=self.paramsMore_err['dP/P']
#
#         if M1*M2==0 and hasattr(self,'systemParams'):
#             if 'M1' in self.systemParams: M1=self.systemParams['M1']
#             if 'M2' in self.systemParams: M2=self.systemParams['M2']
#             if 'M1_err' in self.systemParams: M1_err=self.systemParams['M1_err']
#             if 'M2_err' in self.systemParams: M2_err=self.systemParams['M2_err']
#
#         if M1*M2>0:
#             if M1<M2:
#                 #change M1<->M2 (M1>M2)
#                 mm=M1
#                 M1=M2
#                 M2=mm
#                 mm=M1_err
#                 M1_err=M2_err
#                 M2_err=mm
#
#             self.paramsMore['dM']=M1*M2/(3*(M1-M2))*dP*year
#             output['dM']=self.paramsMore['dM']
#
#             if len(self.params_err)>0:
#                 #calculate error of params
#                 self.paramsMore_err['dM']=self.paramsMore['dM']*np.sqrt((dP_err/dP)**2+\
#                                           ((M1-M2-M1**2)*M1_err/M1)**2+((M1-M2+M2**2)*M2_err/M2)**2)
#
#                 #if some errors = 0, del them; and return only non-zero errors
#                 if self.paramsMore_err['dM']==0: del self.paramsMore_err['dM']
#                 else: output['dM_err']=self.paramsMore_err['dM']
#
#         return output
#
#     def SaveOC(self,name,t0=None,P=None,weight=None):
#         '''saving O-C calculated from given ephemeris to file
#         name - name of file
#         t0 - time of zeros epoch (necessary if not given in model or epoch not calculated)
#         P - period (necessary if not given in model or epoch not calculated)
#         weight - weight of data
#         warning: weights have to be in same order as input date!
#         '''
#
#         #get linear ephemeris
#         if len(self.epoch)==len(self.t): t0=self._t0P[0]
#         elif t0 is None: raise TypeError('t0 is not given!')
#
#         if len(self.epoch)==len(self.t): P=self._t0P[1]
#         elif P is None: raise TypeError('P is not given!')
#
#         old_epoch=self.epoch
#         if not len(self.epoch)==len(self.t): self.Epoch(self.t,t0,P)
#
#         f=open(name,'w')
#         if weight is not None:
#             np.savetxt(f,np.column_stack((self.t,self.epoch,self.oc,np.array(weight)[self._order])),
#                        fmt=["%14.7f",'%10.3f',"%+12.10f","%.10f"],delimiter="    ",
#                        header='Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                        +'    '+'O-C'.ljust(12,' ')+'    '+'Weight')
#         elif self._set_err:
#             if self._corr_err: err=self._old_err
#             else: err=self.err
#             np.savetxt(f,np.column_stack((self.t,self.epoch,self.oc,err)),
#                        fmt=["%14.7f",'%10.3f',"%+12.10f","%.10f"],delimiter="    ",
#                        header='Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                        +'    '+'O-C'.ljust(12,' ')+'    '+'Error')
#         else:
#             np.savetxt(f,np.column_stack((self.t,self.epoch,self.oc)),
#                        fmt=["%14.7f",'%10.3f',"%+12.10f"],delimiter="    ",
#                        header='Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                        +'    '+'O-C')
#         f.close()
#         self.epoch=old_epoch
# =============================================================================

# =============================================================================
#
# class SimpleFit(Common):
#     '''class with common function for FitLinear and FitQuad'''
#     def __init__(self,t,t0,P,oc=None,err=None,dE=0.5):
#         '''input: observed time, time of zeros epoch, period, (O-C values, errors)'''
#         self.t=np.array(t)     #times
#
#         #linear ephemeris of binary
#         self.P=P
#         self.t0=t0
#         self._t0P=[t0,P]   #given linear ephemeris of binary
#
#         self.dE=dE   #diffence in epoch between primary and secondary minima
#
#         if oc is None:
#             #calculate O-C
#             self.Epoch()
#             tC=t0+P*self.epoch
#             self.oc=self.t-tC
#         else: self.oc=np.array(oc)
#
#         if err is None:
#             #errors not given
#             self.err=np.ones(self.t.shape)
#             self._set_err=False
#         else:
#             #errors given
#             self.err=np.array(err)
#             self._set_err=True
#         self._corr_err=False
#         self._calc_err=False
#         self._old_err=[]
#
#         #sorting data...
#         self._order=np.argsort(self.t)
#         self.t=self.t[self._order]      #times
#         self.oc=self.oc[self._order]    #O-Cs
#         self.err=self.err[self._order]  #errors
#         self._min_type=[]       #type of minima (primary=0 / secondary=1)
#
#         self.Epoch()
#         self.params={}         #values of parameters
#         self.params_err={}     #errors of fitted parameters
#         self.paramsMore={}      #values of parameters calculated from model params
#         self.paramsMore_err={}  #errors of calculated parameters
#         self.model=[]          #model O-C
#         self.new_oc=[]         #new O-C (residue)
#         self.chi=0
#         self._fit=''
#         self.tC=[]
#
#     def Epoch(self):
#         '''calculate epoch'''
#         self.epoch,self._min_type=Epoch(self.t,self.t0,self.P,self.dE)
#         return self.epoch
#
#     def PhaseCurve(self,P,t0,plot=False):
#         '''create phase curve'''
#         f=np.mod(self.t-t0,P)/float(P)    #phase
#         order=np.argsort(f)
#         f=f[order]
#         oc=self.oc[order]
#         if plot:
#             mpl.figure()
#             if self._set_err: mpl.errorbar(f,oc,yerr=self.err,fmt='o')
#             else: mpl.plot(f,oc,'.')
#         return f,oc
#
#     def Summary(self,name=None):
#         '''parameters summary, writting to file "name"'''
#         params=list(self.params.keys())
#         units={'t0':'JD','P':'d','Q':'d'}
#
#         text=['original ephemeris']
#         text.append('------------------------------------')
#         text.append('parameter'.ljust(15,' ')+'unit'.ljust(10,' ')+'value')
#         text.append('P'.ljust(15,' ')+'d'.ljust(10,' ')+str(self._t0P[1]))
#         text.append('t0'.ljust(15,' ')+'JD'.ljust(10,' ')+str(self._t0P[0]))
#         text.append('------------------------------------\n')
#
#         text.append('parameter'.ljust(15,' ')+'unit'.ljust(10,' ')+'value'.ljust(30,' ')+'error')
#
#         for p in sorted(params):
#             if p in self.params_err: err=str(self.params_err[p])
#             else: err='fixed'  #fixed params
#
#             text.append(p.ljust(15,' ')+units[p].ljust(10,' ')+str(self.params[p]).ljust(30,' ')+err.ljust(30,' '))
#
#         self.QuadTerm()
#         if len(self.paramsMore)>0: text.append('')
#         params=[]
#         vals=[]
#         err=[]
#         unit=[]
#         for x in sorted(self.paramsMore.keys()):
#             #names, units, values and errors of more params
#             params.append(x)
#             vals.append(str(self.paramsMore[x]))
#             if not len(self.paramsMore_err)==0:
#                 #errors calculated
#                 if x in self.paramsMore_err:
#                     err.append(str(self.paramsMore_err[x]))
#                 else: err.append('---')   #errors not calculated
#             else: err.append('---')  #errors not calculated
#             #add units
#             if x=='dM': unit.append('M_sun/yr')
#             elif x=='dP':
#                 unit.append('d/d')
#                 #also in years
#                 params.append(x)
#                 vals.append(str(self.paramsMore[x]*year))
#                 err.append(str(float(err[-1])*year))
#                 unit.append('d/yr')
#             elif x=='dP/P': unit.append('1/d')
#
#         for i in range(len(params)):
#             text.append(params[i].ljust(15,' ')+unit[i].ljust(10,' ')+vals[i].ljust(30,' ')+err[i].ljust(30,' '))
#
#         text.append('')
#         text.append('Fitting method: '+self._fit)
#         g=len(params)
#         n=len(self.t)
#         text.append('chi2 = '+str(self.chi))
#         if n-g>0: text.append('chi2_r = '+str(self.chi/(n-g)))
#         else: text.append('chi2_r = NA')
#         text.append('AIC = '+str(self.chi+2*g))
#         if n-g-1>0: text.append('AICc = '+str(self.chi+2*g*n/(n-g-1)))
#         else: text.append('AICc = NA')
#         text.append('BIC = '+str(self.chi+g*np.log(n)))
#         if name is None:
#             print('------------------------------------')
#             for t in text: print(t)
#             print('------------------------------------')
#         else:
#             f=open(name,'w')
#             for t in text: f.write(t+'\n')
#             f.close()
#
#     def InfoMCMC(self,db,eps=False):
#         '''statistics about GA fitting'''
#         info=InfoMCClass(db)
#         info.AllParams(eps)
#
#         for p in info.pars: info.OneParam(p,eps)
#
#     def CalcErr(self):
#         '''calculate errors according to current model'''
#         n=len(self.model)
#         err=np.sqrt(sum((self.oc-self.model)**2)/(n*(n-1)))
#         errors=err*np.ones(self.model.shape)*np.sqrt(n-len(self.params))
#         chi=sum(((self.oc-self.model)/errors)**2)
#         print('New chi2:',chi,chi/(n-len(self.params)))
#         self._calc_err=True
#         self._set_err=False
#         self.err=errors
#         return errors
#
#     def CorrectErr(self):
#         '''scaling errors according to current model'''
#         n=len(self.model)
#         chi0=sum(((self.oc-self.model)/self.err)**2)
#         alfa=chi0/(n-2)
#         err=self.err*np.sqrt(alfa)
#         chi=sum(((self.oc-self.model)/err)**2)
#         print('New chi2:',chi,chi/(n-len(self.params)))
#         if self._set_err and len(self._old_err)==0: self._old_err=self.err
#         self.err=err
#         self._corr_err=True
#         return err
#
#     def AddWeight(self,weight):
#         '''adding weight to data + scaling according to current model
#         warning: weights have to be in same order as input date!
#         '''
#         if not len(weight)==len(self.t):
#             print('incorrect length of "w"!')
#             return
#         weight=np.array(weight)[self._order]
#         err=1./weight
#         n=len(self.t)
#         chi0=sum(((self.oc-self.model)/err)**2)
#         alfa=chi0/(n-len(self.params))
#         err*=np.sqrt(alfa)
#         chi=sum(((self.oc-self.model)/err)**2)
#         print('New chi2:',chi,chi/(n-len(self.params)))
#         self._calc_err=True
#         self._set_err=False
#         self.err=err
#         return err
#
#
#     def SaveRes(self,name,weight=None):
#         '''saving residue (new O-C) to file
#         name - name of file
#         weight - weight of data
#         warning: weights have to be in same order as input date!
#         '''
#         f=open(name,'w')
#         if self._set_err:
#             if self._corr_err: err=self._old_err
#             else: err=self.err
#             np.savetxt(f,np.column_stack((self.t,self.epoch,self.new_oc,err)),
#                        fmt=["%14.7f",'%10.3f',"%+12.10f","%.10f"],delimiter="    ",
#                        header='Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                        +'    '+'new O-C'.ljust(12,' ')+'    Error')
#         elif weight is not None:
#             np.savetxt(f,np.column_stack((self.t,self.epoch,self.new_oc,np.array(weight)[self._order])),
#                        fmt=["%14.7f",'%10.3f',"%+12.10f","%.10f"],delimiter="    ",
#                        header='Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                        +'    '+'new O-C'.ljust(12,' ')+'    Weight')
#         else:
#             np.savetxt(f,np.column_stack((self.t,self.epoch,self.new_oc)),
#                        fmt=["%14.7f",'%10.3f',"%+12.10f"],delimiter="    ",
#                        header='Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                        +'    new O-C')
#         f.close()
#
#     def SaveModel(self,name,E_min=None,E_max=None,n=1000):
#         '''save model curve of O-C to file
#         name - name of output file
#         E_min - minimal value of epoch
#         E_max - maximal value of epoch
#         n - number of data points
#         '''
#         #same interval of epoch like in plot
#         if len(self.epoch)<1000: dE=50*(self.epoch[-1]-self.epoch[0])/1000.
#         else: dE=0.05*(self.epoch[-1]-self.epoch[0])
#
#         if E_min is None: E_min=min(self.epoch)-dE
#         if E_max is None: E_max=max(self.epoch)+dE
#
#         E=np.linspace(E_min,E_max,n)
#
#         tC=self._t0P[0]+self._t0P[1]*E
#         p=[]
#         if 'Q' in self.params:
#             #Quad Model
#             p.append(self.params['Q'])
#         p+=[self.params['P']-self._t0P[1],self.params['t0']-self._t0P[0]]
#         new=np.polyval(p,E)
#
#         f=open(name,'w')
#         np.savetxt(f,np.column_stack((tC+new,E,new)),fmt=["%14.7f",'%10.3f',"%+12.10f"]
#                    ,delimiter='    ',header='Obs. Time'.ljust(14,' ')+'    '+'Epoch'.ljust(10,' ')
#                    +'    model O-C')
#         f.close()
#
#     def PlotRes(self,name=None,no_plot=0,no_plot_err=0,eps=False,oc_min=True,
#                 time_type='JD',offset=2400000,trans=True,title=None,epoch=False,
#                 min_type=False,weight=None,trans_weight=False,bw=False,double_ax=False,
#                 fig_size=None):
#         '''plotting residue (new O-C)
#         name - name of file to saving plot (if not given -> show graph)
#         no_plot - count of outlier point which will not be plot
#         no_plot_err - count of errorful point which will not be plot
#         eps - save also as eps file
#         oc_min - O-C in minutes (if False - days)
#         time_type - type of JD in which is time (show in x label)
#         offset - offset of time
#         trans - transform time according to offset
#         title - name of graph
#         epoch - x axis in epoch
#         min_type - distinction of type of minimum
#         weight - weight of data (shown as size of points)
#         trans_weight - transform weights to range (1,10)
#         bw - Black&White plot
#         double_ax - two axes -> time and epoch
#         fig_size - custom figure size - e.g. (12,6)
#
#         warning: weights have to be in same order as input data!
#         '''
#
#         if fig_size:
#             fig=mpl.figure(figsize=fig_size)
#         else:
#             fig=mpl.figure()
#
#         ax1=fig.add_subplot(111)
#         #setting labels
#         if epoch and not double_ax:
#             ax1.set_xlabel('Epoch')
#             x=self.epoch
#         elif offset>0:
#             ax1.set_xlabel('Time ('+time_type+' - '+str(offset)+')')
#             if not trans: offset=0
#             x=self.t-offset
#         else:
#             ax1.set_xlabel('Time ('+time_type+')')
#             offset=0
#             x=self.t
#
#         if oc_min:
#             ax1.set_ylabel('Residue O - C (min)')
#             k=minutes
#         else:
#             ax1.set_ylabel('Residue O - C (d)')
#             k=1
#
#         if title is not None:
#             if double_ax: fig.subplots_adjust(top=0.85)
#             fig.suptitle(title,fontsize=20)
#
#         #primary / secondary minimum
#         if min_type:
#             prim=np.where(self._min_type==0)
#             sec=np.where(self._min_type==1)
#         else:
#             prim=np.arange(0,len(self.epoch),1)
#             sec=np.array([])
#
#         #set weight
#         set_w=False
#         if weight is not None:
#             weight=np.array(weight)[self._order]
#             if trans_weight:
#                 w_min=min(weight)
#                 w_max=max(weight)
#                 weight=9./(w_max-w_min)*(weight-w_min)+1
#             if weight.shape==self.t.shape:
#                 w=[]
#                 levels=[0,3,5,7.9,10]
#                 size=[3,4,5,7]
#                 for i in range(len(levels)-1):
#                     w.append(np.where((weight>levels[i])*(weight<=levels[i+1])))
#                 w[-1]=np.append(w[-1],np.where(weight>levels[-1]))  #if some weight is bigger than max. level
#                 set_w=True
#             else:
#                 warnings.warn('Shape of "weight" is different to shape of "time". Weight will be ignore!')
#
#         if bw: color='k'
#         else: color='b'
#         errors=GetMax(abs(self.new_oc),no_plot)
#         if set_w:
#             #using weights
#             prim=np.delete(prim,np.where(np.in1d(prim,errors)))
#             sec=np.delete(sec,np.where(np.in1d(sec,errors)))
#             if not len(prim)==0:
#                 for i in range(len(w)):
#                     ax1.plot(x[prim[np.where(np.in1d(prim,w[i]))]],
#                              (self.new_oc*k)[prim[np.where(np.in1d(prim,w[i]))]],color+'o',markersize=size[i])
#             if not len(sec)==0:
#                 for i in range(len(w)):
#                     ax1.plot(x[sec[np.where(np.in1d(sec,w[i]))]],
#                              (self.new_oc*k)[sec[np.where(np.in1d(sec,w[i]))]],color+'o',markersize=size[i],
#                              fillstyle='none',markeredgewidth=1,markeredgecolor=color)
#
#         else:
#             #without weight
#             if self._set_err:
#                 #using errors
#                 if self._corr_err: err=self._old_err
#                 else: err=self.err
#                 errors=np.append(errors,GetMax(err,no_plot_err))
#                 prim=np.delete(prim,np.where(np.in1d(prim,errors)))
#                 sec=np.delete(sec,np.where(np.in1d(sec,errors)))
#                 if not len(prim)==0:
#                     ax1.errorbar(x[prim],(self.new_oc*k)[prim],yerr=(err*k)[prim],fmt=color+'o',markersize=5)
#                 if not len(sec)==0:
#                     ax1.errorbar(x[sec],(self.new_oc*k)[sec],yerr=(err*k)[sec],fmt=color+'o',markersize=5,
#                                  fillstyle='none',markeredgewidth=1,markeredgecolor=color)
#
#             else:
#                 #without errors
#                 prim=np.delete(prim,np.where(np.in1d(prim,errors)))
#                 sec=np.delete(sec,np.where(np.in1d(sec,errors)))
#                 if not len(prim)==0:
#                     ax1.plot(x[prim],(self.new_oc*k)[prim],color+'o')
#                 if not len(sec)==0:
#                     ax1.plot(x[sec],(self.new_oc*k)[sec],color+'o',
#                              mfc='none',markeredgewidth=1,markeredgecolor=color)
#
#         if double_ax:
#             #setting secound axis
#             ax2=ax1.twiny()
#             #generate plot to obtain correct axis in epoch
#             l=ax2.plot(self.epoch,self.oc*k)
#             ax2.set_xlabel('Epoch')
#             l.pop(0).remove()
#             lims=np.array(ax1.get_xlim())
#             epoch=np.round((lims-self.t0)/self.P*2)/2.
#             ax2.set_xlim(epoch)
#
#         if name is None: mpl.show()
#         else:
#             mpl.savefig(name+'.png')
#             if eps: mpl.savefig(name+'.eps')
#             mpl.close(fig)
#
#     def Plot(self,name=None,no_plot=0,no_plot_err=0,eps=False,oc_min=True,
#              time_type='JD',offset=2400000,trans=True,title=None,epoch=False,
#              min_type=False,weight=None,trans_weight=False,bw=False,double_ax=False,
#              fig_size=None):
#         '''plotting original O-C with linear fit
#         name - name of file to saving plot (if not given -> show graph)
#         no_plot - count of outlier point which will not be plot
#         no_plot_err - count of errorful point which will not be plot
#         eps - save also as eps file
#         oc_min - O-C in minutes (if False - days)
#         time_type - type of JD in which is time (show in x label)
#         offset - offset of time
#         trans - transform time according to offset
#         title - name of graph
#         epoch - x axis in epoch
#         min_type - distinction of type of minimum
#         weight - weight of data (shown as size of points)
#         trans_weight - transform weights to range (1,10)
#         bw - Black&White plot
#         double_ax - two axes -> time and epoch
#         fig_size - custom figure size - e.g. (12,6)
#
#         warning: weights have to be in same order as input data!
#         '''
#
#         if fig_size:
#             fig=mpl.figure(figsize=fig_size)
#         else:
#             fig=mpl.figure()
#
#         ax1=fig.add_subplot(111)
#         #setting labels
#         if epoch and not double_ax:
#             ax1.set_xlabel('Epoch')
#             x=self.epoch
#         elif offset>0:
#             ax1.set_xlabel('Time ('+time_type+' - '+str(offset)+')')
#             if not trans: offset=0
#             x=self.t-offset
#         else:
#             ax1.set_xlabel('Time ('+time_type+')')
#             offset=0
#             x=self.t
#
#         if oc_min:
#             ax1.set_ylabel('O - C (min)')
#             k=minutes
#         else:
#             ax1.set_ylabel('O - C (d)')
#             k=1
#
#         if title is not None:
#             if double_ax: fig.subplots_adjust(top=0.85)
#             fig.suptitle(title,fontsize=20)
#
#         if not len(self.model)==len(self.t):
#             no_plot=0
#
#         #primary / secondary minimum
#         if min_type:
#             prim=np.where(self._min_type==0)
#             sec=np.where(self._min_type==1)
#         else:
#             prim=np.arange(0,len(self.epoch),1)
#             sec=np.array([])
#
#         #set weight
#         set_w=False
#         if weight is not None:
#             weight=np.array(weight)[self._order]
#             if trans_weight:
#                 w_min=min(weight)
#                 w_max=max(weight)
#                 weight=9./(w_max-w_min)*(weight-w_min)+1
#             if weight.shape==self.t.shape:
#                 w=[]
#                 levels=[0,3,5,7.9,10]
#                 size=[3,4,5,7]
#                 for i in range(len(levels)-1):
#                     w.append(np.where((weight>levels[i])*(weight<=levels[i+1])))
#                 w[-1]=np.append(w[-1],np.where(weight>levels[-1]))  #if some weight is bigger than max. level
#                 set_w=True
#             else:
#                 warnings.warn('Shape of "weight" is different to shape of "time". Weight will be ignore!')
#
#         if bw: color='k'
#         else: color='b'
#         if len(self.new_oc)==len(self.oc): errors=GetMax(abs(self.new_oc),no_plot)  #remove outlier points
#         else: errors=np.array([])
#         if set_w:
#             #using weights
#             prim=np.delete(prim,np.where(np.in1d(prim,errors)))
#             sec=np.delete(sec,np.where(np.in1d(sec,errors)))
#             if not len(prim)==0:
#                 for i in range(len(w)):
#                     ax1.plot(x[prim[np.where(np.in1d(prim,w[i]))]],
#                              (self.oc*k)[prim[np.where(np.in1d(prim,w[i]))]],color+'o',markersize=size[i],zorder=1)
#             if not len(sec)==0:
#                 for i in range(len(w)):
#                     ax1.plot(x[sec[np.where(np.in1d(sec,w[i]))]],
#                              (self.oc*k)[sec[np.where(np.in1d(sec,w[i]))]],color+'o',markersize=size[i],
#                              fillstyle='none',markeredgewidth=1,markeredgecolor=color,zorder=1)
#
#         else:
#             #without weight
#             if self._set_err:
#                 #using errors
#                 if self._corr_err: err=self._old_err
#                 else: err=self.err
#                 errors=np.append(errors,GetMax(err,no_plot_err))  #remove errorful points
#                 prim=np.delete(prim,np.where(np.in1d(prim,errors)))
#                 sec=np.delete(sec,np.where(np.in1d(sec,errors)))
#                 if not len(prim)==0:
#                     ax1.errorbar(x[prim],(self.oc*k)[prim],yerr=(err*k)[prim],fmt=color+'o',markersize=5,zorder=1)
#                 if not len(sec)==0:
#                     ax1.errorbar(x[sec],(self.oc*k)[sec],yerr=(err*k)[sec],fmt=color+'o',markersize=5,
#                                  fillstyle='none',markeredgewidth=1,markeredgecolor=color,zorder=1)
#
#             else:
#                 #without errors
#                 prim=np.delete(prim,np.where(np.in1d(prim,errors)))
#                 sec=np.delete(sec,np.where(np.in1d(sec,errors)))
#                 if not len(prim)==0:
#                     ax1.plot(x[prim],(self.oc*k)[prim],color+'o',zorder=1)
#                 if not len(sec)==0:
#                     ax1.plot(x[sec],(self.oc*k)[sec],color+'o',
#                              mfc='none',markeredgewidth=1,markeredgecolor=color,zorder=1)
#
#         #plot linear model
#         if bw:
#             color='k'
#             lw=2
#         else:
#             color='r'
#             lw=1
#
#         if len(self.model)==len(self.t):
#             #model was calculated
#             if len(self.t)<1000:
#                 dE=(self.epoch[-1]-self.epoch[0])/1000.
#                 E=np.linspace(self.epoch[0]-50*dE,self.epoch[-1]+50*dE,1100)
#             else:
#                 dE=(self.epoch[-1]-self.epoch[0])/len(self.epoch)
#                 E=np.linspace(self.epoch[0]-0.05*len(self.epoch)*dE,self.epoch[-1]+0.05*len(self.epoch)*dE,int(1.1*len(self.epoch)))
#             tC=self._t0P[0]+self._t0P[1]*E
#             p=[]
#             if 'Q' in self.params:
#                 #Quad Model
#                 p.append(self.params['Q'])
#             p+=[self.params['P']-self._t0P[1],self.params['t0']-self._t0P[0]]
#             new=np.polyval(p,E)
#
#             if epoch and not double_ax: ax1.plot(E,new*k,color,linewidth=lw,zorder=2)
#             else: ax1.plot(tC+new-offset,new*k,color,linewidth=lw,zorder=2)
#
#         if double_ax:
#             #setting secound axis
#             ax2=ax1.twiny()
#             #generate plot to obtain correct axis in epoch
#             if len(self.model)==len(self.t): l=ax2.plot(E,new*k,zorder=2)
#             else: l=ax2.plot(self.epoch,self.oc*k,zorder=2)
#             ax2.set_xlabel('Epoch')
#             l.pop(0).remove()
#             lims=np.array(ax1.get_xlim())
#             epoch=np.round((lims-self.t0)/self.P*2)/2.
#             ax2.set_xlim(epoch)
#
#
#         if name is None: mpl.show()
#         else:
#             mpl.savefig(name+'.png')
#             if eps: mpl.savefig(name+'.eps')
#             mpl.close(fig)
#
# =============================================================================

# =============================================================================
#
# class FitLinear(SimpleFit):
#     '''fitting of O-C diagram with linear function'''
#
#     def FitRobust(self,n_iter=10):
#         '''robust regresion
#         return: new O-C'''
#         self.FitLinear()
#         for i in range(n_iter): self.FitLinear(robust=True)
#         self._fit='Robust regression'
#         return self.new_oc
#
#     def FitLinear(self,robust=False):
#         '''simple linear regresion
#         return: new O-C'''
#         if robust:
#             err=self.err*np.exp(((self.oc-self.model)/(5*self.err))**4)
#             k=1
#             while np.inf in err:
#                 k*=10
#                 err=self.err*np.exp(((self.oc-self.model)/(5*k*self.err))**4)
#         else: err=self.err
#         w=1./err
#
#         p,cov=np.polyfit(self.epoch,self.oc,1,cov=True,w=w)
#
#         self.P=p[0]+self._t0P[1]
#         self.t0=p[1]+self._t0P[0]
#
#         self.params['P']=p[0]+self._t0P[1]
#         self.params['t0']=p[1]+self._t0P[0]
#
#         self.Epoch()
#         self.model=np.polyval(p,self.epoch)
#         self.chi=sum(((self.oc-self.model)/self.err)**2)
#
#         if robust:
#             n=len(self.t)*1.06*sum(1./err)/sum(1./self.err)
#             chi_m=1.23*sum(((self.oc-self.model)/err)**2)/(n-2)
#         else: chi_m=self.chi/(len(self.t)-2)
#
#         err=np.sqrt(chi_m*cov.diagonal())
#         self.params_err['P']=err[0]
#         self.params_err['t0']=err[1]
#
#         self.tC=self.t0+self.P*self.epoch
#         self.new_oc=self.oc-self.model
#
#         self._fit='Standard regression'
#         #remove some values calculated from old parameters
#         self.paramsMore={}
#         self.paramsMore_err={}
#
#         return self.new_oc
#
#     def FitMCMC(self,n_iter,limits,steps,fit_params=None,burn=0,binn=1,walkers=0,visible=True,db=None):
#         '''fitting with Markov chain Monte Carlo using emcee
#         n_iter - number of MC iteration - should be at least 1e5
#         limits - limits of parameters for fitting
#         steps - steps (width of normal distibution) of parameters for fitting
#         fit_params - list of fitted parameters
#         burn - number of removed steps before equilibrium - should be approx. 0.1-1% of n_iter
#         binn - binning size - should be around 10
#         walkers - number of walkers - should be at least 2-times number of fitted parameters
#         visible - display status of fitting
#         db - name of database to save MCMC fitting details (could be analysed later using InfoMCMC function)
#         '''
#
#         #setting emcee priors for fitted parameters
#         if fit_params is None: fit_params=['P','t0']
#         vals0={'P': self._t0P[1], 't0': self._t0P[0]}
#         vals1={}
#         priors={}
#         for p in ['P','t0']:
#             if p in self.params: vals1[p]=self.params[p]
#             else: vals1[p]=vals0[p]
#             if p in fit_params:
#                 priors[p]=_Prior("limuniform",lower=limits[p][0],upper=limits[p][1],name=p)
#
#         dims=len(fit_params)
#         if walkers==0: walkers=dims*2
#         elif walkers<dims * 2:
#             walkers=dims*2
#             warnings.warn('Numbers of walkers is smaller than two times number of free parameters. Auto-set to '+str(int(walkers))+'.')
#
#
#         def likeli(names, vals):
#             '''likelihood function for emcee'''
#             pp={n:v for n,v in zip(names,vals)}
#
#             if 'P' in pp: P=pp['P']
#             else: P=vals1['P']
#             if 't0' in pp: t0=pp['t0']
#             else: t0=vals1['t0']
#
#             tC=t0+P*self.epoch
#             chi=np.sum(((self.t-tC)/self.err)**2)
#
#             likeli=-0.5*chi
#             return likeli
#
#         def lnpostdf(values):
#             # Parameter-Value dictionary
#             ps = dict(zip(fit_params,values))
#             # Check prior information
#             prior_sum = 0
#             for name in fit_params: prior_sum += priors[name](ps, name)
#             # If log prior is negative infinity, parameters
#             # are out of range, so no need to evaluate the
#             # likelihood function at this step:
#             pdf = prior_sum
#             if pdf == -np.inf: return pdf
#             # Likelihood
#             pdf += likeli(fit_params, values)
#             return pdf
#
#         # Generate the sampler
#         emceeSampler=emcee.EnsembleSampler(int(walkers),int(dims),lnpostdf)
#
#         # Generate starting values
#         pos = []
#         for j in range(walkers):
#             pos.append(np.zeros(dims))
#             for i, n in enumerate(fit_params):
#                 # Trial counter -- avoid values beyond restrictions
#                 tc = 0
#                 while True:
#                     if tc == 100:
#                         raise ValueError('Could not determine valid starting point for parameter: "'+n+'" due to its limits! Try to change the limits and/or step.')
#                     propval = np.random.normal(vals1[n],steps[n])
#                     if propval < limits[n][0]:
#                         tc += 1
#                         continue
#                     if propval > limits[n][1]:
#                         tc += 1
#                         continue
#                     break
#                 pos[-1][i] = propval
#
#         # Default value for state
#         state = None
#
#         if burn>0:
#             # Run burn-in
#             pos,prob,state=emceeSampler.run_mcmc(pos,int(burn),progress=visible)
#             # Reset the chain to remove the burn-in samples.
#             emceeSampler.reset()
#
#         pos,prob,state=emceeSampler.run_mcmc(pos,int(n_iter),rstate0=state,thin=int(binn),progress=visible)
#
#         if not db is None:
#             sampleArgs={}
#             sampleArgs["burn"] = int(burn)
#             sampleArgs["binn"] = int(binn)
#             sampleArgs["iters"] = int(n_iter)
#             sampleArgs["nwalker"] = int(walkers)
#             np.savez_compressed(open(db,'wb'),chain=emceeSampler.chain,lnp=emceeSampler.lnprobability,                               pnames=list(fit_params),sampleArgs=sampleArgs)
#
#         self.params_err={} #remove errors of parameters
#
#         for p in ['P','t0']:
#             #calculate values and errors of parameters and save them
#             if p in fit_params:
#                 i=fit_params.index(p)
#                 self.params[p]=np.mean(emceeSampler.flatchain[:,i])
#                 self.params_err[p]=np.std(emceeSampler.flatchain[:,i])
#             else:
#                 self.params[p]=vals1[p]
#                 #self.params_err[p]='---'
#
#         self.Epoch()
#         self.tC=self.params['t0']+self.params['P']*self.epoch
#         self.new_oc=self.t-self.tC
#         self.model=self.oc+self.new_oc
#
#         self.chi=sum(((self.oc-self.model)/self.err)**2)
#
#         self._fit='MCMC'
#         #remove some values calculated from old parameters
#         self.paramsMore={}
#         self.paramsMore_err={}
#
#         return self.new_oc
#
#     def FitMCMC_old(self,n_iter,limits,steps,fit_params=None,burn=0,binn=1,visible=True,db=None):
#         '''fitting with Markov chain Monte Carlo using pymc
#         n_iter - number of MC iteration - should be at least 1e5
#         limits - limits of parameters for fitting
#         steps - steps (width of normal distibution) of parameters for fitting
#         fit_params - list of fitted parameters
#         burn - number of removed steps before equilibrium - should be approx. 0.1-1% of n_iter
#         binn - binning size - should be around 10
#         visible - display status of fitting
#         db - name of database to save MCMC fitting details (could be analysed later using InfoMCMC function)
#         '''
#
#         #setting pymc sampling for fitted parameters
#         if fit_params is None: fit_params=['P','t0']
#         vals0={'P': self._t0P[1], 't0': self._t0P[0]}
#         vals={}
#         pars={}
#         for p in ['P','t0']:
#             if p in self.params: vals[p]=self.params[p]
#             else: vals[p]=vals0[p]
#             if p in fit_params:
#                 pars[p]=pymc.Uniform(p,lower=limits[p][0],upper=limits[p][1],value=vals[p])
#
#         def model_fun(**arg):
#             '''model function for pymc'''
#             if 'P' in arg: P=arg['P']
#             else: P=vals['P']
#             if 't0' in arg: t0=arg['t0']
#             else: t0=vals['t0']
#             return t0+P*self.epoch
#
#         #definition of pymc model
#         model=pymc.Deterministic(
#             eval=model_fun,
#             doc='model',
#             name='Model',
#             parents=pars,
#             trace=True,
#             plot=False)
#
#         #final distribution
#         if self._set_err or self._calc_err:
#             #if known errors of data -> normal/Gaussian distribution
#             y=pymc.Normal('y',mu=model,tau=1./self.err**2,value=self.t,observed=True)
#         else:
#             #if unknown errors of data -> Poisson distribution
#             #note: should cause wrong performance of fitting, rather use function CalcErr for obtained errors
#             y=pymc.Poisson('y',mu=model,value=self.t,observed=True)
#
#         #adding final distribution and sampling of parameters to model
#         Model=[y]
#         for v in pars.values():
#             Model.append(v)
#
#         #create pymc object
#         if db is None: R=pymc.MCMC(Model)
#         else:
#             #saving MCMC fitting details
#             path=db.replace('\\','/')   #change dirs in path (for Windows)
#             if path.rfind('/')>0:
#                 path=path[:path.rfind('/')+1]  #find current dir of db file
#                 if not os.path.isdir(path): os.mkdir(path) #create dir of db file, if not exist
#             R=pymc.MCMC(Model,db='pickle',dbname=db)
#
#         #setting pymc method - distribution and steps
#         for p in pars:
#             R.use_step_method(pymc.Metropolis,pars[p],proposal_sd=steps[p],
#                               proposal_distribution='Normal')
#
#         if not visible:
#             #hidden output
#             f = open(os.devnull, 'w')
#             out=sys.stdout
#             sys.stdout=f
#
#         R.sample(iter=n_iter,burn=burn,thin=binn)  #MCMC fitting/simulation
#
#         self.params_err={} #remove errors of parameters
#
#         for p in ['P','t0']:
#             #calculate values and errors of parameters and save them
#             if p in pars:
#                 self.params[p]=np.mean(pars[p].trace())
#                 self.params_err[p]=np.std(pars[p].trace())
#             else:
#                 self.params[p]=vals[p]
#                 #self.params_err[p]='---'
#
#
#         if not visible:
#             #hidden output
#             sys.stdout=out
#             f.close()
#
#         self.Epoch()
#         self.tC=self.params['t0']+self.params['P']*self.epoch
#         self.new_oc=self.t-self.tC
#         self.model=self.oc+self.new_oc
#
#         self.chi=sum(((self.oc-self.model)/self.err)**2)
#
#         self._fit='MCMC_old'
#         #remove some values calculated from old parameters
#         self.paramsMore={}
#         self.paramsMore_err={}
#
#         return self.new_oc
#
# =============================================================================

# =============================================================================
#
# class FitQuad(SimpleFit):
#     '''fitting of O-C diagram with quadratic function'''
#
#     def FitRobust(self,n_iter=10):
#         '''robust regresion
#         return: new O-C'''
#         self.FitQuad()
#         for i in range(n_iter): self.FitQuad(robust=True)
#         self._fit='Robust regression'
#         return self.new_oc
#
#     def FitQuad(self,robust=False):
#         '''simple linear regresion
#         return: new O-C'''
#         if robust:
#             err=self.err*np.exp(((self.oc-self.model)/(5*self.err))**4)
#             k=1
#             while np.inf in err:
#                 k*=10
#                 err=self.err*np.exp(((self.oc-self.model)/(5*k*self.err))**4)
#         else: err=self.err
#         p,cov=np.polyfit(self.epoch,self.oc,2,cov=True,w=1./err)
#
#         self.Q=p[0]
#         self.P=p[1]+self._t0P[1]
#         self.t0=p[2]+self._t0P[0]
#
#         self.params['Q']=p[0]
#         self.params['P']=p[1]+self._t0P[1]
#         self.params['t0']=p[2]+self._t0P[0]
#
#         self.Epoch()
#         self.model=np.polyval(p,self.epoch)
#         self.chi=sum(((self.oc-self.model)/self.err)**2)
#
#         if robust:
#             n=len(self.t)*1.06*sum(1./err)/sum(1./self.err)
#             chi_m=1.23*sum(((self.oc-self.model)/err)**2)/(n-3)
#         else: chi_m=self.chi/(len(self.t)-3)
#
#         err=np.sqrt(chi_m*cov.diagonal())
#         self.params_err['Q']=err[0]
#         self.params_err['P']=err[1]
#         self.params_err['t0']=err[2]
#
#         self.tC=self.t0+self.P*self.epoch+self.Q*self.epoch**2
#         self.new_oc=self.oc-self.model
#
#         self._fit='Standard regression'
#         #remove some values calculated from old parameters
#         self.paramsMore={}
#         self.paramsMore_err={}
#
#         return self.new_oc
#
#     def FitMCMC(self,n_iter,limits,steps,fit_params=None,burn=0,binn=1,walkers=0,visible=True,db=None):
#         '''fitting with Markov chain Monte Carlo using emcee
#         n_iter - number of MC iteration - should be at least 1e5
#         limits - limits of parameters for fitting
#         steps - steps (width of normal distibution) of parameters for fitting
#         fit_params - list of fitted parameters
#         burn - number of removed steps before equilibrium - should be approx. 0.1-1% of n_iter
#         binn - binning size - should be around 10
#         walkers - number of walkers - should be at least 2-times number of fitted parameters
#         visible - display status of fitting
#         db - name of database to save MCMC fitting details (could be analysed later using InfoMCMC function)
#         '''
#
#         #setting emcee priors for fitted parameters
#         if fit_params is None: fit_params=['Q','P','t0']
#         vals0={'P': self._t0P[1], 't0': self._t0P[0], 'Q':0}
#         vals1={}
#         priors={}
#         for p in ['P','t0','Q']:
#             if p in self.params: vals1[p]=self.params[p]
#             else: vals1[p]=vals0[p]
#             if p in fit_params:
#                 priors[p]=_Prior("limuniform",lower=limits[p][0],upper=limits[p][1],name=p)
#
#         dims=len(fit_params)
#         if walkers==0: walkers=dims*2
#         elif walkers<dims * 2:
#             walkers=dims*2
#             warnings.warn('Numbers of walkers is smaller than two times number of free parameters. Auto-set to '+str(int(walkers))+'.')
#
#
#         def likeli(names, vals):
#             '''likelihood function for emcee'''
#             pp={n:v for n,v in zip(names,vals)}
#
#             if 'Q' in pp: Q=pp['Q']
#             else: Q=vals1['Q']
#             if 'P' in pp: P=pp['P']
#             else: P=vals1['P']
#             if 't0' in pp: t0=pp['t0']
#             else: t0=vals1['t0']
#
#             tC=t0+P*self.epoch+Q*self.epoch**2
#             chi=np.sum(((self.t-tC)/self.err)**2)
#
#             likeli=-0.5*chi
#             return likeli
#
#         def lnpostdf(values):
#             # Parameter-Value dictionary
#             ps = dict(zip(fit_params,values))
#             # Check prior information
#             prior_sum = 0
#             for name in fit_params: prior_sum += priors[name](ps, name)
#             # If log prior is negative infinity, parameters
#             # are out of range, so no need to evaluate the
#             # likelihood function at this step:
#             pdf = prior_sum
#             if pdf == -np.inf: return pdf
#             # Likelihood
#             pdf += likeli(fit_params, values)
#             return pdf
#
#         # Generate the sampler
#         emceeSampler=emcee.EnsembleSampler(int(walkers),int(dims),lnpostdf)
#
#         # Generate starting values
#         pos = []
#         for j in range(walkers):
#             pos.append(np.zeros(dims))
#             for i, n in enumerate(fit_params):
#                 # Trial counter -- avoid values beyond restrictions
#                 tc = 0
#                 while True:
#                     if tc == 100:
#                         raise ValueError('Could not determine valid starting point for parameter: "'+n+'" due to its limits! Try to change the limits and/or step.')
#                     propval = np.random.normal(vals1[n],steps[n])
#                     if propval < limits[n][0]:
#                         tc += 1
#                         continue
#                     if propval > limits[n][1]:
#                         tc += 1
#                         continue
#                     break
#                 pos[-1][i] = propval
#
#         # Default value for state
#         state = None
#
#         if burn>0:
#             # Run burn-in
#             pos,prob,state=emceeSampler.run_mcmc(pos,int(burn),progress=visible)
#             # Reset the chain to remove the burn-in samples.
#             emceeSampler.reset()
#
#         pos,prob,state=emceeSampler.run_mcmc(pos,int(n_iter),rstate0=state,thin=int(binn),progress=visible)
#
#         if not db is None:
#             sampleArgs={}
#             sampleArgs["burn"] = int(burn)
#             sampleArgs["binn"] = int(binn)
#             sampleArgs["iters"] = int(n_iter)
#             sampleArgs["nwalker"] = int(walkers)
#             np.savez_compressed(open(db,'wb'),chain=emceeSampler.chain,lnp=emceeSampler.lnprobability,                               pnames=list(fit_params),sampleArgs=sampleArgs)
#
#         self.params_err={} #remove errors of parameters
#
#         for p in ['Q','P','t0']:
#             #calculate values and errors of parameters and save them
#             if p in fit_params:
#                 i=fit_params.index(p)
#                 self.params[p]=np.mean(emceeSampler.flatchain[:,i])
#                 self.params_err[p]=np.std(emceeSampler.flatchain[:,i])
#             else:
#                 self.params[p]=vals1[p]
#                 #self.params_err[p]='---'
#
#         self.Epoch()
#         self.tC=self.t0+self.P*self.epoch+self.Q*self.epoch**2
#         self.new_oc=self.t-self.tC
#         self.model=self.oc+self.new_oc
#
#         self.chi=sum(((self.oc-self.model)/self.err)**2)
#
#         self._fit='MCMC'
#         #remove some values calculated from old parameters
#         self.paramsMore={}
#         self.paramsMore_err={}
#
#         return self.new_oc
#
#     def FitMCMC_old(self,n_iter,limits,steps,fit_params=None,burn=0,binn=1,visible=True,db=None):
#         '''fitting with Markov chain Monte Carlo using pymc
#         n_iter - number of MC iteration - should be at least 1e5
#         limits - limits of parameters for fitting
#         steps - steps (width of normal distibution) of parameters for fitting
#         fit_params - list of fitted parameters
#         burn - number of removed steps before equilibrium - should be approx. 0.1-1% of n_iter
#         binn - binning size - should be around 10
#         visible - display status of fitting
#         db - name of database to save MCMC fitting details (could be analysed later using InfoMCMC function)
#         '''
#
#         #setting pymc sampling for fitted parameters
#         if fit_params is None: fit_params=['Q','P','t0']
#         vals0={'P': self._t0P[1], 't0': self._t0P[0], 'Q':0}
#         vals={}
#         pars={}
#         for p in ['P','t0','Q']:
#             if p in self.params: vals[p]=self.params[p]
#             else: vals[p]=vals0[p]
#             if p in fit_params:
#                 pars[p]=pymc.Uniform(p,lower=limits[p][0],upper=limits[p][1],value=vals[p])
#
#         def model_fun(**arg):
#             '''model function for pymc'''
#             if 'Q' in arg: Q=arg['Q']
#             else: Q=vals['Q']
#             if 'P' in arg: P=arg['P']
#             else: P=vals['P']
#             if 't0' in arg: t0=arg['t0']
#             else: t0=vals['t0']
#             return t0+P*self.epoch+Q*self.epoch**2
#
#         #definition of pymc model
#         model=pymc.Deterministic(
#             eval=model_fun,
#             doc='model',
#             name='Model',
#             parents=pars,
#             trace=True,
#             plot=False)
#
#         #final distribution
#         if self._set_err or self._calc_err:
#             #if known errors of data -> normal/Gaussian distribution
#             y=pymc.Normal('y',mu=model,tau=1./self.err**2,value=self.t,observed=True)
#         else:
#             #if unknown errors of data -> Poisson distribution
#             #note: should cause wrong performance of fitting, rather use function CalcErr for obtained errors
#             y=pymc.Poisson('y',mu=model,value=self.t,observed=True)
#
#         #adding final distribution and sampling of parameters to model
#         Model=[y]
#         for v in pars.values():
#             Model.append(v)
#
#         #create pymc object
#         if db is None: R=pymc.MCMC(Model)
#         else:
#             #saving MCMC fitting details
#             path=db.replace('\\','/')   #change dirs in path (for Windows)
#             if path.rfind('/')>0:
#                 path=path[:path.rfind('/')+1]  #find current dir of db file
#                 if not os.path.isdir(path): os.mkdir(path) #create dir of db file, if not exist
#             R=pymc.MCMC(Model,db='pickle',dbname=db)
#
#         #setting pymc method - distribution and steps
#         for p in pars:
#             R.use_step_method(pymc.Metropolis,pars[p],proposal_sd=steps[p],
#                               proposal_distribution='Normal')
#
#         if not visible:
#             #hidden output
#             f = open(os.devnull, 'w')
#             out=sys.stdout
#             sys.stdout=f
#
#         R.sample(iter=n_iter,burn=burn,thin=binn)  #MCMC fitting/simulation
#
#         self.params_err={} #remove errors of parameters
#
#         for p in ['Q','P','t0']:
#             #calculate values and errors of parameters and save them
#             if p in pars:
#                 self.params[p]=np.mean(pars[p].trace())
#                 self.params_err[p]=np.std(pars[p].trace())
#             else:
#                 self.params[p]=vals[p]
#                 #self.params_err[p]='---'
#
#
#         if not visible:
#             #hidden output
#             sys.stdout=out
#             f.close()
#
#         self.Epoch()
#         self.tC=self.t0+self.P*self.epoch+self.Q*self.epoch**2
#         self.new_oc=self.t-self.tC
#         self.model=self.oc+self.new_oc
#         self.chi=sum(((self.oc-self.model)/self.err)**2)
#
#         self._fit='MCMC_old'
#         #remove some values calculated from old parameters
#         self.paramsMore={}
#         self.paramsMore_err={}
#
#         return self.new_oc
# =============================================================================


# =============================================================================
# class ComplexFit():
#     '''class with common function for OCFit and RVFit'''
#     def KeplerEQ(self,M,e,eps=1e-10):
#         '''solving Kepler Equation using Newton-Raphson method
#         with starting formula S9 given by Odell&Gooding (1986)
#         M - Mean anomaly (np.array, float or list) [rad]
#         e - eccentricity
#         (eps - accurancy)
#         output in rad in same format as M
#         '''
#         #if input is not np.array
#         len1=False
#         if isinstance(M,int) or isinstance(M,float):
#             #M is float
#             if M==0.: return 0.
#             M=np.array(M)
#             len1=True
#         lst=False
#         if isinstance(M,list):
#             #M is list
#             lst=True
#             M=np.array(M)
#
#         E0=M+e*np.sin(M)/np.sqrt(1-2*e*np.cos(M)+e**2)  #starting formula S9
#         E=E0-(E0-e*np.sin(E0)-M)/(1-e*np.cos(E0))
#         while (abs(E-E0)>eps).any():
#             E0=E
#             E=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
#         while (E<0).any(): E[np.where(E<0)]+=2*np.pi
#         while (E>2*np.pi).any(): E[np.where(E>2*np.pi)]-=2*np.pi
#         if len1: return E[0]  #output is float
#         if lst: return list(E)  #output is list
#         return E
#
#
#     def KeplerEQMarkley(self,M,e):
#         '''solving Kepler Equation - Markley (1995): Kepler Equation Solver
#         M - Mean anomaly (np.array, float or list) [rad]
#         e - eccentricity
#         output in rad in same format as M
#         '''
#         #if input is not np.array
#         len1=False
#         if isinstance(M,int) or isinstance(M,float):
#             #M is float
#             if M==0.: return 0.
#             M=np.array(M)
#             len1=True
#         lst=False
#         if isinstance(M,list):
#             #M is list
#             lst=True
#             M=np.array(M)
#
#         pi2=np.pi**2
#         pi=np.pi
#
#         #if somewhere is M=0 or M=pi
#         M=M-(np.floor(M/(2*pi))*2*pi)
#         flip=np.where(M>pi)
#         M[flip]=2*pi-M[flip]
#         M_0=np.where(np.round_(M,14)==0)
#         M_pi=np.where(np.round_(M,14)==np.round_(pi,14))
#
#         alpha=(3.*pi2+1.6*pi*(pi-abs(M))/(1.+e))/(pi2-6.)
#         d=3*(1-e)+alpha*e
#         r=3*alpha*d*(d-1+e)*M+M**3
#         q=2*alpha*d*(1-e)-M**2
#         w=(abs(r)+np.sqrt(q**3+r**2))**(2./3.)
#         E1=(2*r*w/(w**2+w*q+q**2)+M)/d
#         s=e*np.sin(E1)
#         f0=E1-s-M
#         f1=1-e*np.cos(E1)
#         f2=s
#         f3=1-f1
#         f4=-f2
#         d3=-f0/(f1-0.5*f0*f2/f1)
#         d4=-f0/(f1+0.5*d3*f2+(d3**2)*f3/6.)
#         d5=-f0/(f1+0.5*d4*f2+d4**2*f3/6.+d4**3*f4/24.)
#         E=E1+d5
#         E[flip]=2*pi-E[flip]
#         E[M_0]=0.
#         E[M_pi]=pi
#         if len1: return E[0]  #output is float
#         if lst: return list(E)  #output is list
#         return E
#
#
#
#
#     def LiTE(self,t,a_sin_i3,e3,w3,t03,P3):
#         '''model of O-C by Light-Time effect given by Irwin (1952)
#         t - times of minima (np.array or float) [days]
#         a_sin_i3 - semimayor axis original binary around center of mass of triple system [AU]
#         e3 - eccentricity of 3rd body
#         w3 - longitude of pericenter of 3rd body [rad]
#         P3 - period of 3rd body [days]
#         t03 - time of pericenter passage of 3rd body [days]
#         output in days
#         '''
#
#         M=2*np.pi/P3*(t-t03)  #mean anomally
#         if e3<0.9: E=self.KeplerEQ(M,e3)   #eccentric anomally
#         else: E=self.KeplerEQMarkley(M,e3)
#         nu=2*np.arctan(np.sqrt((1+e3)/(1-e3))*np.tan(E/2))  #true anomally
#         dt=a_sin_i3*AU/c*((1-e3**2)/(1+e3*np.cos(nu))*np.sin(nu+w3)+e3*np.sin(w3))
#         return dt/day
#
# =============================================================================

class TransitFit():
    '''class for fitting transits'''
    def __init__(self,t,flux,err=None):
        '''loading times, fluxes, (errors)'''
        self.t=np.array(t)
        self.flux=np.array(flux)
        if err is None:
            #if unknown (not given) errors of data
            #note: should cause wrong performance of fitting using MC, rather use function CalcErr for obtained errors after GA fitting
            self.err=np.ones(self.t.shape)/1440.
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
        self.res=[]             #residua
        self._fit=''            #used algorithm for fitting (GA/DE/MCMC)
        self.availableModels=['Uniform','Linear','Quadratic','SquareRoot','Logarithmic','Exponential','Power2','Nonlinear']   #list of available models


    def AvailableModels(self):
        '''print available models for fitting O-Cs'''
        print('Available Models:')
        for s in self.availableModels: print(s)

    def ModelParams(self,model=None,allModels=False):
        '''display parameters of model'''

        def Display(model):
            s=model+': '
            if 'Transit' in model:
                s+='t0, P, Rp, a, i, e, w, '
                if 'Uniform' not in model: s+='c1, '
                if ('Linear' not in model) and ('Power2' not in model): s+='c2, '
                if 'Nonlinear' in model: s+='c3, c4, '
            print(s[:-2])

        if model is None: model=self.model
        if allModels:
            for m in self.availableModels: Display(m)
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

    def Transit(self,t,t0,P,Rp,a,i,e,w,u):
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

        flux=m.light_curve(params)                    #calculates light curve

        return flux


    def PhaseCurve(self,P,t0,plot=False):
        '''create phase curve'''
        f=np.mod(self.t-t0,P)/float(P)    #phase
        order=np.argsort(f)
        f=f[order]
        flux=self.flux[order]
        if plot:
            mpl.figure()
            if self._set_err: mpl.errorbar(f,flux,yerr=self.err,fmt='o')
            else: mpl.plot(f,flux,'.')
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
            if x[0]=='c': continue
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
            elif x[0]=='P':
                unit.append('d')
                #also in years
                params.append(x)
                vals.append(str(self.params[x]/year))
                if err[-1]=='---' or err[-1]=='fixed': err.append(err[-1])  #error not calculated
                else: err.append(str(float(err[-1])/year)) #error calculated
                unit.append('yr')
            elif x[0]=='t': unit.append('JD')
            elif x[0]=='e' or x[0]=='c': unit.append('')
            elif x[0]=='w' or x[0]=='i': unit.append('deg')

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

        #calculate some more parameters, if not calculated
        self.DepthDur()
        R=0
        R_err=0
        if 'R' in self.systemParams:
            R=self.systemParams['R']
            if 'R_err' in self.systemParams: R_err=self.systemParams['R_err']

        if R>0: self.AbsoluteParam(R,R_err)

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
            elif x[0]=='T':
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


    def DepthDur(self):
        '''calculate depth and duration of transit'''
        output={}
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
            if self.paramsMore_err['delta']==0: del self.paramsMore_err['delta']
            else: output['delta_err']=self.paramsMore_err['delta']
            if self.paramsMore_err['T14']==0: del self.paramsMore_err['T14']
            else: output['T14_err']=self.paramsMore_err['T14']
            if self.paramsMore_err['T23']==0: del self.paramsMore_err['T23']
            else: output['T23_err']=self.paramsMore_err['T23']

        return output

    def AbsoluteParam(self,R,R_err=0):
        '''calculate absolute radius and semi-mayor axis of planet from radius of star'''
        output={}
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
            if self.paramsMore_err['a']==0: del self.paramsMore_err['a']
            else: output['a_err']=self.paramsMore_err['a']
            if self.paramsMore_err['Rp']==0: del self.paramsMore_err['Rp']
            else: output['Rp_err']=self.paramsMore_err['Rp']

        return output


    def Model(self,t=None,param=None):
        ''''calculate model curve of transit in given times based on given set of parameters'''
        if t is None: t=self.t
        if param is None: param=self.params
        if 'Transit' in self.model:
            u=[]
            if 'Uniform' not in self.model: u.append(param['c1'])
            if ('Linear' not in self.model) or ('Power2' not in self.model): u.append(param['c2'])
            if 'Nonlinear' in self.model:
                u.append(param['c3'])
                u.append(param['c4'])

            model=self.Transit(t,param['t0'],param['P'],param['Rp'],param['a'],param['i'],param['e'],param['w'],u)
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


    #TODO!
    def Plot(self,name=None,no_plot=0,no_plot_err=0,params=None,eps=False,
             time_type='JD',offset=2400000,trans=True,center=True,title=None,hours=False,
             phase=False,weight=None,trans_weight=False,model2=False,with_res=False,
             bw=False,double_ax=False,legend=None,fig_size=None):
        '''plotting original O-C with model O-C based on current parameters set
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
        weight - weight of data (shown as size of points)
        trans_weight - transform weights to range (1,10)
        model2 - plot 2 models - current params set and set given in "params"
        with_res - common plot with residue
        bw - Black&White plot
        double_ax - two axes -> time and phase
        legend - labels for data and model(s) - give '' if no show label, 2nd model given in "params" is the last
        fig_size - custom figure size - e.g. (12,6)

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

        self.Phase(params['t0'],params['P'])
        l=''
        if hours:
            center=True
            l=' [h]'
        if center:
            E=np.round((self.t-params['t0'])/params['P'])
            E=E[len(E)//2]
            offset=params['t0']+params['P']*E
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
        ax1.set_ylabel('Flux')

        if title is not None:
            if double_ax: fig.subplots_adjust(top=0.85)
            fig.suptitle(title,fontsize=20)

        model=self.Model(self.t,params)
        res=self.flux-model

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

        errors=GetMax(abs(model-self.flux),no_plot)  #remove outlier points
        if bw: color='k'
        else: color='b'
        if set_w:
            #using weights
            #prim=np.delete(prim,np.where(np.in1d(prim,errors)))
            for i in range(len(w)):
                ax1.plot(k*x[np.where(w[i])],
                        (self.flux)[np.where(w[i])],color+'o',markersize=size[i],label=legend[0],zorder=1)

        else:
            #without weight
            if self._set_err:
                #using errors
                if self._corr_err: err=self._old_err
                else: err=self.err
                errors=np.append(errors,GetMax(err,no_plot_err))  #remove errorful points
                #prim=np.delete(prim,np.where(np.in1d(prim,errors)))
                ax1.errorbar(k*x,self.flux,yerr=err,fmt=color+'o',markersize=5,label=legend[0],zorder=1)
            else:
                #without errors
                #prim=np.delete(prim,np.where(np.in1d(prim,errors)))
                ax1.plot(k*x,self.flux,color+'o',label=legend[0],zorder=1)

        #expand time interval for model O-C
        if len(self.t)<1000:
            dt=(self.t[-1]-self.t[0])/1000.
            t1=np.linspace(self.t[0]-50*dt,self.t[-1]+50*dt,1100)
        else:
            dt=(self.t[-1]-self.t[0])/len(self.t)
            t1=np.linspace(self.t[0]-0.05*len(self.t)*dt,self.t[-1]+0.05*len(self.t)*dt,int(1.1*len(self.t)))

        if bw:
            color='k'
            lw=2
        else:
            color='r'
            lw=1

        model_long=self.Model(t1,params)
        if phase and not double_ax: ax1.plot(self.Phase(params['t0'],params['P'],t1),model_long,color,linewidth=lw,label=legend[1],zorder=2)
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
            if phase and not double_ax: ax1.plot(self.Phase(params['t0'],params['P'],t1),model_set,color+lt,linewidth=lw,label=legend[2],zorder=3)
            else: ax1.plot(k*(t1-offset),model_set,color+lt,linewidth=lw,label=legend[2],zorder=3)

        if show_legend: ax1.legend()

        if double_ax:
            #setting secound axis
            ax3=ax1.twiny()
            #generate plot to obtain correct axis in phase
            #expand time interval for model O-C
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
            ph=self.Phase(params['t0'],params['P'],lims)
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

        if name is None: mpl.show()
        else:
            mpl.savefig(name+'.png')
            if eps: mpl.savefig(name+'.eps')
            mpl.close(fig)

    #TODO!
    def PlotRes(self,name=None,no_plot=0,no_plot_err=0,params=None,eps=False,oc_min=True,
                time_type='JD',offset=2400000,trans=True,title=None,epoch=False,
                min_type=False,weight=None,trans_weight=False,bw=False,double_ax=False,
                fig_size=None):
        '''plotting residue (new O-C)
        name - name of file to saving plot (if not given -> show graph)
        no_plot - count of outlier point which will not be plot
        no_plot_err - count of errorful point which will not be plot
        params - set of params of current model (if not given -> current parameters set)
        eps - save also as eps file
        oc_min - O-C in minutes (if False - days)
        time_type - type of JD in which is time (show in x label)
        offset - offset of time
        trans - transform time according to offset
        title - name of graph
        epoch - x axis in epoch
        min_type - distinction of type of minimum
        weight - weight of data (shown as size of points)
        trans_weight - transform weights to range (1,10)
        bw - Black&White plot
        double_ax - two axes -> time and epoch
        fig_size - custom figure size - e.g. (12,6)

        warning: weights have to be in same order as input data!
        '''

        if epoch:
            if not len(self.epoch)==len(self.t):
                raise NameError('Epoch not callculated! Run function "Epoch" before it.')

        if params is None: params=self.params

        if fig_size:
            fig=mpl.figure(figsize=fig_size)
        else:
            fig=mpl.figure()

        ax1=fig.add_subplot(1,1,1)
        ax1.yaxis.set_label_coords(-0.11,0.5)

        #setting labels
        if epoch and not double_ax:
            ax1.set_xlabel('Epoch')
            x=self.epoch
        elif offset>0:
            ax1.set_xlabel('Time ('+time_type+' - '+str(offset)+')')
            if not trans: offset=0
            x=self.t-offset
        else:
            ax1.set_xlabel('Time ('+time_type+')')
            offset=0
            x=self.t

        if oc_min:
            ax1.set_ylabel('Residue O - C (min)')
            k=minutes
        else:
            ax1.set_ylabel('Residue O - C (d)')
            k=1
        if title is not None:
            if double_ax: fig.subplots_adjust(top=0.85)
            fig.suptitle(title,fontsize=20)

        model=self.Model(self.t,params)
        self.res=self.flux-model

        #primary / secondary minimum
        if min_type:
            if not len(self.epoch)==len(self.t):
                raise NameError('Epoch not callculated! Run function "Epoch" before it.')
            prim=np.where(self._min_type==0)
            sec=np.where(self._min_type==1)
        else:
            prim=np.arange(0,len(self.t),1)
            sec=np.array([])

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


        errors=GetMax(abs(self.res),no_plot)  #remove outlier points
        if bw: color='k'
        else: color='b'
        if set_w:
            #using weights
            prim=np.delete(prim,np.where(np.in1d(prim,errors)))
            sec=np.delete(sec,np.where(np.in1d(sec,errors)))
            if not len(prim)==0:
                for i in range(len(w)):
                    mpl.plot(x[prim[np.where(np.in1d(prim,w[i]))]],
                             (self.res*k)[prim[np.where(np.in1d(prim,w[i]))]],color+'o',markersize=size[i])
            if not len(sec)==0:
                for i in range(len(w)):
                    mpl.plot(x[sec[np.where(np.in1d(sec,w[i]))]],
                             (self.res*k)[sec[np.where(np.in1d(sec,w[i]))]],color+'o',markersize=size[i],
                             fillstyle='none',markeredgewidth=1,markeredgecolor=color)

        else:
            #without weight
            if self._set_err:
                #using errors
                if self._corr_err: err=self._old_err
                else: err=self.err
                errors=np.append(errors,GetMax(err,no_plot_err))  #remove errorful points
                prim=np.delete(prim,np.where(np.in1d(prim,errors)))
                sec=np.delete(sec,np.where(np.in1d(sec,errors)))
                if not len(prim)==0:
                    mpl.errorbar(x[prim],(self.res*k)[prim],yerr=(err*k)[prim],fmt=color+'o',markersize=5)
                if not len(sec)==0:
                    mpl.errorbar(x[sec],(self.res*k)[sec],yerr=(err*k)[sec],fmt=color+'o',markersize=5,
                                 fillstyle='none',markeredgewidth=1,markeredgecolor=color)

            else:
                #without errors
                prim=np.delete(prim,np.where(np.in1d(prim,errors)))
                sec=np.delete(sec,np.where(np.in1d(sec,errors)))
                if not len(prim)==0:
                    mpl.plot(x[prim],(self.res*k)[prim],color+'o')
                if not len(sec)==0:
                    mpl.plot(x[sec],(self.res*k)[sec],color+'o',
                             mfc='none',markeredgewidth=1,markeredgecolor=color)

        if double_ax:
            #setting secound axis
            if not len(self.epoch)==len(self.t):
                raise NameError('Epoch not callculated! Run function "Epoch" before it.')
            ax2=ax1.twiny()
            #generate plot to obtain correct axis in epoch
            l=ax2.plot(self.epoch,self.res*k)
            ax2.set_xlabel('Epoch')
            l.pop(0).remove()
            lims=np.array(ax1.get_xlim())
            epoch=np.round((lims-self._t0P[0])/self._t0P[1]*2)/2.
            ax2.set_xlim(epoch)

        if name is None: mpl.show()
        else:
            mpl.savefig(name+'.png')
            if eps: mpl.savefig(name+'.eps')
            mpl.close(fig)


    #TODO!
    def SaveModel(self,name,t_min=None,t_max=None,n=1000,phase=False,params=None):
        '''save model curve of transit to file
        name - name of output file
        t_min - minimal value of time
        t_max - maximal value of time
        n - number of data points
        phase - export phase curve (min/max value give in t_min/t_max as phase)
        params - parameters of model (if not given, used "params" from class)
        '''

        if params is None: params=self.params

        #same interval of epoch like in plot
        #TODO!
        if len(self.t)<1000: dt=50*(self.t[-1]-self.t[0])/1000.
        else: dt=0.05*(self.t[-1]-self.t[0])

        if t_min is None: t_min=min(self.t)-dt
        if t_max is None: t_max=max(self.t)+dt

        t=np.linspace(t_min,t_max,n)
        phase=self.Phase(params['t0'],params['P'],t)

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

        model=self.Model(self.t,params)
        phase=self.Phase(params['t0'],params['P'])

        res=self.flux-model
        f=open(name,'w')
        if self._set_err:
            if self._corr_err: err=self._old_err
            else: err=self.err
            np.savetxt(f,np.column_stack((self.t,phase,res,err)),
                       fmt=["%14.7f",'%8.5f',"%12.10f","%.10f"],delimiter="    ",
                       header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(10,' ')
                       +'    '+'Residue'.ljust(10,' ')+'    Error')
        elif weight is not None:
            np.savetxt(f,np.column_stack((self.t,phase,res,np.array(weight)[self._order])),
                       fmt=["%14.7f",'%8.5f',"%12.10f","%.10f"],delimiter="    ",
                       header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(10,' ')
                       +'    '+'Residue'.ljust(12,' ')+'    Weight')
        else:
            np.savetxt(f,np.column_stack((self.t,phase,res)),
                       fmt=["%14.7f",'%8.5f',"%12.10f"],delimiter="    ",
                       header='Time'.ljust(12,' ')+'    '+'Phase'.ljust(10,' ')
                       +'    '+'Residue')
        f.close()



class TransitFitLoad(TransitFit):
    '''loading saved data, model... from TransitFit class'''
    def __init__(self,path):
        '''loading data, model, parameters... from file'''
        super().__init__([0],[0],[0])

        self.Load(path)

