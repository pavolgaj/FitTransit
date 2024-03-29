import FitTransit
import numpy as np
import matplotlib.pyplot as mpl
import batman

def transit(t,t0,p,r,a,i,e,w,u,ld='quadratic'):
    params=batman.TransitParams()       #object to store transit parameters

    params.t0=t0                       #time of inferior conjunction
    params.per=p                       #orbital period
    params.rp=r                       #planet radius (in units of stellar radii)
    params.a=a                        #semi-major axis (in units of stellar radii)
    params.inc=i                      #orbital inclination (in degrees)
    params.ecc=e                     #eccentricity
    params.w=w                        #longitude of periastron (in degrees)

    #ld_options = ["uniform", "linear", "quadratic", "nonlinear"]
    #ld_coefficients = [[], [0.3], [0.1, 0.3], [0.5, 0.1, 0.1, -0.1]]

    params.limb_dark=ld        #limb darkening model
    params.u=u      #limb darkening coefficients [u1, u2, u3, u4]

    #t = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
    m=batman.TransitModel(params,t)    #initializes model

    flux=m.light_curve(params)                    #calculates light curve

    return flux

t0=2.4e6
param={'t0':t0,'P':15,'Rp':1e-2,'a':18,'i':89.8,'e':0,'w':47,'c1':0.15,'c2':0.45,'p0':-1.025,'p1':9e-2,'p2':-1e-3}
#param={'t0':t0,'P':15,'Rp':1e-2,'a':18,'i':89.8,'e':0,'w':47,'c1':0.15,'c2':0.45,'p0':10,'p1':0,'p2':0}
limits={'t0':[t0-1e-2,t0+1e-2],'P':[14.5,15.5],'Rp':[5e-3,5e-2],'a':[15,20],'i':[85,90],'e':[0,0.1],'w':[0,360],'c1':[0.1,0.2],'c2':[0.4,0.5],'p0':[-1.0251,-1.0249],'p1':[8.5e-2,9.5e-2],'p2':[-1.5e-3,-0.5e-3]}
steps={'t0':1e-4,'P':1e-2,'Rp':1e-5,'a':0.1,'i':0.1,'e':1e-2,'w':0.1,'c1':1e-3,'c2':1e-3,'p0':1e-6,'p1':1e-4,'p2':1e-5}

t0+=45
t=np.linspace(-0.3,0.3,100)+t0

pp=np.polyval([param['p2'],param['p1'],param['p0']],t-param['t0'])
flux0=transit(t,param['t0'],param['P'],param['Rp'],param['a'],param['i'],param['e'],param['w'],[param['c1'],param['c2']])*pp
flux=flux0+1e-5*np.random.normal(size=t.shape)*np.mean(pp)
err=1e-5*np.ones(t.shape)*np.mean(pp)


tf=FitTransit.FitTransit(t,flux,err)

tf.Plot0()

tf.Normalise([t0-0.15,t0+0.15],reverse=True,plot=False)
param['p0']=1
param['p1']=0
param['p2']=0

tf.Plot0()

tf.params=dict(param)
tf.systemParams={'R':1.2,'R_err':0.1,'mag':8.76}
#tf.Phase(tf.params['t0'],tf.params['P'])
tf.Summary()

tf.limits=limits
tf.steps=steps

tf.fit_params=['Rp','a','i','c1','c2']  #t0,P,Rp,a,i,e,w,c1,c2,p0,p1,p2
#tf.fit_params=['p0','p1','p2']

tf.FitGA(100,100)
tf.Summary()

#tf.FitDE(100,100)
#tf.Summary()

tf.FitMCMC(1e3)

flux1=tf.Model()

tf.Summary()

#tf.Save('test')
#tf.SaveModel('model.dat')
#tf.SaveModel('phase.dat',t_min=-0.4,t_max=0.4,phase=True)
#tf.SaveRes('res.dat')


tf.Plot()
#tf.Plot(double_ax=True)
#tf.Plot(with_res=True)
#tf.Plot(model2=True,params=param)
#tf.Plot(hours=True)
#tf.Plot(hours=True,double_ax=True,with_res=True)
#tf.Plot(hours=True,detrend=True)
tf.Plot(model2=True,params=param,hours=True,double_ax=True,with_res=True)
tf.PlotRes(hours=True,double_ax=True)

mpl.show()
