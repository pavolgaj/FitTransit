import TransitFit
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

param={'t0':0,'P':15,'Rp':1e-2,'a':18,'i':89.8,'e':0,'w':47,'c1':0.15,'c2':0.45}
limits={'t0':[-1e-2,1e-2],'P':[14.5,15.5],'Rp':[5e-3,5e-2],'a':[15,20],'i':[85,90],'e':[0,0.1],'w':[0,360],'c1':[0.1,0.2],'c2':[0.4,0.5]}
steps={'t0':1e-4,'P':1e-2,'Rp':1e-5,'a':0.1,'i':0.1,'e':1e-2,'w':0.1,'c1':1e-3,'c2':1e-3}

t=np.linspace(-0.3,0.3,100)
flux0=transit(t,param['t0'],param['P'],param['Rp'],param['a'],param['i'],param['e'],param['w'],[param['c1'],param['c2']])
flux=flux0+1e-5*np.random.normal(size=t.shape)
err=1e-5*np.ones(t.shape)

mpl.errorbar(t*24,flux,yerr=err,fmt='o')
mpl.show()

tf=TransitFit.TransitFit(t,flux,err)
tf.params=param
tf.systemParams={'R':1.2,'R_err':0.1}
tf.Phase(tf.params['t0'],tf.params['P'])
tf.Summary()

tf.limits=limits
tf.steps=steps
tf.fit_params=['Rp','a','i','c1','c2','e','w']  #t0,P,Rp,a,i,e,w,c1,c2

tf.FitGA(100,100)
tf.Summary()

#tf.FitDE(100,100)
#tf.Summary()

tf.FitMCMC(1e3)

flux1=tf.Model()

tf.Summary()

#tf.Save('test')

mpl.errorbar(t*24,flux,yerr=err,fmt='o')
mpl.plot(t*24,flux0)
mpl.plot(t*24,flux1)

mpl.show()
