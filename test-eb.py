import TransitFit
import numpy as np
import matplotlib.pyplot as mpl


t0=2.4e6
param={'t0':t0,'P':15,'A':-0.2,'tC':t0+45,'s':0.05,'p0':-3,'p1':0.1,'p2':0}
#param={'t0':t0,'P':15,'A':-0.2,'tC':t0+45,'s':0.05,'p0':-2024,'p1':90,'p2':-1}
#param={'t0':t0,'P':15,'A':-0.2,'tC':t0+45,'s':0.05,'p0':-2046.5,'p1':90.5,'p2':-1}

#param={'t0':t0,'P':15,'Rp':1e-2,'a':18,'i':89.8,'e':0,'w':47,'c1':0.15,'c2':0.45,'p0':10,'p1':0,'p2':0}
limits={'t0':[t0-1e-2,t0+1e-2],'P':[14.5,15.5],'A':[-0.3,-0.1],'tC':[t0+45-1/24,t0+45+1/24],'s':[1e-3,1e-1],'p0':[-10,0],'p1':[0,1],'p2':[-2,0]}
steps={'t0':1e-4,'P':1e-2,'A':1e-3,'tC':1e-5,'s':1e-5,'p0':1e-2,'p1':1e-3,'p2':1e-2}

t0+=45
t=np.linspace(-0.3,0.3,100)+t0

g=1+param['A']*np.exp(-(t-param['tC'])**2/param['s']**2)

pp=np.polyval([param['p2'],param['p1'],param['p0']],t-param['t0'])
flux0=g*pp
flux=flux0+5e-3*np.random.normal(size=t.shape)*np.mean(pp)
err=5e-3*np.ones(t.shape)*np.mean(pp)


tf=TransitFit.TransitFit(t,flux,err)

tf.Plot0()

tf.Normalise([t0-0.15,t0+0.15],reverse=True,plot=False)
param['p0']=1
param['p1']=0
param['p2']=0

tf.Plot0()

tf.model='Gauss'
tf.params=dict(param)
tf.Phase(tf.params['t0'],tf.params['P'])
tf.Summary()

tf.limits=limits
tf.steps=steps

tf.fit_params=['A','tC','s']  #t0,P,Rp,a,i,e,w,c1,c2,p0,p1,p2
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
tf.Plot(model2=True,params=param,hours=True,double_ax=True,with_res=True,detrend=True)
tf.PlotRes(hours=True,double_ax=True)

mpl.show()
