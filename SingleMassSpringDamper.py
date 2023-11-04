# Single Mass-Spring-Damper system Solved by PINN
# This software is distributed under the BSD 3-clause license.
# License file included in the script directory.
# This script solves the single mass-spring-damper system explained in the IntroductionToPINN.pdf file.
# Using both:
#    -   Runge-Kutta Numerical solver
#    -   Physics Informed Neural Network (PINN)
# Written by:       MehdiYaghouti, 2023
# Contact Info.:    MehdiYaghouti@gmail.com

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import numba




m = 1.5
k = 4
b = 0.5


@numba.njit
def RK4(odefun,ics,h,span,degree):

    N= int( (span[1]-span[0])/h )

    tY = np.zeros((N+1,degree+1))
    tY[0,1:] = ics
    for  i in range(N):
        tY[i+1,0] = tY[i,0] + h

        k1= odefun(tY[i,0]       , tY[i,1:])
        k2= odefun(tY[i,0] +(h/2), tY[i,1:] +(h*k1)/2 )
        k3= odefun(tY[i,0] +(h/2), tY[i,1:] +(h*k2)/2)
        k4= odefun(tY[i,0] +(h)  , tY[i,1:] +(h*k3))

        tY[i+1,1:] = tY[i,1:] + h*(1/6) * (k1+2*k2+2*k3+k4)

    return tY[:,0],tY[:,1:]



@numba.njit
def system_of_ode(t,V):
  dx, x = V[0],V[1]
  return np.array([ (-k*x-b*dx)/m ,dx])


t,y=RK4(system_of_ode, ics=np.array([0,6]), h=1e-5, span=np.array([1e-4,10*np.pi]), degree =2)
plt.figure(figsize=(10,6))
plt.plot(t,y[:,1],'-r',label='Runge-Kutta x[1]')

plt.grid()
plt.legend()



N_b = 1000
N_c = 100000

tmin,tmax=0*jnp.pi , 10*jnp.pi





y1_t0 = jnp.zeros([N_b,1],dtype='float32')
y1_ic = jnp.ones_like(y1_t0)*0
Y1_IC = jnp.concatenate([y1_t0,y1_ic],axis=1)


y2_t0 = jnp.zeros([N_b,1],dtype='float32')
y2_ic = jnp.ones_like(y2_t0) * 6
Y2_IC = jnp.concatenate([y2_t0,y2_ic],axis=1)



cnds = [Y1_IC,Y2_IC ]



key=jax.random.PRNGKey(0)

t_c = jax.random.uniform(key,minval=tmin,maxval=tmax,shape=(N_c,1))
pnts_train = t_c

def ODE(t,y1,dy,d2y):
  return m*d2y(t)-(-k*y1(t)-b*dy(t))

def init_params(layers):
  keys = jax.random.split(jax.random.PRNGKey(0),len(layers)-1)
  params = list()
  for key,n_in,n_out in zip(keys,layers[:-1],layers[1:]):
    lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
    W = lb + (ub-lb) * jax.random.uniform(key,shape=(n_in,n_out))
    B = jax.random.uniform(key,shape=(n_out,))
    params.append({'W':W,'B':B})
  return params

def fwd(params,t):
  X = jnp.concatenate([t],axis=1)
  *hidden,last = params
  for layer in hidden :
    X = jax.nn.tanh(X@layer['W']+layer['B'])
  return X@last['W'] + last['B']

@jax.jit
def MSE(true,pred):
  return jnp.mean((true-pred)**2)

def Loss(params,pnts_train,cnds):
  t_c =pnts_train[:,[0]]
  y1_func   = lambda t : fwd(params,t)[:,[0]]
  y1_func_t = lambda t:jax.grad(lambda t:jnp.sum(y1_func(t)))(t)
  d2y1_func_t = lambda t:jax.grad(lambda t: jnp.sum(jax.grad(lambda t:jnp.sum(y1_func(t)))(t))       )(t)
  loss_y1  = ODE(t_c,y1_func, y1_func_t, d2y1_func_t)
  loss = jnp.mean( loss_y1 **2)
  t_ic,y1_ic = cnds[0][:,[0]],cnds[0][:,[1]]
  loss += MSE(y1_ic,y1_func_t(t_ic))
  t_ic,y2_ic = cnds[1][:,[0]],cnds[1][:,[1]]
  loss += MSE(y2_ic,y1_func(t_ic))



  return  loss

@jax.jit
def update(opt_state,params,pnts_train,cnds):
  grads=jax.jit(jax.grad(Loss,0))(params,pnts_train,cnds)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return opt_state,params



params = init_params([1] + [4]+[9]*3+[4] +[1])



optimizer = optax.adam(2e-2)
opt_state = optimizer.init(params)




epochs = 20000
loss=[]
for _ in range(epochs):

  key=jax.random.PRNGKey(_)
  t_c = jax.random.uniform(key,minval=tmin,maxval=tmax,shape=(N_c,1))
  pnts_train = t_c
  opt_state,params = update(opt_state,params,pnts_train,cnds)


  if _ %(50) ==0:
    loss.append(Loss(params,pnts_train,cnds))
    print(f'Epoch={_}\tloss={loss[-1]:.3e}')




dT = 1e-3
Tf = 10*jnp.pi
T = np.arange(0,Tf+dT,dT)
plt.plot(T,fwd(params,T.reshape(-1,1))[:,0],'--k',label='NN[x1]',linewidth=2)
plt.legend()
plt.savefig('SingleMassSpringDamper.png')
