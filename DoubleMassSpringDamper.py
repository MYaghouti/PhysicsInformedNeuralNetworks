# Double Mass-Spring-Damper system Solved by PINN
# This software is distributed under the BSD 3-clause license.
# License file included in the script directory.
# This script solves the double mass-spring-damper system explained in the IntroductionToPINN.pdf file.
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





m1 = 6.4
m2 = 4.4
k1 = 5.0
k2 = 2.0
b1 = 0.1
b2 = 0.08

t_max = 50

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
def SysODE(t,V):
  y1, y2, y3, y4 = V[0],V[1],V[2],V[3]
  return np.array([y3,y4,(-k1*y1-b1*y3+k2*(y2-y1)+b2*(y4-y3))/m1,(-k2*(y2-y1)-b2*(y4-y3))/m2])


t,y=RK4(SysODE, ics=np.array([0.5,3.25,0,0]), h=1e-5, span=np.array([0,t_max]), degree =4)
plt.plot(t,y[:,0],'-r',label='x[1]')
plt.plot(t,y[:,1],'-g',label='x[2]')




N_b = 1000
N_c = 10000

tmin,tmax=0 , t_max





y1_t0 = jnp.zeros([N_b,1],dtype='float32')
y1_ic = jnp.ones_like(y1_t0)*0.5
Y1_IC = jnp.concatenate([y1_t0,y1_ic],axis=1)


y2_t0 = jnp.zeros([N_b,1],dtype='float32')
y2_ic = jnp.ones_like(y2_t0) * 3.25
Y2_IC = jnp.concatenate([y2_t0,y2_ic],axis=1)


y3_t0 = jnp.zeros([N_b,1],dtype='float32')
y3_ic = jnp.ones_like(y3_t0) * 0
Y3_IC = jnp.concatenate([y3_t0,y3_ic],axis=1)


y4_t0 = jnp.zeros([N_b,1],dtype='float32')
y4_ic = jnp.ones_like(y4_t0) * 0
Y4_IC = jnp.concatenate([y4_t0,y4_ic],axis=1)

conds = [Y1_IC,Y2_IC,Y3_IC,Y4_IC ]


key=jax.random.PRNGKey(0)

t_c = jax.random.uniform(key,minval=tmin,maxval=tmax,shape=(N_c,1))
colloc = t_c

def ODE(t,y1,dy1,d2y1, y2, dy2, d2y2):
  return m1*d2y1(t)-(-k1*y1(t)-b1*dy1(t)+k2*(y2(t)-y1(t))+b2*(dy2(t)-dy1(t)))  , m2*d2y2(t)-(-k2*(y2(t)-y1(t))-b2*(dy2(t)-dy1(t)))


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

def loss_fun(params,colloc,conds):
  t_c =colloc[:,[0]]

  y1_func   = lambda t : fwd(params,t)[:,[0]]
  y1_func_t = lambda t:jax.grad(lambda t:jnp.sum(y1_func(t)))(t)
  d2y1_func_t = lambda t:jax.grad(lambda t:jnp.sum(y1_func_t(t)))(t)


  y2_func   = lambda t : fwd(params,t)[:,[1]]
  y2_func_t = lambda t:jax.grad(lambda t:jnp.sum(y2_func(t)))(t)
  d2y2_func_t = lambda t:jax.grad(lambda t:jnp.sum(y2_func_t(t)))(t)

  loss_y1, loss_y2  = ODE(t_c,y1_func, y1_func_t, d2y1_func_t, y2_func, y2_func_t, d2y2_func_t)


  loss = jnp.mean( loss_y1 **2)
  loss+= jnp.mean( loss_y2 **2)

  t_ic,y1_ic = conds[0][:,[0]],conds[0][:,[1]]
  loss += MSE(y1_ic,y1_func(t_ic))
  t_ic,y2_ic = conds[1][:,[0]],conds[1][:,[1]]
  loss += MSE(y2_ic,y2_func(t_ic))
  t_ic,y3_ic = conds[2][:,[0]],conds[2][:,[1]]
  loss += MSE(y3_ic,y1_func_t(t_ic))
  t_ic,y4_ic = conds[3][:,[0]],conds[3][:,[1]]
  loss += MSE(y4_ic,y2_func_t(t_ic))

  return  loss

@jax.jit
def update(opt_state,params,colloc,conds):
  grads=jax.jit(jax.grad(loss_fun,0))(params,colloc,conds)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return opt_state,params



params = init_params([1] + [64]+ [64]+ [64] + [2])
optimizer = optax.adam(2e-3)
opt_state = optimizer.init(params)




epochs = 60000
loss=[]
for _ in range(epochs):
  colloc = t_c
  opt_state,params = update(opt_state,params,colloc,conds)
  if _ %(100) ==0:
    loss.append(loss_fun(params,colloc,conds))
    print(f'Epoch={_}\tloss={loss[-1]:.3e}')




dT = 1e-3
Tf = t_max
T = np.arange(0,Tf+dT,dT)
plt.plot(T,fwd(params,T.reshape(-1,1))[:,0],'--k',label='PINN[x1]',linewidth=2)
plt.plot(T,fwd(params,T.reshape(-1,1))[:,1],'--k',label='PINN[x2]',linewidth=2)
plt.grid()
plt.legend()
plt.savefig('DoubleMassSpringDamper.png')
