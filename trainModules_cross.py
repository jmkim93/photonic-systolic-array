#%%
import matplotlib
import matplotlib.pylab as plt
import numpy as np

import jax
import jax.numpy as jnp

import tidy3d as td

import optax
import pickle
from datetime import datetime

from n_eff_calculation import n_eff_slab

import defineAdjointOptimization as ao
from defineAdjointOptimization import l_des, pixel_size, nx, ny, w_wg, h_wg, radius, dist_wg, val_grad_fn, min_step_per_wvl


module = "cross"
num_steps = 100
lr = 0.05

folder_name = "PE_module_{}".format(module)

x_param = np.linspace(-(l_des-pixel_size)/2, (l_des-pixel_size)/2, nx)
y_param = np.linspace(-(l_des-pixel_size)/2, (l_des-pixel_size)/2, ny)
y_param, x_param = np.meshgrid(y_param, x_param)



## Cross, BS
w_junction = l_des/2
params_init = (
    (np.abs(x_param)<w_wg/2) + 
    (np.abs(y_param)<w_wg/2) + 
    (np.abs(x_param)<w_junction/2) * (np.abs(y_param)<w_junction/2) * 
    ((np.abs(x_param)-w_junction/2)**2 + (np.abs(y_param)-w_junction/2)**2 >= ((w_junction-w_wg)**2/4)) + 0.0
)
## 1
# params_init = (
#     ((x_param-l_des/2)**2+(y_param+l_des/2)**2 >= (l_des-w_wg)**2/4) * 
#     ((x_param-l_des/2)**2+(y_param+l_des/2)**2 <= (l_des+w_wg)**2/4) + 0.0
# )

## >2 ver2
# params_init = (
#     ((x_param-l_des/2)**2+(y_param+l_des/2)**2 >= (l_des-w_wg)**2/4) * 
#     ((x_param-l_des/2)**2+(y_param+l_des/2)**2 <= (l_des+w_wg)**2/4) + 
#     (np.abs(x_param)<w_wg/2) + 0.0
# )


params_init += (np.random.uniform(low=-1, high=1, size=(nx,ny)))
params_init = np.clip(params_init, 0, 1)
params = params_init.copy()

plt.matshow(params_init, cmap=plt.cm.viridis, vmin=0, vmax=1)
plt.matshow(ao.smoothen.evaluate(params_init), cmap=plt.cm.viridis, vmin=0, vmax=1)

optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(params)

dphase = jnp.array([0.0])
optimizer_dphase = optax.adam(learning_rate=lr)
opt_state_dphase = optimizer_dphase.init(dphase)


beta0 = 10
beta_increment = 0.5
beta_max = 50


alpha = 0.5
gamma = [1, 0.05, 0.05, 0.05] #cross
# gamma = [0.5, 0.5, 0.05, 0.05] #BS
# gamma = [0, 1, 0.05, 0] #1
# gamma = [0.5, 0.5, 0.05, 0] #>=2

history = dict(
    params=[],
    # dphase=[],
    grad=[],
    # grad_dphase=[],
    obj_tot=[],
    obj_port=[],
    obj_rel=[],
    alpha=alpha,
    beta=[],
    gamma=gamma,
    coeff=[],
    radius=radius,
    lr=lr,
    l_des=l_des,
    dist_wg = dist_wg,
    w_wg = w_wg,
    h_wg = h_wg,
    min_step_per_wvl=min_step_per_wvl,
    n_eff=[],
    iterations=0
)

#%% Run epoch

for ii in range(history["iterations"], 250):
    print("---------------------------------")
    print("Step: ({}/{})".format(ii, num_steps))

    # Too large beta makes it unstable, 30 is good.
    beta = min(beta0 + ii*beta_increment, beta_max)
    (val, aux_data), grad = val_grad_fn(
          params,
          alpha=alpha,
          beta=beta,
          gamma=gamma,
          verbose=False, step_num=ii,
          module=module,
        #   cross_efficiency=0.9938450312283263,
        #   split_efficiency=0.98,
          folder_name=folder_name
    )

    #print
    print("beta={} | objective={} | n_eff={}".format(beta, round(val.item(),4), round(aux_data["n_eff"],4)))
    print("port1,2 = ({}, {}) | |grad| = {}".format( round(np.abs(aux_data["coeff"][0])**2,4), round(np.abs(aux_data["coeff"][1])**2,4), round(np.linalg.norm(grad[0]),4)))

    #history
    history["params"].append(params)
    history["grad"].append(grad[0])
    # history["dphase"].append(dphase)
    # history["grad_dphase"].append(grad[1])
    history["obj_tot"].append(val)
    history["beta"].append(beta)
    history["obj_port"].append(aux_data["obj_port"])
    history["coeff"].append(aux_data["coeff"])
    history["n_eff"].append(aux_data["n_eff"])

    # if ii<num_steps-1:
    history["iterations"] = ii+1
    updates, opt_state = optimizer.update(grad[0], opt_state, params)
    params = optax.apply_updates(params, updates)
    params = jnp.clip(params, 0, 1)
    # updates_dphase, opt_state_dphase = optimizer_dphase.update(grad[1], opt_state_dphase, dphase)
    # dphase = optax.apply_updates(dphase, updates_dphase)



now = datetime.now().strftime("%m%d%H%M")
with open("Log/{}_{}.pickle".format(folder_name, now), 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% plot result
import defineAdjointOptimization as ao
import plot_setting

target = aux_data["coeff_tar"]
coeff = np.array(history["coeff"])
r, theta = np.abs(coeff), np.angle(coeff)
power = r**2
coeff_rel = coeff[:,0]/coeff[:,1]
# r_rel, theta_rel = np.abs(coeff_rel), np.angle(coeff_rel)

opt_state_index = np.argmin(np.array(history["obj_tot"]))
opt_state_index = -1
# opt_state_index = 300+np.argmin(np.abs(coeff_rel[300:]-1j)) # BS

params_opt = history["params"][opt_state_index]

beta_opt = history["beta"][opt_state_index]
sim_final = ao.get_simulation(
    params=params_opt, beta=beta_opt,
    waveguide="+" if module in ["cross", "BS"] else ("L" if module=="1" else "T"), #cross, BS
    symmetric=(module=="cross" or module=="BS"),
    symmetric2=(module=="cross" or module=="BS" or module=="1"),
)

sim_data_opt = td.web.api.webapi.load(
    task_id="fdve-bb4130c8-e013-4938-b3f4-ff6f5991a373", # "1" at 199
)

fig, ax = plt.subplots(2, 1,figsize=(3,3), sharex=True,  tight_layout=True)
ax[0].plot(history["obj_tot"], 'k', label="Total", lw=0.75)
# ax[0].plot(np.array(history["obj_port"])[:,[0,1]], label=[1,2])
ax[0].legend(frameon=False)
ax[0].set_yscale('log')
ax[0].set(ylabel=r'Objective functions, $\mathcal{L}$')
ax[1].plot(power[:,[0,1]], label=[1,2], lw=0.75)
ax[1].axhline(y=np.abs(target[0])**2, color='k', ls='--', lw=0.5)
ax[1].axhline(y=np.abs(target[1])**2, color='k', ls='--', lw=0.5)
ax[1].axvline(x=opt_state_index, color='k', ls='--', lw=0.5)
ax[1].set(ylim=(0,1), xlabel=r'Iteration', ylabel=r'Power (arb. u.)')
ax[1].legend(frameon=False, ncol=2)


fig, ax = plt.subplots(figsize=(3,3))
ax.set(xlim=(-dist_wg/2,dist_wg/2), ylim=(-dist_wg/2,dist_wg/2))
box_plot = td.Box(
    center=(0,0,0),
    size=(dist_wg, dist_wg, 0)
)
eps = np.real(sim_final.to_simulation()[0].epsilon(box=box_plot,freq=ao.freq0).to_numpy()[:,:,0])
x = sim_final.to_simulation()[0].epsilon(box=box_plot,freq=ao.freq0).x
y = sim_final.to_simulation()[0].epsilon(box=box_plot,freq=ao.freq0).y
Y, X = np.meshgrid(y,x)
level = (eps.max()+eps.min())/2
ax.contour(X, Y, eps, [level], colors='k', linewidths=0.5)

x, y = sim_data_opt["field"].Hz.x, sim_data_opt["field"].Hz.y
Y, X = np.meshgrid(y,x)
field = sim_data_opt["field"].Hz
vmax = np.abs(field).max()
ax.pcolormesh(X, Y, np.real(field[:,:,0,0]), cmap=plt.cm.RdBu, vmin=-vmax,vmax=vmax, shading='gouraud')
ax.set(xlabel=r'$x$ ($\mu$m)', ylabel=r'$y$ ($\mu$m)')

fig, ax = plt.subplots(figsize=(3,3), subplot_kw={'projection': 'polar'})
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
cmap3 = plt.cm.cividis
for i in range(0, len(coeff)):
    if i>0:
        ax.plot([theta1_prev, theta[i,0]], [r1_prev, r[i,0]], '-', color=cmap1(i/len(coeff)), lw=0.5, alpha=0.5)
        ax.plot([theta2_prev, theta[i,1]], [r2_prev, r[i,1]], '-', color=cmap2(i/len(coeff)), lw=0.5, alpha=0.5)
    ax.plot(theta[i,0], r[i,0], 'o', color=cmap1(i/len(coeff)), ms=1)
    ax.plot(theta[i,1], r[i,1], 'o', color=cmap2(i/len(coeff)), ms=1)
    ax.plot(theta[i,2], r[i,2], 'o', color=cmap3(i/len(coeff)), ms=1)

    # ax.plot(theta_rel[i], r_rel[i], 'o', color=cmap2(i/len(coeff)), ms=1)

    r1_prev, theta1_prev = r[i,0], theta[i,0]
    r2_prev, theta2_prev = r[i,1], theta[i,1]

ax.plot(np.angle(target[0]), np.abs(target[0]), 'ro', ms=5, mfc="None", mew=0.75)
ax.plot(np.angle(target[1]), np.abs(target[1]), 'ro', ms=5, mfc="None", mew=0.75)
ax.set(ylim=(0,1), yticks=[0,0.5,1], xticks=np.arange(0,2*np.pi,np.pi/4))


#%% save and load aux data

# now = datetime.now().strftime("%m%d%H%M")
# with open("log_cell{}_{}.pickle".format(cell, now), 'wb') as handle:
#     pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open("log_3d_cell_33_04250952.pickle", 'rb') as handle:
    history_load = pickle.load(handle)


#%%

# history_load = dict()

# for key, value in history.items():
#     np.savez("unit_22_{}.npz".format(key), np.array(value))
#     # history_load[key] = np.load("unit_33_{}.npz".format(key))["arr_0"]



#%%

n_temp = 100
array_temp = np.arange(int(n_temp*(n_temp+1)/2))

mat_sym = np.zeros([n_temp]*2)
k = 0 
for i in range(n_temp):
    for j in range(n_temp):
        if i<=j:
            mat_sym[i,j] = array_temp[k]
            mat_sym[j,i] = array_temp[k]
            k += 1



#%% contour
            
box_plot = td.Box(
    center=(0,0,0),
    size=(l_des*1.4, l_des*1.4, 0)
)
eps = np.real(sim_final.to_simulation()[0].epsilon(box=box_plot,freq=ao.freq0).to_numpy()[:,:,0])

x = sim_final.to_simulation()[0].epsilon(box=box_plot,freq=ao.freq0).x
y = sim_final.to_simulation()[0].epsilon(box=box_plot,freq=ao.freq0).y

Y, X = np.meshgrid(y,x)

fig, ax = plt.subplots(figsize=(3.5,3.5))
level = (eps.max()+eps.min())/2
ax.contour(X, Y, eps, [level], colors='k', linewidths=1)



#%%

r = np.arange(0, 2, 0.01)+0.01
theta = np.linspace(0, 2*np.pi, 201)
R, Theta = np.meshgrid(r, theta)
Z = R * np.exp(1j * Theta)

a = 1

L = 0.5*(np.abs(a)**2-np.abs(Z)**2)*(np.abs(Z)<=a) + np.abs(Z-a)**2

X, Y = np.real(Z), np.imag(Z)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, L)

fig, ax =plt.subplots(figsize=(4,4), tight_layout=True)
ax.contour(X, Y, L, levels=np.arange(0,9.01,0.2))
ax.set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"$\mathrm{Im}(z)$")
ax.set_title(r"$\mathcal{L}(z, a=1) = (|a|^2-|z|^2)(|z|\leq|a|) + |z-a|^2$")
