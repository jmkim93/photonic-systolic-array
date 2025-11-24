#%%
import plot_setting
import matplotlib
import matplotlib.pylab as plt
import numpy as np

import jax
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins import waveguide

from tidy3d.plugins.adjoint.utils.filter import ConicFilter, BinaryProjector

import optax
import pickle
from datetime import datetime

from n_eff_calculation import n_eff_slab

#%% Simulation base

dimension = 3

## Spectral parameters
wavelength = 1.55
k0 = 2*np.pi/wavelength
freq0 = td.C_0/wavelength
fwidth = freq0/20
run_time = 100/fwidth


## Geometric parameters
w_wg = 0.3
h_wg = 0.22
dist_wg = 4.8
l_des = 3.5
radius = 120e-3 #minimum fab size
lattice_const = 6*wavelength
buffer_pml_xy = 0.3*wavelength
buffer_pml_z = 0.5 * wavelength
min_step_per_wvl = 20
pixel_size = l_des/350


## Material parameters
Si = td.material_library["cSi"]["Palik_Lossless"]
SiO2 = td.material_library["SiO2"]["Palik_Lossless"]

## Total simulation size
Lx = 2*buffer_pml_xy + dist_wg
Ly = 2*buffer_pml_xy + dist_wg
Lz = 2*buffer_pml_z + h_wg if dimension==3 else 0

## Source and monitor locations
x_src1, y_src1 = 0, -dist_wg/2
x_src2, y_src2 = -dist_wg/2, 0

x_mnt1, y_mnt1 = 0, dist_wg/2
x_mnt2, y_mnt2 = dist_wg/2, 0
x_mnt3, y_mnt3 = 0, -dist_wg/2 - buffer_pml_xy/5
x_mnt4, y_mnt4 = -dist_wg/2, 0

## Number of design pixels
nx = int(np.ceil(l_des/pixel_size))
ny = int(np.ceil(l_des/pixel_size))

background = td.Structure(
    geometry=td.Box(center=(0,0,0),size=(td.inf, td.inf, td.inf)),
    medium=SiO2
)

substrate = td.Structure(
    geometry = td.Box(
        center=(0,0,-(h_wg+1.5*buffer_pml_z)/2),
        size=(td.inf,td.inf,1.5*buffer_pml_z)
    ),
    medium=SiO2
)

waveguide1 = td.Structure(
    geometry=td.Box(
        center=(0, 0, 0),
        size=(w_wg, td.inf, h_wg)
    ),
    medium=Si
)
waveguide2 = td.Structure(
    geometry=td.Box(
        center=(0, 0, 0),
        size=(td.inf, w_wg, h_wg)
    ),
    medium=Si
)

waveguide1_half = td.Structure(
    geometry=td.Box(
        center=(0, -Ly*1.5/2, 0),
        size=(w_wg, Ly*1.5, h_wg)
    ),
    medium=Si
)
waveguide2_half = td.Structure(
    geometry=td.Box(
        center=(Lx*1.5/2, 0, 0),
        size=(Lx*1.5, w_wg, h_wg)
    ),
    medium=Si
)

design_region_geo = td.Box(
    center=(0,0,0),
    size=(l_des, l_des, h_wg)
)
designspace = td.Structure(
    geometry=design_region_geo,
    medium=Si
)

base_structures = [
    # substrate,
    # background,
    waveguide1, 
    waveguide2, 
]

base_structures_half = [
    # substrate,
    # background,
    waveguide1, 
    waveguide2_half, 
]

base_structures_Lshape = [
    # substrate,
    # background,
    waveguide1_half, 
    waveguide2_half, 
]

mnt_field = td.FieldMonitor(
    size=(td.inf,td.inf, 0),
    center=(0,0,0),
    freqs=[freq0],
    name="field"
)

mnt_field_out1 = td.FieldMonitor(
    size=(td.inf, 0, td.inf),
    center=(0,dist_wg/2,0),
    freqs=[freq0],
    name="field_out1"
)
mnt_field_out2 = td.FieldMonitor(
    size=(0, td.inf, td.inf),
    center=(dist_wg/2,0,0),
    freqs=[freq0],
    name="field_out2"
)

sim_base = tda.JaxSimulation(
    size=(Lx,Lx,Lz),
    structures=base_structures,
    monitors=[mnt_field, mnt_field_out1, mnt_field_out2],
    grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
    boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
    run_time=run_time,
    symmetry=(0,0,1),
    medium=SiO2,
)

fig, ax = plt.subplots()
sim_base.plot(z=0, ax=ax)
ax.plot(dist_wg*np.array([-0.5,0.5,0.5,-0.5,-0.5]),
        dist_wg*np.array([-0.5,-0.5,0.5,0.5,-0.5]), 'k--', lw=0.75)
ax.plot(l_des*np.array([-0.5,0.5,0.5,-0.5,-0.5]),
        l_des*np.array([-0.5,-0.5,0.5,0.5,-0.5]), 'r--', lw=0.75)

if dimension==3:
    fig, ax = plt.subplots()
    sim_base.plot(x=dist_wg/4, ax=ax)

#%% Mode inspection

w_mode = dist_wg*1.0

plane_in = td.Box(
    size=(w_mode, 0, Lz or 1.0),
    center=(x_src1, y_src1, 0)
)

num_modes = 3
mode_spec = td.ModeSpec(num_modes=num_modes)
mode_solver = ModeSolver(
    simulation=sim_base.to_simulation()[0],
    plane=plane_in,
    mode_spec=mode_spec,
    freqs=[freq0],
)

mode_solver_group = ModeSolver(
    simulation=sim_base.to_simulation()[0],
    plane=plane_in,
    mode_spec=mode_spec,
    freqs=[freq0*0.999, freq0, freq0*1.001],
)


mode_data = mode_solver.solve()
mode_data_group = mode_solver_group.solve()

n_eff_mode = mode_data_group.n_eff[1,0].data.item()
n_eff_mode_group = n_eff_mode + ((mode_data_group.n_eff[2,0]-mode_data_group.n_eff[0,0])/0.002).data.item()

fig, ax = plt.subplots(num_modes, 3, figsize=(8,4), sharey=True, sharex=True, tight_layout=True)
for mode_index in range(num_modes):
    for pol, field_name in enumerate(["Ex", "Ey", "Ez"]):
        field = mode_data.field_components[field_name].sel(mode_index=mode_index)
        # field.abs.plot(ax=ax[mode_index, pol], vmin=0,vmax=60)
        if dimension==3:
            X,Z = np.meshgrid(field.x, field.z)
            ax[mode_index,pol].pcolormesh(X,Z, field.abs.as_numpy()[:,0,:,0].T**2, vmin=0,vmax=4000, cmap=plt.cm.inferno)
            ax[mode_index,pol].plot(w_wg*np.array([-0.5,0.5,0.5,-0.5,-0.5]), h_wg*np.array([-0.5,-0.5,0.5,0.5,-0.5]), 'w', lw=0.5)
            ax[mode_index,pol].axhline(y=-h_wg/2,color='w', lw=0.5)
            ax[mode_index,pol].set(xlim=(-dist_wg/4,dist_wg/4))
        else:
            ax[mode_index,pol].plot(field.x, field.real.as_numpy()[:,0,0,0])
            ax[mode_index,pol].plot(field.x, field.imag.as_numpy()[:,0,0,0], '--')
            ax[mode_index,pol].axvline(x=-w_wg/2, lw=1, color='gray', ls='--')
            ax[mode_index,pol].axvline(x=w_wg/2, lw=1, color='gray', ls='--')
ax[0,0].set_title("In-plane, $x$")
ax[0,1].set_title("In-plane, $y$")
ax[0,2].set_title("Out-of-plane, $z$")

#Mode profile
Hz = -mode_data.field_components["Hz"].sel(mode_index=0)[:,0,:,0]
Z, X = np.meshgrid(Hz.z, Hz.x)
fig, ax = plt.subplots(figsize=(2.5, 1.1), tight_layout=True)
vmax= np.max(np.abs(np.real(Hz)))
pc = ax.pcolormesh(X, Z, np.real(Hz), cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax, linewidth=0, shading='gouraud')
ax.set(xlim=(-1.25,1.25), ylim=(-0.5,0.5), xlabel=r'$x$ ($\mu$m)', ylabel=r'$z$ ($\mu$m)')

ax.plot(w_wg*np.array([-0.5,0.5,0.5,-0.5,-0.5]), h_wg*np.array([-0.5,-0.5,0.5,0.5,-0.5]), 'k', lw=0.5)
ax.axhline(y=-h_wg/2,color='k', lw=0.5)

cb = plt.colorbar(pc, shrink=1)
cb.ax.set_yticks([-vmax,0,vmax])
cb.ax.set_yticklabels([-1,0,1])
cb.ax.set_ylabel(r"$H_z$ (Arb.U.)")
fig.savefig("mode_profile.pdf")



src = mode_solver.to_source(
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    direction="+",
    mode_index=0
)
src = src.updated_copy(mode_spec=mode_spec)

# fig, ax = plt.subplots()
# src.source_time.plot(times=np.linspace(0,run_time/10,1000), ax=ax)


mnt1 = mode_solver.to_monitor(
    name="MNT_MODE1",
)
mnt1 = mnt1.updated_copy(
    mode_spec=mode_spec,
    center=(x_mnt1,y_mnt1,0)
)
mnt2 = mnt1.updated_copy(
    mode_spec=mode_spec,
    size=(0, w_mode, Lz or 1.0),
    center=(x_mnt2,y_mnt2,0),
    name="MNT_MODE2",
)
mnt3 = mnt1.updated_copy(
    mode_spec=mode_spec,
    center=(x_mnt3,y_mnt3,0),
    name="MNT_MODE3",
)
mnt4 = mnt1.updated_copy(
    mode_spec=mode_spec,
    size=(0, w_mode, Lz or 1.0),
    center=(x_mnt4,y_mnt4,0),
    name="MNT_MODE4",
)


#%% Design parameters

## Filter and Binarization for fabrication constraint

beta = 4.0
beta_start = 10.0
smoothen = ConicFilter(radius=radius, design_region_dl=pixel_size)
binarize = BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=beta)

param_array = np.arange(0, 1, 0.01)
fig, ax = plt.subplots()
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=1).evaluate(param_array), label=r'$\beta=1$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=2).evaluate(param_array), label=r'$\beta=2$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=5).evaluate(param_array), label=r'$\beta=5$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=10).evaluate(param_array), label=r'$\beta=10$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=20).evaluate(param_array), label=r'$\beta=20$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=30).evaluate(param_array), label=r'$\beta=30$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=40).evaluate(param_array), label=r'$\beta=40$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=50).evaluate(param_array), label=r'$\beta=50$')
ax.plot(param_array, BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=100).evaluate(param_array), label=r'$\beta=100$')
ax.legend()

def get_structure(
        params: jnp.ndarray, 
        beta: float=beta_start, 
        symmetric: bool=False,
        symmetric2: bool=False,
        eta: float=0.5,
        pos: list=[0,0],
        l_des: float=l_des,
        pixel_size: float=pixel_size) -> tda.JaxStructureStaticGeometry:

    if symmetric:
        params = (params + params.T)/2
    if symmetric2:
        params = (params + params[::-1,::-1].T)/2
    params_filtered = smoothen.evaluate(params)
    
    binarize = BinaryProjector(vmin=0,vmax=1, eta=eta, beta=beta)
    params_binarized = binarize.evaluate(params_filtered)
    eps_values = SiO2.eps_model(freq0) + ( Si.eps_model(freq0)-SiO2.eps_model(freq0)) * params_binarized
    xs = np.linspace(pos[0]-(l_des-pixel_size)/2, pos[0]+(l_des-pixel_size)/2, nx).tolist()
    ys = np.linspace(pos[1]-(l_des-pixel_size)/2, pos[1]+(l_des-pixel_size)/2, ny).tolist()
    coords = dict(x=xs, y=ys, z=[0], f=[freq0])
    eps_data = tda.JaxDataArray(values=eps_values.reshape(nx,ny,1,1), coords=coords)

    field_components = {f"eps_{dim}{dim}": eps_data for dim in "xyz"}
    eps_dataset = tda.JaxPermittivityDataset(**field_components)
    custom_medium = tda.JaxCustomMedium(eps_dataset=eps_dataset)
    custom_structure = tda.JaxStructureStaticGeometry(geometry=design_region_geo, medium=custom_medium)
    return custom_structure

def get_simulation(
        params: jnp.ndarray, 
        beta: float=beta_start, 
        eta: float=0.5,
        sources: list=[src],
        waveguide: str="+", 
        symmetric: bool=False,
        symmetric2: bool=False) -> tda.JaxSimulation:
    design_structure = get_structure(params=params, beta=beta, eta=eta, symmetric=symmetric, symmetric2=symmetric2)
    return tda.JaxSimulation(
        size=(Lx,Lx,Lz),
        structures=base_structures if waveguide=="+" else (base_structures_half if waveguide=="T" else base_structures_Lshape),
        input_structures=[design_structure],
        sources=sources,
        monitors=[mnt_field, mnt_field_out1, mnt_field_out2],
        output_monitors=[mnt1,mnt2,mnt3,mnt4],
        grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
        boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
        run_time=run_time,
        subpixel=True,
        medium=SiO2,
        symmetry=(0,0,1),
    )

# params0 = np.random.rand(nx,ny)

# fig, ax = plt.subplots(2,1, figsize=(20,10))
# sim0 = get_simulation(params=params0, beta=5.0,  symmetric=True, waveguide="+")
# # sim0 = get_simulation(params=params_init, beta=5.0,  symmetric=False)
# sim0.plot_structures_eps(z=0.01, freq=freq0, ax=ax[0])
# ax[0].set(xlim=(-dist_wg/2,dist_wg/2), ylim=(-dist_wg/2,dist_wg/2))
# sim0.plot(z=0, ax=ax[1])


# sim_data = tda.web.run(
#     sim0, 
#     folder_name="test3d", 
#     task_name="test_source_step0", 
#     verbose=True
# )

#%% Define Objective function

def get_coupling_coefficients(sim_data: tda.JaxSimulationData, num_monitor: int=4)->jnp.ndarray:
    c_mode0 = jnp.squeeze(jnp.array([sim_data["MNT_MODE{}".format(port)].amps.sel(mode_index=0, direction="+" if port<=2 else "-").values for port in range(1,1+num_monitor)]))
    return c_mode0, sim_data["MNT_MODE3"].n_eff.data[0,0]

def objective(
        params: jnp.ndarray,
        alpha: float=0.5,
        beta: float=beta_start,
        gamma: list=[1.0]*5,
        verbose: bool=True, 
        step_num: int=0, 
        module: str="cross",
        cross_efficiency: float=1.0,
        split_efficiency: float=1.0,
        folder_name: str="default")->float:

    symmetric = (module=="cross" or module=="BS")
    symmetric2 = (module=="cross" or module=="BS" or module=="1")

    if not symmetric:
        efficiency_tot = cross_efficiency * split_efficiency
        n = int(module)
        if efficiency_tot==1.0:
            split = 1/n
        else:
            split = efficiency_tot**(n-1) * (1-efficiency_tot) / (1-efficiency_tot**n)

    sim_src = get_simulation(
        params=params, 
        beta=beta, 
        symmetric=symmetric,
        symmetric2=symmetric2,
        waveguide="+" if symmetric else ("L" if symmetric2 else "T"),
    )
    sim_data = tda.web.run(
        sim_src, 
        folder_name=folder_name, 
        task_name="test_source_step{}".format(step_num), 
        verbose=verbose
    )
    coeff_mode0, n_eff_mode = get_coupling_coefficients(sim_data)
    phase0 = n_eff_mode * k0 * dist_wg
    if module=="cross":
        target = jnp.array([1,0,0,0]) * jnp.exp(1j*(phase0))
    elif module=="BS":
        target = jnp.array([split_efficiency*0.5, -split_efficiency*0.5, 0, 0j])**0.5 * jnp.exp(1j*(phase0))
    else:
        target = jnp.array([split_efficiency*(1-split), split_efficiency*split,0,0j])**0.5 * jnp.exp(1j*phase0)
    
    target = target * jnp.array([1,-1,1,-1]) ## x, ydirections have different sign of coefficient

    obj_port_small = jnp.abs(target)**2 - jnp.abs(coeff_mode0)**2
    obj_port_large = jnp.abs(coeff_mode0 - target)**2
    obj_port = (obj_port_small>0)*obj_port_small*alpha + obj_port_large

    obj_tot = jnp.sum(jnp.array(gamma[:4]) * obj_port) # minimize this

    aux_data = dict(
        obj_port = obj_port,
        coeff=coeff_mode0,
        n_eff=n_eff_mode,
        coeff_tar=target
    )
    return obj_tot, aux_data

val_grad_fn = jax.value_and_grad(objective, argnums=[0], has_aux=True)

# (val_cross, aux_data_cross), grad_cross = val_grad_fn(
#     params0,
#     0.0,
#     beta=10.0,
#     verbose=True, 
#     module='cross', 
#     folder_name="test3d"
# )

