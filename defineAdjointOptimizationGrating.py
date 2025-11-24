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

l_des = 3.5
radius = 120e-3 #minimum fab size
lattice_const = 6*wavelength
buffer_pml_xy = 2.5*wavelength
buffer_pml_z = 0.51 * wavelength
min_step_per_wvl = 20
pixel_size = l_des/350

h_box = 2.0
h_clad = 0.78
h_sub = 500
h_air = 5


## Material parameters
Si = td.material_library["cSi"]["Palik_Lossless"]
SiO2 = td.material_library["SiO2"]["Palik_Lossless"]
air = td.Medium(permittivity=1.0)


## Total simulation size
Lx = 2*buffer_pml_xy + l_des
Ly = 2*buffer_pml_xy + l_des
Lz = h_wg + 2.1* h_clad + buffer_pml_z
z_offset = Lz/2 - (buffer_pml_z+h_clad + h_wg/2)


## Source and monitor locations
x_src1, y_src1 = 0, -l_des*3/4


## Number of design pixels
nx = int(np.ceil(l_des/pixel_size))
ny = int(np.ceil(l_des/pixel_size))

background = td.Structure(
    geometry=td.Box(center=(0,0,0),size=(td.inf, td.inf, td.inf)),
    medium=SiO2
)

BOx = td.Structure(
    geometry = td.Box(
        center=(0,0,-(h_wg+h_box)/2 +z_offset),
        size=(td.inf,td.inf,h_box)
    ),
    medium=SiO2
)

clad = td.Structure(
    geometry = td.Box(
        center=(0,0,h_clad/2 +z_offset),
        size=(td.inf,td.inf,h_clad+h_wg)
    ),
    medium=SiO2
)
substrate = td.Structure(
    geometry = td.Box(
        center=(0,0,-(h_wg+h_sub)/2-h_box +z_offset),
        size=(td.inf,td.inf,h_sub)
    ),
    medium=Si
)


waveguide1_half = td.Structure(
    geometry=td.Box(
        center=(0, -Ly*1.5/2, +z_offset),
        size=(w_wg, Ly*1.5, h_wg)
    ),
    medium=Si
)

design_region_geo = td.Box(
    center=(0,0,h_wg/4 +z_offset),
    size=(l_des, l_des, h_wg/2)
)
designspace = td.Structure(
    geometry=design_region_geo,
    medium=td.Medium(permittivity=2)
)

design_region_lower_geo = td.Box(
    center=(0,0,-h_wg/4+z_offset),
    size=(l_des, l_des, h_wg/2)
)
designspace_lower = td.Structure(
    geometry=design_region_lower_geo,
    medium=Si
)

base_structures = [
    BOx, clad,
    waveguide1_half,
    designspace_lower,
]


mnt_field1 = td.FieldMonitor(
    size=(td.inf,td.inf, 0),
    center=(0,0,0+z_offset),
    freqs=[freq0],
    name="field_z=0"
)

mnt_field2 = td.FieldMonitor(
    size=(0, td.inf,td.inf),
    center=(0,0,0+z_offset),
    freqs=[freq0],
    name="field_x=0"
)


mnt_trans = td.FluxMonitor(
    size=(l_des,l_des, 0),
    center=(0,0,h_wg/2+h_clad*1.1 +z_offset),
    freqs=[freq0],
    normal_dir="+",
    name="transmission",
)


sim_base = tda.JaxSimulation(
    size=(Lx,Lx,Lz),
    structures=base_structures,
    monitors=[mnt_field1, mnt_field2, mnt_trans],
    grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
    boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=(dimension==3)),
    run_time=run_time,
    medium=air,
    symmetry=(-1,0,0) # PEC symmetry
)

fig, ax = plt.subplots(tight_layout=True)
sim_base.plot(z=z_offset-0.01, ax=ax)
ax.plot(l_des*np.array([-0.5,0.5,0.5,-0.5,-0.5]),
        l_des*np.array([-0.5,-0.5,0.5,0.5,-0.5]), 'r--', lw=0.75)


fig, ax = plt.subplots(tight_layout=True)
sim_base.plot(x=0.001, ax=ax)


#%% Mode inspection

w_mode = 4.8

plane_in = td.Box(
    size=(w_mode, 0, wavelength + h_wg),
    center=(x_src1, y_src1, 0 + z_offset)
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
            ax[mode_index,pol].plot(w_wg*np.array([-0.5,0.5,0.5,-0.5,-0.5]), h_wg*np.array([-0.5,-0.5,0.5,0.5,-0.5])+z_offset, 'w', lw=0.5)
            ax[mode_index,pol].axhline(y=-h_wg/2+z_offset,color='w', lw=0.5)
            ax[mode_index,pol].set(xlim=(-4.5/4,4.5/4))

ax[0,0].set_title("In-plane, $x$")
ax[0,1].set_title("In-plane, $y$")
ax[0,2].set_title("Out-of-plane, $z$")

#Mode profile
Hz = -mode_data.field_components["Hz"].sel(mode_index=0)[:,0,:,0]
Z, X = np.meshgrid(Hz.z, Hz.x)
fig, ax = plt.subplots(figsize=(2.5, 1.1), tight_layout=True)
vmax= np.max(np.abs(np.real(Hz)))
pc = ax.pcolormesh(X, Z, np.real(Hz), cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax, linewidth=0, shading='gouraud')
ax.set(xlim=(-1.25,1.25), ylim=(-0.5+z_offset,0.5+z_offset), xlabel=r'$x$ ($\mu$m)', ylabel=r'$z$ ($\mu$m)')

ax.plot(w_wg*np.array([-0.5,0.5,0.5,-0.5,-0.5]), h_wg*np.array([-0.5,-0.5,0.5,0.5,-0.5])+z_offset, 'k', lw=0.5)
ax.axhline(y=-h_wg/2+z_offset,color='k', lw=0.5)

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


mnt_refl = mode_solver.to_monitor(name="reflection")
mnt_refl = mnt_refl.updated_copy(
    mode_spec=mode_spec,
    center=(x_src1, y_src1*1.1, 0 + z_offset),
)


# mnt_angle = td.FieldProjectionAngleMonitor(
#     center=(0,0,h_wg/2+h_clad),
#     size=(td.inf,td.inf,0),
#     freqs=[freq0],
#     name="n2f_angle",
#     medium=air,
#     far_field_approx=True,
#     theta=np.arange(0,np.pi/2, np.pi/180),
#     phi=np.arange(0,2*np.pi, np.pi/180)
# )

mnt_nearfield = td.FieldMonitor(
    size=(Lx,Ly, 0),
    center=(0,0,h_wg/2+h_clad*1.1 +z_offset),
    freqs=[freq0],
    name="nearfield",
    colocate=False,
)

# mnt_backfield = td.FieldMonitor(
#     size=(Lx,Ly, 0),
#     center=(0,0, -h_wg/2-h_clad*1 +z_offset),
#     freqs=[freq0],
#     name="backfield",
#     colocate=False,
# )

# mnt_focalfield = td.FieldMonitor(
#     size=(0, 0, 0),
#     center=(0,0,h_wg/2+h_clad*1.15),
#     freqs=[freq0],
#     name="focalfield"
# )


# fig, ax = plt.subplots()
# src.source_time.plot(times=np.linspace(0,run_time/10,1000), ax=ax)




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
        pos: list=[0,0],
        l_des: float=l_des,
        pixel_size: float=pixel_size) -> tda.JaxStructureStaticGeometry:
    
    xs = np.linspace(pos[0]-(l_des-pixel_size)/2, pos[0]+(l_des-pixel_size)/2, nx).tolist()
    ys = np.linspace(pos[1]-(l_des-pixel_size)/2, pos[1]+(l_des-pixel_size)/2, ny).tolist()
    # Ys, Xs = np.meshgrid(ys, xs)
    
    params = (params + params[::-1])/2
    params_filtered = smoothen.evaluate(params)
    binarize = BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=beta)
    params_binarized = binarize.evaluate(params_filtered)
    
    # params_binarized = params_binarized * (Ys>= l_des/2 + (2*l_des)/(l_des-w_wg) * (np.abs(Xs)-l_des/2))

    eps_values = SiO2.eps_model(freq0) + ( Si.eps_model(freq0)-SiO2.eps_model(freq0)) * params_binarized    
    coords = dict(x=xs, y=ys, z=[h_wg/4+z_offset], f=[freq0])
    eps_data = tda.JaxDataArray(values=eps_values.reshape(nx,ny,1,1), coords=coords)

    field_components = {f"eps_{dim}{dim}": eps_data for dim in "xyz"}
    eps_dataset = tda.JaxPermittivityDataset(**field_components)
    custom_medium = tda.JaxCustomMedium(eps_dataset=eps_dataset)
    custom_structure = tda.JaxStructureStaticGeometry(geometry=design_region_geo, medium=custom_medium)
    return custom_structure

def get_simulation(
        params: jnp.ndarray, 
        beta: float=beta_start, 
        sources: list=[src],) -> tda.JaxSimulation:
    design_structure = get_structure(params=params, beta=beta)
    return tda.JaxSimulation(
        size=(Lx,Lx,Lz),
        structures=base_structures,
        input_structures=[design_structure],
        sources=sources,
        monitors=[mnt_field1, mnt_field2, mnt_refl, mnt_trans],
        output_monitors=[mnt_nearfield],
        grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
        boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
        run_time=run_time,
        symmetry=(-1,0,0),
        subpixel=True,
        medium=air,
    )

# params0 = np.random.rand(nx,ny)
# fig, ax = plt.subplots(2,1, figsize=(20,10))
# sim0 = get_simulation(params=params0, beta=5.0)
# # sim0 = get_simulation(params=params_init, beta=5.0,  symmetric=False)
# sim0.plot(z=z_offset+0.001, freq=freq0, ax=ax[0])
# # ax[0].set(xlim=(-4.5/2,4.5/2), ylim=(-4.5/2,4.5/2))
# sim0.plot(x=0.001, ax=ax[1])


# sim_data = tda.web.run(
#     sim0, 
#     folder_name="test_grating", 
#     task_name="test_source_step0", 
#     verbose=True
# )


#%% Define Objective function

def get_coupling_gaussian(
        sim_data:tda.JaxSimulationData,
        beam_waist: float=l_des/np.sqrt(2)):
    Ex_near = sim_data["nearfield"].Ex.values[:,:,0,0]
    x_near = jnp.array(sim_data["nearfield"].Ex.coords["x"])
    y_near = jnp.array(sim_data["nearfield"].Ex.coords["y"])
    y, x = jnp.meshgrid(y_near,x_near)
    dy, dx = jnp.meshgrid(np.gradient(y_near),np.gradient(x_near))

    w0 = beam_waist
    psi_Gauss = jnp.exp(-(x**2+y**2)/w0**2)
    norm_factor = jnp.sum(np.abs(psi_Gauss)**2*dx*dy)
    C_gauss = jnp.sum(Ex_near*psi_Gauss*dx*dy)/norm_factor
    return C_gauss, (Ex_near, x_near, y_near)

def get_farfield(
        sim_data:tda.JaxSimulationData, 
        dist:float=50*wavelength,
        theta:float=0, 
        phi:float=np.pi/2):
    x_near = jnp.array(sim_data["nearfield"].Ex.coords["x"])
    y_near = jnp.array(sim_data["nearfield"].Ex.coords["y"])
    z_near = jnp.array(sim_data["nearfield"].Ex.coords["z"])
    Y_near,X_near = np.meshgrid(y_near,x_near)
    dx, dy = jnp.gradient(x_near),jnp.gradient(y_near)
    dY, dX = np.meshgrid(dy, dx)

    Ex_near = sim_data["nearfield"].Ex.values[:,:,0,0]

    theta = jnp.array(theta).reshape(-1,1,1,1)
    phi = jnp.array(phi).reshape(1,-1,1,1)

    x_far = dist*np.sin(theta)*np.cos(phi)
    y_far = dist*np.sin(theta)*np.sin(phi)
    z_far = dist*np.cos(theta) + h_wg/2 + h_clad +z_offset
    
    R = jnp.sqrt((X_near-x_far)**2+(Y_near-y_far)**2+ (z_near-z_far)**2)
    Ex_far = jnp.sum(
        1/(1j*wavelength)*Ex_near*jnp.exp(1j*k0*R)*(z_far-z_near)/R**2 *(1 + 1j/k0/R)*dX*dY,
        axis=(2,3)
    )
    Ex_far = jnp.squeeze(Ex_far)
    return Ex_far, (Ex_near, x_near, y_near)


def objective(
        params: jnp.ndarray,
        beta: float=10,
        verbose: bool=True, 
        step_num: int=0,
        # dist:float=50*wavelength,
        # theta_tar:float=0.0, 
        # phi_tar:float=np.pi/2,
        beam_waist: float=l_des,
        folder_name: str="default")->float:

    sim_src = get_simulation(
        params=params, 
        beta=beta, 
    )
    sim_data = tda.web.run(
        sim_src, 
        folder_name=folder_name, 
        task_name="test_source_step{}".format(step_num), 
        verbose=verbose
    )

    c_gauss, (Ex_near, x_near, y_near) = get_coupling_gaussian(sim_data, beam_waist=beam_waist)
    obj_tot = -jnp.abs(c_gauss)**2

    power_refl = np.abs(sim_data["reflection"].amps.sel(mode_index=0, direction="-").values.item())**2
    power_det = sim_data["transmission"].flux.data.item()

    # Ex_near = sim_data["nearfield"].Ex.values[:,:,0,0]
    # x_near = jnp.array(sim_data["nearfield"].Ex.coords["x"])
    # y_near = jnp.array(sim_data["nearfield"].Ex.coords["y"])
    # dy, dx = jnp.meshgrid(np.gradient(y_near),np.gradient(x_near))

    # x_near_in = x_near[jnp.abs(x_near)<=l_des/2]
    # y_near_in = y_near[jnp.abs(y_near)<=l_des/2]
    # dy_in, dx_in = jnp.meshgrid(jnp.gradient(y_near_in),jnp.gradient(x_near_in))
    # Ex_near_in = Ex_near[jnp.abs(x_near)<=l_des/2][:,jnp.abs(y_near)<=l_des/2]

    # intensity_all = jnp.sum(dx*dy*jnp.abs(Ex_near)**2)
    # intensity_in = jnp.sum(dx_in*dy_in*jnp.abs(Ex_near_in)**2)
    # obj_tot = intensity_all - 2*intensity_in



    theta_list = np.linspace(-np.pi/2, np.pi/2, 181)
    Ex_far_polar1, (Ex_near, x_near, y_near) = get_farfield(sim_data, dist=20*wavelength, theta=theta_list, phi=np.pi/2)
    Ex_far_polar2, (Ex_near, x_near, y_near) = get_farfield(sim_data, dist=20*wavelength, theta=theta_list, phi=0.0)

    # Ex_far_obj, (Ex_near, x_near, y_near) = get_farfield(sim_data, dist=dist, theta=theta_tar, phi=phi_tar)
    # obj_tot = -jnp.abs(Ex_far_obj)**2

    aux_data = dict(
        # theta=theta_list,
        Ex_polar1=Ex_far_polar1,
        Ex_polar2=Ex_far_polar2,
        # int_all = intensity_all,
        # int_in = intensity_in,
        Ex_near=Ex_near,
        x_near=x_near,
        y_near=y_near,
        power_det=power_det,
        power_refl=power_refl
    )
    return obj_tot, aux_data

val_grad_fn = jax.value_and_grad(objective, argnums=0, has_aux=True)

# (val_cross, aux_data_cross), grad_cross = val_grad_fn(
#     params0,
#     0.0,
#     beta=10.0,
#     verbose=True, 
#     module='cross', 
#     folder_name="test3d"
# )

