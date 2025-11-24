
#%%
import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import plot_setting

import numpy as np

import jax
import jax.numpy as jnp

import tidy3d as td
import tidy3d.web as web
import tidy3d.plugins.adjoint as tda
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins import waveguide

from tidy3d.plugins.adjoint.utils.filter import ConicFilter, BinaryProjector

import optax
import pickle
from datetime import datetime

from n_eff_calculation import n_eff_slab

import defineAdjointOptimization as ua
from defineAdjointOptimization import get_structure, smoothen, SiO2, Si, freq0

def env(t, signal):
    dt = t[1]-t[0]
    T_period = dt*len(t)
    df = 1/T_period
    f = df.data * np.arange(len(t))
    FFT_signal = np.fft.fft(signal)
    FFT_signal_filtered = FFT_signal * ((f<freq0)+(f>f[-1]-freq0))
    signal_env = np.fft.ifft(FFT_signal_filtered)*2
    return np.real(signal_env)

## Obtain design parameters

module_names = ["cross", "BS", "1", "2", "3", "4"]
eps_opt = dict()
binarize = BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=50.0)
for mod in module_names:
    with open("Data_Fig2/set3/PE_module_{}.pickle".format(mod), 'rb') as handle:
        trainData = pickle.load(handle)
    coeff = np.array(trainData["coeff"])
    coeff_rel = coeff[:,0]/coeff[:,1]
    idx_opt = (200+np.argmin(np.abs(coeff_rel[200:]-1j)) if mod=='BS' 
               else np.argmin(np.array(trainData["obj_tot"])))   
    param =  trainData["params"][idx_opt]
    sym1 = mod in ["cross", "BS"]
    sym2 = mod in ["cross", "BS", "1"]
    if sym1:
        param = (param + param.T)/2
    if sym2:
        param = (param + param[::-1,::-1].T)/2
    param = binarize.evaluate(smoothen.evaluate(param))
    eps_opt[mod] = SiO2.eps_model(freq0) + ( Si.eps_model(freq0)-SiO2.eps_model(freq0)) * param
    


with open("Data_Fig2/set3/PE_module_grating_smallbw.pickle", 'rb') as handle:
    trainData = pickle.load(handle)
    idx_opt = np.argmin(np.array(trainData["obj_tot"]))
    param = trainData["params"][idx_opt]

param = (param + param[::-1])/2
param = binarize.evaluate(smoothen.evaluate(param))
eps_opt["grating"] = SiO2.eps_model(freq0) + ( Si.eps_model(freq0)-SiO2.eps_model(freq0)) * param

np.savez_compressed("eps_opt", **eps_opt)

sim_data_neff = td.web.api.webapi.load(task_id="fdve-387c7701-2ef5-456d-b2ce-8c038d6526c1")

del trainData

## effective index calculation

freq_list = freq0*np.linspace(0.95,1.05,101)
n_eff_phase = sim_data_neff["mode_inspect"].n_eff[:,0].data
n_eff_group = n_eff_phase + freq_list * np.gradient(n_eff_phase, freq_list)

fig, ax = plt.subplots(figsize=(3,2))
ax.plot(freq_list/1e12, n_eff_phase, lw=0.75)
ax.plot(freq_list/1e12, n_eff_group, lw=0.75)
print("n_eff_phase = {}".format(n_eff_phase[50]))
print("n_eff_phase = {}".format(n_eff_group[50]))
ax.set(xlim=(freq_list[0]/1e12, freq_list[-1]/1e12), xlabel=r"Frequency (THz)", ylabel=r"$n_\mathrm{eff}$")
ax.axvline(x=freq_list[50]/1e12, color='k', lw=0.5, ls='--')

del sim_data_neff

#%% Simulation

from defineAdjointOptimization import w_wg, h_wg, l_des, w_mode
from defineAdjointOptimization import min_step_per_wvl, pixel_size

def get_structure(eps: np.ndarray, center: tuple=(0,0,0), size: tuple=(l_des,l_des,h_wg)):
    xs = np.linspace(center[0]-(l_des-pixel_size)/2, center[0]+(l_des-pixel_size)/2, eps.shape[0]).tolist()
    ys = np.linspace(center[1]-(l_des-pixel_size)/2, center[1]+(l_des-pixel_size)/2, eps.shape[1]).tolist()
    coords = dict(x=xs, y=ys, z=[center[2]], f=[freq0])
    eps_data = td.ScalarFieldDataArray(eps.reshape(eps.shape[0],eps.shape[1],1,1), coords=coords)
    field_components = {f"eps_{dim}{dim}": eps_data for dim in "xyz"}
    eps_dataset = td.PermittivityDataset(**field_components)
    custom_medium = td.CustomMedium(eps_dataset=eps_dataset)
    return td.Structure(
        geometry=td.Box(center=center, size=size), 
        medium=custom_medium
    )

### Geometry
wavelength = 1.55
k0 = 2*np.pi/wavelength
freq0 = td.C_0/wavelength

h_box = 2.0
h_clad = 0.78
h_sub = 500

### Lattice definition
N_cell = (2, 2) # Change as (1,4) first to calculate time delay
lattice_const = wavelength/n_eff_phase[50]*30 # Too short? might degrade grating coupler
dist_module = 4.0
x_cell_center = (N_cell[0]/2 - 1/2 - np.arange(N_cell[0])) * lattice_const
y_cell_center = (N_cell[1]/2 - 1/2 - np.arange(N_cell[1])) * lattice_const


buffer_pml_xy = 1.5 * wavelength
buffer_pml_z = 1 * wavelength
Lx = N_cell[0] * lattice_const + 2*buffer_pml_xy
Ly = N_cell[1] * lattice_const + 2*buffer_pml_xy
Lz = 2*buffer_pml_z + h_clad + h_wg
z_offset = Lz/2 - (buffer_pml_z + h_clad + h_wg/2)

## Spectral parameters
pulse_width = 100
fwidth = freq0/pulse_width
t_delay = 3.486268405754292e-13


run_time = (
    15/(2*np.pi*fwidth) + 
    t_delay * (np.sum(N_cell)-1) + 
    (h_clad+h_wg)*SiO2.nk_model(freq_list)[0][50]/td.C_0
)

BOx = td.Structure(
    geometry=td.Box(
        size=(td.inf, td.inf, h_box), 
        center=(0, 0, -(h_wg+h_box)/2+ z_offset)
    ),
    medium=SiO2
)
cladding = td.Structure(
    geometry=td.Box(
        size=(td.inf, td.inf, h_clad+h_wg), 
        center=(0,0,h_clad/2 +z_offset)
    ),
    medium=SiO2
)

substrate = td.Structure(
    geometry = td.Box(
        center=(0,0,-(h_wg+h_sub)/2-h_box+z_offset),
        size=(td.inf,td.inf,h_sub)
    ),
    medium=Si
)
base_structure = [BOx, cladding]
wg_x = [
    td.Structure(
        geometry=td.Box(
            center=(-dist_module + x_cell_center[i], -lattice_const/2, z_offset),
            size=(w_wg, N_cell[1]*lattice_const, h_wg)
        ),
        medium=Si
    ) for i in range(N_cell[0])
]
wg_y = [
    td.Structure(
        geometry=td.Box(
            center=(-lattice_const/2, -dist_module + y_cell_center[i], z_offset),
            size=(N_cell[0]*lattice_const, w_wg, h_wg)
        ),
        medium=Si
    ) for i in range(N_cell[1])
]
wg_intra_x = [
    td.Structure(
        geometry=td.Box(
            center=(x_cell_center[i], y_cell_center[j], z_offset),
            size=(w_wg, 2*dist_module, h_wg),
        ),
        medium=Si
    ) for i in range(N_cell[0]) for j in range(N_cell[1])
]
wg_intra_y = [
    td.Structure(
        geometry=td.Box(
            center=(x_cell_center[i], y_cell_center[j], z_offset),
            size=(2*dist_module, w_wg, h_wg),
        ),
        medium=Si
    ) for i in range(N_cell[0]) for j in range(N_cell[1])
]
crossings =[
    get_structure(
        eps_opt["cross"], 
        center=(x_cell_center[i]-dist_module,y_cell_center[j]-dist_module,z_offset)
    ) for i in range(N_cell[0]) for j in range(N_cell[1])
]
splitter = [
    get_structure(
        eps_opt[str(j+1)] * p + (eps_opt[str(i+1)].T) * (1-p), 
        center=(x_cell_center[i]-dist_module*p, y_cell_center[j]-dist_module*(1-p),z_offset)
    ) for i in range(N_cell[0]) for j in range(N_cell[1]) for p in range(2)
]
BSs = [
    get_structure(
        eps_opt["BS"], 
        center=(x_cell_center[i] ,y_cell_center[j],z_offset)
    ) for i in range(N_cell[0]) for j in range(N_cell[1])
]
grating = [
    get_structure(
        eps_opt["grating"].T * p + eps_opt["grating"]* (1-p), 
        center=(x_cell_center[i]+dist_module*p,y_cell_center[j]+dist_module*(1-p),h_wg/4+z_offset),
        size=(l_des,l_des,h_wg/2)
    ) for i in range(N_cell[0]) for j in range(N_cell[1]) for p in range(2)
]
grating_sub = [
    td.Structure(
        geometry=td.Box(
            center=(x_cell_center[i]+dist_module*p, y_cell_center[j]+dist_module*(1-p), -h_wg/4+z_offset),
            size=(l_des,l_des,h_wg/2)
        ),
        medium=Si        
    ) for i in range(N_cell[0]) for j in range(N_cell[1]) for p in range(2)
]
base_structure = (
    base_structure
    + wg_x + wg_y + wg_intra_x + wg_intra_y 
    + crossings 
    + splitter 
    + BSs 
    + grating + grating_sub
)

grid_spec = td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl)
sim_base = td.Simulation(
    size=(Lx,Ly,Lz),
    structures=base_structure,
    grid_spec=grid_spec,
    boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
    run_time=run_time,
    subpixel=True,
)

plane_in = td.Box(
    size=(w_mode, 0, wavelength + h_wg),
    center=(x_cell_center[-1]-dist_module, y_cell_center[-1]-lattice_const/2, z_offset)
)
num_modes = 1
mode_spec = td.ModeSpec(num_modes=num_modes, target_neff=n_eff_phase[50])
mode_solver = ModeSolver(
    simulation=sim_base,
    plane=plane_in,
    mode_spec=mode_spec,
    freqs=[freq0],
)
mode_data = mode_solver.solve()

mnt_thru_A_flux = [td.FluxTimeMonitor(
    center=(x-dist_module, y-lattice_const/2, z_offset),
    size=(w_wg, 0, h_wg),
    name="thruA_{}_{}_flux".format(nx, ny),
    interval=5,
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center)]

mnt_thru_B_flux = [td.FluxTimeMonitor(
    center=(x-lattice_const/2, y-dist_module,  z_offset),
    size=(0, w_wg, h_wg),
    name="thruB_{}_{}_flux".format(nx, ny),
    interval=5,
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center)]

mnt_det_C_flux = [td.FluxTimeMonitor(
    center=(x, y+dist_module,  h_wg/2+h_clad*1.1+z_offset),
    size=(l_des, l_des, 0),
    name="detC_{}_{}_flux".format(nx, ny),
    interval=5,
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center)]

mnt_det_D_flux = [td.FluxTimeMonitor(
    center=(x+dist_module, y,  h_wg/2+h_clad*1.1+z_offset),
    size=(l_des, l_des, 0),
    name="detD_{}_{}_flux".format(nx, ny),
    interval=5,
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center)]


mnt_det_C_field = [td.FieldTimeMonitor(
    center=(x+delx, y+dist_module+dely,  h_wg/2+h_clad*1.1+z_offset),
    size=(0, 0, 0),
    name="detC_{}_{}_{}_{}_field".format(nx, ny, ndx, ndy),
    interval=5,
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center) for ndx, delx in enumerate(np.arange(-l_des/2+l_des/10, l_des/2, l_des/5)) for ndy, dely in enumerate(np.arange(-l_des/2+l_des/10, l_des/2, l_des/5))]

mnt_det_D_field = [td.FieldTimeMonitor(
    center=(x+dist_module+delx, y+dely,  h_wg/2+h_clad*1.1+z_offset),
    size=(0, 0, 0),
    name="detD_{}_{}_{}_{}_field".format(nx, ny, ndx, ndy),
    interval=5,
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center) for ndx, delx in enumerate(np.arange(-l_des/2+l_des/10, l_des/2, l_des/5)) for ndy, dely in enumerate(np.arange(-l_des/2+l_des/10, l_des/2, l_des/5))]

freq_monitor = mode_solver.to_monitor(name='temp')
mnt_thru_A_freq = [freq_monitor.updated_copy(
    mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff_phase[50]),
    size=(w_mode, 0, wavelength + h_wg),
    center=(x-dist_module, y-lattice_const/2, z_offset),
    name="thruA_{}_{}_freq".format(nx, ny),
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center)]
mnt_thru_B_freq = [freq_monitor.updated_copy(
    mode_spec=mode_spec,
    size=(0, w_mode, wavelength + h_wg),
    center=(x-lattice_const/2, y-dist_module,  z_offset),
    name="thruB_{}_{}_freq".format(nx, ny),
) for nx, x in enumerate(x_cell_center) for ny, y in enumerate(y_cell_center)]

mnt_landscape = td.FieldMonitor(
    center=(0,0,z_offset),
    size=(Lx, Ly, 0),
    name='field_center',
    fields=["Hz", "Ex", "Ey"],
    freqs=[freq0],
    colocate=False,
)
mnt_landscape_air = td.FieldMonitor(
    center=(0,0,h_wg/2+h_clad*1.1+z_offset),
    size=(Lx, Ly, 0),
    name='field_air',
    freqs=[freq0],
    fields=["Ex", "Ey", "Hx", "Hy"],
    colocate=False,
)
mnt_field_freq = [mnt_landscape, mnt_landscape_air]
monitors = (
    mnt_thru_A_flux + 
    mnt_thru_B_flux + 
    mnt_det_C_flux + 
    mnt_det_D_flux + 
    mnt_det_C_field + 
    mnt_det_D_field + 
    mnt_thru_A_freq + 
    mnt_thru_B_freq +
    mnt_field_freq
)

src_amp = [1, 0, 0, 0]

def make_sim(src_amp: tuple=(1,0,0,0)):
    amp_A = src_amp[:N_cell[0]]
    amp_B = src_amp[N_cell[0]:]
    
    src_A = [td.ModeSource(
        size=(w_mode, 0, wavelength + h_wg),
        center=(x-dist_module, y_cell_center[-1]-lattice_const/2-0.1*wavelength, z_offset),
        source_time = td.GaussianPulse(
            freq0=freq0, fwidth=fwidth,
            amplitude=1e-14+np.abs(amp_A[nx])*np.sqrt(N_cell[1]), 
            phase=0+np.pi*(amp_A[nx]<0),
            # offset=4 + 2*np.pi*fwidth * t_delay_prac[nx],
            offset=4 + 2*np.pi*fwidth * t_delay*(N_cell[0]-1-nx)
        ),
        mode_spec=mode_spec,
        mode_index=0,
        direction="+",
        num_freqs=9,
    ) for nx, x in enumerate(x_cell_center)]

    src_B = [td.ModeSource(
        size=(0, w_mode, wavelength + h_wg),
        center=(x_cell_center[-1]-lattice_const/2-0.1*wavelength, y-dist_module, z_offset),
        source_time = td.GaussianPulse(
            freq0=freq0, fwidth=fwidth,
            amplitude=1e-14+np.abs(amp_B[ny])*np.sqrt(N_cell[0]), 
            phase=-np.pi/2+np.pi*(amp_B[ny]<0),
            # offset=4 + 2*np.pi*fwidth * t_delay_prac[ny],
            offset=4 + 2*np.pi*fwidth * t_delay*(N_cell[1]-1-ny)
        ),
        mode_spec=mode_spec,
        mode_index=0,
        direction="+",
        num_freqs=9,
    ) for ny, y in enumerate(y_cell_center)]
    sources = src_A+src_B
    return sim_base.updated_copy(sources=sources, monitors=monitors)


### change config to (4,1) first and run to measure the precise time delay 
# sim_time_measure = make_sim([1,0,0])
# sim_time_measure_data = td.web.run(
#     sim_time_measure, 
#     folder_name="Fig6_mmm",
#     task_name="time_measure", 
#     verbose=True
# )
# t0 = sim_time_measure_data["thruA_0_0_flux"].flux.t
# flux0 = sim_time_measure_data["thruA_0_0_flux"].flux.data
# t1 = sim_time_measure_data["thruA_0_1_flux"].flux.t
# flux1 = sim_time_measure_data["thruA_0_1_flux"].flux.data
# t_delay = (t0[np.argmax(env(t0,flux0))]-t1[np.argmax(env(t1,flux1))]).data.item()



## Real simulation 
sims = {
    "A0": make_sim([1,0,0,0]),
    'A1': make_sim([0,1,0,0]), 
    "B0": make_sim([0,0,1,0]),
    "B1": make_sim([0,0,0,1]),
}
# batch = web.Batch(simulations=sims, folder_name="Fig6_mmm")
# batch_data = batch.run(path_dir="Data_Fig6")


#%% Data Load and organize, save

batch_data = {
    'A0': web.load(task_id='fdve-cc92f70f-1813-42e5-96e0-33e906eaf52f', path="Data_Fig6/fdve-cc92f70f-1813-42e5-96e0-33e906eaf52f.hdf5", replace_existing=False), 
    'A1': web.load(task_id='fdve-34147e2f-4edc-493d-999a-4dc464d2d12b', path="Data_Fig6/fdve-34147e2f-4edc-493d-999a-4dc464d2d12b.hdf5", replace_existing=False), 
    'B0': web.load(task_id='fdve-efe27570-4197-404a-86e5-68ddfa88e0a7', path="Data_Fig6/fdve-efe27570-4197-404a-86e5-68ddfa88e0a7.hdf5", replace_existing=False), 
    'B1': web.load(task_id='fdve-cba529a0-4ab5-4062-9b6c-eb84755985c6', path="Data_Fig6/fdve-cba529a0-4ab5-4062-9b6c-eb84755985c6.hdf5", replace_existing=False), 
}

src_list = ["A0","A1","B0","B1"]
y, x = np.meshgrid(batch_data["A0"]["field_air"].Ex.y, batch_data["A0"]["field_air"].Ex.x)
Ex_air_freq = np.stack([batch_data[src]["field_air"].Ex.data[:,:,0,0] for src in src_list])
Ey_air_freq = np.stack([batch_data[src]["field_air"].Ey.data[:,:,0,0] for src in src_list])
Hx_air_freq = np.stack([batch_data[src]["field_air"].Hx.data[:,:,0,0] for src in src_list])
Hy_air_freq = np.stack([batch_data[src]["field_air"].Hy.data[:,:,0,0] for src in src_list])
Hz_center_freq = np.stack([batch_data[src]["field_center"].Hz.data[:,:,0,0] for src in src_list])

Field_freq = {
    "x": x,
    "y": y,
    "Ex_air": Ex_air_freq,
    "Ey_air": Ey_air_freq,
    "Hx_air": Hx_air_freq,
    "Hy_air": Hy_air_freq,
    "Hz_center": Hz_center_freq
}

thru_flux_time = {
    "{}{}{}".format(direction, nx, ny): np.stack([batch_data[src]["thru{}_{}_{}_flux".format(direction, nx, ny)].flux.data for src in src_list])
    for direction in "AB" for nx in range(N_cell[0]) for ny in range(N_cell[1])
}
thru_flux_time["t"] = batch_data["A0"]["thruA_0_0_flux"].flux.t.data

det_flux_time = {
    "{}{}{}".format(direction, nx, ny): np.stack([batch_data[src]["det{}_{}_{}_flux".format(direction, nx, ny)].flux.data for src in src_list])
    for direction in "CD" for nx in range(N_cell[0]) for ny in range(N_cell[1])
}
det_flux_time["t"] = batch_data["A0"]["detC_0_0_flux"].flux.t.data

det_field_time = {
    "{}{}{}".format(det, nx, ny): np.stack([
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Ex.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2),
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Ey.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2),
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Hx.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2),
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Hy.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2)
    ], axis=2)
    for det in "CD" for nx in range(N_cell[0]) for ny in range(N_cell[1])
} # (Num src, time steps, num_field-Ex,Ey,Hx,Hy, num det points)
det_field_time["t"] = batch_data["A0"]["detC_0_0_0_0_field"].Ex.t.data


# with open("Data_Fig6/field_freq.pickle", 'wb') as handle:
#     pickle.dump(Field_freq, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open("Data_Fig6/thru_flux.pickle", 'wb') as handle:
#     pickle.dump(thru_flux_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open("Data_Fig6/det_flux.pickle", 'wb') as handle:
#     pickle.dump(det_flux_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open("Data_Fig6/det_field.pickle", 'wb') as handle:
#     pickle.dump(det_field_time, handle, protocol=pickle.HIGHEST_PROTOCOL)



#%% Fig 6 signal routing
from defineAdjointOptimization import w_wg, h_wg, l_des, w_mode
from defineAdjointOptimization import min_step_per_wvl, pixel_size
from matplotlib.colors import ListedColormap

# color_a = (100/256,100/256,256/256)
# color_b = (247/256,147/256,30/256)
# cmap_a = ListedColormap(np.arange(256).reshape(-1,1)/256 * np.array(color_a).reshape(1,-1))
# cmap_b = ListedColormap(np.arange(256).reshape(-1,1)/256 * np.array(color_b).reshape(1,-1))


color_a = (46/256,49/256,146/256)
color_b = (247/256,147/256,30/256)
cmap_a = ListedColormap(1+np.arange(256).reshape(-1,1)/256 * (np.array(color_a)-1).reshape(1,-1))
cmap_b = ListedColormap(1+np.arange(256).reshape(-1,1)/256 * (np.array(color_b)-1).reshape(1,-1))



### Geometry
wavelength = 1.55
k0 = 2*np.pi/wavelength
freq0 = td.C_0/wavelength

h_box = 2.0
h_clad = 0.78
h_sub = 500

### Lattice definition
N_cell = (2,2)
lattice_const = wavelength/n_eff_phase[50]*30
# lattice_const = wavelength/n_eff_phase[50]*18
dist_module = 4.0
x_cell_center = (N_cell[0]/2 - 1/2 - np.arange(N_cell[0])) * lattice_const
y_cell_center = (N_cell[1]/2 - 1/2 - np.arange(N_cell[1])) * lattice_const

buffer_pml_xy = 1.5 * wavelength
buffer_pml_z = 1 * wavelength
Lx = N_cell[0] * lattice_const + 2*buffer_pml_xy
Ly = N_cell[1] * lattice_const + 2*buffer_pml_xy
Lz = 2*buffer_pml_z + h_clad + h_wg
z_offset = Lz/2 - (buffer_pml_z + h_clad + h_wg/2)



signal = np.array((1, -1, -1,1))
with open("Data_Fig6/field_freq.pickle", 'rb') as handle:
    field_freq = pickle.load(handle)

x, y = field_freq["x"], field_freq["y"]
Hz = field_freq["Hz_center"]
vmax = (np.abs(Hz).max())**2 /20

fig, ax = plt.subplots(1,4, figsize=(7, 2.1), sharex=True, sharey=True, tight_layout=True)
ax[0].pcolormesh(x, y, np.abs(Hz[0])**2, vmin=0 ,vmax=vmax, cmap=cmap_a, rasterized=True)
pca = ax[1].pcolormesh(x, y, np.abs(Hz[1])**2, vmin=0 ,vmax=vmax, cmap=cmap_a, rasterized=True)
ax[2].pcolormesh(x, y, np.abs(Hz[2])**2, vmin=0 ,vmax=vmax, cmap=cmap_b, rasterized=True)
pcb = ax[3].pcolormesh(x, y, np.abs(Hz[3])**2, vmin=0,vmax=vmax, cmap=cmap_b, rasterized=True)

ax[0].set(
    xticks=(-20,0, 20), yticks=(-20,0, 20),
    xlim=(-lattice_const*N_cell[0]/2,lattice_const*N_cell[0]/2), 
    ylim=(-lattice_const*N_cell[1]/2,lattice_const*N_cell[1]/2)
)

# [ax[i].axvline(x=x-lattice_const/2, color='gray', ls='--', lw=0.5) for x in x_cell_center for i in range(4)]
# [ax[i].axhline(y=y-lattice_const/2, color='gray', ls='--', lw=0.5) for y in y_cell_center for i in range(4)]

x_module = l_des/2*np.array([1,1,-1,-1,1])
y_module = l_des/2*np.array([1,-1,-1,1,1])


[ax[i].plot(x+dist_module+x_module, y+y_module, color=plt.cm.PRGn(0.2), ls='--', lw=0.4) for x in x_cell_center for y in y_cell_center for i in range(4)]
[ax[i].plot(x+x_module, y+dist_module+y_module, color=plt.cm.PRGn(0.8), ls='--', lw=0.4) for x in x_cell_center for y in y_cell_center for i in range(4) ]
[ax[i].plot(x-dist_module+x_module, y+y_module, color='gray', ls='--', lw=0.4) for x in x_cell_center for y in y_cell_center for i in range(4) ]
[ax[i].plot(x+x_module, y-dist_module+y_module, color='gray', ls='--', lw=0.4) for x in x_cell_center for y in y_cell_center for i in range(4)]
[ax[i].plot(x+x_module, y+y_module, color='gray', ls='--', lw=0.4) for x in x_cell_center for y in y_cell_center for i in range(4)]
[ax[i].plot(x-dist_module+x_module, y-dist_module+y_module, color='gray', ls='--', lw=0.4) for x in x_cell_center for y in y_cell_center for i in range(4) ]


cbaxs_a = ax[3].inset_axes([1.05, 0, 0.04,1])
cbar_a = plt.colorbar(pca, cax=cbaxs_a, extend="max")
cbar_a.ax.set_yticks([0, vmax/2, vmax])
cbar_a.ax.set_yticklabels([])
cbar_a.ax.set_title(r"   $|H_z|^2$", fontsize=7)
cbaxs_b = ax[3].inset_axes([1.1, 0, 0.04,1])
cbar_b = plt.colorbar(pcb, cax=cbaxs_b, extend="max")
cbar_b.ax.set_yticks([0, vmax/2, vmax])
cbar_b.ax.set_yticklabels([0, r"$\frac{\mathrm{max}}{40}$", r"$\frac{\mathrm{max}}{20}$"])

ax[0].set(xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)")
ax[1].set(xlabel=r"$x$ ($\mu$m)")
ax[2].set(xlabel=r"$x$ ($\mu$m)")
ax[3].set(xlabel=r"$x$ ($\mu$m)")

ax[0].set_title(r"$A_1$", fontsize=7)
ax[1].set_title(r"$A_2$", fontsize=7)
ax[2].set_title(r"$B_1$", fontsize=7)
ax[3].set_title(r"$B_2$", fontsize=7)

# fig.savefig('Figures/Fig6/fig6a_routing.pdf')


#%% for Cleo plot

fig, ax = plt.subplots(figsize=(3.3,3.1), tight_layout=True)
vmax = np.abs(Hz[0]).max()**2 /20
pca = ax.pcolormesh(x, y, np.abs(Hz[0])**2, vmin=0, vmax=vmax, cmap=plt.cm.inferno, rasterized=True)
[ax.plot(x+dist_module+x_module, y+y_module, color=plt.cm.PRGn(0.2), ls='--', lw=0.8) for x in x_cell_center for y in y_cell_center]
[ax.plot(x+x_module, y+dist_module+y_module, color=plt.cm.PRGn(0.8), ls='--', lw=0.8) for x in x_cell_center for y in y_cell_center]
[ax.plot(x-dist_module+x_module, y+y_module, color='gray', ls='--', lw=0.8) for x in x_cell_center for y in y_cell_center]
[ax.plot(x+x_module, y-dist_module+y_module, color='gray', ls='--', lw=0.8) for x in x_cell_center for y in y_cell_center]
[ax.plot(x+x_module, y+y_module, color='gray', ls='--', lw=0.8) for x in x_cell_center for y in y_cell_center]
[ax.plot(x-dist_module+x_module, y-dist_module+y_module, color='gray', ls='--', lw=0.8) for x in x_cell_center for y in y_cell_center]

ax.set(
    xticks=(-20,0, 20), yticks=(-20,0, 20),
    xlim=(-lattice_const*N_cell[0]/2,lattice_const*N_cell[0]/2), 
    ylim=(-lattice_const*N_cell[1]/2,lattice_const*N_cell[1]/2),
    xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)"
)

# cbaxs_a = ax.inset_axes([0.2, 0.97, 0.6,0.03])
cbaxs_a = ax.inset_axes([1.01, 0.0, 0.03,1])
cbar_a = plt.colorbar(pca, cax=cbaxs_a, extend="max")
cbar_a.ax.set_yticks([0, vmax/2, vmax])
cbar_a.ax.set_yticklabels([0, r"$\frac{\mathrm{max}}{40}$", r"$\frac{\mathrm{max}}{20}$"], fontsize=8)
cbar_a.ax.set_title(r"$|H_z|^2$", fontsize=7)
fig.savefig('Figures/Fig6/fig_cleo1c.pdf')

#%% postprocessing - time domain signals & linear superposition

with open("Data_Fig6/det_field.pickle", 'rb') as handle:
    det_field_time = pickle.load(handle)
    t = det_field_time["t"]
    dt = t[1]-t[0]

with open("Data_Fig6/thru_flux.pickle", 'rb') as handle:
    thru_flux = pickle.load(handle)



t = thru_flux["t"]
dt = t[1]-t[0]
delay_idx = int(t_delay/dt)
t_offset = t[np.argmax(env(t, thru_flux["A11"][1]))]

def delayed(signal, N_delay:int=0, N_pulse:int=2):
    if N_pulse==1:
        return signal
    else:
        if len(signal.shape)==1:
            return np.concat([
                np.zeros(delay_idx*N_delay), 
                signal, 
                np.zeros(delay_idx*(N_pulse-1-N_delay))
            ])
        else:
            return np.concat([
                np.zeros([delay_idx*N_delay, *signal.shape[1:]]),
                signal,
                np.zeros([delay_idx*(N_pulse-1-N_delay), *signal.shape[1:]]), 
            ], axis=0)


def time_signals(signal_A, signal_B):
    N_pulse = len(signal_A)
    t_ext = dt * np.arange(len(t)+delay_idx*(N_pulse-1))

    input_signal_A = [signal_A[pulse,nx]**2 * delayed(env(t, thru_flux[f"A{nx}1"][nx]), N_delay=pulse, N_pulse=N_pulse) for pulse in range(N_pulse) for nx in range(N_cell[0])]
    input_signal_A = np.array(input_signal_A).reshape(N_pulse, N_cell[0], -1).sum(axis=0)
    input_signal_B = [signal_B[pulse,ny]**2 * delayed(env(t, thru_flux[f"B1{ny}"][N_cell[0]+ny]), N_delay=pulse, N_pulse=N_pulse) for pulse in range(N_pulse) for ny in range(N_cell[1])]
    input_signal_B = np.array(input_signal_B).reshape(N_pulse, N_cell[1], -1).sum(axis=0)

    output_signal_C = []
    output_signal_D = []
    output_signal_raw = []

    for nx in range(N_cell[0]):
        for ny in range(N_cell[1]):
            field =  det_field_time[f"C{nx}{ny}"]
            field = [delayed(np.sum(field * np.concat([signal_A[pulse,:], signal_B[pulse,:]]).reshape(-1,1,1,1), axis=0), N_delay=pulse, N_pulse=N_pulse) for pulse in range(N_pulse)]
            field = np.sum(field, axis=0)
            Ex, Ey, Hx, Hy = field[:,0],field[:,1],field[:,2],field[:,3]
            flux_C = env(t_ext, np.average(Ex*Hy - Ey*Hx, axis=1)* l_des**2)
            output_signal_C.append(flux_C)

            field =  det_field_time[f"D{nx}{ny}"]
            field = [delayed(np.sum(field * np.concat([signal_A[pulse,:], signal_B[pulse,:]]).reshape(-1,1,1,1), axis=0), N_delay=pulse, N_pulse=N_pulse) for pulse in range(N_pulse)]
            field = np.sum(field, axis=0)
            Ex, Ey, Hx, Hy = field[:,0],field[:,1],field[:,2],field[:,3]
            flux_D = env(t_ext, np.average(Ex*Hy - Ey*Hx, axis=1)* l_des**2)
            output_signal_D.append(flux_D)
            output_signal_raw.append(flux_C-flux_D)

    output_signal_C = np.array(output_signal_C).reshape(2,2, -1)
    output_signal_D = np.array(output_signal_D).reshape(2,2, -1)
    output_signal_raw = np.array(output_signal_raw).reshape(2,2, -1)
    return t_ext, (input_signal_A, input_signal_B), (output_signal_C, output_signal_D, output_signal_raw)


signal_A = np.random.uniform(low=-1, high=1, size=(2,2))
signal_B = np.random.uniform(low=-1, high=1, size=(2,2))


## Example
# signal_A = np.array([
#     [1, 0],
#     [-0.5, 0.5],
#     [0, -1]
# ])
# signal_B = np.array([
#     [0,1],
#     [0.5, -0.5],
#     [1, 0]
# ])

# t_ext, (input_signal_A, input_signal_B), (output_signal_C, output_signal_D, output_signal_raw) = time_signals(signal_A, signal_B)
# fig, ax = plt.subplots(4,3, sharex=True, tight_layout=True)
# ax[0,0].plot(t_ext-t_offset, input_signal_A[0])
# ax[1,0].plot(t_ext-t_offset, input_signal_A[1])
# ax[2,0].plot(t_ext-t_offset, input_signal_B[0])
# ax[3,0].plot(t_ext-t_offset, input_signal_B[1])
# [ax[i,0].set(xlim=(-t_offset,2e-12), ylim=(0,2)) for i in range(4)]

# ax[0,1].plot(t_ext-t_offset, output_signal_C[0,0])
# ax[1,1].plot(t_ext-t_offset, output_signal_C[0,1])
# ax[2,1].plot(t_ext-t_offset, output_signal_C[1,0])
# ax[3,1].plot(t_ext-t_offset, output_signal_C[1,1])

# ax[0,1].plot(t_ext-t_offset, output_signal_D[0,0], '--')
# ax[1,1].plot(t_ext-t_offset, output_signal_D[0,1], '--')
# ax[2,1].plot(t_ext-t_offset, output_signal_D[1,0], '--')
# ax[3,1].plot(t_ext-t_offset, output_signal_D[1,1], '--')

# [ax[i,1].set(xlim=(-t_offset,2e-12), ylim=(0,2)) for i in range(4)]


# ax[0,2].plot(t_ext-t_offset, output_signal_raw[0,0])
# ax[1,2].plot(t_ext-t_offset, output_signal_raw[0,1])
# ax[2,2].plot(t_ext-t_offset, output_signal_raw[1,0])
# ax[3,2].plot(t_ext-t_offset, output_signal_raw[1,1])
# [ax[i,2].set(xlim=(-t_offset,2e-12), ylim=(-1,1)) for i in range(4)]


## Computing normalization value for each cell
normalizer = []
for nx in range(N_cell[0]):
    for ny in range(N_cell[1]):

        sig_A = 0.0 + (np.arange(N_cell[0])==nx).reshape(1,-1)
        sig_B = 0.0 + (np.arange(N_cell[1])==ny).reshape(1,-1)

        _, _, (_, _, output_raw) = time_signals(sig_A, sig_B)
        normalizer.append(output_raw[nx,ny])

normalizer = np.array(normalizer).reshape(*N_cell, -1).sum(axis=-1)

fig, ax = plt.subplots()
ax.matshow(normalizer)



#%% pulse train plot

N_pulse = 10
np.random.seed(1)
sig_A, sig_B = np.random.uniform(low=-1, high=1, size=(2,N_pulse,2))

t_ext, inputs, outputs = time_signals(sig_A, sig_B)

color_a = (46/256,49/256,146/256)
color_b = (247/256,147/256,30/256)
color_c = plt.cm.PRGn(0.9)
color_d = plt.cm.PRGn(0.1)

fig, ax = plt.subplots(4, 2, sharex=True, tight_layout=True, figsize=(4.3,2.3))

ax[0,0].plot((t_ext-t_offset)/1e-12, inputs[0][0], color=color_a, lw=0.75)
ax[1,0].plot((t_ext-t_offset)/1e-12, inputs[0][1], color=color_a, lw=0.75)
ax[2,0].plot((t_ext-t_offset)/1e-12, inputs[1][0], color=color_b, lw=0.75)
ax[3,0].plot((t_ext-t_offset)/1e-12, inputs[1][1], color=color_b, lw=0.75)
ax[0,0].fill_between((t_ext-t_offset)/1e-12, inputs[0][0], color=color_a, lw=0.01, alpha=0.2)
ax[1,0].fill_between((t_ext-t_offset)/1e-12, inputs[0][1], color=color_a, lw=0.01, alpha=0.2)
ax[2,0].fill_between((t_ext-t_offset)/1e-12, inputs[1][0], color=color_b, lw=0.01, alpha=0.2)
ax[3,0].fill_between((t_ext-t_offset)/1e-12, inputs[1][1], color=color_b, lw=0.01, alpha=0.2)

ax[0,1].plot((t_ext-t_offset)/1e-12, outputs[0][0,0], color=color_c, lw=0.75)
ax[1,1].plot((t_ext-t_offset)/1e-12, outputs[0][0,1], color=color_c, lw=0.75)
ax[2,1].plot((t_ext-t_offset)/1e-12, outputs[0][1,0], color=color_c, lw=0.75)
ax[3,1].plot((t_ext-t_offset)/1e-12, outputs[0][1,1], color=color_c, lw=0.75)
ax[0,1].fill_between((t_ext-t_offset)/1e-12, outputs[0][0,0], color=color_c, lw=0.01, alpha=0.2)
ax[1,1].fill_between((t_ext-t_offset)/1e-12, outputs[0][0,1], color=color_c, lw=0.01, alpha=0.2)
ax[2,1].fill_between((t_ext-t_offset)/1e-12, outputs[0][1,0], color=color_c, lw=0.01, alpha=0.2)
ax[3,1].fill_between((t_ext-t_offset)/1e-12, outputs[0][1,1], color=color_c, lw=0.01, alpha=0.2)

ax[0,1].plot((t_ext-t_offset)/1e-12, outputs[1][0,0], ls='--', color=color_d, lw=0.75)
ax[1,1].plot((t_ext-t_offset)/1e-12, outputs[1][0,1], ls='--', color=color_d, lw=0.75)
ax[2,1].plot((t_ext-t_offset)/1e-12, outputs[1][1,0], ls='--', color=color_d, lw=0.75)
ax[3,1].plot((t_ext-t_offset)/1e-12, outputs[1][1,1], ls='--', color=color_d, lw=0.75)
ax[0,1].fill_between((t_ext-t_offset)/1e-12, outputs[1][0,0], ls='--', color=color_d, lw=0.01, alpha=0.2)
ax[1,1].fill_between((t_ext-t_offset)/1e-12, outputs[1][0,1], ls='--', color=color_d, lw=0.01, alpha=0.2)
ax[2,1].fill_between((t_ext-t_offset)/1e-12, outputs[1][1,0], ls='--', color=color_d, lw=0.01, alpha=0.2)
ax[3,1].fill_between((t_ext-t_offset)/1e-12, outputs[1][1,1], ls='--', color=color_d, lw=0.01, alpha=0.2)

ax[3,0].set(xlim=(-0.5,4.5), xlabel=r"$t$ (ps)")
ax[3,1].set(xlim=(-0.5,4.5), xlabel=r"$t$ (ps)")

CD_max = max(outputs[0].max(), outputs[1].max())
[ax[i,0].set(ylim=(0,2)) for i in range(4)]
[ax[i,1].set(ylim=(0,CD_max*1.1)) for i in range(4)]

ax[0,0].set(ylabel=r"$P^A_1(t)$")
ax[1,0].set(ylabel=r"$P^A_2(t)$")
ax[2,0].set(ylabel=r"$P^B_1(t)$")
ax[3,0].set(ylabel=r"$P^B_2(t)$")

ax[0,1].set(ylabel=r"$P^{C,D}_{1,1}(t)$")
ax[1,1].set(ylabel=r"$P^{C,D}_{1,2}(t)$")
ax[2,1].set(ylabel=r"$P^{C,D}_{2,1}(t)$")
ax[3,1].set(ylabel=r"$P^{C,D}_{2,2}(t)$")


result_truth = sig_A.T@sig_B
result_output = outputs[2].sum(axis=2)/normalizer


for nx in range(2):
    for ny in range(2):
        str_output_c = f"{round(outputs[0][nx,ny].sum()/normalizer[nx,ny],3)}"
        ax[nx*2+ny,1].text(0, 0.5, str_output_c, ha="left", fontsize=6, color=color_c)
        str_output_d = f"- {round(outputs[1][nx,ny].sum()/normalizer[nx,ny],3)}"
        ax[nx*2+ny,1].text(0.7, 0.5, str_output_d, ha="left", fontsize=6, color=color_d)
        str_output_out = f"= {round(outputs[2][nx,ny].sum()/normalizer[nx,ny],3)}"
        ax[nx*2+ny,1].text(1.55, 0.5, str_output_out, ha="left", fontsize=6, color="k")


[ax[i,0].axvline(x=pulse*t_delay/1e-12, color='gray', lw=0.5, ls='--') for pulse in range(2) for i in range(4)]

ax[0,0].set_title("Input pulse trains", fontsize=7)
ax[0,1].set_title("Output pulse trains", fontsize=7)

[ax[i,j].spines['top'].set_visible(False) for i in range(4) for j in range(2)]
[ax[i,j].spines['right'].set_visible(False) for i in range(4) for j in range(2)]

fig.savefig('Figures/Fig6/fig6b_train.pdf')


fig, ax = plt.subplots(2,1, sharex=True, sharey=True, tight_layout=True, figsize=(1.3, 2.3))
ax[0].matshow(result_truth[::-1].T, vmin=-1.5,vmax=1.5, cmap=plt.cm.PRGn)
ms = ax[1].matshow(result_output[::-1].T, vmin=-1.5,vmax=1.5, cmap=plt.cm.PRGn)

[ax[0].text(n,m, round(result_truth[::-1].T[m,n],3), va="top", ha="center") for m in range(2) for n in range(2)]
[ax[1].text(n,m, round(result_output[::-1].T[m,n],3), va="top", ha="center") for m in range(2) for n in range(2)]

# ax[0].xaxis.set_ticks_position('bottom')
ax[1].xaxis.set_ticks_position('bottom')

ax[1].set(xticks=(0,1), xticklabels=[r"$A_2$", r"$A_1$"])
ax[1].set(yticks=(1,0), yticklabels=[r"$B_2$", r"$B_1$"])
ax[0].set(yticks=(1,0), yticklabels=[r"$B_2$", r"$B_1$"])

cax0 = ax[0].inset_axes([1.05, 0.2, 0.05, 0.6])
cax1 = ax[1].inset_axes([1.05, 0.2, 0.05, 0.6])
cbar = plt.colorbar(ms, cax=cax0)
cbar = plt.colorbar(ms, cax=cax1)

ax[0].set_title("Ground truth", fontsize=7)
ax[1].set_title("Device output", fontsize=7)
fig.savefig('Figures/Fig6/fig6c_matrix.pdf')



#%%  pulse train 3d

fig3d = plt.figure(figsize=(4.3, 2.3))
ax3d = fig3d.add_subplot(projection="3d")

ax3d.plot((t_ext-t_offset)/1e-12, 0, inputs[0][0], color=color_a, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 1, inputs[0][1], color=color_a, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 2, inputs[1][0], color=color_b, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 3, inputs[1][1], color=color_b, lw=0.5)
ax3d.fill_between((t_ext-t_offset)/1e-12, 0, inputs[0][0], (t_ext-t_offset)/1e-12, 0, 0, color=color_a, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 1, inputs[0][1], (t_ext-t_offset)/1e-12, 1, 0, color=color_a, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 2, inputs[1][0], (t_ext-t_offset)/1e-12, 2, 0, color=color_b, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 3, inputs[1][1], (t_ext-t_offset)/1e-12, 3, 0, color=color_b, lw=0.01, alpha=0.2)

ax3d.plot((t_ext-t_offset)/1e-12, 4, outputs[0][0,0], color=color_c, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 5, outputs[0][0,1], color=color_c, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 6, outputs[0][1,0], color=color_c, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 7, outputs[0][1,1], color=color_c, lw=0.5)
ax3d.fill_between((t_ext-t_offset)/1e-12, 4, outputs[0][0,0], (t_ext-t_offset)/1e-12, 4, 0, color=color_c, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 5, outputs[0][0,1], (t_ext-t_offset)/1e-12, 5, 0, color=color_c, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 6, outputs[0][1,0], (t_ext-t_offset)/1e-12, 6, 0, color=color_c, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 7, outputs[0][1,1], (t_ext-t_offset)/1e-12, 7, 0, color=color_c, lw=0.01, alpha=0.2)

ax3d.plot((t_ext-t_offset)/1e-12, 4, outputs[1][0,0], ls='--', color=color_d, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 5, outputs[1][0,1], ls='--', color=color_d, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 6, outputs[1][1,0], ls='--', color=color_d, lw=0.5)
ax3d.plot((t_ext-t_offset)/1e-12, 7, outputs[1][1,1], ls='--', color=color_d, lw=0.5)
ax3d.fill_between((t_ext-t_offset)/1e-12, 4, outputs[1][0,0], (t_ext-t_offset)/1e-12, 4, 0, ls='--', color=color_d, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 5, outputs[1][0,1], (t_ext-t_offset)/1e-12, 5, 0, ls='--', color=color_d, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 6, outputs[1][1,0], (t_ext-t_offset)/1e-12, 6, 0, ls='--', color=color_d, lw=0.01, alpha=0.2)
ax3d.fill_between((t_ext-t_offset)/1e-12, 7, outputs[1][1,1], (t_ext-t_offset)/1e-12, 7, 0, ls='--', color=color_d, lw=0.01, alpha=0.2)

ax3d.set(
    xlabel=r"$t$ (ps)", yticks=np.arange(8), 
    # yticklabels=[r"$A_1$", r"$A_2$", r"$B_1$", r"$B_2$", r"$C,D_{11}$", r"$C,D_{12}$", r"$C,D_{21}$", r"$C,D_{22}$"]
    yticklabels=[],    
)
ax3d.set(xlim=(-0.5,4.5), ylim=(-0.5,7.5), zlim=(0,2.2))
ax3d.view_init(60,60)
ax3d.grid(False)
fig3d.savefig("Figures/Figpulsetrain_3d.pdf")

#%% Computation density

Density = round(1/(lattice_const*1e-3)**2/t_delay / 1e15, 3)
print(f"Computing density = {Density} PMACS/mm^2/s")

reduce_factor = (1e-11/t_delay)
Density = round(1/(lattice_const*1e-3)**2/t_delay / reduce_factor**1 / 1e12, 3)
print(f"Computing density = {Density} TMACS/mm^2/s")
