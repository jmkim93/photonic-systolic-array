
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
N_cell = (4,4) # Change as (1,4) first to calculate time delay
lattice_const = wavelength/n_eff_phase[50]*15 # Too short? might degrade grating coupler
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

## Spectral parameters
pulse_width = 200
fwidth = freq0/pulse_width
t_delay = lattice_const*n_eff_group[50]/td.C_0

# t03 = 5.266490570394706e-13
# t13 = 3.519235482877214e-13
# t23 = 1.763738626078979e-13
# t03 = 5.264430128074521e-13 ## new
# t13 = 3.515114598236842e-13
# t23 = 1.757557299118421e-13
# t03 = 6.314843622905203e-13 ## large lattice const
# t13 = 4.209895748603468e-13
# t23 = 2.1016511665894367e-13

t03 = 5.262369685754336e-13
t13 = 3.513054155916657e-13
t23 = 1.755496856798236e-13

t_delay_prac = [t03, t13, t23, 0]

run_time = (
    8.5/(2*np.pi*fwidth) + 
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
sim_base = td.Simulation(
    size=(Lx,Ly,Lz),
    structures=base_structure,
    grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
    boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
    run_time=run_time,
    subpixel=True,
    # medium=air,
    # symmetry=(0,0,1),
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

src_amp = [1, 0, 0, 0, 0, 0, 0, 0]

def make_sim(src_amp: tuple=(1,0,0,0,0,0,0,0)):
    amp_A = src_amp[:N_cell[0]]
    amp_B = src_amp[N_cell[0]:]
    
    src_A = [td.ModeSource(
        size=(w_mode, 0, wavelength + h_wg),
        center=(x-dist_module, y_cell_center[-1]-lattice_const/2-0.1*wavelength, z_offset),
        source_time = td.GaussianPulse(
            freq0=freq0, fwidth=fwidth,
            amplitude=1e-14+np.abs(amp_A[nx])*np.sqrt(N_cell[1]), 
            phase=0+np.pi*(amp_A[nx]<0),
            offset=4 + 2*np.pi*fwidth * t_delay_prac[nx]
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
            offset=4 + 2*np.pi*fwidth * t_delay_prac[ny]
        ),
        mode_spec=mode_spec,
        mode_index=0,
        direction="+",
        num_freqs=9,
    ) for ny, y in enumerate(y_cell_center)]
    sources = src_A+src_B
    return sim_base.updated_copy(sources=sources, monitors=monitors)


### change config to (4,1) first and run to measure the precise time delay 
# sim_time_measure = make_sim([1,0,0,0,0])
# sim_time_measure_data = td.web.run(
#     sim_time_measure, 
#     folder_name="Fig4_landscape",
#     task_name="time_measure", 
#     verbose=True
# )
# t0 = sim_time_measure_data["thruA_0_0_flux"].flux.t
# flux0 = sim_time_measure_data["thruA_0_0_flux"].flux.data
# t1 = sim_time_measure_data["thruA_0_1_flux"].flux.t
# flux1 = sim_time_measure_data["thruA_0_1_flux"].flux.data
# t2 = sim_time_measure_data["thruA_0_2_flux"].flux.t
# flux2 = sim_time_measure_data["thruA_0_2_flux"].flux.data
# t3 = sim_time_measure_data["thruA_0_3_flux"].flux.t
# flux3 = sim_time_measure_data["thruA_0_3_flux"].flux.data
# t03 = (t0[np.argmax(env(t0,flux0))]-t3[np.argmax(env(t3,flux3))]).data.item()
# t13 = (t1[np.argmax(env(t1,flux1))]-t3[np.argmax(env(t3,flux3))]).data.item()
# t23 = (t2[np.argmax(env(t2,flux2))]-t3[np.argmax(env(t3,flux3))]).data.item()


## Real simulation 
sims = {
    "A0": make_sim([1,0,0,0,0,0,0,0]),
    # 'A1': make_sim([0,1,0,0,0,0,0,0]), 
    # "A2": make_sim([0,0,1,0,0,0,0,0]),
    # "A3": make_sim([0,0,0,1,0,0,0,0]),
    # "B0": make_sim([0,0,0,0,1,0,0,0]),
    # "B1": make_sim([0,0,0,0,0,1,0,0]),
    # 'B2': make_sim([0,0,0,0,0,0,1,0]),
    # "B3": make_sim([0,0,0,0,0,0,0,1]),
}
# batch = web.Batch(simulations=sims, folder_name="Fig4_landscape")
# batch_data = batch.run()

#%% SuppFig4: simulation schematic

sim = sims["A0"]
fig, ax = plt.subplots(2,1, figsize=(6,9), tight_layout=True)
sim.plot_eps(freq=freq0, z=z_offset+0.001, ax=ax[0])
sim.plot_eps(freq=freq0, x=x_cell_center[-1]-dist_module, ax=ax[1])

ax[0].set(
    xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)",
)
ax[1].set(
    xlabel=r"$y$ ($\mu$m)", ylabel=r"$z$ ($\mu$m)",
    yticks=np.linspace(-2,2,3)+z_offset,
    yticklabels=np.linspace(-2,2,3),
    xlim=(-Lx/2, -Lx/2*0)
)

ax[0].set_title("Cross section, $z=0$", fontsize=7)
ax[1].set_title("Cross section, $x={}$".format(round(x_cell_center[-1]-dist_module, 2)), fontsize=7)
ax[1].text(-Lx/3, 1.5, "air", horizontalalignment='center', verticalalignment='center')
ax[1].text(-Lx/3, -1.5, "SiO$_2$", horizontalalignment='center', verticalalignment='center')
ax[1].text(-Lx/3, 0, "Si", horizontalalignment='center', verticalalignment='center')
ax[1].text(-7, 1, "Detection plane monitor", color="orange", horizontalalignment='center', verticalalignment='center')
ax[1].text(-7, z_offset-0.5, "Waveguide plane monitor", color="orange", horizontalalignment='center', verticalalignment='center')
fig.savefig("Figures/Fig4/SuppFig4_simschematic.png", transparent=True, dpi=300)

#%% Data Load and organize, save

batch_data = {
    'A0': web.load(task_id='fdve-00b5aa45-b48c-40ff-8549-9f1ce40f878e'), 
    'A1': web.load(task_id='fdve-42178129-9c42-433d-ae42-9df27113bffa'), 
    'A2': web.load(task_id='fdve-4616ecde-a471-447e-8e64-4ec787107352'), 
    'A3': web.load(task_id='fdve-a79ae83f-eb55-4b17-9260-f297715eb08e'), 
    'B0': web.load(task_id='fdve-1ec11207-950f-4852-9a67-792327fbd73c'), 
    'B1': web.load(task_id='fdve-8104ed61-ad5a-41be-90d9-3f9a5a0dca8f'), 
    'B2': web.load(task_id='fdve-815cd7f6-3cd5-4c2c-a2d4-d44c14245e14'), 
    'B3': web.load(task_id='fdve-decd76be-02a6-4cbe-a81e-2a92be1e07b9'),
}

src_list = ["A0","A1","A2","A3","B0","B1","B2","B3"]

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
    for direction in "AB" for nx in range(4) for ny in range(4)
}
thru_flux_time["t"] = batch_data["A0"]["thruA_0_0_flux"].flux.t.data

det_flux_time = {
    "{}{}{}".format(direction, nx, ny): np.stack([batch_data[src]["det{}_{}_{}_flux".format(direction, nx, ny)].flux.data for src in src_list])
    for direction in "CD" for nx in range(4) for ny in range(4)
}
det_flux_time["t"] = batch_data["A0"]["detC_0_0_flux"].flux.t.data

det_field_time = {
    "{}{}{}".format(det, nx, ny): np.stack([
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Ex.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2),
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Ey.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2),
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Hx.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2),
        np.stack([np.stack([batch_data[src]["det{}_{}_{}_{}_{}_field".format(det, nx, ny, ndx,ndy)].Hy.data[0,0,0,:] for src in src_list]) for ndx in range(5) for ndy in range(5)], axis=2)
    ], axis=2)
    for det in "CD" for nx in range(4) for ny in range(4)
} # (Num src, time steps, num_field-Ex,Ey,Hx,Hy, num det points)
det_field_time["t"] = batch_data["A0"]["detC_0_0_0_0_field"].Ex.t.data



# with open("Data_Fig4/field_freq.pickle", 'wb') as handle:
#     pickle.dump(Field_freq, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open("Data_Fig4/thru_flux.pickle", 'wb') as handle:
#     pickle.dump(thru_flux_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open("Data_Fig4/det_flux.pickle", 'wb') as handle:
#     pickle.dump(det_flux_time, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open("Data_Fig4/det_field.pickle", 'wb') as handle:
#     pickle.dump(det_field_time, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% Fig 4 ABC
from defineAdjointOptimization import w_wg, h_wg, l_des, w_mode
from defineAdjointOptimization import min_step_per_wvl, pixel_size

color_a = (46/256,49/256,146/256)
color_b = (247/256,147/256,30/256)


### Geometry
wavelength = 1.55
k0 = 2*np.pi/wavelength
freq0 = td.C_0/wavelength

h_box = 2.0
h_clad = 0.78
h_sub = 500

### Lattice definition
N_cell = (4,4)
lattice_const = wavelength/n_eff_phase[50]*15
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

## Spectral parameters
pulse_width = 200
fwidth = freq0/pulse_width
t_delay = lattice_const*n_eff_group[50]/td.C_0

# signal = np.random.uniform(low=0.2, high=0.95, size=8)
# signal = signal * (-1+2*(np.arange(8)%2==0))
signal = np.array([ 0.42265912, -0.50070461,  0.65285966, -0.49873527,  0.7648223 ,
       -0.26871357,  0.64854911, -0.72943543])

signal_orthogonal = np.array([0, 1, 0, 0, 0, 0, 1, 0])


with open("Data_Fig4/field_freq.pickle", 'rb') as handle:
    field_freq = pickle.load(handle)

y, x = field_freq["y"], field_freq["x"]
Hz = np.sum(signal.reshape(-1,1,1) * field_freq["Hz_center"], axis=0)

Ex = np.sum(signal.reshape(-1,1,1) * field_freq["Ex_air"], axis=0)
Ey = np.sum(signal.reshape(-1,1,1) * field_freq["Ey_air"], axis=0)
Hx = np.sum(signal.reshape(-1,1,1) * field_freq["Hx_air"], axis=0)
Hy = np.sum(signal.reshape(-1,1,1) * field_freq["Hy_air"], axis=0)
Sz = np.conj(Ex)*Hy - np.conj(Ey)*Hx

Sz1 = np.conj(Ex)*Hy 
Sz2 = - np.conj(Ey)*Hx

fig, ax = plt.subplots(1,2, figsize=(5.3,2.6), tight_layout=True, sharey=True)
vmax = np.max(np.abs(Hz))
pc0 = ax[0].pcolormesh(x, y, np.real(Hz)/vmax, vmin=-1, vmax=1, cmap=plt.cm.bwr, rasterized=True)
ax[0].set(ylabel=r'$y$ ($\mu$m)', xlabel=r'$x$ ($\mu$m)', xlim=(-2*lattice_const,2*lattice_const), ylim=(-2*lattice_const,2*lattice_const), xticks=np.linspace(-20,20,5), yticks=np.linspace(-20,20,5))
[ax[0].axvline(x=x-lattice_const/2, color='gray', ls='--', lw=0.5) for x in x_cell_center]
[ax[0].axhline(y=y-lattice_const/2, color='gray', ls='--', lw=0.5) for y in y_cell_center]

cbax = ax[0].inset_axes([1.02, 0, 0.03, 1])
cb = plt.colorbar(pc0, cax=cbax)
cb.ax.set_yticks([-1, 0, 1])
cb.ax.set_ylabel(r"$\mathrm{Re}(H_z)$ (Arb. U.)")
ax[0].set_title(r"Waveguide plane, $z=0$", fontsize=7)


vmax = np.max(np.abs(Sz))
pc1 = ax[1].pcolormesh(x, y, np.abs(Sz)/vmax, vmin=0, vmax=1, cmap=plt.cm.magma, rasterized=True)
ax[1].set(
    xlabel=r'$x$ ($\mu$m)',  
    xlim=(-2*lattice_const,2*lattice_const), 
    ylim=(-2*lattice_const,2*lattice_const), 
    xticks=np.linspace(-20,20,5), yticks=np.linspace(-20,20,5)
)
x_module = l_des/2*np.array([1,1,-1,-1,1])
y_module = l_des/2*np.array([1,-1,-1,1,1])
[ax[1].plot(x+dist_module+x_module, y+y_module, color=plt.cm.PRGn(0.35), ls='--', lw=0.5) for x in x_cell_center for y in y_cell_center]
[ax[1].plot(x+x_module, y+dist_module+y_module, color=plt.cm.PRGn(0.65), ls='--', lw=0.5) for x in x_cell_center for y in y_cell_center]

color_module = "k"
ax[0].plot(x_cell_center[2]+x_module+dist_module, y_cell_center[2]+y_module, ls='-', lw=0.5, color='crimson')
ax[0].plot(x_cell_center[2]+x_module, y_cell_center[2]+y_module+dist_module, ls='-', lw=0.5, color='crimson')
ax[0].plot(x_cell_center[2]+x_module-dist_module, y_cell_center[2]+y_module, ls='-', lw=0.5, color=color_module)
ax[0].plot(x_cell_center[2]+x_module, y_cell_center[2]+y_module-dist_module, ls='-', lw=0.5, color=color_module)
ax[0].plot(x_cell_center[2]+x_module, y_cell_center[2]+y_module, ls='-', lw=0.5, color=color_module)
ax[0].plot(x_cell_center[2]+x_module-dist_module, y_cell_center[2]+y_module-dist_module, ls='-', lw=0.5, color=color_module)

cbax = ax[1].inset_axes([1.02, 0, 0.03, 1])
cb = plt.colorbar(pc1, cax=cbax)
cb.ax.set_yticks([0, 0.5,  1])
cb.ax.set_ylabel(r"$\mathrm{Re}(S_z)$ (Arb. U.)")
ax[1].set_title(r"Detection plane, $z = 0.968~\mu\mathrm{m}$", fontsize=7)
fig.savefig('Figures/Fig4/fig4a.pdf')

#%%

## nomalization flux calculation: one hot x one hot
with open("Data_Fig4/det_field.pickle", 'rb') as handle:
    det_field = pickle.load(handle)
    t = det_field["t"]
    dt = t[1]-t[0]

with open("Data_Fig4/thru_flux.pickle", 'rb') as handle:
    thru_flux = pickle.load(handle)

power_CD_diff = []
power_CD_sum = []
power_AB_sum = []
for nx in range(4):
    for ny in range(4):
        signal_two_hot = np.hstack([np.arange(4)==nx, np.arange(4)==ny])+0.0
        fields = np.sum(signal_two_hot.reshape(-1,1,1,1)*det_field["C{}{}".format(nx,ny)], axis=0)
        Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
        flux_C = np.average(Ex*Hy - Ey*Hx, axis=1)* l_des**2

        fields = np.sum(signal_two_hot.reshape(-1,1,1,1)*det_field["D{}{}".format(nx,ny)], axis=0)
        Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
        flux_D = np.average(Ex*Hy - Ey*Hx, axis=1)* l_des**2
        plt.plot(t, env(t, flux_C-flux_D))

        power_CD_diff.append(np.sum( (flux_C-flux_D)*dt))
        power_CD_sum.append(np.sum( (flux_C+flux_D)*dt))

        input_power = np.sum(thru_flux["A{}3".format(nx)][nx]*dt) + np.sum(thru_flux["B3{}".format(ny)][ny+4]*dt)
        power_AB_sum.append(input_power)
        
power_CD_diff = np.array(power_CD_diff).reshape(4,4)
power_CD_sum = np.array(power_CD_sum).reshape(4,4)
power_AB_sum = np.array(power_AB_sum).reshape(4,4)

power_C = (power_CD_sum + power_CD_diff)/2
power_D = (power_CD_sum - power_CD_diff)/2

##### Two-hot input powers #####
fig, ax = plt.subplots(1,2, figsize=(5, 2.5), sharex=True, sharey=True, tight_layout=True)
vmax = power_CD_diff.max()
ax[0].matshow(power_CD_sum[::-1].T/vmax,cmap=plt.cm.inferno, vmin=0.9, vmax=1)
ms = ax[1].matshow(power_CD_diff[::-1].T/vmax,cmap=plt.cm.inferno, vmin=0.9, vmax=1)
ax[0].set(xticks=np.arange(4),xticklabels=np.arange(4,0,-1))
ax[0].set(yticks=np.arange(4),yticklabels=np.arange(1,5))
ax[0].set(xlabel=r"$m$", ylabel=r"$n$")
ax[0].set_title(r"$P_C + P_D$", fontsize=7)
ax[0].xaxis.set_ticks_position('bottom')
ax[1].set(xticks=np.arange(4),xticklabels=np.arange(4,0,-1))
ax[1].set(yticks=np.arange(4),yticklabels=np.arange(1,5))
ax[1].set(xlabel=r"$m$")
ax[1].set_title(r"$P_C - P_D$", fontsize=7)
ax[1].xaxis.set_ticks_position('bottom')
for nx in range(4):
    for ny in range(4):
        ax[0].text(nx, ny, round(power_CD_sum[3-nx,ny]/vmax,3), fontsize=7, color="k" if np.abs(power_CD_sum[3-nx,ny]/vmax)>0.95 else 'w', horizontalalignment="center", verticalalignment="center")
        ax[1].text(nx, ny, round(power_CD_diff[3-nx,ny]/vmax,3), fontsize=7, color="k" if np.abs(power_CD_diff[3-nx,ny]/vmax)>0.95 else 'w', horizontalalignment="center", verticalalignment="center")

cbax = ax[1].inset_axes([1.05,0,0.05,1])
cbar = plt.colorbar(ms, cax=cbax)
cbar.ax.set_yticks([0.9, 1])
cbar.ax.set_yticklabels([r"$0.9 P_\mathrm{max}$", r"$P_\mathrm{max}$"])
fig.savefig("Figures/Fig4/SuppFig4_twohotpowers.pdf")


##### Input & output pulse result #####
fig, ax = plt.subplots(4,1, figsize=(1.8,2.6), sharex=True, tight_layout=True)
t = thru_flux["t"]
t0 = t[np.argmax(thru_flux["A33"][3])]

[ax[0].plot((t-t0)/1e-12, env(t, signal[nx]**2 * thru_flux["A{}3".format(nx)][nx]), color=color_a, lw=0.5) for nx in range(4)]
[ax[1].plot((t-t0)/1e-12, env(t, signal[ny+4]**2 * thru_flux["B3{}".format(ny)][ny+4]) , color=color_b, lw=0.5) for ny in range(4)]
[ax[0].fill_between((t-t0)/1e-12, env(t, signal[nx]**2 * thru_flux["A{}3".format(nx)][nx]), color=color_a, lw=0, alpha=0.1) for nx in range(4)]
[ax[1].fill_between((t-t0)/1e-12, env(t, signal[ny+4]**2 * thru_flux["B3{}".format(ny)][ny+4]) , color=color_b, lw=0, alpha=0.1) for ny in range(4)]
det_result = []
det_result_orthogonal = []
t = det_field["t"]
for nx in range(4):
    for ny in range(4):
        fields = np.sum(signal.reshape(-1,1,1,1)*det_field["C{}{}".format(nx,ny)], axis=0)
        Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
        flux_C = np.average(Ex*Hy-Ey*Hx, axis=1)*l_des**2
        fields = np.sum(signal.reshape(-1,1,1,1)*det_field["D{}{}".format(nx,ny)], axis=0)
        Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
        flux_D = np.average(Ex*Hy-Ey*Hx, axis=1)*l_des**2
        det_result.append(np.sum( (flux_C-flux_D)*dt))
        ax[2].fill_between((t-t0)/1e-12, env(t,flux_C), color=plt.cm.PRGn(0.8), lw=0, alpha=0.1)
        ax[2].plot((t-t0)/1e-12, env(t,flux_C), color=plt.cm.PRGn(0.8), lw=0.5)
        ax[3].fill_between((t-t0)/1e-12, env(t,flux_D), color=plt.cm.PRGn(0.2), lw=0, alpha=0.1)
        ax[3].plot((t-t0)/1e-12, env(t,flux_D), color=plt.cm.PRGn(0.2), lw=0.5)

        if nx==ny:
            ax[2].axvline(x=(t[np.argmax(env(t,flux_C))]-t0)/1e-12, color='k', lw=0.5, ls='--', alpha=0.5)
        elif nx==ny+1:   
            ax[3].axvline(x=(t[np.argmax(env(t,flux_D))]-t0)/1e-12, color='k', lw=0.5, ls='--', alpha=0.5)

        fields = np.sum(signal_orthogonal.reshape(-1,1,1,1)*det_field["C{}{}".format(nx,ny)], axis=0)
        Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
        flux_C = np.average(Ex*Hy-Ey*Hx, axis=1)*l_des**2
        fields = np.sum(signal_orthogonal.reshape(-1,1,1,1)*det_field["D{}{}".format(nx,ny)], axis=0)
        Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
        flux_D = np.average(Ex*Hy-Ey*Hx, axis=1)*l_des**2
        det_result_orthogonal.append(np.sum( (flux_C-flux_D)*dt))

ax[0].set(ylim=(0,4.4*np.max(np.abs(signal[:4]))**2))
ax[0].set_title("Input & output pulses", fontsize=7)
ax[1].set(ylim=(0,4.4*np.max(np.abs(signal[4:]))**2), yticks=(0,1,2))
ax[2].set(ylabel=r"                   Power (Arb. U.)", xlim=(-0.3,1.6), ylim=(0,0.42), yticks=(0,0.2, 0.4))
ax[3].set(xlabel=r"$t$ (ps)",xlim=(-0.3,1.6), ylim=(0,0.42), yticks=(0,0.2, 0.4))
fig.savefig('Figures/Fig4/fig4b.pdf')

det_result = np.array(det_result).reshape(4,4)
norm_result = det_result/power_CD_diff
det_result_orthogonal = np.array(det_result_orthogonal).reshape(4,4)
norm_result_orthogonal = det_result_orthogonal/power_CD_diff
ground_result = signal[:4].reshape(-1,1) * signal[4:].reshape(1,-1)

print("mean abs. error before norm. = ", np.mean(np.abs(ground_result-det_result/np.abs(power_CD_diff).max())))
print("mean abs. error after norm. = ", np.mean(np.abs(ground_result-norm_result)))

##### Outer product result plot #####
fig, ax = plt.subplots(1,4, figsize=(7.0,2.2), tight_layout=True)
ax[0].matshow(ground_result[::-1].T, vmin=-1,vmax=1, cmap=plt.cm.PRGn)
ax[1].matshow(det_result[::-1].T/np.abs(power_CD_diff).max(), vmin=-1,vmax=1, cmap=plt.cm.PRGn)
ax[2].matshow(norm_result[::-1].T, vmin=-1,vmax=1, cmap=plt.cm.PRGn)
pc = ax[3].matshow(norm_result_orthogonal[::-1].T, vmin=-1,vmax=1, cmap=plt.cm.PRGn)
# pc2 = ax[3].matshow((power_CD_sum/power_AB_sum)[::-1].T, vmin=0,vmax=0.1, cmap=plt.cm.cividis)

ax[0].set(xticks=np.arange(4),xticklabels=[round(signal[3-nx],2) for nx in range(4)])
ax[0].set(yticks=np.arange(4),yticklabels=[round(signal[4+ny],2) for ny in range(4)])
ax[0].set(xlabel=r"Input, $A$", ylabel=r"Input, $B$")
ax[0].set_title("Ground truth", fontsize=7)
ax[0].xaxis.set_ticks_position('bottom')

ax[1].set(xticks=np.arange(4),xticklabels=[round(signal[3-nx],2) for nx in range(4)])
ax[1].set(yticks=np.arange(4),yticklabels=[round(signal[4+ny],2) for ny in range(4)])
ax[1].set(xlabel=r"Input, $A$")
ax[1].set_title(r"Output raw data ($P_\mathrm{max}$)", fontsize=7)
ax[1].xaxis.set_ticks_position('bottom')

ax[2].set(xticks=np.arange(4),xticklabels=[round(signal[3-nx],2) for nx in range(4)])
ax[2].set(yticks=np.arange(4),yticklabels=[round(signal[4+ny],2) for ny in range(4)])
ax[2].set(xlabel=r"Input, $A$")
ax[2].set_title("Normalized output", fontsize=7)
ax[2].xaxis.set_ticks_position('bottom')

ax[3].set(xticks=np.arange(4),xticklabels=[round(signal_orthogonal[3-nx],2) for nx in range(4)])
ax[3].set(yticks=np.arange(4),yticklabels=[round(signal_orthogonal[4+ny],2) for ny in range(4)])
ax[3].set(xlabel=r"Input, $A$")
ax[3].set_title("Norm. output (two-hot)", fontsize=7)
ax[3].xaxis.set_ticks_position('bottom')

cbax = ax[3].inset_axes([1.04, 0, 0.04, 1])
cb = plt.colorbar(pc, cax=cbax)
cb.ax.set_yticks([-1,0, 1])
cb.ax.set_ylabel(r"$A\times B$")

for nx in range(4):
    for ny in range(4):
        ax[0].text(nx, ny, round(ground_result[3-nx,ny],3), fontsize=7, color="w" if np.abs(ground_result[3-nx,ny])>0.5 else 'k', horizontalalignment="center", verticalalignment="center")
        ax[1].text(nx, ny, round(det_result[3-nx,ny]/np.abs(power_CD_diff).max(),3), fontsize=7, color="w" if np.abs(ground_result[3-nx,ny])>0.5 else 'k', horizontalalignment="center", verticalalignment="center")
        ax[2].text(nx, ny, round(norm_result[3-nx,ny],3), fontsize=7, color="w" if np.abs(norm_result[3-nx,ny])>0.5 else 'k', horizontalalignment="center", verticalalignment="center")
        ax[3].text(nx, ny, round(norm_result_orthogonal[3-nx,ny],3), fontsize=7, color="w" if np.abs(norm_result_orthogonal[3-nx,ny])>0.5 else 'k', horizontalalignment="center", verticalalignment="center")
        # ax[3].text(nx, ny, round((power_CD_sum/power_AB_sum*100)[3-nx,ny],2), fontsize=7, color="k", horizontalalignment="center", verticalalignment="center")

fig.savefig('Figures/Fig4/fig4c.pdf')


#%% Supplementary : error distribution
from tqdm import tqdm
N_stat = 10000
signal_set = np.random.uniform(low=-1, high=1, size=(N_stat,8))

raw_error = []
norm_error = []

#### Long time. just load precalculated data
# for signal in tqdm(signal_set):
#     det_result = []
#     t = det_field["t"]
#     for nx in range(4):
#         for ny in range(4):
#             fields = np.sum(signal.reshape(-1,1,1,1)*det_field["C{}{}".format(nx,ny)], axis=0)
#             Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
#             flux_C = np.average(Ex*Hy-Ey*Hx, axis=1)*l_des**2
#             fields = np.sum(signal.reshape(-1,1,1,1)*det_field["D{}{}".format(nx,ny)], axis=0)
#             Ex, Ey, Hx, Hy = fields[:,0], fields[:,1], fields[:,2], fields[:,3]
#             flux_D = np.average(Ex*Hy-Ey*Hx, axis=1)*l_des**2
#             det_result.append(np.sum( (flux_C-flux_D)*dt))
            
#     det_result = np.array(det_result).reshape(4,4)
#     raw_result, norm_result = det_result/np.abs(power_CD_diff).max(), det_result/power_CD_diff
#     ground_result = signal[:4].reshape(-1,1) * signal[4:].reshape(1,-1)

#     raw_error.append(raw_result-ground_result)
#     norm_error.append(norm_result-ground_result)

# raw_error = np.array(raw_error).reshape(N_stat, 4, 4)
# norm_error = np.array(norm_error).reshape(N_stat, 4, 4)

# np.savez_compressed(
#     'Data_Fig4/stat.npz', 
#     signal_set=signal_set, 
#     raw_error=raw_error,
#     norm_error=norm_error
# )

with np.load('Data_Fig4/stat.npz') as stat_data:
    signal_set = stat_data["signal_set"]
    raw_error = stat_data["raw_error"]
    norm_error = stat_data["norm_error"]

MAE_raw = np.mean(np.abs(raw_error), axis=0)
MAE_norm = np.mean(np.abs(norm_error), axis=0)

fig, ax = plt.subplots(1,2, figsize=(3.5,1.8), sharey=True,tight_layout=True)
ax[0].matshow(MAE_raw[::-1].T, vmin=0.01, vmax=0.025, cmap=plt.cm.viridis)
ms = ax[1].matshow(MAE_norm[::-1].T, vmin=0.01, vmax=0.025, cmap=plt.cm.viridis)

cbax = ax[1].inset_axes([1.05, 0, 0.05, 1])
cb = plt.colorbar(ms, cax=cbax)

ax[0].set(
    xlabel=r"$m$", ylabel=r"$n$",
    xticks=np.arange(4), xticklabels=4-np.arange(4), 
    yticks=np.arange(4), yticklabels=1+np.arange(4)
)
ax[1].set(
    xlabel=r"$m$",
    xticks=np.arange(4), xticklabels=4-np.arange(4), 
)
ax[0].xaxis.set_ticks_position('bottom')
ax[1].xaxis.set_ticks_position('bottom')

ax[0].set_title("MAE (raw output)", fontsize=7)
ax[1].set_title("MAE (norm. output)", fontsize=7)

for nx in range(4):
    for ny in range(4):
        ax[0].text(nx, ny, round(MAE_raw[3-nx,ny],3), fontsize=6, color="w" if np.abs(MAE_raw[3-nx,ny])>0.5 else 'k', horizontalalignment="center", verticalalignment="center")
        ax[1].text(nx, ny, round(MAE_norm[3-nx,ny],3), fontsize=6, color="w" if np.abs(MAE_norm[3-nx,ny])>0.5 else 'k', horizontalalignment="center", verticalalignment="center")

fig.savefig('Figures/Fig4/Supp_Fig4_errorstat.pdf')

print("Total MAE, raw =", np.mean(np.abs(raw_error)) )
print("Total MAE, norm =", np.mean(np.abs(norm_error)) )


fig, ax = plt.subplots(4,4, figsize=(6,4), sharex=True, sharey=True, tight_layout=True)
for ii in range(4):
    ax[ii,0].set(ylabel="PDF")
    for jj in range(4):
        ax[3,jj].set(xlabel="Error")
        ax[ii,jj].hist(raw_error[:, 3-jj, ii], bins=np.linspace(-0.12,0.12,61), alpha=0.4, density=True, label=r"Raw")
        ax[ii,jj].hist(norm_error[:, 3-jj, ii], bins=np.linspace(-0.12,0.12,61),alpha=0.6, density=True, label=r"Norm.")
        ax[ii,jj].set(xlim=(-0.11,0.11), ylim=(0,22))
        ax[ii,jj].set_title(r"$({},{})$".format(4-jj, ii+1), fontsize=7)
ax[0,3].legend(frameon=False, fontsize=6)
fig.savefig('Figures/Fig4/Supp_Fig4_errordistrib.pdf')





#%% CLEO version

#%% Fig 4 ABC
from defineAdjointOptimization import w_wg, h_wg, l_des, w_mode
from defineAdjointOptimization import min_step_per_wvl, pixel_size

color_a = (46/256,49/256,146/256)
color_b = (247/256,147/256,30/256)


### Geometry
wavelength = 1.55
k0 = 2*np.pi/wavelength
freq0 = td.C_0/wavelength

h_box = 2.0
h_clad = 0.78
h_sub = 500

### Lattice definition
N_cell = (4,4)
lattice_const = wavelength/n_eff_phase[50]*15
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

## Spectral parameters
pulse_width = 200
fwidth = freq0/pulse_width
t_delay = lattice_const*n_eff_group[50]/td.C_0

signal = np.random.uniform(low=0.2, high=0.95, size=8)
signal = signal * (-1+2*(np.arange(8)%2==0))
# signal = np.array([ 0.42265912, -0.50070461,  0.65285966, -0.49873527,  0.7648223 ,
#        -0.26871357,  0.64854911, -0.72943543])

signal_orthogonal = np.array([0, 1, 0, 0, 0, 0, 1, 0])


with open("Data_Fig4/field_freq.pickle", 'rb') as handle:
    field_freq = pickle.load(handle)

y, x = field_freq["y"], field_freq["x"]
Hz = np.sum(signal.reshape(-1,1,1) * field_freq["Hz_center"], axis=0)

Ex = np.sum(signal.reshape(-1,1,1) * field_freq["Ex_air"], axis=0)
Ey = np.sum(signal.reshape(-1,1,1) * field_freq["Ey_air"], axis=0)
Hx = np.sum(signal.reshape(-1,1,1) * field_freq["Hx_air"], axis=0)
Hy = np.sum(signal.reshape(-1,1,1) * field_freq["Hy_air"], axis=0)
Sz = np.conj(Ex)*Hy - np.conj(Ey)*Hx

Sz1 = np.conj(Ex)*Hy 
Sz2 = - np.conj(Ey)*Hx

fig, ax = plt.subplots(1,2, figsize=(5.6,2.6), tight_layout=True, sharey=True)
vmax = np.max(np.abs(Hz))
pc0 = ax[0].pcolormesh(x, y, np.real(Hz)/vmax, vmin=-1, vmax=1, cmap=plt.cm.RdBu, rasterized=True)
ax[0].set(
    # ylabel=r'$y$ ($\mu$m)', 
    # xlabel=r'$x$ ($\mu$m)', 
    xlim=(-2*lattice_const,2*lattice_const), 
    ylim=(-2*lattice_const,2*lattice_const), 
    # xticks=np.linspace(-20,20,5), yticks=np.linspace(-20,20,5),
    xticks=[], yticks=[],
)
[ax[0].axvline(x=x-lattice_const/2, color='gray', ls='--', lw=0.5) for x in x_cell_center]
[ax[0].axhline(y=y-lattice_const/2, color='gray', ls='--', lw=0.5) for y in y_cell_center]

cbax = ax[0].inset_axes([1.02, 0, 0.03, 1])
cb = plt.colorbar(pc0, cax=cbax)
cb.ax.set_yticks([-1, 0, 1])
cb.ax.set_ylabel(r"$\mathrm{Re}(H_z)$ (Arb. U.)")
ax[0].set_title(r"Waveguide plane", fontsize=7)


vmax = np.max(np.abs(Sz))
pc1 = ax[1].pcolormesh(x, y, np.abs(Sz)/vmax, vmin=0, vmax=1, cmap=plt.cm.GnBu, rasterized=True)
ax[1].set(
    # xlabel=r'$x$ ($\mu$m)',  
    xlim=(-2*lattice_const,2*lattice_const), 
    ylim=(-2*lattice_const,2*lattice_const), 
    # xticks=np.linspace(-20,20,5), yticks=np.linspace(-20,20,5)
    xticks=[], yticks=[],
)
x_module = l_des/2*np.array([1,1,-1,-1,1])
y_module = l_des/2*np.array([1,-1,-1,1,1])
[ax[1].plot(x+dist_module+x_module, y+y_module, color="crimson", ls='--', lw=0.5) for x in x_cell_center for y in y_cell_center]
[ax[1].plot(x+x_module, y+dist_module+y_module, color="crimson", ls='-', lw=0.5) for x in x_cell_center for y in y_cell_center]

color_module = "k"
ax[0].plot(x_cell_center[2]+x_module+dist_module, y_cell_center[2]+y_module, ls='--', lw=0.5, color='crimson')
ax[0].plot(x_cell_center[2]+x_module, y_cell_center[2]+y_module+dist_module, ls='-', lw=0.5, color='crimson')
ax[0].plot(x_cell_center[2]+x_module-dist_module, y_cell_center[2]+y_module, ls='-', lw=0.5, color="darkorange")
ax[0].plot(x_cell_center[2]+x_module, y_cell_center[2]+y_module-dist_module, ls='--', lw=0.5, color="darkorange")
ax[0].plot(x_cell_center[2]+x_module, y_cell_center[2]+y_module, ls='-', lw=0.5, color="navy")
ax[0].plot(x_cell_center[2]+x_module-dist_module, y_cell_center[2]+y_module-dist_module, ls='-', lw=0.5, color='teal')



cbax = ax[1].inset_axes([1.02, 0, 0.03, 1])
cb = plt.colorbar(pc1, cax=cbax)
cb.ax.set_yticks([0, 0.5,  1])
cb.ax.set_ylabel(r"$\mathrm{Re}(S_z)$ (Arb. U.)")
ax[1].set_title(r"Detection plane", fontsize=7)
fig.savefig('Figures/Fig4/fig_cleo.pdf')