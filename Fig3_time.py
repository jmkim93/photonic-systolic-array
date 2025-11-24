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


#%% Obtain design parameters

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
    

#%% FDTD simulation

for cell in [(2,2)]:
    for pulse_width in [10, 50, 100]:
        for inpt in ["A"]:  # ["A", "B", "AB", "neff"]
            ## Geometric parameters
            from defineAdjointOptimization import w_wg, h_wg, l_des, w_mode, buffer_pml_z
            from defineAdjointOptimization import min_step_per_wvl, pixel_size
            # buffer_pml_xy = 1.5
            # dist_wg = l_des*1.2
            buffer_pml_xy = 1.0
            dist_wg = 4.0
            lattice_const = (dist_wg + l_des)+2

            ## Material parameters
            from defineAdjointOptimization import Si, SiO2

            ## Total simulation size
            Lx = 2*buffer_pml_xy + lattice_const
            Ly = 2*buffer_pml_xy + lattice_const
            Lz = 2*buffer_pml_z + h_wg 

            ## Spectral parameters
            from defineAdjointOptimization import wavelength, k0, freq0
            fwidth = freq0/pulse_width
            run_time = 10/(2*np.pi*fwidth) + 9*(Lx+dist_wg)/td.C_0

            source_time1 = td.GaussianPulse(
                freq0=freq0, fwidth=fwidth,
                amplitude=1, phase=0,
            )
            source_time2 = td.GaussianPulse(
                freq0=freq0, fwidth=fwidth,
                amplitude=1, phase=-np.pi/2,
            )

            time = np.linspace(0, run_time, 10000)
            source_time1.plot(time)
            source_time2.plot(time)

            def get_structure(eps: np.ndarray, pos: list=[0,0]):
                xs = np.linspace(pos[0]-(l_des-pixel_size)/2, pos[0]+(l_des-pixel_size)/2, eps.shape[0]).tolist()
                ys = np.linspace(pos[1]-(l_des-pixel_size)/2, pos[1]+(l_des-pixel_size)/2, eps.shape[1]).tolist()
                coords = dict(x=xs, y=ys, z=[0], f=[freq0])
                eps_data = td.ScalarFieldDataArray(eps.reshape(eps.shape[0],eps.shape[1],1,1), coords=coords)
                field_components = {f"eps_{dim}{dim}": eps_data for dim in "xyz"}
                eps_dataset = td.PermittivityDataset(**field_components)
                custom_medium = td.CustomMedium(eps_dataset=eps_dataset)
                return td.Structure(
                    geometry=td.Box(center=(pos[0],pos[1],0), size=(l_des,l_des,h_wg)), 
                    medium=custom_medium
                )

            wg1 = td.Structure(
                geometry=td.Box(
                    center=(-dist_wg/2,-(Ly*1.1-dist_wg)/4,0) if cell[0]==1 else (-dist_wg/2, 0, 0),
                    size=(w_wg, (Ly*1.1+dist_wg)/2, h_wg) if cell[0]==1 else (w_wg, td.inf, h_wg)
                ),
                medium=Si
            )
            wg2 = td.Structure(
                geometry=td.Box(
                    center=(-(Ly*1.1-dist_wg)/4,-dist_wg/2,0) if cell[1]==1 else (0, -dist_wg/2,  0),
                    size=((Ly*1.1+dist_wg)/2, w_wg,  h_wg) if cell[1]==1 else (td.inf, w_wg,  h_wg)
                ),
                medium=Si
            )
            wg3 = td.Structure(
                geometry=td.Box(
                    center=(dist_wg/2, (Ly*1.1-dist_wg)/4, 0),
                    size=(w_wg, (Ly*1.1+dist_wg)/2, h_wg)
                ),
                medium=Si
            )
            wg4 = td.Structure(
                geometry=td.Box(
                    center=((Ly*1.1-dist_wg)/4, dist_wg/2,  0),
                    size=((Ly*1.1+dist_wg)/2, w_wg,  h_wg)
                ),
                medium=Si
            )

            cross = get_structure(eps_opt["cross"], pos=(-dist_wg/2,-dist_wg/2))
            branch1 = get_structure(eps_opt[str(cell[0])], pos=(-dist_wg/2,dist_wg/2))
            branch2 = get_structure(eps_opt[str(cell[1])].T, pos=(dist_wg/2,-dist_wg/2))
            BS = get_structure(eps_opt["BS"], pos=(dist_wg/2,dist_wg/2))

            sim_base = td.Simulation(
                size=(Lx,Lx,Lz),
                structures=[wg1, wg2, wg3, wg4, cross, branch1, branch2, BS],
                grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
                boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
                run_time=run_time,
                subpixel=True,
                medium=SiO2,
                symmetry=(0,0,1),
            )

            sim_base.plot_eps(z=0, freq=freq0)

            plane_in = td.Box(size=(w_mode, 0, Lz),center=(-dist_wg/2, -lattice_const/2, 0))
            num_modes = 3
            mode_spec = td.ModeSpec(num_modes=num_modes)
            mode_solver = ModeSolver(
                simulation=sim_base,
                plane=plane_in,
                mode_spec=mode_spec,
                freqs=[freq0],
            )
            mode_data = mode_solver.solve()

            src_A = td.ModeSource(
                size=(w_mode, 0, Lz),
                center=(-dist_wg/2, -lattice_const/2-buffer_pml_xy/5, 0),
                source_time=source_time1,
                mode_spec=mode_spec,
                mode_index=0,
                direction="+",
                num_freqs=9,
            )
            src_B = td.ModeSource(
                size=(0, w_mode, Lz),
                center=(-lattice_const/2-buffer_pml_xy/5, -dist_wg/2, 0),
                source_time=source_time2,
                mode_spec=mode_spec,
                mode_index=0,
                direction="+",
                num_freqs=9,
            )
            src = []
            if inpt[0]=="A" or inpt=="neff":
                src.append(src_A)
            if inpt[-1]=="B":
                src.append(src_B)

            mnt_thru1_time = td.FieldTimeMonitor(
                center=(-dist_wg/2, lattice_const/2, 0),
                size=(w_wg, 0, h_wg),
                name="thru1_time",
                interval=10,
                start=3.5*lattice_const/td.C_0,
                # stop=3.5*lattice_const/td.C_0+12/(2*np.pi*fwidth)
            )
            mnt_det1_time = td.FieldTimeMonitor(
                center=(dist_wg/2, lattice_const/2, 0),
                size=(w_wg, 0, h_wg),
                name="det1_time",
                interval=10,
                start=3.5*(lattice_const+dist_wg)/td.C_0,
                # stop=3.5*(lattice_const+dist_wg)/td.C_0+12/(2*np.pi*fwidth)
            )
            mnt_thru2_time = td.FieldTimeMonitor(
                center=(lattice_const/2, -dist_wg/2, 0),
                size=(0, w_wg, h_wg),
                name="thru2_time",
                interval=10,
                start=3.5*lattice_const/td.C_0,
                # stop=3.5*lattice_const/td.C_0+12/(2*np.pi*fwidth)
            )
            mnt_det2_time = td.FieldTimeMonitor(
                center=(lattice_const/2, +dist_wg/2, 0),
                size=(0, w_wg, h_wg),
                name="det2_time",
                interval=10,
                start=3.5*(lattice_const+dist_wg)/td.C_0,
                # stop=3.5*(lattice_const+dist_wg)/td.C_0+12/(2*np.pi*fwidth)
            )

            mnt_thru1_freq = mode_solver.to_monitor(
                name="thru1_freq",
            )
            mnt_thru1_freq = mnt_thru1_freq.updated_copy(
                mode_spec=mode_spec,
                center=(-dist_wg/2, lattice_const/2, 0),
                size=(w_mode, 0, Lz),
            )

            mnt_det1_freq = mnt_thru1_freq.updated_copy(
                mode_spec=mode_spec,
                center=(dist_wg/2, lattice_const/2, 0),
                size=(w_mode, 0, Lz),
                name="det1_freq",
            )

            mnt_thru2_freq = mnt_thru1_freq.updated_copy(
                mode_spec=mode_spec,
                center=(lattice_const/2, -dist_wg/2, 0),
                size=(0, w_mode, Lz),
                name="thru2_freq",
            )
            mnt_det2_freq = mnt_thru1_freq.updated_copy(
                mode_spec=mode_spec,
                center=(lattice_const/2, +dist_wg/2, 0),
                size=(0, w_mode, Lz),
                name="det2_freq",
            )

            mnt_field_time = td.FieldTimeMonitor(
                size=(td.inf,td.inf, 0),
                center=(0,0,0),
                interval=100,
                # interval_space=(2,2,1),
                name="field_time",
            )

            mnt_field_freq = td.FieldMonitor(
                size=(td.inf,td.inf, 0),
                center=(0,0,0),
                freqs=[freq0],
                name="field_freq"
            )

            mnt_src1_time = td.FieldTimeMonitor(
                center=(-dist_wg/2, -lattice_const/2, 0),
                size=(w_wg, 0, h_wg),
                name="src1_time",
                interval=10,
                stop=11/(2*np.pi*fwidth),
            )

            mnt_src2_time = td.FieldTimeMonitor(
                center=(-lattice_const/2, -dist_wg/2, 0),
                size=(0, w_wg, h_wg),
                name="src2_time",
                interval=10,
                stop=11/(2*np.pi*fwidth),
            )

            monitors_det = [
                mnt_src1_time, mnt_src2_time,
                mnt_det1_time, mnt_det2_time,
                mnt_det1_freq, mnt_det2_freq
            ]
            monitors_thru1 = [mnt_thru1_time, mnt_thru1_freq]
            monitors_thru2 = [mnt_thru2_time, mnt_thru2_freq]

            monitors = (
                monitors_det + 
                (monitors_thru1 if cell[0]>1 else []) + 
                (monitors_thru2 if cell[1]>1 else []) + 
                ([mnt_field_freq] if inpt=="A" else [])
            )
            if inpt=="AB":
                monitors=[mnt_field_time]

            if inpt=="neff":
                mode_solver_group = ModeSolver(
                    simulation=sim_base,
                    plane=plane_in,
                    mode_spec=mode_spec,
                    freqs=freq0*np.linspace(0.95,1.05,101),
                )
                mnt_mode_inspect = mode_solver_group.to_monitor(
                    name="mode_inspect",
                )
                mnt_mode_inspect = mnt_mode_inspect.updated_copy(
                    mode_spec=mode_spec,
                    center=(-dist_wg/2, lattice_const/2, 0),
                    size=(w_mode, 0, Lz),
                )
                monitors=[mnt_mode_inspect]

            sim = sim_base.updated_copy(sources=src, monitors=monitors)

            sim.plot(z=0)
            # sim_data = td.web.run(
            #     sim, 
            #     folder_name="Fig3_ip_set3",
            #     task_name="test_cell{}{}_pulsewidth{}_type{}_gap".format(cell[0],cell[1],pulse_width, inpt), 
            #     verbose=True
            # )

#%% SuppFig - simulation schematic

## Ensure type A

fig, ax = plt.subplots(1,2, figsize=(7,3.5), tight_layout=True)
sim.plot(z=0.001, ax=ax[0])
sim.plot(x=-dist_wg/2, ax=ax[1])

ax[0].set(
    xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)",
    xticks=np.linspace(-5,5,11), yticks=np.linspace(-5,5,11)
)
ax[1].set(
    xlabel=r"$y$ ($\mu$m)", ylabel=r"$z$ ($\mu$m)",
    xticks=np.linspace(-5,5,11),
    yticks=np.linspace(-1.5,1.5,7),
)

ax[0].set_title("Cross section, $z=0$")
ax[1].set_title("Cross section, $x=-2$")

ax[0].text(-2,-2, "Crossing", horizontalalignment='center', verticalalignment='center')
ax[0].text(-2,2, r"Branch, $n$", horizontalalignment='center', verticalalignment='center')
ax[0].text(2,-2, r"Branch$^T$, $m$", horizontalalignment='center', verticalalignment='center')
ax[0].text(2,2, r"Beam splitter", horizontalalignment='center', verticalalignment='center')
ax[0].text(-4,-4.5, r"Pulse source", color="green", horizontalalignment='center',verticalalignment='center', )
ax[0].text(0,4.3, r"Freq. monitors", color="orange", horizontalalignment='center',verticalalignment='center', )
ax[0].text(-4.5,-1.5, r"Time mnts.", color="orange", horizontalalignment='center',verticalalignment='center', )

fig.savefig("Figures/Fig3/SuppFig3_simschematic.pdf")




#%% saved Data load

# For main fig
cell = (2,2)
sim_data_A = dict()
sim_data_B = dict()
sim_data_A_cell = dict()

sim_data_A["neff"] = td.web.api.webapi.load(task_id="fdve-387c7701-2ef5-456d-b2ce-8c038d6526c1")
sim_data_A["10"] = td.web.api.webapi.load(task_id="fdve-b7203e7f-9875-4c50-9fc3-a10b819d5452")
sim_data_A["15"] = td.web.api.webapi.load(task_id="fdve-b229b575-f4ba-440b-ba3d-d2ec1ef9f35c")
sim_data_A["20"] = td.web.api.webapi.load(task_id="fdve-3cd73818-cd6b-4f14-9bc2-b62718a9d2fb")
sim_data_A["30"] = td.web.api.webapi.load(task_id="fdve-7e480ccf-e771-4621-a444-766b1df94f83")
sim_data_A["50"] = td.web.api.webapi.load(task_id="fdve-df2cbcbb-de7b-4f31-bc6d-310628fa745a")
sim_data_A["75"] = td.web.api.webapi.load(task_id="fdve-1302111d-d4ca-4f14-bc4d-1aca9bde0b6e")
sim_data_A["100"] = td.web.api.webapi.load(task_id="fdve-9979b7a4-cf66-4848-a04b-7387b79bf5b9")
sim_data_A["150"] = td.web.api.webapi.load(task_id="fdve-8d908e8c-e168-4617-955c-d1b6a512b197")
sim_data_A["200"] = td.web.api.webapi.load(task_id="fdve-c30a54d1-682b-4b8f-aca2-d6749235a067")

sim_data_B["10"] = td.web.api.webapi.load(task_id="fdve-2898277e-eb86-4077-b5a8-cbd433ff1ee6")
sim_data_B["100"] = td.web.api.webapi.load(task_id="fdve-a67a9c82-760b-4050-83e7-d3412c81edf6")
sim_data_B["200"] = td.web.api.webapi.load(task_id="fdve-fb601300-a412-406e-bef9-a56458ed3eb7")

sim_data_A_cell["42"] = td.web.api.webapi.load(task_id="fdve-29423c0c-05aa-451a-84e1-3af056c22d94")
sim_data_A_cell["32"] = td.web.api.webapi.load(task_id="fdve-2dafdcd0-a411-4e58-a24c-3da75896bce7")
sim_data_A_cell["22"] = sim_data_A["200"]
sim_data_A_cell["12"] = td.web.api.webapi.load(task_id="fdve-48ffdc0d-2b42-4e91-8240-483a5e013f52")


# For animation new
# sim_data_AB = dict()
# sim_data_AB["10"] = td.web.api.webapi.load(task_id="fdve-1aae5379-22a8-46ca-8513-116d1fbdcec0")
# sim_data_AB["50"] = td.web.api.webapi.load(task_id="fdve-c0b96fe9-491d-4454-aca8-a6fee91d132c")
# sim_data_AB["100"] = td.web.api.webapi.load(task_id="fdve-acdb0942-bb8a-4340-9494-284400a07880")



### Old 300 by 300
# sim_data_A["neff"] = td.web.api.webapi.load(task_id="fdve-555252cc-2956-487b-9ee1-a42d052f9484")
# sim_data_A["10"] = td.web.api.webapi.load(task_id="fdve-85a0d133-3d23-4692-b4af-738d2f92e53d")
# sim_data_A["15"] = td.web.api.webapi.load(task_id="fdve-9e4ce761-1dcb-4e80-84b8-27aee432b4eb")
# sim_data_A["20"] = td.web.api.webapi.load(task_id="fdve-7ac644c1-fe5d-447f-9cdc-f891ab5f53e0")
# sim_data_A["30"] = td.web.api.webapi.load(task_id="fdve-382511e9-e3cd-43a2-839a-e57de0570bf1")
# sim_data_A["50"] = td.web.api.webapi.load(task_id="fdve-3d34dfb5-ab64-468e-9951-6d6857989447")
# sim_data_A["75"] = td.web.api.webapi.load(task_id="fdve-41329ef9-b56e-4f97-b062-c1e0b1d37dc1")
# sim_data_A["100"] = td.web.api.webapi.load(task_id="fdve-343a9b70-5d4c-40a4-9ba7-5fc3ea5e8ef6")
# sim_data_A["150"] = td.web.api.webapi.load(task_id="fdve-17fd95d1-989e-4bee-8deb-4d9ab4e8cd2b")
# sim_data_A["200"] = td.web.api.webapi.load(task_id="fdve-53f52e43-b238-4d40-a681-07d52f9255ef")

# sim_data_B["100"] = td.web.api.webapi.load(task_id="fdve-94b00959-9bc0-422c-affd-db1d5b8de365")
# sim_data_B["10"] = td.web.api.webapi.load(task_id="fdve-1be90c65-d3a4-4080-80f0-38ca9bd1673f")


# sim_data_A_cell["42"] = td.web.api.webapi.load(task_id="fdve-3f11745b-57fe-4253-943f-27bc01c2be7a")
# sim_data_A_cell["32"] = td.web.api.webapi.load(task_id="fdve-16e57887-47ff-444c-9ff0-67cfc28b8104")
# sim_data_A_cell["22"] = sim_data_A["200"]
# sim_data_A_cell["12"] = td.web.api.webapi.load(task_id="fdve-957277ca-67e2-42b1-970d-ebed6bc02bf4")
# sim_data_A_cell["neff"] = td.web.api.webapi.load(task_id="fdve-555252cc-2956-487b-9ee1-a42d052f9484")



# For animation new
# sim_data_AB = dict()
# sim_data_AB["10"] = td.web.api.webapi.load(task_id="fdve-c17a1e6f-69e9-4f0f-89de-b5af068d027d")
# sim_data_AB["50"] = td.web.api.webapi.load(task_id="fdve-e7c533e8-75ef-4987-884d-c91e064e7617")
# sim_data_AB["100"] = td.web.api.webapi.load(task_id="fdve-3124dae8-a461-4938-99ef-93b6916e7c9c")



#%% Fig 1: Mode dispersion spec

freq_list = freq0*np.linspace(0.95,1.05,101)
n_eff_phase = sim_data_A["neff"]["mode_inspect"].n_eff[:,0].data
n_eff_group = n_eff_phase + freq_list * np.gradient(n_eff_phase, freq_list)

n_Si = np.real(Si.nk_model(freq_list))[0]
n_SiO2 = np.real(SiO2.nk_model(freq_list))[0]

fig, ax = plt.subplots(figsize=(3.3, 1.8), tight_layout=True)
ax.plot(freq_list/1e12, n_Si, lw=1, ls='-', color='gray')
ax.plot(freq_list/1e12, n_SiO2, lw=1, ls='-', color='gray' )
ax.plot(freq_list/1e12, n_eff_phase, lw=1, color='cornflowerblue')
ax.plot(freq_list/1e12, n_eff_group, lw=1, color='royalblue')
ax.axvline(x=freq0/1e12, color='k', lw=0.5, ls='--')
ax.set(
    xlim=(freq0*0.95/1e12, freq0*1.05/1e12), 
    xlabel=r"Frequency (THz)", ylabel=r'Refractive index',
    xticks=np.arange(184, 203, 2),
    yticks=[1.5, 2, 2.5, 3, 3.5, 4,4.5]
)

ax.text(184.5, 1.85, "Waveguide, phase", color='cornflowerblue')
ax.text(184.5, 1.51, "SiO$_2$", color='gray')
ax.text(184.5, 3.98, "Waveguide, group", color='royalblue')
ax.text(184.5, 3.2, "Si", color='gray')

n_eff_group_wg = n_eff_group[50]

axins = ax.inset_axes([0.56, 0.1, 0.323, 0.72])
axins.set(xticks=[], yticks=[])

x, z, Hz = mode_data.Hz.x, mode_data.Hz.z, mode_data.Hz.data[:,0,:,0,0]
Z, X = np.meshgrid(z, x)
x_offset = x[np.argmax(np.abs(Hz)[:,22])].data
vmax = np.max(np.abs(Hz)**2)
pc = axins.pcolormesh(X-x_offset, Z, np.abs(Hz)**2, cmap=plt.cm.inferno, shading="gouraud")
axins.axhline(y=-h_wg/2, color='w', lw=0.3)
axins.plot(w_wg/2*np.array([-1,-1,1,1]), h_wg/2*np.array([-1,1,1,-1]), color='w', lw=0.3)

axins.text(-0.1,0.17, r"$w_\mathrm{wg}$", color='w')
axins.text(0.16,-0.04, r"$h_\mathrm{wg}$", color='w')
axins.text(-0.5,0.4, "cladding", color='w')
axins.text(-0.5,-0.48, "BOx", color='w')
axins.text(-0.5,-0.03, "core", color='w')

axins.set(
    xlim=(-0.5,0.5), ylim=(-0.5,0.5), 
    xticks=[], yticks=[]
)
[axins.spines[pos].set_visible(False) for pos in ["top","bottom", "right", "left"]]

cbax = axins.inset_axes([1.03, 0.0, 0.05, 1])
cb = plt.colorbar(pc, cax=cbax)
cb.ax.set_yticks([0,vmax])
cb.ax.set_yticklabels([0,"max"])
# cb.ax.set_title(r"       $|H_z|^2$", fontsize=7)

fig.savefig("Figures/Fig1/fig1_dispersion.pdf")

#%% Fig 3A: Time-domain dispersion 

color_a = (46/256,49/256,146/256)
color_b = (247/256,147/256,30/256)
color_a2 = np.array(color_a)*1.5


def env(t, signal):
    dt = t[1]-t[0]
    T_period = dt*len(t)
    df = 1/T_period
    f = df.data * np.arange(len(t))
    FFT_signal = np.fft.fft(signal)
    FFT_signal_filtered = FFT_signal * ((f<freq0)+(f>f[-1]-freq0))
    signal_env = np.fft.ifft(FFT_signal_filtered)*2
    return np.real(signal_env)


# for cell_number in ["42", "32", "22"]:
#     print("cell ", cell_number)
#     t = sim_data_A_cell[cell_number]["src1_time"].flux.t
#     t_offset = t[np.argmax(env(t,sim_data_A_cell[cell_number]["src1_time"].flux))].data
#     t_det1 = t[np.argmax(env(t,sim_data_A_cell[cell_number]["det1_time"].flux))].data
#     t_det2 = t[np.argmax(env(t,sim_data_A_cell[cell_number]["det2_time"].flux))].data
#     # print("Time dealy = ", round((t_thru-t_offset)/1e-12, 6))
#     if cell_number[0]!="1":
#         t_thru = t[np.argmax(env(t,sim_data_A_cell[cell_number]["thru1_time"].flux))].data
#         print("Thru, n_eff_group = ", round(td.C_0*(t_thru-t_offset)/lattice_const, 6) )
#     print("Det1, n_eff_group = ", round(td.C_0*(t_det1-t_offset)/(lattice_const+dist_wg), 6) )
#     print("Det2, n_eff_group = ", round(td.C_0*(t_det2-t_offset)/(lattice_const+dist_wg), 6) )

#     print("wg, neff_phase = ", sim_data_A_cell[cell_number]["det1_freq"].n_eff[0,0].data)


def plot_line_area(ax, t, signal, color:str='k', lw:float=0.75, ls:str='-', range:float=(0,1), norm:float=1):
    # norm = signal.max()
    ax.plot(t/1e-12, range[0]+(range[1]-range[0])*signal/norm, lw=lw, ls=ls, color=color)
    ax.fill_between(t/1e-12, range[0]+(range[1]-range[0])*signal/norm, range[0], lw=0, color=color, alpha=0.15)

# cross_efficiency = 0.99064654
# split_efficiency = 0.96
cross_efficiency = 0.9938450312283263
split_efficiency = 0.98
efficiency_tot = cross_efficiency * split_efficiency
n = cell[0]
split = efficiency_tot**(n-1) * (1-efficiency_tot) / (1-efficiency_tot**n)


# fig = plt.figure(figsize=(6.5, 3.4), constrained_layout=True, tight_layout=True)
# gs = fig.add_gridspec(100, 200)
# ax = [
#     fig.add_subplot(gs[0:42, 0:90]), fig.add_subplot(gs[0:42, 110:180]), 
#     fig.add_subplot(gs[58:100, 0:45]), fig.add_subplot(gs[58:100, 65:110]), fig.add_subplot(gs[58:100, 130:175])]
# ax_cb = fig.add_subplot(gs[58:100, 190:])

fig = plt.figure(figsize=(6.85, 1.9), constrained_layout=True, tight_layout=True)
gs = fig.add_gridspec(1,13)
ax = [fig.add_subplot(gs[0, :7]), fig.add_subplot(gs[0, 8:])]
for pwidth, ls in zip([200, 100, 50, 10], ["-", "--", "-.", ":"]):
    # ls = "-" if pwidth==200 else ("--" if pwidth==50 else "-.")
    # n_eff = sim_data_A[str(pwidth)]["det1_freq"].n_eff.data[0,0]
    # t = sim_data_A[str(pwidth)]["src1_time"].flux.t - 5*pwidth/(2*np.pi*freq0)
    t = sim_data_A[str(pwidth)]["src1_time"].flux.t
    t_offset = t[np.argmax(env(t,sim_data_A[str(pwidth)]["src1_time"].flux))].data
    norm = env(t,sim_data_A[str(pwidth)]["src1_time"].flux).max()
    plot_line_area(ax[0], t-t_offset, env(t,sim_data_A[str(pwidth)]["src1_time"].flux), ls=ls, color=color_a, range=(1/2,1), norm=norm)

    t = sim_data_A[str(pwidth)]["thru1_time"].flux.t
    t_thru = t[np.argmax(env(t,sim_data_A[str(pwidth)]["thru1_time"].flux))].data
    plot_line_area(ax[0], t-t_offset, env(t,sim_data_A[str(pwidth)]["thru1_time"].flux), ls=ls, color=color_a2,  range=(1/4,1/2), norm=norm*split_efficiency*(1-split))
    
    t = sim_data_A[str(pwidth)]["det1_time"].flux.t
    plot_line_area(ax[0], t-t_offset, env(t,sim_data_A[str(pwidth)]["det1_time"].flux), ls=ls, color='k',  range=(1/8,1/4), norm=norm*split_efficiency*split/2)
    t = sim_data_A[str(pwidth)]["det2_time"].flux.t
    plot_line_area(ax[0], t-t_offset, env(t,sim_data_A[str(pwidth)]["det2_time"].flux), ls=ls, color='k',  range=(0,1/8), norm=norm*split_efficiency*split/2)

    print("Time dealy = ", round((t_thru-t_offset)/1e-12, 4))
    print("n_eff_group = ", round(td.C_0*(t_thru-t_offset)/lattice_const, 4) )

    if pwidth==200:
        ax[0].axvline(x=0, color='gray', lw=0.5)
        ax[0].axvline(x=(t_thru-t_offset)/1e-12, color='gray', lw=0.5, ls='--')


ax[0].set(
    ylim=(0,1), xlim=(-0.08,0.28), xlabel=r'$t$ (ps)', ylabel=r'Power envelope (Arb. U.)',
    yticks=(
        0, 1/8*0.1/(norm*split_efficiency*split/2),
        1/8, 1/8+ 1/8*0.1/(norm*split_efficiency*split/2),
        1/4,1/4+1/4*0.2/(norm*split_efficiency*(1-split)), 1/4+1/4*0.4/(norm*split_efficiency*(1-split)),
        1/2, 3/4, 1
    ), 
    yticklabels=(
        0, 0.1,
        0, 0.1,
        0,0.2, 0.4,
        0, 0.5, 1
    )    
)
# ax[0].set_title("Module (2, 2)", fontsize=7)

def overlap(t, signal1, signal2):
    signal1 = env(t, signal1)
    signal2 = env(t, signal2)
    # return np.sum(np.minimum(signal1,signal2))/np.sum(np.maximum(signal1,signal2))
    return np.sum(np.minimum(signal1,signal2))/np.sqrt(np.sum(signal1)*np.sum(signal2))

pwidth_list = [10, 15, 20, 30, 50, 75, 100, 150, 200]
similarity = [overlap(
    sim_data_A[str(pwidth)]["det1_time"].flux.t, 
    sim_data_A[str(pwidth)]["det1_time"].flux, 
    sim_data_A[str(pwidth)]["det2_time"].flux
) for pwidth in pwidth_list]

def measure_FWHM(t, signal):
    signal = env(t,signal)
    signal = signal/np.max(signal)
    idx_center = np.argmax(signal)
    t_left = t[:idx_center][np.argmin(np.abs(signal[:idx_center]-0.5))]
    t_right = t[idx_center:][np.argmin(np.abs(signal[idx_center:]-0.5))]
    return t_right-t_left

FWHM_thru = np.array([measure_FWHM(
    sim_data_A[str(pwidth)]["thru1_time"].flux.t, 
    sim_data_A[str(pwidth)]["thru1_time"].flux
) for pwidth in pwidth_list])

FWHM_inc = np.sqrt(np.log(2))/(np.pi*freq0) * np.array(pwidth_list)
ax[1].plot(FWHM_inc/1e-12, similarity, color='k', marker='o', mfc='w', mew=0.5, lw=0.5, ms=3)
ax[1].set(xlabel=r'Incident pulse width, $T_\mathrm{FWHM}$ (ps)', ylabel=r'Overlap coefficient')
ax[1].set(yticks=(0.85, 0.9, 0.95, 1), ylim=(0.85,1))
# ax[0].plot([-FWHM_inc[-1]/2/1e-12, FWHM_inc[-1]/2/1e-12], [0.75,0.75])

# ax[1].set_title("Dispersion characteristics", fontsize=7)

axt = ax[1].twinx()
axt.plot(FWHM_inc/1e-12, FWHM_thru/FWHM_inc, color=color_a2, marker='s', mfc='w', mew=0.5, lw=0.5, ms=3)
axt.set_ylabel(r"Dispersion factor", color=color_a2)
axt.spines["right"].set_edgecolor(color_a2)
axt.tick_params(axis='y', colors=color_a2)
axt.set(ylim=(1,1.4))
fig.savefig("Figures/Fig3/fig3A.pdf")


#%% Plot Fig. 3b
wg1 = td.Structure(geometry=td.Box(center=(-dist_wg/2, 0, 0),size=(w_wg, td.inf, h_wg)),medium=Si)
wg2 = td.Structure(geometry=td.Box(center=(0, -dist_wg/2,  0),size=(td.inf, w_wg,  h_wg)),medium=Si)
wg3 = td.Structure(geometry=td.Box(center=(dist_wg/2, (Ly*1.1-dist_wg)/4, 0),size=(w_wg, (Ly*1.1+dist_wg)/2, h_wg)),medium=Si)
wg4 = td.Structure(geometry=td.Box(center=((Ly*1.1-dist_wg)/4, dist_wg/2,  0),size=((Ly*1.1+dist_wg)/2, w_wg,  h_wg)),medium=Si)
branch1 = get_structure(eps_opt["2"], pos=(-dist_wg/2,dist_wg/2))
branch2 = get_structure(eps_opt["2"].T, pos=(dist_wg/2,-dist_wg/2))
BS = get_structure(eps_opt["BS"], pos=(dist_wg/2,dist_wg/2))
cross = get_structure(eps_opt["cross"], pos=(-dist_wg/2,-dist_wg/2))
sim = td.Simulation(
    size=(Lx,Lx,Lz),
    structures=[wg1, wg2, wg3, wg4, cross, branch1, branch2, BS],
    grid_spec=td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_step_per_wvl),
    boundary_spec=td.BoundarySpec.pml(x=True,y=True,z=True),
    run_time=run_time,
    subpixel=True,
    medium=SiO2,
    symmetry=(0,0,1),
)

# fig = plt.figure(figsize=(6.12, 1.9), constrained_layout=True, tight_layout=True)
# gs = fig.add_gridspec(1,4)
# ax = [fig.add_subplot(gs[0, 0]),fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]

fig, ax = plt.subplots(1, 4, figsize=(6.85, 1.8), tight_layout=True)
box_plot = td.Box(
    center=(0,0,0),
    size=(lattice_const, lattice_const, 0)
)
eps = np.real(sim.epsilon(box=box_plot,freq=ua.freq0).to_numpy()[:,:,0])
x = sim.epsilon(box=box_plot,freq=ua.freq0).x
y = sim.epsilon(box=box_plot,freq=ua.freq0).y
Y, X = np.meshgrid(y,x)
level = (eps.max()+eps.min())/2
ax[0].contour(X, Y, eps, [level], colors='k', linewidths=0.4)
ax[1].contour(X, Y, eps, [level], colors='k', linewidths=0.4)

x, y, Hz_base = (
    sim_data_A["100"]["field_freq"].Hz.x, 
    sim_data_A["100"]["field_freq"].Hz.y, 
    sim_data_A["100"]["field_freq"].Hz.data[:,:,0,0]
)
Y,X = np.meshgrid(y,x)
amp1, amp2 = (1,0)
Hz_superposed = amp1*Hz_base + 1j*amp2*Hz_base.T
vmax = np.max(np.abs(Hz_superposed))
pc = ax[0].pcolormesh(X, Y, np.real(Hz_superposed), cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, shading='auto', linewidth=0)
ax[0].set(xlim=(-lattice_const*0.85/2,lattice_const*0.85/2), ylim=(-lattice_const*0.85/2,lattice_const*0.85/2))
[ax[0].spines[pos].set_visible(False) for pos in ["top","bottom", "right", "left"]]
ax[0].set(xticks=[], yticks=[])
# ax[0].set_title(r'$(a,b)=(1,0)$', fontsize=7)

cbax = ax[0].inset_axes([1.05, 0.2, 0.05, 0.6])
cb = plt.colorbar(pc, cax=cbax)
cb.ax.set_yticks(vmax*np.linspace(-1,1,3))
cb.ax.set_yticklabels([-1,0,1])
cb.ax.set_ylabel(r"$\mathrm{Re}(H_z)$ (Arb.U.)")

amp1, amp2 = (1,1)
Hz_superposed = amp1*Hz_base + 1j*amp2*Hz_base.T
vmax = np.max(np.abs(Hz_superposed))
pc = ax[1].pcolormesh(X, Y, np.real(Hz_superposed), cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, shading='auto', linewidth=0)
ax[1].set(xlim=(-lattice_const*0.85/2,lattice_const*0.85/2), ylim=(-lattice_const*0.85/2,lattice_const*0.85/2))
[ax[1].spines[pos].set_visible(False) for pos in ["top","bottom", "right", "left"]]
ax[1].set(xticks=[], yticks=[])
# ax[1].set_title(r'$(a,b)=(1,1)$', fontsize=7)


pwidth = "100"
dx = (sim_data_A[pwidth]["det1_time"].Hz.x[1]-sim_data_A[pwidth]["det1_time"].Hz.x[0]).data
dy = (sim_data_A[pwidth]["det2_time"].Hz.y[1]-sim_data_A[pwidth]["det2_time"].Hz.y[0]).data
dz = (sim_data_A[pwidth]["det2_time"].Hz.z[1]-sim_data_A[pwidth]["det2_time"].Hz.z[0]).data
dt = (sim_data_A[pwidth]["det1_time"].Hz.t[1]-sim_data_A[pwidth]["det1_time"].Hz.t[0]).data
ampA = np.linspace(-1,1,51).reshape(-1,1)
ampB = np.linspace(-1,1,51).reshape(1,-1)
flux1 = (
    ampA**2 * np.sum(sim_data_A[pwidth]["det1_time"].Ez*sim_data_A[pwidth]["det1_time"].Hx - sim_data_A[pwidth]["det1_time"].Ex*sim_data_A[pwidth]["det1_time"].Hz).data
    + ampA*ampB * np.sum(sim_data_A[pwidth]["det1_time"].Ez*sim_data_B[pwidth]["det1_time"].Hx + sim_data_B[pwidth]["det1_time"].Ez*sim_data_A[pwidth]["det1_time"].Hx - sim_data_A[pwidth]["det1_time"].Ex*sim_data_B[pwidth]["det1_time"].Hz - sim_data_B[pwidth]["det1_time"].Ex*sim_data_A[pwidth]["det1_time"].Hz).data
    + ampB**2 * np.sum(sim_data_B[pwidth]["det1_time"].Ez*sim_data_B[pwidth]["det1_time"].Hx - sim_data_B[pwidth]["det1_time"].Ex*sim_data_B[pwidth]["det1_time"].Hz).data
)*dx*dz*dt
flux2 = -1 * (
    ampA**2 * np.sum(sim_data_A[pwidth]["det2_time"].Ez*sim_data_A[pwidth]["det2_time"].Hy - sim_data_A[pwidth]["det2_time"].Ey*sim_data_A[pwidth]["det2_time"].Hz).data
    + ampA*ampB * np.sum(sim_data_A[pwidth]["det2_time"].Ez*sim_data_B[pwidth]["det2_time"].Hy + sim_data_B[pwidth]["det2_time"].Ez*sim_data_A[pwidth]["det2_time"].Hy - sim_data_A[pwidth]["det2_time"].Ey*sim_data_B[pwidth]["det2_time"].Hz - sim_data_B[pwidth]["det2_time"].Ey*sim_data_A[pwidth]["det2_time"].Hz).data
    + ampB**2 * np.sum(sim_data_B[pwidth]["det2_time"].Ez*sim_data_B[pwidth]["det2_time"].Hy - sim_data_B[pwidth]["det2_time"].Ey*sim_data_B[pwidth]["det2_time"].Hz).data
)*dy*dz*dt
AdotB_time100 = flux1-flux2

detA1 = sim_data_A[pwidth]["det1_freq"].amps.data[0,0,0]
detA2 = -sim_data_A[pwidth]["det2_freq"].amps.data[0,0,0]
flux1 = np.abs(ampA* detA1 + 1j * ampB * detA2)**2
flux2 = np.abs(ampA* detA2 + 1j * ampB * detA1)**2
AdotB_freq = flux1-flux2

B, A = np.meshgrid(np.linspace(-1,1,51), np.linspace(-1,1,51))
vmax = np.max(np.abs(AdotB_time100))
ax[2].contour(A, B, A*B, levels=np.arange(-1,0,0.1), colors='k', linewidths=0.5, linestyles='--')
ax[2].contour(A, B, A*B, levels=np.arange(0,1,0.1)+0.1, colors='k', linewidths=0.5, linestyles='--')
ax[2].axhline(y=0, color='k', lw=0.5, ls='--')
ax[2].axvline(x=0, color='k', lw=0.5, ls='--')
ax[2].contourf(A, B, AdotB_time100, levels=vmax*np.linspace(-1,1,21), vmin=-vmax,vmax=vmax, cmap=plt.cm.PRGn)
ax[2].set(
    xlim=(-1,1), ylim=(-1,1), 
    xticks=np.linspace(-1,1,5), yticks=np.linspace(-1,1,5),
    xlabel=r'Input, $a$ (Arb.U.)', ylabel=r'Input, $b$ (Arb.U.)'
)
ax[2].set_title(r"$T_\mathrm{FWHM} = $ 0.14 ps", fontsize=7)


vmax = np.max(np.abs(AdotB_freq))
ax[3].contour(A, B, A*B, levels=np.arange(-1,0,0.1), colors='k', linewidths=0.5, linestyles='--')
ax[3].contour(A, B, A*B, levels=np.arange(0,1,0.1)+0.1, colors='k', linewidths=0.5, linestyles='--')
ax[3].axhline(y=0, color='k', lw=0.5, ls='--')
ax[3].axvline(x=0, color='k', lw=0.5, ls='--')
cf = ax[3].contourf(A, B, AdotB_freq, levels=vmax*np.linspace(-1,1,21), vmin=-vmax,vmax=vmax, cmap=plt.cm.PRGn)
ax[3].set(
    xlim=(-1,1), ylim=(-1,1), 
    xticks=np.linspace(-1,1,5), yticks=np.linspace(-1,1,5),
    xlabel=r'Input, $a$ (Arb.U.)'
)
ax[3].set_title("Continuous wave", fontsize=7)

cbax = ax[3].inset_axes([1.05, 0, 0.05, 1])
cb = plt.colorbar(cf, cax=cbax)
cb.ax.set_yticks(vmax*np.linspace(-1,1,5))
cb.ax.set_yticklabels(np.linspace(-1,1,5))
cb.ax.set_ylabel("Output (Arb.U.)")

fig.savefig("Figures/Fig3/fig3B.pdf")

#%% for 3D schematic in Fig.1

eps = np.real(sim.epsilon(box=box_plot,freq=ua.freq0).to_numpy()[:,:,0])
x = sim.epsilon(box=box_plot,freq=ua.freq0).x.data
y = sim.epsilon(box=box_plot,freq=ua.freq0).y.data

y, x = np.meshgrid(y,x)
colors = np.array([
    [1,1,1, 0.0],
    [0,0,0, 0.5]
])

cmap = matplotlib.colors.ListedColormap(colors)

fig, ax = plt.subplots(figsize=(3.35,3.35), tight_layout=True)
ax.contourf(x, y, eps, levels=np.linspace(eps.min(), eps.max(),3), cmap=cmap)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.savefig("Figures/Fig1/3D_vector_base.svg", dpi=600, transparent=True, format="svg")

#%% Error plot Supple
result_time100 = AdotB_time100/np.max(np.abs(AdotB_time100))
result_freq = AdotB_freq/np.max(np.abs(AdotB_freq))
fig, ax = plt.subplots(figsize=(2.45,2), tight_layout=True)
pc = ax.pcolormesh(
    A, B, np.abs((result_time100-result_freq)/result_freq)*100, 
    cmap=plt.cm.cividis,linewidth=0,rasterized=True
)
ax.set(xlabel=r"Input, $a$ (Arb. U.)", ylabel=r"Input, $b$ (Arb. U.)", xlim=(-1,1), ylim=(-1,1))

cb = plt.colorbar(pc)
cb.ax.set_ylabel("Error (%)")
fig.savefig("Figures/Fig3/Suppfig_error.png")

#%% Inset schematic - dispersion / overlap

t = np.arange(-500,500,0.001)
dt = t[1]-t[0]
T_tot = len(t)*dt
domega = 2*np.pi/T_tot
omega =  t*domega/dt

omega0 = 2*np.pi/2
T_width = 10


signal = np.exp(-1j * omega0 * t) * np.exp(-0.5 * (t/T_width)**2)

F_signal = np.fft.fftshift(np.fft.fft(signal))

distance = 2

ref_index = np.ones_like(omega) 
ref_index = 1 +2 * (omega/omega0)**2 +5  * (omega/omega0)**4 

ref_index2 = 1 +4 * (omega/omega0)**2 +3  * (omega/omega0)**4 

F_signal_thru = F_signal * np.exp(-1j *ref_index* omega * distance)
F_signal_det1 = F_signal * np.exp(-1j *ref_index* omega * distance*1.5)
F_signal_det2 = F_signal * np.exp(-1j *ref_index2* omega * distance*1.5)
signal_thru = np.fft.ifft(np.fft.ifftshift(F_signal_thru))
signal_det1 = np.fft.ifft(np.fft.ifftshift(F_signal_det1))
signal_det2 = np.fft.ifft(np.fft.ifftshift(F_signal_det2))


fig, ax = plt.subplots(2,1, figsize=(1.5,1.2), tight_layout=True)

ax[0].plot(t, np.abs(signal_det1)**2, color='k', lw=0.5)
ax[0].plot(t, np.abs(signal_det2)**2, color='k', lw=0.5)
ax[0].fill_between(t, np.minimum(np.abs(signal_det1),np.abs(signal_det2))**2, color='k', lw=0, alpha=0.2)
ax[0].set(xlim=(50,130), ylim=(-0.02,0.02+np.max(np.abs(signal_det2)**2)), yticks=[],xticks=[])
[ax[0].spines[pos].set_visible(False) for pos in ["top","bottom", "right", "left"]]


ax[1].plot(t, np.abs(signal)**2, color=color_a, lw=0.5)
t_left = t[t<0][np.argmin(np.abs(np.abs(signal[t<0])**2-0.5))]
t_right = t[t>0][np.argmin(np.abs(np.abs(signal[t>0])**2-0.5))]
t_FWHM_inc = t_right-t_left
ax[1].plot([t_left,t_right],[0.5,0.5], '.-', color=color_a, alpha=0.5, lw=0.5, ms=1)

ax[1].plot(t, np.abs(signal_thru)**2, color=color_a2, lw=0.5)
t_max = t[np.argmax(np.abs(signal_thru)**2)]
max_val = np.max(np.abs(signal_thru)**2)
t_left = t[t<t_max][np.argmin(np.abs(np.abs(signal_thru[t<t_max])**2-0.5*max_val))]
t_right = t[t>t_max][np.argmin(np.abs(np.abs(signal_thru[t>t_max])**2-0.5*max_val))]
t_FWHM_disp = t_right-t_left
print(t_FWHM_disp/t_FWHM_inc)
ax[1].plot([t_left,t_right],[0.5*max_val,0.5*max_val], '.-', color=color_a2, alpha=0.5, lw=0.5, ms=1)

ax[1].set(xlim=(-60+t_max/2,60+t_max/2), ylim=(-0.02,1.02), xticks=[], yticks=[])
[ax[1].spines[pos].set_visible(False) for pos in ["top","bottom", "right", "left"]]


fig.savefig("Figures/Fig3/fig3_inset.pdf")

# ax[1].plot(omega, np.real(F_signal))
# ax[1].plot(omega, np.abs(F_signal))






#%% Animation_ pulse

for pwidth in [10, 50, 100]:
    for plot_type in ["power", "field"]: 

        fig, ax = plt.subplots(figsize=(2,2.15), tight_layout=True)
        box_plot = td.Box(
            center=(0,0,0),
            size=((dist_wg+l_des)*1.2, (dist_wg+l_des)*1.2, 0),
        )
        eps = np.real(sim.epsilon(box=box_plot,freq=ua.freq0).to_numpy()[:,:,0])

        x = sim.epsilon(box=box_plot,freq=ua.freq0).x
        y = sim.epsilon(box=box_plot,freq=ua.freq0).y
        Y, X = np.meshgrid(y,x)
        level = (eps.max()+eps.min())/2
        ax.contour(X, Y, eps, [level], colors='k' if plot_type=="field" else "w", linewidths=0.5, alpha=0.5)
        ax.set(
            xlim=(-0.6*(dist_wg+l_des),+0.6*(dist_wg+l_des)),
            ylim=(-0.6*(dist_wg+l_des),+0.6*(dist_wg+l_des)),
            xlabel=r'$x$ ($\mu$m)', ylabel=r'$y$ ($\mu$m)'
        )

        rect_x = np.array([-0.5, -0.5, 0.5, 0.5, -0.5]) * l_des
        rect_y = np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * l_des
        ax.plot(-dist_wg/2+rect_x, -dist_wg/2+rect_y, ls='--', color='crimson', alpha=0.5, lw=0.5)
        ax.plot(-dist_wg/2+rect_x, +dist_wg/2+rect_y, ls='--', color='crimson', alpha=0.5, lw=0.5)
        ax.plot(+dist_wg/2+rect_x, +dist_wg/2+rect_y, ls='--', color='crimson', alpha=0.5, lw=0.5)
        ax.plot(+dist_wg/2+rect_x, -dist_wg/2+rect_y, ls='--', color='crimson', alpha=0.5, lw=0.5)

        x, y = sim_data_AB[str(pwidth)]["field_time"].Hz.x, sim_data_AB[str(pwidth)]["field_time"].Hz.y
        t = sim_data_AB[str(pwidth)]["field_time"].Hz.t - 5*pwidth/(2*np.pi*freq0)
        Y, X = np.meshgrid(y,x)
        if plot_type=="field":
            Hz_time = sim_data_AB[str(pwidth)]["field_time"].Hz.data[:,:,0]
            vmax = np.max(np.abs(Hz_time))
            pc = ax.pcolormesh(X, Y, Hz_time[:,:,0], vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
        else:
            Sx_time = (
                sim_data_AB[str(pwidth)]["field_time"].Ey.data
                *sim_data_AB[str(pwidth)]["field_time"].Hz.data 
                - sim_data_AB[str(pwidth)]["field_time"].Ez.data
                *sim_data_AB[str(pwidth)]["field_time"].Hy.data
            )[:,:,0]
            Sy_time = (
                sim_data_AB[str(pwidth)]["field_time"].Ez.data
                *sim_data_AB[str(pwidth)]["field_time"].Hx.data
                - sim_data_AB[str(pwidth)]["field_time"].Ex.data
                *sim_data_AB[str(pwidth)]["field_time"].Hz.data
            )[:,:,0]
            S_time = np.sqrt(Sx_time**2 + Sy_time**2)
            vmax = np.max(S_time)
            pc = ax.pcolormesh(X, Y, S_time[:,:,0], vmin=0, vmax=vmax, cmap=plt.cm.inferno)


        # ax.set_title("tet", fontsize=7)

        def update(frame):
            if plot_type=="field":
                pc.set_array(Hz_time[:,:,frame])
            else:
                pc.set_array(S_time[:,:,frame])
            ax.set_title(r"$t = {}".format(round(t[frame].data/1e-12, 2))+"$ ps", fontsize=7)
            return pc

        fps = 100
        ani = animation.FuncAnimation(
            fig=fig, func=update, 
            frames=np.arange(0, len(t), 1), 
            interval=1000/fps
        )
        ani.save("Figures/Fig3/anim_{}_pwidth_{}.gif".format(plot_type, pwidth))
