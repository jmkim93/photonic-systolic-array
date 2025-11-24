#%%
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from matplotlib.patches import Rectangle
import plot_setting
import pickle
import defineAdjointOptimization as ua
from defineAdjointOptimization import l_des, pixel_size, nx, ny, w_wg, h_wg, radius, dist_wg, val_grad_fn, min_step_per_wvl, freq0
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.adjoint.utils.filter import BinaryProjector


module_list = ["cross", "BS", "1", "2", "3", "4"]
task_id_list = [
    "fdve-475d065a-4107-4b95-90dd-4473e0360a3f",
    "fdve-5dfbccaa-412c-468a-bf8d-b17a355bfa9e",
    "fdve-d7e618fe-dd3a-4fee-b33b-4a73b46200cd",
    "fdve-591832b6-d47b-4fab-8ca0-babc2596a669",
    "fdve-d7be5852-7d44-4e8e-b5e6-64280bfeb763",
    "fdve-9f4d1b0d-5d22-4531-ba24-41e1f557faee", ## set3

]
task_id_list_eroded = [
    "fdve-976ce641-dbbc-47d8-bb56-b1f6745e0a05",
    "fdve-cc5fb80a-1f78-4897-a8b6-5cf47534631a",
    "fdve-2a15c5c0-7bb7-4f35-b536-04278c199ae3",
    "fdve-6a49cc2e-7cf3-4b9a-8e4c-d66e436b7531",
    "fdve-5f192aa3-3b45-44ba-90f1-75fae73b8ac1",
    "fdve-65881410-069a-418c-b5d8-18df4e474ffb"
]
task_id_list_dilated = [
    "fdve-68f68105-68f7-4b01-af64-ebd4ed440a73",
    "fdve-4ee5453e-e520-4f7f-a88c-64fc9b75553c",
    "fdve-1db68f5b-0514-47ed-bb49-e5122e5b974e",
    "fdve-8cf98526-5e92-483c-b027-27a6029d5e23",
    "fdve-08e33cc2-e18a-47c9-bd8e-7c1fd6fa2ac1",
    "fdve-87c4d82e-58ca-4b8d-b1dc-d25fee1c5203"
]

train_data = dict()

sim_dict = dict()
sim_dict_eroded = dict()
sim_dict_dilated = dict()


sim_data_dict = dict()
sim_data_dict_eroded = dict()
sim_data_dict_dilated = dict()

for ii, mod in enumerate(module_list):
    with open("Data_Fig2/set3/PE_module_{}.pickle".format(mod), 'rb') as handle:
        train_data[mod] = pickle.load(handle)
    coeff = np.array(train_data[mod]["coeff"])
    r, theta = np.abs(coeff), np.angle(coeff)
    coeff_rel = coeff[:,0]/coeff[:,1]
    power = r**2
    opt_state_index = (200+np.argmin(np.abs(coeff_rel[200:]-1j)) if mod=='BS' 
                       else np.argmin(np.array(train_data[mod]["obj_tot"])))    
    # print(opt_state_index)
    params_opt = train_data[mod]["params"][opt_state_index]
    beta_opt = train_data[mod]["beta"][opt_state_index]
    sym1 = (mod=="cross" or mod=="BS")
    sym2 = (mod=="cross" or mod=="BS" or mod=="1")
    sim = ua.get_simulation(
        symmetric=sym1,
        symmetric2=sym2,
        params=params_opt, beta=beta_opt,
        waveguide="+" if sym1 else ("L" if sym2 else "T"),
    ).to_simulation_fwd()[0]

    sim_eroded = ua.get_simulation(
        symmetric=sym1,
        symmetric2=sym2,
        params=params_opt, beta=beta_opt, eta=0.52,
        waveguide="+" if sym1 else ("L" if sym2 else "T"),
    ).to_simulation_fwd()[0]

    sim_dilated = ua.get_simulation(
        symmetric=sym1,
        symmetric2=sym2,
        params=params_opt, beta=beta_opt, eta=0.48,
        waveguide="+" if sym1 else ("L" if sym2 else "T"),
    ).to_simulation_fwd()[0]

    freq_measure = freq0 * np.linspace(0.95, 1.05, 51)
    src = td.ModeSource(
        size=ua.plane_in.size,
        center=ua.plane_in.center,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=ua.fwidth),
        mode_spec=ua.mode_spec,
        mode_index=0,
        direction="+",
        num_freqs=9,
    )
    mode_solver = ModeSolver(
        simulation=sim,
        plane=ua.plane_in,
        mode_spec=ua.mode_spec,
        freqs=freq_measure,
    )
    mnt1 = mode_solver.to_monitor(
        name="MNT_MODE1",
    )
    mnt1 = mnt1.updated_copy(
        mode_spec=ua.mode_spec,
        center=(ua.x_mnt1,ua.y_mnt1,0)
    )
    mnt2 = mnt1.updated_copy(
        mode_spec=ua.mode_spec,
        size=(0, ua.w_mode, ua.Lz or 1.0),
        center=(ua.x_mnt2,ua.y_mnt2,0),
        name="MNT_MODE2",
    )
    mnt3 = mnt1.updated_copy(
        mode_spec=ua.mode_spec,
        center=(ua.x_mnt3,ua.y_mnt3,0),
        name="MNT_MODE3",
    )
    mnt_field = td.FieldMonitor(
        size=(td.inf,td.inf, 0),
        center=(0,0,0),
        freqs=[freq0],
        name="field"
    )
    sim = sim.updated_copy(
        sources=[src],
        monitors=[mnt_field, mnt1, mnt2, mnt3]
    )

    sim_eroded = sim_eroded.updated_copy(
        sources=[src],
        monitors=[mnt_field, mnt1, mnt2, mnt3]
    )
    sim_dilated = sim_dilated.updated_copy(
        sources=[src],
        monitors=[mnt_field, mnt1, mnt2, mnt3]
    )

    sim_dict[mod] = sim
    sim_dict_eroded[mod] = sim_eroded
    sim_dict_dilated[mod] = sim_dilated

    # sim_data = td.web.run(
    #     sim, 
    #     folder_name="Fig2_set3_freq_analysis", 
    #     task_name=mod, 
    #     verbose=True
    # )
    # sim_data_dict[mod] = sim_data
    sim_data_dict[mod] = td.web.api.webapi.load(task_id=task_id_list[ii])


    # sim_data_eroded = td.web.run(
    #     sim_eroded, 
    #     folder_name="Fig2_set3_freq_analysis_eroded", 
    #     task_name=mod, 
    #     verbose=True
    # )
    # sim_data_dict_eroded[mod] = sim_data_eroded
    sim_data_dict_eroded[mod] = td.web.api.webapi.load(task_id=task_id_list_eroded[ii])

    # sim_data_dilated = td.web.run(
    #     sim_dilated, 
    #     folder_name="Fig2_set3_freq_analysis_dilated", 
    #     task_name=mod, 
    #     verbose=True
    # )
    # sim_data_dict_dilated[mod] = sim_data_dilated
    sim_data_dict_dilated[mod] = td.web.api.webapi.load(task_id=task_id_list_dilated[ii])

#%% Supp fig: Design schematic

fig, ax = plt.subplots(1,2, figsize=(7,3.5), tight_layout=True)
sim_dict["cross"].plot(z=0.001, ax=ax[0])
sim_dict["cross"].plot(x=0, ax=ax[1])

ax[0].set(
    xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)",
    xticks=np.linspace(-3,3,7), yticks=np.linspace(-3,3,7)
)
ax[1].set(
    xlabel=r"$y$ ($\mu$m)", ylabel=r"$z$ ($\mu$m)",
    xticks=np.linspace(-3,3,7),
    yticks=np.linspace(-1.5,1.5,7),
)

ax[0].plot([-l_des/2,l_des/2], [-l_des/2,l_des/2], 'k--', lw=0.75)
ax[0].plot([-l_des/2,l_des/2], [l_des/2,-l_des/2], 'k--', lw=0.75)


ax[0].set_title("Cross section, $z=0$")
ax[1].set_title("Cross section, $x=0$")

ax[0].text(-2.7,2, r"$\mathrm{SiO}_2$ background",horizontalalignment='left', verticalalignment='center',)
ax[0].text(-2.7,0, r"Si wg", color="royalblue", horizontalalignment='left',verticalalignment='center', )
ax[0].text(0,0, r"Design region", color="crimson", horizontalalignment='center',verticalalignment='center',)
ax[0].text(-1.8,-2.2, r"Mode source", color="green", horizontalalignment='center',verticalalignment='center', )
ax[0].text(-1.8,-2.7, r"Mode monitors", color="orange", horizontalalignment='center',verticalalignment='center', )

ax[0].text(0, 1.2, r"Symmetry ($y=\pm x$)", color="k", horizontalalignment='center',verticalalignment='center', zorder=1000)

ax[1].text(0,1.25, r"PML", color="k", horizontalalignment='center',verticalalignment='center', zorder=1000)
ax[1].text(0,-0.5, r"Symmetry ($z=0$, PMC)", color="k", horizontalalignment='center',verticalalignment='center', zorder=1000)

fig.savefig("Figures/Fig2/SuppFig2_designschematic.pdf")


#%% Plot figures
    
fig, ax = plt.subplots(2,6, figsize=(7.1,2.8), tight_layout=True)
# fig, ax = plt.subplots(3,4, figsize=(6,4.35), tight_layout=True)
# ax = ax.reshape(-1)
fig_fabtol, ax_fabtol = plt.subplots(4,6, figsize=(7.1,4.8), tight_layout=True)

fig_supp = plt.figure(figsize=(7, 8), tight_layout=True)
ax_supp = [fig_supp.add_subplot(4,3,i+1, polar=(i>=6)) for i in range(12)]

for ii, mod in enumerate(module_list):

    coeff = np.array(train_data[mod]["coeff"])
    r, theta = np.abs(coeff), np.angle(coeff)
    coeff_rel = coeff[:,0]/coeff[:,1]
    power = r**2
    opt_state_index = (200+np.argmin(np.abs(coeff_rel[200:]-1j)) if mod=='BS' 
                       else np.argmin(np.array(train_data[mod]["obj_tot"])))    
    print(opt_state_index, power[opt_state_index, :2])
    params_opt = train_data[mod]["params"][opt_state_index]
    beta_opt = train_data[mod]["beta"][opt_state_index]


    ax_supp[ii].add_patch(Rectangle((0, 0), 50, 1, linewidth=0, edgecolor=None, facecolor='k', alpha=0.1))
    ax_supp[ii].plot(r[:,0]**2, color='darkorange', lw=0.75, label=r'port 1')
    ax_supp[ii].plot(r[:,1]**2, color='darkcyan', lw=0.75, label=r'port 2')
    ax_supp[ii].axvline(x=opt_state_index, color='k', lw=0.5, ls='--')
    ax_supp[ii].set(
        xlabel=r'Epoch', 
        xlim=(0, train_data[mod]["iterations"]),
        ylim=(0,1), yticks=[0,0.5, 1], 
        yticklabels=[0, 0.5, 1] if ii==0 or ii==3 else []
    )
    
    axt = ax_supp[ii].twinx()
    axt.semilogy(np.array(train_data[mod]["obj_tot"]), 'crimson', lw=0.75, alpha=0.75)
    axt.set(ylim=(5e-6,5e0), yticks=(1e-4, 1e-2, 1e0), yticklabels=["10$^{-4}$", "10$^{-2}$", "10$^{0}$"] if ii==2 or ii==5 else [])
    if ii==2 or ii==5:
        axt.set_ylabel("Objective function, $L$", color='crimson')
    if ii==0 or ii==3:
        ax_supp[ii].set_ylabel(r"Transmission, $|S|^2$")
    if ii==0:
        ax_supp[ii].legend(frameon=False, fontsize=7)
    axt.yaxis.label.set_color("crimson")
    axt.spines["right"].set_edgecolor("crimson")
    axt.tick_params(axis='y', colors="crimson")
    
    ax_supp[ii].set_title(
        "Crossing" if ii==0 else ("Beam splitter" if ii==1 else (r"Branch, $n={}$".format(mod))), 
        fontsize=7
    )
    ax_supp[ii+6].set_title(
        "Crossing" if ii==0 else ("Beam splitter" if ii==1 else (r"Branch, $n={}$".format(mod))), 
        fontsize=7
    )
    
    cmap1 = plt.cm.viridis
    cmap2 = plt.cm.cividis
    for i in range(0, len(coeff)):
        if i>0:
            ax_supp[ii+6].plot([theta1_prev, theta[i,0]], [r1_prev, r[i,0]], '-', color=cmap1(i/len(coeff)), lw=0.5, alpha=0.5)
            ax_supp[ii+6].plot([theta2_prev, theta[i,1]], [r2_prev, r[i,1]], '-', color=cmap2(i/len(coeff)), lw=0.5, alpha=0.5)
        ax_supp[ii+6].plot(theta[i,0], r[i,0], 'o', color=cmap1(i/len(coeff)), ms=1)
        ax_supp[ii+6].plot(theta[i,1], r[i,1], 'o', color=cmap2(i/len(coeff)), ms=1)
        r1_prev, theta1_prev = r[i,0], theta[i,0]
        r2_prev, theta2_prev = r[i,1], theta[i,1]
    ax_supp[ii+6].set(ylim=(0,1), yticks=[0,0.5,1], xticks=np.arange(0,2*np.pi,np.pi/6))


    ax[0,ii].set(xlim=(-dist_wg/2,dist_wg/2), ylim=(-dist_wg/2,dist_wg/2))
    ax_fabtol[0,ii].set(xlim=(-0.8,0.8), ylim=(-0.8,0.8))
    ax_fabtol[1,ii].set(xlim=(-0.8,0.8), ylim=(-0.8,0.8))
    ax_fabtol[2,ii].set(xlim=(-0.8,0.8), ylim=(-0.8,0.8))
    if ii==2:
        ax_fabtol[1,ii].plot([-0.3, -0.2],[0,0], lw=1, color='k')
        ax_fabtol[1,ii].text(-0.25, 0.07, "100 nm", fontsize=7, ha="center")
    
    box_plot = td.Box(
        center=(0,0,0),
        size=(dist_wg, dist_wg, 0)
    )
    eps = np.real(sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).to_numpy()[:,:,0])
    x = sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).x
    y = sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).y
    Y, X = np.meshgrid(y,x)
    level = (eps.max()+eps.min())/2
    ax[0,ii].contour(X, Y, eps, [level], colors='k', linewidths=0.4)
    ax_fabtol[1,ii].contour(X, Y, eps, [level], colors='k', linewidths=0.4)
    
    eps = np.real(sim_dict_eroded[mod].epsilon(box=box_plot,freq=ua.freq0).to_numpy()[:,:,0])
    x = sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).x
    y = sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).y
    Y, X = np.meshgrid(y,x)
    level = (eps.max()+eps.min())/2
    ax_fabtol[0,ii].contour(X, Y, eps, [level], colors='k', linewidths=0.4)

    eps = np.real(sim_dict_dilated[mod].epsilon(box=box_plot,freq=ua.freq0).to_numpy()[:,:,0])
    x = sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).x
    y = sim_dict[mod].epsilon(box=box_plot,freq=ua.freq0).y
    Y, X = np.meshgrid(y,x)
    level = (eps.max()+eps.min())/2
    ax_fabtol[2,ii].contour(X, Y, eps, [level], colors='k', linewidths=0.4)


    x, y = sim_data_dict[mod]["field"].Hz.x, sim_data_dict[mod]["field"].Hz.y
    Y, X = np.meshgrid(y,x)
    field = sim_data_dict[mod]["field"].Hz
    vmax = np.abs(field).max()
    pc = ax[0,ii].pcolormesh(X, Y, np.real(field[:,:,0,0]), cmap=plt.cm.bwr, vmin=-vmax,vmax=vmax, shading='auto', linewidth=0)
    ax[0,ii].set(xticks=[], yticks=[])
    pc = ax_fabtol[1,ii].pcolormesh(X, Y, np.real(field[:,:,0,0]), cmap=plt.cm.bwr, vmin=-vmax,vmax=vmax, shading='auto', linewidth=0)
    ax_fabtol[1,ii].set(xticks=[], yticks=[])

    field = sim_data_dict_eroded[mod]["field"].Hz
    vmax = np.abs(field).max()
    pc = ax_fabtol[0,ii].pcolormesh(X, Y, np.real(field[:,:,0,0]), cmap=plt.cm.bwr, vmin=-vmax,vmax=vmax, shading='auto', linewidth=0)
    ax_fabtol[0,ii].set(xticks=[], yticks=[])

    field = sim_data_dict_dilated[mod]["field"].Hz
    vmax = np.abs(field).max()
    pc = ax_fabtol[2,ii].pcolormesh(X, Y, np.real(field[:,:,0,0]), cmap=plt.cm.bwr, vmin=-vmax,vmax=vmax, shading='auto', linewidth=0)
    ax_fabtol[2,ii].set(xticks=[], yticks=[])

    ax[0,ii].plot(l_des/2*np.array([-1,-1,1,1,-1]), l_des/2*np.array([-1,1,1,-1,-1]), '--', lw=0.4, color='crimson')
    ax_fabtol[0,ii].plot(l_des/2*np.array([-1,-1,1,1,-1]), l_des/2*np.array([-1,1,1,-1,-1]), '--', lw=0.4, color='crimson')
    ax_fabtol[1,ii].plot(l_des/2*np.array([-1,-1,1,1,-1]), l_des/2*np.array([-1,1,1,-1,-1]), '--', lw=0.4, color='crimson')
    ax_fabtol[2,ii].plot(l_des/2*np.array([-1,-1,1,1,-1]), l_des/2*np.array([-1,1,1,-1,-1]), '--', lw=0.4, color='crimson')
    
    if ii==1:
        center = (-0.43,-0.08)
        xcoord = np.array([-0.5,-0.5,0.5,0.5,-0.5])/5
        ycoord = np.array([-0.5,0.5,0.5,-0.5,-0.5])/5
        [ax_fabtol[jj,ii].plot(center[0]+xcoord, center[1]+ycoord, lw=0.75, color="cyan") for jj in range(3)]
    
    if ii==3:
        center = (0.24,-0.52)
        xcoord = np.array([-0.5,-0.5,0.5,0.5,-0.5])/5
        ycoord = np.array([-0.5,0.5,0.5,-0.5,-0.5])/5
        [ax_fabtol[jj,ii].plot(center[0]+xcoord, center[1]+ycoord, lw=0.75, color="cyan") for jj in range(3)]


    if ii==0:
        axins = ax[0,ii].inset_axes([-0.34, 0.1, 0.05, 0.8])
        cb = plt.colorbar(pc, orientation='vertical', cax=axins, extend=None)
        cb.ax.set_yticks([-vmax, 0, vmax])
        cb.ax.set_yticklabels([-1, 0, 1])
        cb.ax.set_ylabel(r'$\mathrm{Re}(H_z)$ (Arb. U.)')

        axins = ax_fabtol[1,ii].inset_axes([-0.55, 0.1, 0.05, 0.8])
        cb = plt.colorbar(pc, orientation='vertical', cax=axins, extend=None)
        cb.ax.set_yticks([-vmax, 0, vmax])
        cb.ax.set_yticklabels([-1, 0, 1])
        cb.ax.set_ylabel(r'$\mathrm{Re}(H_z)$ (Arb. U.)')

    [ax[0,ii].spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]
    [ax_fabtol[0,ii].spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]
    [ax_fabtol[1,ii].spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]
    [ax_fabtol[2,ii].spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]

    # ax[0,ii].plot([-l_des/5, l_des/5], [-dist_wg/2, -dist_wg/2], 'indigo', lw=2)
    # if ii != 2:
    #     ax[0,ii].plot([-l_des/5, l_des/5], [dist_wg/2, dist_wg/2], 'darkorange', lw=2)
    # ax[0,ii].plot([dist_wg/2, dist_wg/2], [-l_des/5, l_des/5], 'darkcyan', lw=2)

    coeff = []
    for port in range(1,4):
        coeff.append(sim_data_dict[mod]["MNT_MODE{}".format(port)].amps.sel(mode_index=0, direction="+" if port<=2 else "-").values)
    coeff = np.array(coeff)
    ax[1,ii].plot(freq_measure/freq0, np.abs(coeff[0])**2, color='darkorange', lw=0.75, label=r'port 1')
    ax[1,ii].plot(freq_measure/freq0, np.abs(coeff[1])**2, color='darkcyan', lw=0.75, label=r'port 2')
    ax_fabtol[3,ii].plot(freq_measure/freq0, np.abs(coeff[0])**2, color='darkorange', lw=0.75, label=r'normal')
    ax_fabtol[3,ii].plot(freq_measure/freq0, np.abs(coeff[1])**2, color='darkcyan', lw=0.75)

    coeff = []
    for port in range(1,4):
        coeff.append(sim_data_dict_eroded[mod]["MNT_MODE{}".format(port)].amps.sel(mode_index=0, direction="+" if port<=2 else "-").values)
    coeff = np.array(coeff)
    ax_fabtol[3,ii].plot(freq_measure/freq0, np.abs(coeff[0])**2, color='darkorange', lw=0.75, ls=':', label=r'eroded')
    ax_fabtol[3,ii].plot(freq_measure/freq0, np.abs(coeff[1])**2, color='darkcyan', lw=0.75, ls=':')

    coeff = []
    for port in range(1,4):
        coeff.append(sim_data_dict_dilated[mod]["MNT_MODE{}".format(port)].amps.sel(mode_index=0, direction="+" if port<=2 else "-").values)
    coeff = np.array(coeff)
    ax_fabtol[3,ii].plot(freq_measure/freq0, np.abs(coeff[0])**2, color='darkorange', lw=0.75, ls='--', label=r'dilated')
    ax_fabtol[3,ii].plot(freq_measure/freq0, np.abs(coeff[1])**2, color='darkcyan', lw=0.75, ls='--')

    phasefactor = np.exp(1j * train_data[mod]["n_eff"][0]* (2*np.pi/1.55)*dist_wg )
    if mod=="BS":
        ax[1,ii].axhline(y=0.5, lw=0.5, ls='--', color='k', alpha=0.75)
        ax_supp[ii].axhline(y=0.5, lw=0.75, ls='--', color='k')
        ax[0,ii].set_title("Beam splitter", fontsize=7)
        ax_fabtol[0,ii].set_title("Beam splitter", fontsize=7)
        IL = 1 - (np.abs(coeff[:2,])**2).sum(axis=0)
        ax_supp[ii+6].plot(np.angle(phasefactor), np.sqrt(0.5), 's', color='darkorange', mfc='None', mew=0.9)
        ax_supp[ii+6].plot(np.angle(phasefactor)-np.pi/2, np.sqrt(0.5), 's', color='darkcyan', mfc='None', mew=0.9)
    elif mod in [ "1", "2", "3", "4"]:
        n = int(mod)
        # cross_efficiency = 0.99064654
        # split_efficiency = 0.96
        cross_efficiency = 0.9938450312283263
        split_efficiency = 0.98
        efficiency_tot = cross_efficiency*split_efficiency
        split = efficiency_tot**(n-1) * (1-efficiency_tot) / (1-efficiency_tot**n)
        ax[1,ii].axhline(y=split_efficiency*(1-split), lw=0.5, ls='--', color='darkorange', alpha=0.75)
        ax[1,ii].axhline(y=split_efficiency*split, lw=0.5, ls='--', color='darkcyan', alpha=0.75)
        ax_supp[ii].axhline(y=split_efficiency*(1-split), lw=0.75, ls='--', color='darkorange')
        ax_supp[ii].axhline(y=split_efficiency*split, lw=0.75, ls='--', color='darkcyan')
        ax_supp[ii+6].plot(np.angle(phasefactor), np.sqrt(split_efficiency*(1-split)), 's', color='darkorange', mfc='None', mew=0.9)
        ax_supp[ii+6].plot(np.angle(phasefactor)+np.pi, np.sqrt(split_efficiency*(split)), 's',color='darkcyan', mfc='None', mew=0.9)
        ax[0,ii].set_title("Branch, $n={}$".format(mod), fontsize=7)
        ax_fabtol[0,ii].set_title("Branch, $n={}$".format(mod), fontsize=7)
        IL = 1- (np.abs(coeff[:2,])**2).sum(axis=0)
    else:
        ax[0,ii].set_title("Crossing", fontsize=7)
        ax_fabtol[0,ii].set_title("Crossing", fontsize=7)
        IL = 1 - (np.abs(coeff[:1,])**2).sum(axis=0)
        ax_supp[ii+6].plot(np.angle(phasefactor), 1, 's', color='darkorange', mfc='None', mew=0.9)
        ax_supp[ii+6].plot(np.angle(phasefactor), 0, 's', color='darkcyan', mfc='None', mew=0.9)

    if ii==5:
        ax[1,ii].legend(frameon=False, fontsize=6)
        ax_fabtol[3,ii].legend(frameon=False, fontsize=6)

    ax[1,ii].axvline(x=1.0, lw=0.5, ls='--', color='k')
    ax[1,ii].set(
        xlabel=r'Frequency, $f/f_0$',
        xlim=(0.95, 1.05), ylim=(0,1),
        xticks=(0.96 ,1, 1.04), 
        yticks=np.linspace(0,1,5), yticklabels=[0, "", 0.5, "", 1] if ii==0 else []
    )
    ax_fabtol[3,ii].axvline(x=1.0, lw=0.5, ls='--', color='k')
    ax_fabtol[3,ii].set(
        xlabel=r'Frequency, $f/f_0$',
        xlim=(0.95, 1.05), ylim=(0,1),
        xticks=(0.96 ,1, 1.04), 
        yticks=np.linspace(0,1,5), yticklabels=[0, "", 0.5, "", 1] if ii==0 else []
    )

ax[1,0].set(ylabel=r"Transmission, $|S|^2$")
ax_fabtol[3,0].set(ylabel=r"Transmission, $|S|^2$")

ax_fabtol[0,0].set(ylabel=r"Eroded")
ax_fabtol[1,0].set(ylabel=r"Normal")
ax_fabtol[2,0].set(ylabel=r"Dilated")

fig.savefig("Figures/Fig2/fig2.pdf")
fig_supp.savefig("Figures/Fig2/SuppFig2_training.pdf")

fig_fabtol.savefig("Figures/Fig2/SuppFig2_fabtolerance.pdf")


#%% Adjoint schematic

taskid = "fdve-bcd36d19-f081-46bb-9d84-3b76234c7c00"
simdata = td.web.api.webapi.load(task_id=taskid)

param_0 = train_data["cross"]["params"][0]

bn = BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=5)
eps_0 = bn.evaluate(ua.smoothen.evaluate(param_0))

fig, ax = plt.subplots(figsize=(1,1))
ax.matshow(param_0.T, cmap=plt.cm.Greys, vmin=0, vmax=1)
ax.set(xticks=[], yticks=[])
[ax.spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]
fig.savefig("Figures/Fig2/param.pdf")

fig, ax = plt.subplots(figsize=(1,1))
ax.matshow(eps_0.T, cmap=plt.cm.Greys, vmin=0, vmax=1)
ax.set(xticks=[], yticks=[])
[ax.spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]
fig.savefig("Figures/Fig2/eps.pdf")

fig, ax = plt.subplots(figsize=(1,1))
ax.set(xlim=(-dist_wg/2,dist_wg/2), ylim=(-dist_wg/2,dist_wg/2))
x, y = np.meshgrid(
    np.arange(-l_des/2, l_des/2, 0.01)+0.005,
    np.arange(-l_des/2, l_des/2, 0.01)+0.005
)
ax.contour(x, y, eps_0.T, levels=[0.5], colors='k', linewidths=0.4)
ax.contour(x, y, eps_0.T, levels=[0.05], colors='gray', linewidths=0.4)
ax.contour(x, y, eps_0.T, levels=[0.95], colors='gray', linewidths=0.4)

ax.plot([-dist_wg/2, -l_des/2], [w_wg/2,w_wg/2], 'k', lw=0.4)
ax.plot([-dist_wg/2, -l_des/2], [-w_wg/2,-w_wg/2], 'k', lw=0.4)
ax.plot([dist_wg/2, l_des/2], [w_wg/2,w_wg/2], 'k', lw=0.4)
ax.plot([dist_wg/2, l_des/2], [-w_wg/2,-w_wg/2], 'k', lw=0.4)
ax.plot([w_wg/2,w_wg/2], [-dist_wg/2, -l_des/2],'k', lw=0.4)
ax.plot([-w_wg/2,-w_wg/2], [-dist_wg/2, -l_des/2], 'k', lw=0.4)
ax.plot([w_wg/2,w_wg/2], [dist_wg/2, l_des/2], 'k', lw=0.4)
ax.plot([-w_wg/2,-w_wg/2], [dist_wg/2, l_des/2], 'k', lw=0.4)
x, y = simdata["field"].Hz.x, simdata["field"].Hz.y
Y, X = np.meshgrid(y,x)
field = simdata["field"].Hz
vmax = np.abs(field).max()
ax.pcolormesh(X, Y, np.real(field[:,:,0,0]), cmap=plt.cm.bwr, vmin=-vmax,vmax=vmax, shading='auto',linewidth=0)
ax.set(xticks=[], yticks=[])
ax.plot(l_des/2*np.array([-1,-1,1,1,-1]), l_des/2*np.array([-1,1,1,-1,-1]), '--', lw=0.4, color='crimson')
[ax.spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]
fig.savefig("Figures/Fig2/field.pdf")

print("S-param", train_data["cross"]["coeff"][0])
print("power", np.abs(train_data["cross"]["coeff"][0])**2)
print("angle", np.angle(train_data["cross"]["coeff"][0])*180/np.pi)

#%% evolution - animation
import matplotlib.animation as animation

for ii, mod in enumerate(module_list):
    symmetric = (mod=="cross" or mod=="BS")
    symmetric2 = (mod=="cross" or mod=="BS" or mod=="1")
    fig, ax = plt.subplots(figsize=(1.5,1.5))
    ax.set(xticks=[], yticks=[])
    [ax.spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]

    bn = BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=5)
    im = ax.imshow(
        bn.evaluate(ua.smoothen.evaluate(train_data[mod]["params"][0])).T[::-1], 
        cmap=plt.cm.Greys, vmin=0, vmax=1
    )
    title = ax.text(5,15, r"Iteration = {}".format(0), fontsize=5, color='r')
    def update(idx):
        param = train_data[mod]["params"][idx]
        if symmetric:
            param = (param + param.T)/2
        if symmetric2:
            param = (param + param[::-1,::-1].T)/2
        bn = BinaryProjector(vmin=0,vmax=1, eta=0.5, beta=min(5 + idx*0.5, 30))
        im_data = bn.evaluate(ua.smoothen.evaluate(param)).T[::-1]
        im.set_array(im_data)
        title.set_text(r"Iteration = {}".format(idx))
        return [im]
    fps = 30
    anim = animation.FuncAnimation(
        fig, 
        func=update, 
        frames = train_data[mod]["iterations"],
        interval = 1000/fps, # in ms
    )
    anim.save('Figures/anim_{}_eps.gif'.format(mod), fps=fps, dpi=300)


    fig, ax = plt.subplots(figsize=(1.5,1.5))
    ax.set(xticks=[], yticks=[])
    [ax.spines[pos].set_visible(False) for i in range(2) for pos in ["top","bottom", "right", "left"]]

    im = ax.imshow(
        train_data[mod]["params"][0].T[::-1], 
        cmap=plt.cm.Greys, vmin=0, vmax=1
    )
    title = ax.text(5,15, r"Iteration = {}".format(0), fontsize=5, color='r')
    def update(idx):
        beta = min(5 + idx*0.5, 30)
        im_data = train_data[mod]["params"][idx].T[::-1]
        im.set_array(im_data)
        title.set_text(r"Iteration = {}".format(idx))
        return [im]
    fps = 30
    anim = animation.FuncAnimation(
        fig, 
        func=update, 
        frames = train_data[mod]["iterations"],
        interval = 1000/fps, # in ms
    )
    anim.save('Figures/anim_{}_params.gif'.format(mod), fps=fps, dpi=300)
