# plot figure the change in induction factor $a$  calculated by the Pitt-peters model for a sinusoidal change in
# thrust coefficient
# $C_{T}= C_{T_0}+\Delta C_T \sin\left(\omega t \right)$, with $C_{T_0}=0.5$ and $\Delta C_{T}=0.35$.
# The values are made non-dimensioned by the steady-solution values of $a$
# for the minimum and maximum values of $C_T$. Time $t$ is made non-dimensional by the radius of the actuator $R$ and
# the unperturbed wind speed $U_\infty$, and set to zero at the moment of the step change.

# plot figure
plt.rcParams.update({'font.size': 14})  # define fontsize for the figures
plt.rcParams["font.family"] = "serif"  # a nice font for the figures
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # a nice font for the latex expressions
# cmap = plt.get_cmap('BuGn')
fig1, ax1 = plt.subplots(figsize=[6, 6])  # create figure, axis and define the size of the figure

# plot steady solution of induction as a function of $C_T$
Ctsteady = np.arange(-(Ct0 - deltaCt), -(Ct0 + deltaCt) + .005, .01)  # define an array of $C_T$
asteady = ainduction(Ctsteady)  # calculate steady solution of induction as a function of $C_T$

# we plot the steady solution of induction as a function of $C_T$
ax1.plot(Ctsteady, asteady, label='Steady', color='blue')

# we will now plot the unsteady solution
for j, omegaval in enumerate(omega):
    ind = (-np.floor(2 * np.pi / (omegaval * R / Uinf) / dt) - 1).astype(
        int)  # indices of the last full cycle to only plot 1 cycle
    label1 = r'$\omega \frac{R}{U_\infty}=' + np.str(omegaval) + '$'  # define label for the legend
    # plot unsteady solution
    plt1 = ax1.plot(-Ct[j, ind:], -vind[j, ind:] / Uinf, label=label1, linestyle=(0, (j + 1, j + 1)),
                    linewidth=(6 / (j + 2)))
    color = plt1[0].get_color()

    # we will plot arrows to see the direction of the cycle
    phase_of_cycle = np.mod(time[ind:] * omegaval * R / Uinf,
                            2 * np.pi)  # calculate the phase of the different points of the cycle
    i1 = np.argmin(np.abs(phase_of_cycle - 0)) + j * 30  # index of start of cycle plotted
    i2 = np.argmin(np.abs(phase_of_cycle - np.pi)) + j * 30  # index of 180 degrees
    scale_arrow = .1  # scale od arrow
    dx = -(Ct[j, ind + i1 + 1] - Ct[j, ind + i1])  # dx of arrow
    dy = -(vind[j, ind + i1 + 1] - vind[j, ind + i1]) / Uinf  # dy of arrow
    ax1.arrow(-Ct[j, ind + i1], -vind[j, ind + i1] / Uinf, scale_arrow * dx / np.sqrt(dx ** 2 + dy ** 2),
              scale_arrow * dy / np.sqrt(dx ** 2 + dy ** 2), color=color, width=scale_arrow * .04,
              shape='left')  # plot arrow at 0 degrees of cycle
    dx = -(Ct[j, ind + i2 + 1] - Ct[j, ind + i2])  # dx of arrow
    dy = -(vind[j, ind + i2 + 1] - vind[j, ind + i2]) / Uinf  # dy of arrow
    ax1.arrow(-Ct[j, ind + i2], -vind[j, ind + i2] / Uinf, scale_arrow * dx / np.sqrt(dx ** 2 + dy ** 2),
              scale_arrow * dy / np.sqrt(dx ** 2 + dy ** 2), color=color, width=scale_arrow * .04,
              shape='left')  # plot arrow at 190 degrees of cycle

# define properties of axis, plot grid and show figure
ax1.set_xlabel(r'$C_t$')  # label of the x-axis
ax1.set_ylabel(r'$a$')  # label of the y-axis
ax1.set_xlim(0, 1)  # set limits of x-axis
ax1.set_ylim(0, .35)  # set limits of x-axis
plt.legend(fontsize=12)  # plot the legend, change fontsize to fit better
plt.grid()  # plot grid
plt.show()  # show figure

filename = 'figures_tutorial_dynamic_inflow/sinusoidal_ct_induction_oye'
fig1.savefig(filename + '.svg')  # save figure
fig1.savefig(filename + '.pdf')  # save figure
fig1.savefig(filename + '.png', dpi=300)  # save figure
