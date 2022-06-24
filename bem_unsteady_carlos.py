# Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.interpolate as sp
import matplotlib
from matplotlib.ticker import MaxNLocator


#matplotlib.use('Qt5Agg')

# BEM
def BEM(TSR, dr, r_R_curr, c_curr, area_curr, twist_curr, add_Prandtl_correction, r_Rtip, r_Rroot, tol,
        max_n_iterations, Uinf, Radius, Omega, alpha, CL, CD, blades):
    """
        Solves the flow around an annulus of the actuator disk using BEM theory

        Parameters
        ----------
        TSR : integer
            tip speed ratio
        dr : float
            width of the annulus
        r_R_curr : float
            adimensionalized radial"(middle) position of the annulus
        c_curr : float
            chord length of middle of the annulus [m]
        area_curr : float
            area of the current annulus
        twist_curr : float
            twist at the middle of the current annulus
        add_Prandtl_correction : BOOLEAN
            parameter which tells whether prandtl corrections should be applied
        r_Rtip : float
            adimensionalized position of the tip of the turbine blade
        r_Rroot : float
            adimensionalized position of the root of the turbine blade
        tol : float
            accepted difference between the values of the axial induction factor of 2 consecutive iterations
        max_n_iterations : integer
            maximum number of iterations taken to calculate the streamtube loading
        Uinf : integer
            freestream velocity [m/s]
        Radius : float
            Radius of the wind turbine [m]
        Omega : float
            Rotational velocity of the wind turbine [m/s]
        alpha : float
            sample values of the AoA for the provided airfoil polar [deg]
        CL :  float
            Lift coefficient values from the provided airfoil polar
        CD : float
            Drag coefficient values from the provided airfoil polar
        blades : integer
            Total number of blades of the wind turbine

        Returns
        -------
        1D array containing:
        a_curr_cor = axial induction factor of the streamtube
        a_prime_curr = tangential induction factor of the streamtube
        r_R_curr = adimensionalized radial position of the annulus
        Faxial = axial force acting on the wind turbine [N]
        Fazim = tangential force acting on the wind turbine
        gamma_curr = bound circulation produced of the streamtube [m^2/s]
        CT = thrust coefficient of the streamtube
        area_curr = area of the streamtube at the actuator disk location
        alpha_distrib = angle of attack of the middle point of the streamtube [deg]
        inflowangle_distrib = inflow angle of the middle point of the streamtube
    """
    r_curr = r_R_curr * Radius
    a_curr = 0.3  # seeding value of the axial induction factor
    a_curr_cor = 1  # initial guess for the axial factor at the new iteration
    a_prime_curr = 0  # seeding value of the tangential induction factor
    current_iter = 1  # integer value counting the number of iterations undertaken
    # by the BEM code
    results_forces = np.zeros(8)

    while (np.abs(a_curr_cor - a_curr) > tol and current_iter < max_n_iterations):
        if current_iter > 1:
            a_curr = 0.25 * a_curr_cor + 0.75 * a_curr
        # if the iterations went over the initial guess, use a weighted average
        # of the previous 2 iterations to obtain a new guess value

        # Compute flow properties and forces for each annulus

        # Determine the local velocity components and the effective AoA of each annulus
        Uaxial = Uinf * (1 - a_curr)  # Axial flow
        Utan = (1 + a_prime_curr) * Omega * r_curr  # Tangential flow
        V = np.sqrt(Uaxial ** 2 + Utan ** 2)  # Effective velocity
        phi = np.arctan2(Uaxial, Utan) * 180 / np.pi  # Inflow angle
        AoA = phi - twist_curr  # AoA in degrees
        # Determine forces on each blade element, given by the AoA
        Cl = np.interp(AoA, alpha, CL)  # Interpolate lift coefficient
        Cd = np.interp(AoA, alpha, CD)  # Interpolate drag coefficient
        Lift = 0.5 * c_curr * V ** 2 * Cl  # lift force at each annulus, taken in the local reference system with
        # respect to the effective velocity
        Drag = 0.5 * c_curr * V ** 2 * Cd  # drag force at each annulus, taken in the local reference system with
        # respect to the effective velocity
        Fazim = Lift * np.sin(phi * np.pi / 180) - Drag * np.cos(phi * np.pi / 180)
        # azimuthal force at each annulus, taken  in a reference system aligned with the axis of rotation of the rotor
        Faxial = Lift * np.cos(phi * np.pi / 180) + Drag * np.sin(phi * np.pi / 180)
        # axial force at each annulus, taken in a reference system aligned with the axis of rotation of the rotor
        CT = Faxial * blades * dr * Radius / (0.5 * Uinf ** 2 * area_curr)
        # thrust coefficient at each annulus, corresponding to the approximated aerodynamic load
        a_prime_curr = Fazim * blades / (2 * np.pi * Uinf * (1 - a_curr) * Omega * 2 * (r_R_curr * Radius) ** 2)
        gamma_curr = 0.5 * V * Cl * c_curr
        """
        Glauert correction for heavily loaded rotors is implemented and computed
        """
        # From thrust coefficient, determine axial induction factor, using Glauert correction
        CT1 = 1.816
        CT2 = 2 * np.sqrt(CT1) - CT1
        if CT < CT2:
            a_curr_cor = 0.5 - np.sqrt(1 - CT) / 2
        elif CT >= CT2:
            a_curr_cor = 1 + (CT - CT1) / (4 * np.sqrt(CT1) - 4)

        """
        Since there are a finite number of blades instead of an idealised actuator disk,
        Prandtl correction is applied.
        """
        if add_Prandtl_correction == True:
            # determine values of the correction at the center of each annulus
            arg_tip = -blades / 2 * (r_Rtip - r_R_curr) / r_R_curr * np.sqrt(
                1 + ((TSR * r_R_curr) ** 2) / ((1 - a_curr_cor) ** 2))
            Ftip = 2 / np.pi * np.arccos(np.exp(arg_tip))
            arg_root = blades / 2 * (r_Rroot - r_R_curr) / r_R_curr * np.sqrt(
                1 + ((TSR * r_R_curr) ** 2) / ((1 - a_curr_cor) ** 2))
            Froot = np.array(2 / np.pi * np.arccos(np.exp(arg_root)))
            # bound case of "not a number"
            if np.isnan(Froot):
                Froot = 0
            elif np.isnan(Ftip):
                Ftip = 0
            # combine the corections for the tip and the root
            Ftotal = Ftip * Froot

            # bound case of "divide by zero"
            if Ftotal < 0.0001:
                Ftotal = 0.0001

            a_curr_cor = a_curr_cor / Ftotal  # Correct axial induction factor
            a_prime_curr = a_prime_curr / Ftotal  # Correct tangential induction factor
        # writing the results
        CT = Faxial * blades * dr * Radius / (0.5 * Uinf ** 2 * area_curr)
        Cn = Fazim * blades * dr * Radius / (0.5 * Uinf ** 2 * area_curr)
        Cq = Fazim * blades * r_curr * dr * Radius / (0.5 * Uinf ** 2 * area_curr * c_curr)
        alpha_distrib = AoA
        inflowangle_distrib = phi
        current_iter = current_iter + 1
    results_BEM = [a_curr_cor, a_prime_curr, r_R_curr, Faxial, Fazim, gamma_curr, CT, area_curr, alpha_distrib,
                   inflowangle_distrib, Cn, Cq]
    return results_BEM

def tip_root_correct(no, a_in):
    arg_tip = -blades / 2 * (r_Rtip - r_R[no]) / r_R[no] * np.sqrt(1 + ((TSR * r_R[no]) ** 2) / ((1 - a_in) ** 2))
    Ftip = 2 / np.pi * np.arccos(np.exp(arg_tip))
    arg_root = blades / 2 * (r_Rroot - r_R[no]) / r_R[no] * np.sqrt(1 + ((TSR * r_R[no]) ** 2) / ((1 - a_in) ** 2))
    Froot = np.array(2 / np.pi * np.arccos(np.exp(arg_root)))
    # bound case of "not a number"
    if np.isnan(Froot):
        Froot = 0
    elif np.isnan(Ftip):
        Ftip = 0
    # combine the corections for the tip and the root
    Ftotal = Ftip * Froot

    # bound case of "divide by zero"
    if Ftotal < 0.0001:
        Ftotal = 0.0001

    a = a_in / Ftotal  # Correct axial induction factor

    return a, Ftotal

def time2semichord(time):
    return 2*Uinf*time/chord

def semichord2time(s):
    return s/2/Uinf*chord

# define properties of the system
dt=.1
time =np.arange(0,500,dt)
Uinf=1
TSR = 8
dr = 1/50
r_R = 0.3
pitch = -2
Uinf = 10
k_red = 0.3
Radius = 50
twist = 14 * (1 - r_R) + pitch
chord = 3 * (1 - r_R) + 1
Area = np.pi * ((( r_R + dr/2)* Radius) ** 2 - ((r_R-dr/2) * Radius) ** 2)  # Area of each streamtube
add_Prandtl_correction = True
Radius = 50  # Radius in meters (from requirements)
r_Rroot = 0.2  # Scaled location of root of blade
r_Rtip = 1  # Scaled location of tip of blade
blades = 3  # Number of blades
rho = 1  # density of air
N = 100  # Number of annuli

# properties of the airfoil
dCn_dalpha=2*np.pi # lift slope
alpha0=-1.945/180*np.pi # alpha for which normal load is zero in steady flow

# Import lift and drag polars for the DU_airfoil, used for the wind turbine case
data = np.genfromtxt("DU_airfoil.txt", delimiter=",")
alpha = data[:, 0]  # Angle of attack in degrees
CL = data[:, 1]  # Lift coefficient polar
CD = data[:, 2]  # Drag coefficient polar

# # pitching motion of the airfoil
# k=.3 # reduced frequency of the pitching motion
# omega=k*2*chord/Uinf # frequency of the piching motion
# Amplitude_alpha=10/180*np.pi # amplitude of the pitching motion
# alpha_t0=15/180*np.pi # alpha at time=0
# alpha=Amplitude_alpha*np.sin(omega*time)+alpha_t0 # calculate alpha
# dalpha_dt=np.gradient(alpha,time) # calculate the time derivative of alpha

max_n_iterations = 1000
tol = 1e-3
U1 = Uinf*1
DeltaU = 0.5
te = 100
Nt = 500
t_array = np.linspace(0,te,Nt)

Omega = k_red * Uinf / Radius
Uinf = U1 + DeltaU*np.cos(Omega*t_array)
alphaqs = np.zeros(len(t_array))
for j, t in enumerate(t_array):
    Uinf_val = Uinf[j]
    Omega = k_red * Uinf_val / Radius
    results_BEM = BEM(TSR, dr, r_R, chord, Area, twist, add_Prandtl_correction, r_Rtip, r_Rroot, tol,
                    max_n_iterations, Uinf_val, Radius, Omega, alpha, CL, CD, blades)
    ## note that the last one will not get overwritten, might cause problems
    alphaqs[j] = results_BEM[8]

print(alpha.shape)
alpha_x = np.linspace(0,len(alpha), len(alpha))
alpha = np.interp(t_array, alpha_x, alpha)
print(alpha.shape)
print(alphaqs.shape)
# define the array semi-chord time scale
sarray = time2semichord(t_array)

# # calculate quasi-steady alpha
# alpha0=-1.945/180*np.pi # alpha for which normal load is zero in steady flow
# alphaqs = alpha + dalpha_dt*(chord/2)/Uinf #- dhplg_dt/Uinf
# dalphaqs_dt=np.gradient(alphaqs,time) # calculate the time derivative of the quasi-steady alpha
#
# # calculate the coefficient of normal force assuming quasi-steady flow asuming potential flow
# Cnormal_quasisteady = 2*np.pi*(alphaqs-alpha0)

# we plot the effective quasi-steady angle of attack \alpha_{qs}

# plot figure
plt.rcParams.update({'font.size': 14}) #, 'figure.dpi':150, 'savefig.dpi':150})
plt.rcParams["font.family"] = "serif" # define font
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # define font
cmap = plt.get_cmap('BuGn')  # define colormap
fig,ax = plt.subplots(figsize=[6,6]) # define pointers for the figure and axes
ax.plot(alpha*180/np.pi, alphaqs*180/np.pi,color='black', linewidth=1) # plot equivalent quasi-steady angle of attack
ax.set_xlabel(r'$\alpha (^\circ)$') # set x-label
ax.set_ylabel(r'$\alpha_{qs} (^\circ)$') # set y-label
# add arrows to indicate the direction of the cycle
parr1=ax.annotate('', xy=(17.5, 20), xytext=(10,12.5),
            arrowprops=dict(color='black', shrink=0.05, width=.5, headwidth=3,headlength=4, linewidth=.2))
parr1=ax.annotate('', xy=(10, 7.5), xytext=(17.7,15),
            arrowprops=dict(color='black', shrink=0.05, width=.5, headwidth=3,headlength=4, linewidth=.2))
plt.grid() # add a grid
ax.set_xlim(0,30) # define limits of the axis
ax.set_ylim(0,30) # define limits of the axis
plt.tight_layout() # all elements of figure inside plot area
plt.show() # show figure