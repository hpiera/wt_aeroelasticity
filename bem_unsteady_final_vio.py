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
                   inflowangle_distrib, Cn, Cq,Cl]
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

def get_s_from_time(time, Uinf, chord):
    s = 2*Uinf_steady*time/chord
    return s

def get_time_from_s(s, Uinf, chord):
    time = s*chord/2/Uinf
    return time

# # Determine quasi-steady angle of attack
# def alpha_quasisteady(alpha, chord, Uinf, theta, h, time):
#     alpha_qs = alpha + chord/(2*Uinf) * np.gradient(theta, time) - 1/Uinf * np.gradient(h, time)
#     return alpha_qs

def duhamel_approx(time, Uinf, chord, alphaqs):
    s_array = get_s_from_time(time, Uinf, chord)

    # Define Wagner constants
    A1 = 0.165
    A2 = 0.335
    b1 = 0.0455
    b2 = 0.3

    # Determine lag states
    Xlag = np.zeros(np.shape(time))
    Ylag = np.zeros(np.shape(time))
    for i, s in enumerate(s_array[:-1]):
        ds = s_array[i+1] - s_array[i]
        delta_alpha = alphaqs[i+1] - alphaqs[i]
        Xlag[i+1] = Xlag[i] * np.exp(-b1 * ds) + A1 * delta_alpha * np.exp(-b1*ds/2)
        Ylag[i+1] = Ylag[i] * np.exp(-b2 * ds) + A2 * delta_alpha * np.exp(-b2*ds/2)

    return Xlag, Ylag

def nc_normal_force(dalpha_qs_dt, c, time, Uinf, v_sos = 343):
    dt = time[1] - time[0]
    Ka = 0.75
    Dnc = np.zeros(np.shape(time))
    for i,t in enumerate(time[:-1]):
        Dnc[i+1] = Dnc[i]*np.exp(-v_sos*dt/(Ka*c)) + (dalpha_qs_dt[i+1] - dalpha_qs_dt[i])*np.exp(-v_sos*dt/(2*Ka*c))

    Cn_nc = 4*Ka*c/Uinf*(dalpha_qs_dt - Dnc)

    return Cn_nc

def pressure_lag(Cn_p, time, Uinf, chord):
    s_array = get_s_from_time(time, Uinf, chord)
    Tp = 1.7
    Dpf = np.zeros(np.shape(time))
    for i, s in enumerate(s_array[:-1]):
        ds = s_array[i+1] - s_array[i]
        Dpf[i+1] = Dpf[i] * np.exp(-ds/Tp) + (Cn_p[i+1] - Cn_p[i])*np.exp(-ds/(2*Tp))

    return Dpf

def boundary_layer_lag(F_p, time, Uinf, chord):
    s_array = get_s_from_time(time, Uinf, chord)
    Tf = 3.0
    Dbl = np.zeros(np.shape(time))
    for i, s in enumerate(s_array[:-1]):
        ds = s_array[i+1] - s_array[i]
        Dbl[i + 1] = Dbl[i] * np.exp(-ds / Tf) + (F_p[i + 1] - F_p[i]) * np.exp(-ds / (2 * Tf))

    return Dbl

def leading_edge(Cn_p_prime, da, time, Uinf, chord):
    Cn1 = 1.0093
    s_array = get_s_from_time(time, Uinf, chord)
    tau_v = np.zeros(np.shape(time))
    for i in range(len(s_array)-1):
        ds = s_array[i+1] - s_array[i]
        # I think his code is a bit weird, because it really depends on which reduced frequency you take
        if Cn_p_prime[i+1] > Cn1:
            tau_v[i + 1] = tau_v[i] + 0.45* ds
        else:
            if da[i+1]<0 and tau_v[i]>0:
                tau_v[i + 1] = tau_v[i] + 0.45* ds
            else:
                tau_v[i + 1] = 0
    return tau_v

def vortex_shedding(tau_v, C_v, time, Uinf, chord):
    s_array = get_s_from_time(time, Uinf, chord)
    Tv = 6
    # this value below is really dependent on the frequency you put in, Im not sure what to take
    Tvl = 5
    Cn_v = np.zeros(np.shape(time))
    for i in range(len(s_array)-1):
        ds = s_array[i + 1] - s_array[i]
        if 0.001 < tau_v[i] < Tvl:
            Cn_v[i+1] = Cn_v[i]*np.exp(-ds/Tv) + (C_v[i+1] - C_v[i])*np.exp(-ds/(2*Tv))
        else:
            Cn_v[i+1] = Cn_v[i]*np.exp(-ds/Tv)
    return Cn_v

def find_nearest(value, y_array):
    array = np.asarray(y_array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_total_results(i1,i3):
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_BEM[freq_pos,i1:i3,an_pos],label=r"$Cn_{steady}$")
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_c[freq_pos,i1:i3,an_pos],label=r"$Cn_c$")
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_nc[freq_pos,i1:i3,an_pos],label=r"$Cn_{nc}$")
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_p_prime[freq_pos,i1:i3,an_pos],label=r"$Cn_{p_{prime}}$")
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_f[freq_pos,i1:i3,an_pos],label=r"$Cn_f$")
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_v[freq_pos,i1:i3,an_pos],label=r"$Cn_v$")
    ax.plot(np.rad2deg(alpha_qs[freq_pos,i1:i3,an_pos]),Cn_t[freq_pos,i1:i3,an_pos],label=r"$Cn_t$")
    ax.legend()
    ax.set_xlabel(r'$\alpha [^\circ]$')
    ax.set_ylabel(r'$C_n$ [ ]')
    fig.savefig(name + "_Cn" + ".pdf")
    plt.tight_layout()
    plt.show()

def find_max(array, tolerance):
    max_value = max(array)
    indices = [index for index, value in enumerate(array) if abs(value - max_value) < tolerance]
    double_ind = []
    for i in range(len(indices)-1):
        if np.abs(indices[i] - indices[i+1]) == 1:
            double_ind.append(i)
    indices_copy = indices.copy()
    for i,ind in enumerate(double_ind):
        value = indices[ind]
        indices_copy.remove(value)
    return indices_copy
#
# def find_min(array, tolerance):
#     min_value = min(array)
#     indices = [index for index, value in enumerate(array) if abs(value - min_value) < tolerance]
#     double_ind = []
#     for i in range(len(indices) - 1):
#         if np.abs(indices[i] - indices[i + 1]) == 1:
#             double_ind.append(i)
#     indices_copy = indices.copy()
#     print(indices_copy)
#     print(double_ind)
#     for i in range(len(double_ind)):
#         print(indices[i])
#         indices_copy.remove(indices[i])
#     return indices_copy

if __name__ == "__main__":
    # Import lift and drag polars for the DU_airfoil, used for the wind turbine case
    data = np.genfromtxt("DU_airfoil.txt", delimiter=",")

    alpha = data[:, 0]  # Angle of attack in degrees
    CL = data[:, 1]  # Lift coefficient polar
    CD = data[:, 2]  # Drag coefficient polar
    CM = data[:, 3]  # Moment coefficient polar

    # Define gradient of Cl to alpha in degrees
    dCl_dalpha = np.gradient(CL, alpha)

    # Plotted the polar and get alpha at which Cl=0
    alpha0 = np.deg2rad(-1.945) #degrees

    # Amount of iterations
    max_n_iterations = 100

    # Allowed tolerance, below which convergence is considered to be achieved
    tol = 0.0000001

    Uinf_steady = 10  # Free stream velocity (from requirements)
    Radius = 50  # Radius in meters (from requirements)
    r_Rroot = 0.2  # Scaled location of root of blade
    r_Rtip = 1  # Scaled location of tip of blade
    blades = 3  # Number of blades
    rho = 1  # density of air
    N = 100  # Number of annuli

    add_Prandtl_correction = True

    # Define tip speed ratio and rotational speed omega
    TSR = 8
    Omega_rotor = TSR*Uinf_steady/Radius

    # Define inflow conditions
    U1 = 1*Uinf_steady

    # it is actually nicer if you start at a lower wind speed
    DeltaU = 0.5*Uinf_steady

    # Define reduced frequency
    k_reduced = [0., 0.3]
    te = 300
    Nt = 3000

    t_array = np.linspace(0,te,Nt)
    dt = t_array[1]-t_array[0]

    # Define annuli

    dr = 1/50
    r_R_range = [0.3, 0.5, 0.7, 0.9]
    pitch = -2

    # TODO Change these for plotting.
    an_pos = 0 # [0, 1, 2, 3]  r_R_range = [0.3, 0.5, 0.7, 0.9]
    freq_pos = 1 # [0, 1] k_reduced = [0,0.3]

    # Initialize solution space
    alpha_qs = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    cl_alpha = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    alpha_eq = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    dalpha_qs_dt = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_c = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_nc = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_p = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_p_prime = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_st = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Dpf = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    alpha_f = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    F_p = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Dbl = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    F_bl = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_f = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    tau_v = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    C_v = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_v = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_t = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_BEM = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    Cn_alpha_eq_grad = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    CT_BEM = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    a_BEM = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    v_induction = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    a = np.zeros((len(k_reduced), len(t_array), len(r_R_range)))
    results_BEM = np.zeros((len(r_R_range), 13))
    omega = np.zeros(len(k_reduced))

    for k, k_red in enumerate(k_reduced):
        print(k)
        # loop over all reduced frequencies of the inflow conditions
        omega[k] = k_red * Uinf_steady / Radius

        # Initialize lag states
        Xlag = np.zeros((len(t_array), len(r_R_range)))
        Ylag = np.zeros((len(t_array), len(r_R_range)))

        # loop over time. I added azimuthal like this, ik makes things a bit harder but okay
        # TODO check if azimuthal discretization like this is correct
        Uinf = U1 + DeltaU*np.cos(omega[k]*t_array)#*np.cos(Omega_rotor*t_array)
        Uinf_no_azi = U1 + DeltaU*np.cos(omega[k]*t_array)
        for j, t in enumerate(t_array[:-1]):
            Uinf_val = Uinf[j]
            for i, r_R in enumerate(r_R_range):
                # Iterate over different annuli
                twist = 14 * (1 - r_R) + pitch
                chord = 3 * (1 - r_R) + 1  # Chord in meters (from requirements)
                r = r_R * Radius  # Middle of annulus in meters
                Area = np.pi * ((( r_R + dr/2)* Radius) ** 2 - ((r_R-dr/2) * Radius) ** 2)  # Area of each streamtube
                results_BEM[i, :] = BEM(TSR, dr, r_R, chord, Area, twist, add_Prandtl_correction, r_Rtip, r_Rroot, tol,
                                        max_n_iterations, Uinf_val, Radius, Omega_rotor, alpha, CL, CD, blades)
            # Here we get our angle of attack per annuli in time for the quasi steady case
            alpha_qs[k, j, :] = np.deg2rad(results_BEM[:,8]) # AoA value for all annuli, at all times, at all frequency
            # TODO Slide 19 For the given angle of attack, the steady lift coefficient was take from BEM
            Cn_BEM[k,j,:] = results_BEM[:,12]
            CT_BEM[k,j, :] = results_BEM[:, 6]
            a_BEM[k,j, :] = results_BEM[:, 0]
            # cl_alpha[k, j, :] = np.interp(alpha_qs[k, j, :], alpha, dCl_dalpha) # gradient of cl to alpha for alpha_qs

        for i, r_R in enumerate(r_R_range):
            chord = 3 * (1 - r_R) + 1
            # here all lag states are introduced from aeroelasticity. All angles are in radians
            Xlag[:,i], Ylag[:,i] = duhamel_approx(t_array, Uinf, chord, alpha_qs[k, :, i])
            alpha_eq[k, :, i] = alpha_qs[k, :, i] - Xlag[:, i] - Ylag[:, i]
            # Cn_alpha_eq_grad[k, :, i] = np.gradient(Cn_BEM[k, :, i], alpha_qs[k, :, i])

        # TODO SLIDE 13 not sure if this should be 2pi or the actual steady CL-alpha gradient
        Cn_c[k, :, :] = 2*np.pi*(alpha_eq[k,:,:] - alpha0)
        # Cn_c[k, :, :] = Cn_alpha_eq_grad[k, :, :]*(alpha_eq[k,:,:] - alpha0)

        for i, r_R in enumerate(r_R_range):
            # TODO SLIDE 16. I think it is correct. but it really depends on the frequency if the nc contributes
            dalpha_qs_dt[k,:,i] = np.gradient(alpha_qs[k,:,i],t_array)
            chord = 3 * (1 - r_R) + 1  # Chord in meters (from requirements)
            Cn_nc[k,:,i] = nc_normal_force(dalpha_qs_dt[k,:,i], chord, t_array, Uinf, v_sos=343)

        # TODO SLIDE 17 circ + non-circ
        Cn_p[k, :, :] = Cn_c[k, :, :] + Cn_nc[k, :, :]


        for i, r_R in enumerate(r_R_range):
            chord = 3 * (1 - r_R) + 1  # Chord in meters (from requirements)

            # TODO SLIDE 20. Since we already had our steady Cn, we do this first. Leading-edge pressure lage
            Dpf[k,:,i] = pressure_lag(Cn_p[k,:,i], t_array, Uinf, chord)

            Cn_p_prime[k,:,i] = Cn_p[k,:,i] - Dpf[k,:,i]
            # not sure if this should be 2pi or the actual steady CL-alpha gradient
            alpha_f[k,:,i] = Cn_p_prime[k,:,i]/(2*np.pi) + alpha0

            # TODO SLIDE 19. TE separation: Get the sep. factor data from the steady case from BEM for the new alpha_f
            f_trial = (2 * np.sqrt(Cn_BEM[k, :, i] / (2*np.pi * (alpha_f[k, :, i] - alpha0))) - 1) ** 2
            # probably right way to get F. Interpolation seems unnecessary

            # it is really not necessary to save this in time. It just by choice hase the same length, but it is
            # irrespective of both frequency and annulus. So it is fine, but confusing.
            F_p[k,:,i] = f_trial #np.interp(alpha_f[k,:,i], alpha, F_sep)

            # TODO SLIDE 21: Boundary layer development lag
            Dbl[k,:,i] = boundary_layer_lag(F_p[k,:,i], t_array, Uinf, chord)
            F_bl[k,:,i] = F_p[k,:,i] - Dbl[k,:,i]
            # unsteady nonlinear normal force coefficient
            Cn_f[k, :, i] = 2*np.pi * ((1 + np.sqrt(F_bl[k, :, i])) / 2) ** 2 * (
                        alpha_eq[k, :, i] - alpha0) + Cn_nc[k, :, i]

            # TODO SLIDE 23. LE flow separation. This module behaves weirdly and is much dependent on the frequency
            tau_v[k,:,i] = leading_edge(Cn_p_prime[k, :, i], dalpha_qs_dt[k, :, i], t_array, Uinf, chord)
            # TODO SLIDE 24 Vortex shedding. His code puts in Cn_c, but his slide Cn_p. Im not sure
            C_v[k,:,i] = Cn_c[k,:,i] * (1-((1+np.sqrt(F_bl[k,:,i]))/2)**2)
            Cn_v[k,:,i] = vortex_shedding(tau_v[k,:,i], C_v[k,:,i], t_array, Uinf, chord)
            # TODO SLIDE 25 Total unsteady non-linear normal force coefficient
            Cn_t[k,:,i] = Cn_f[k,:,i] + Cn_v[k,:,i]

    # TODO, see below
    # check this plot to find indexes between peaks (high and low frequencies)
    # run the file with python console, change i1 and i3 to the required values necessary in the console
    # fill in the plotting function above in the python console with desired i1 and i3, then see the results

    azi = "no_azi"
    title1 = "an_pos"
    title2 = "freq_pos"
    if an_pos == 0:
        antitle = "0"
    elif an_pos == 1:
        antitle = "1"
    elif an_pos == 2:
        antitle = "2"
    elif an_pos == 3:
        antitle = "3"
    elif an_pos == 4:
        antitle = "4"
    if freq_pos ==0:
        freqtitle = "0"
    elif freq_pos ==1:
        freqtitle = "1"

    name = f"figures/{title1}{antitle}_{title2}{freqtitle}_{azi}"

    fig, ax = plt.subplots()
    ax.plot(t_array, Uinf)
    ax.set_ylabel(r'$U_\infty$ [m/s]')
    ax.set_xlabel('t [s]')
    fig.savefig(name + "_Uinf" + ".pdf")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(Uinf)
    plt.ylabel(r'$U_\infty$ [m/s]')
    plt.show()

    # Find indices i1 and i3
    difference = 5e-2
    max_indices = find_max(Uinf, difference)
    i1 = max_indices[-3]
    i3 = max_indices[-1]
    # min_indices = find_min(Uinf_no_azi, difference)
    #
    # if max_indices[-1] > min_indices[-1]:
    #     i3_no_azi = max_indices[-1]
    #     i1_no_azi = max_indices[-2]
    # else:
    #     i3_no_azi = min_indices[-1]
    #     i1_no_azi = min_indices[-2]
    #
    # i3_no_azi = max_indices[-1]
    # i1_no_azi = max_indices[-2]
    # i3_new = np.argmax(Uinf[i3_no_azi - 25:i3_no_azi])
    # i3 = (i3_no_azi-25) + i3_new
    # i1_new = np.argmax(Uinf[i1_no_azi - 25:i1_no_azi])
    # i1 = (i1_no_azi-25) + i1_new
    print(i1, i3)

    plot_total_results(i1, i3)
