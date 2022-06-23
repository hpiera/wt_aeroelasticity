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

def get_s_from_time(time, Uinf, chord):
    s = 2*Uinf*time/chord
    return s

def get_time_from_s(s, Uinf, chord):
    time = s*chord/2/Uinf
    return time

# # Determine quasi-steady angle of attack
# def alpha_quasisteady(alpha, chord, Uinf, theta, h, time):
#     alpha_qs = alpha + chord/(2*Uinf) * np.gradient(theta, time) - 1/Uinf * np.gradient(h, time)
#     return alpha_qs

def duhamel_approx(Xlag, Ylag, delta_s, delta_alpha):
    # Define Wagner constants
    A1 = 0.165
    A2 = 0.335
    b1 = 0.0455
    b2 = 0.3

    # Determine lag states
    Xlag1 = Xlag*np.exp(-b1*delta_s)+A1*delta_alpha
    Ylag1 = Ylag * np.exp(-b2 * delta_s) + A2 * delta_alpha

    return Xlag1, Ylag1

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
    ds = np.zeros(np.shape(time))
    for i, s in enumerate(s_array[:-1]):
        ds[i] = s_array[i+1] - s_array[i]
        Dbl[i + 1] = Dbl[i] * np.exp(-ds[i] / Tf) + (F_p[i + 1] - F_p[i]) * np.exp(-ds[i] / (2 * Tf))

    # plt.figure()
    # plt.plot(ds)
    # plt.show()
    return Dbl

def leading_edge(Cn_f, alpha_eq, time, Uinf, chord):
    Cn1 = 1.0093
    s_array = get_s_from_time(time, Uinf, chord)
    tau_v = np.zeros(np.shape(time))
    for i in range(len(s_array)-1):
        ds = s_array[i+1] - s_array[i]
        dalpha_eq = alpha_eq[i+1] - alpha_eq[i]
        if Cn_f[i] < Cn1 and dalpha_eq > 0:
            tau_v[i+1] = 0
        else:
            tau_v[i+1] = tau_v[i] + 0.45*ds

    return tau_v

def vortex_shedding(tau_v, C_v, time, Uinf, chord):
    s_array = get_s_from_time(time, Uinf, chord)
    Tv = 6
    Tvl = 5
    Cn_v = np.zeros(np.shape(time))
    for i in range(len(s_array)-1):
        ds = s_array[i + 1] - s_array[i]
        if tau_v[i]>0 and tau_v[i]<Tvl:
            Cn_v[i+1] = Cn_v[i]*np.exp(-ds/Tv) + (C_v[i+1] - C_v[i])*np.exp(-ds/(2*Tv))
        else:
            Cn_v[i+1] = Cn_v[i]*np.exp(-ds/Tv)
    return Cn_v

def find_nearest(value, y_array):
    array = np.asarray(y_array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
    alpha0 = -1.945 #degrees

    F = 2*np.pi*np.pi/180*(alpha-alpha0) - CL
    f_trial = (2*np.sqrt(CL/(2*np.pi*(alpha-alpha0)))-1)**2

    # Amount of iterations
    max_n_iterations = 100

    # Allowed tolerance, below which convergence is considered to be achieved
    tol = 0.0000001

    Uinf = 10  # Free stream velocity (from requirements)
    Radius = 50  # Radius in meters (from requirements)
    r_Rroot = 0.2  # Scaled location of root of blade
    r_Rtip = 1  # Scaled location of tip of blade
    blades = 3  # Number of blades
    rho = 1  # density of air
    N = 100  # Number of annuli

    add_Prandtl_correction = True

    # Define tip speed ratio and omega
    TSR = 8
    Omega = Uinf * TSR / Radius

    # Define inflow conditions
    U1 = 1*Uinf
    DeltaU = 0.5

    # Define reduced frequency
    k_reduced = [0, 0.3]
    te = 100
    Nt = 500
    t_array = np.linspace(0,te,Nt)
    dt = t_array[1]-t_array[0]

    # Define annuli
    dr = 1/50
    r_R_range = [0.3, 0.5, 0.7, 0.9]
    pitch = -2

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
    results_BEM = np.zeros((len(r_R_range), 12))

    for k, k_red in enumerate(k_reduced):
        Uinf = 10
        omega = k_red * Uinf / Radius

        # Initialize lag states
        Xlag = Ylag = np.zeros((len(t_array), len(r_R_range)))

        # loop over time
        Uinf = U1 + DeltaU*np.cos(omega*t_array)
        for j, t in enumerate(t_array[:-1]):
            Uinf_val = Uinf[j]
            for i, r_R in enumerate(r_R_range):
                twist = 14 * (1 - r_R) + pitch
                chord = 3 * (1 - r_R) + 1  # Chord in meters (from requirements)
                s_array = get_s_from_time(t_array, Uinf_val, chord)
                r = r_R * Radius  # Middle of annulus in meters
                Area = np.pi * ((( r_R + dr/2)* Radius) ** 2 - ((r_R-dr/2) * Radius) ** 2)  # Area of each streamtube
                results_BEM[i, :] = BEM(TSR, dr, r_R, chord, Area, twist, add_Prandtl_correction, r_Rtip, r_Rroot, tol,
                    max_n_iterations, Uinf_val, Radius, Omega, alpha, CL, CD, blades)
            ## note that the last one will not get overwritten, might cause problems
            alpha_qs[k, j, :] = results_BEM[:,8]
            cl_alpha[k, j, :] = np.interp(alpha_qs[k, j, :], alpha, dCl_dalpha)

            # Determine lag states
            for i, r_R in enumerate(r_R_range):
                Xlag[j+1,i], Ylag[j+1,i] = duhamel_approx(Xlag[j,i], Ylag[j,i], (s_array[j+1]-s_array[j])/dt, (alpha_qs[k, j+1, i]-alpha_qs[k, j, i])/dt)
                alpha_eq[k, j, i] = alpha_qs[k, j, i] - Xlag[j, i] - Ylag[j, i]

        Cn_c[k, :, :] = cl_alpha[k, :, :]*(alpha_eq[k,:,:] - alpha0)
        for i, r_R in enumerate(r_R_range):
            dalpha_qs_dt[k,:,i] = np.gradient(alpha_qs[k,:,i])
            chord = 3 * (1 - r_R) + 1  # Chord in meters (from requirements)
            Cn_nc[k,:,i] = nc_normal_force(dalpha_qs_dt[k,:,i], chord, t_array, Uinf, v_sos=343)

        Cn_p[k, :, :] = Cn_c[k, :, :] + Cn_nc[k, :, :]

        alpha_new = np.linspace(alpha[0], alpha[-1], len(alpha_eq[0, :, 0]))
        dCl_dalpha_new = np.interp(alpha_new, alpha, dCl_dalpha)
        F_new = np.interp(alpha_new, alpha, F)
        f_trial_new = np.interp(alpha_new, alpha, f_trial)
        F_new[F_new<0] = 0
        F_new = (1 - F_new/abs(F_new[-1]))

        F_new = f_trial_new
        steady = dCl_dalpha_new*((1+np.sqrt(F_new))/2)**2*(alpha_new-alpha0)

        for i, r_R in enumerate(r_R_range):
            chord = 3 * (1 - r_R) + 1  # Chord in meters (from requirements)
            Dpf[k,:,i] = pressure_lag(Cn_p[k,:,i], t_array, Uinf, chord)

            Cn_p_prime[k,:,i] = Cn_p[k,:,i] - Dpf[k,:,i]
            alpha_f[k,:,i] = Cn_p_prime[k,:,i]/dCl_dalpha_new + alpha0
            # for x in range(len(t_array)):
            #     ind = find_nearest(alpha_f[k,x,i], alpha_new)
            #     F_p[k,x,i] = F_new[ind]
            F_p[k,:,i] = F_new #np.interp(alpha_f[k,:,i], alpha, F)

            Dbl[k,:,i] = boundary_layer_lag(F_p[k,:,i], t_array, Uinf, chord)
            F_bl[k,:,i] = F_p[k,:,i] - Dbl[k,:,i]

            for x in range(len(t_array)):
                value = alpha_eq[k,x,i] - alpha0
                ind = find_nearest(value, alpha_new)
                Cn_f[k,x,i] = dCl_dalpha_new[x] * ((1+np.sqrt(F_bl[k,ind,i]))/2)**2 * (alpha_eq[k,x,i] - alpha0) + Cn_nc[k,x,i]

            tau_v[k,:,i] = leading_edge(Cn_f[k,:,i], alpha_eq[k,:,i], t_array, Uinf, chord)

            C_v[k,:,i] = Cn_p[k,:,i] * (1-((1+np.sqrt(F_bl[k,:,i]))/2)**2)
            Cn_v[k,:,i] = vortex_shedding(tau_v[k,:,i], C_v[k,:,i], t_array, Uinf, chord)

            Cn_t[k,:,i] = Cn_f[k,:,i] + Cn_v[k,:,i]

    # Remove last point
    F_bl[F_bl < 0] = 0
    Cn_t = Cn_t[:,:-1,:]
    alpha_new = alpha_new[:-1]


    # plt.figure()
    # plt.plot(Cn_c[1,:,0], label="Cn_c")
    # plt.plot(Cn_nc[1, :, 0], label="Cn_nc")
    # plt.plot(Cn_p[1, :, 0], label="Cn_p")
    # plt.show()

    # plt.figure()
    # plt.plot(Cn_nc[1,:,0])
    # #plt.plot(alpha, 2*np.pi*np.pi/180*(alpha-alpha0))
    # #plt.plot(alpha, F)
    # # plt.show()
    #
    # plt.figure()
    # plt.plot(F_p[1,:,0], Dbl[1,:,0], '.')
    # plt.show()
    #

    ind = -(np.floor(2 * np.pi / (k_reduced[1] * 10 / (r_R_range[0] * Radius)) / dt) + 1).astype(int)


    # we will only plot the last cycle
    Ncycles = np.floor(t_array[-1] * Omega / (2 * np.pi))
    n_of_cycle = t_array * Omega / (2 * np.pi)  # calculate the phase of the different points of the cycle
    i1 = np.argmin(np.abs(n_of_cycle - (Ncycles - 1)))  # index of start of cycle plotted
    i2 = np.argmin(np.abs(n_of_cycle - (Ncycles - .5)))  # index of 180 degrees
    i3 = np.argmin(np.abs(n_of_cycle - (Ncycles)))  # index of 360 degrees
    print(i1)
    print(i3)

    plt.figure()
    plt.plot(alpha_eq[1,i1:i3,0], Cn_t[1,i1:i3,0],'.')
    plt.show()
    #
    # plt.figure()
    # plt.plot(alpha_new, F_bl[1, :-1, 0])
    # plt.plot(alpha_new, F_new[:-1])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(alpha_new, F_p[1,:-1,0])
    # plt.plot(alpha_new, F_new[:-1])
    # plt.show()
