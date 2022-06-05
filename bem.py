# Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.interpolate as sp
import matplotlib
from matplotlib.ticker import MaxNLocator

# matplotlib.use('Qt5Agg')

# BEM
def BEM(TSR, dr, r_R_curr, c_curr, area_curr, twist_curr, add_Prandtl_correction, r_Rtip, r_Rroot, tol, max_n_iterations, Uinf, Radius, Omega, alpha, CL, CD, blades):
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
    a_curr = 0.3 # seeding value of the axial induction factor
    a_curr_cor = 1 # initial guess for the axial factor at the new iteration
    a_prime_curr = 0 # seeding value of the tangential induction factor
    current_iter = 1 # integer value counting the number of iterations undertaken
    # by the BEM code
    results_forces = np.zeros(8)

    while (np.abs(a_curr_cor - a_curr) > tol and current_iter < max_n_iterations):
        if current_iter > 1:
            a_curr = 0.25 * a_curr_cor + 0.75 * a_curr
        # if the iterations went over the initial guess, use a weighted average
        # of the previous 2 iterations to obtain a new guess value
        
        # Compute flow properties and forces for each annulus
        
        # Determine the local velocity components and the effective AoA of each annulus
        Uaxial = Uinf*(1-a_curr) # Axial flow
        Utan = (1+a_prime_curr)*Omega*r_curr # Tangential flow
        V = np.sqrt(Uaxial**2 + Utan**2) # Effective velocity
        phi = np.arctan2(Uaxial, Utan)*180/np.pi # Inflow angle
        AoA = phi - twist_curr # AoA in degrees
        # Determine forces on each blade element, given by the AoA
        Cl = np.interp(AoA, alpha, CL) # Interpolate lift coefficient
        Cd = np.interp(AoA, alpha, CD) # Interpolate drag coefficient
        Lift = 0.5*c_curr*V**2*Cl # lift force at each annulus, taken in the local reference system with
        # respect to the effective velocity
        Drag = 0.5*c_curr*V**2*Cd # drag force at each annulus, taken in the local reference system with
        # respect to the effective velocity
        Fazim = Lift*np.sin(phi*np.pi/180) - Drag*np.cos(phi*np.pi/180)
        # azimuthal force at each annulus, taken  in a reference system aligned with the axis of rotation of the rotor
        Faxial = Lift*np.cos(phi*np.pi/180) + Drag*np.sin(phi*np.pi/180)
        # axial force at each annulus, taken in a reference system aligned with the axis of rotation of the rotor
        CT = Faxial*blades*dr*Radius/(0.5*Uinf**2*area_curr)
        # thrust coefficient at each annulus, corresponding to the approximated aerodynamic load
        a_prime_curr = Fazim*blades/(2*np.pi*Uinf*(1-a_curr)*Omega*2*(r_R_curr*Radius)**2)
        gamma_curr = 0.5 * V * Cl *c_curr
        """
        Glauert correction for heavily loaded rotors is implemented and computed
        """
        # From thrust coefficient, determine axial induction factor, using Glauert correction
        CT1 = 1.816
        CT2 = 2*np.sqrt(CT1) - CT1
        if CT < CT2:
            a_curr_cor = 0.5 - np.sqrt(1-CT)/2
        elif CT >= CT2:
            a_curr_cor = 1 + (CT- CT1)/(4*np.sqrt(CT1) -4)
        
        """
        Since there are a finite number of blades instead of an idealised actuator disk,
        Prandtl correction is applied.
        """
        if add_Prandtl_correction == True:
            # determine values of the correction at the center of each annulus
            arg_tip = -blades/2*(r_Rtip-r_R_curr)/r_R_curr*np.sqrt( 1+ ((TSR*r_R_curr)**2)/((1-a_curr_cor)**2))
            Ftip = 2/np.pi*np.arccos(np.exp(arg_tip))
            arg_root = blades/2*(r_Rroot-r_R_curr)/r_R_curr*np.sqrt( 1+ ((TSR*r_R_curr)**2)/((1-a_curr_cor)**2))
            Froot = np.array(2/np.pi*np.arccos(np.exp(arg_root)))
            # bound case of "not a number"
            if np.isnan(Froot):
                Froot = 0
            elif np.isnan(Ftip):
                Ftip = 0
            # combine the corections for the tip and the root
            Ftotal = Ftip*Froot
            
            # bound case of "divide by zero"
            if Ftotal < 0.0001:
                Ftotal = 0.0001
                
            a_curr_cor = a_curr_cor/Ftotal # Correct axial induction factor
            a_prime_curr = a_prime_curr/Ftotal # Correct tangential induction factor
        # writing the results
        CT = Faxial * blades * dr * Radius/(0.5*Uinf**2*area_curr)
        Cn = Fazim * blades* dr * Radius / (0.5*Uinf**2*area_curr)
        Cq = Fazim * blades * r_curr * dr * Radius / (0.5 *Uinf**2*area_curr * c_curr)
        alpha_distrib = AoA
        inflowangle_distrib = phi
        current_iter = current_iter + 1
    results_BEM = [ a_curr_cor, a_prime_curr, r_R_curr, Faxial, Fazim, gamma_curr, CT, area_curr, alpha_distrib, inflowangle_distrib, Cn, Cq ]
    return results_BEM

# Momentum Theory
def get_a_from_ct(ct,glauert=False):
  if glauert:
    ct1 = 1.816
    ct2 = 2*np.sqrt(ct1)-ct1
  else:
    ct1 = 0
    ct2 = 100

  a = np.zeros(np.shape(ct))
  a[ct>=ct2] = 1 + (ct[ct>=ct2]-ct1)/(4*(np.sqrt(ct1)-1))
  a[ct<ct2] = 0.5 - 0.5*np.sqrt(1-ct[ct<ct2])
  return a

def get_ct_from_a(a, glauert=False):
  ct = np.zeros(np.shape(a))
  ct += 4*a*(1-a)
  if glauert:
    ct1 = 1.816
    a1 = 1 - np.sqrt(ct1)/2
    ct[a>a1] = ct1 - 4 * (np.sqrt(ct1) - 1) * (1 - a[a>a1])

  return ct

# Unsteady models
def pittpeters(ct, v_induction, Uinf, R, dt, glauert=False):
  # Determine induction and thrust coefficient
  a = -v_induction / Uinf
  ct_new = - get_ct_from_a(a,glauert)

  # Calculate the time derivative of the induced velocity
  dv_inductiondt = (ct-ct_new) / (16 / (3*np.pi)) * (Uinf**2/R)

  # Perform time integration
  v_induction_new = v_induction + dv_inductiondt*dt

  return v_induction_new, dv_inductiondt

def oye(Ct1, Ct2, v_induced, v_int, Uinf, R, r_R,dt,glauert=False):
  # Find quasi-steady induction velocity at current and at the next time step
  v_qs1 = - get_a_from_ct(-Ct1,glauert)*Uinf
  v_qs2 = - get_a_from_ct(-Ct2,glauert)*Uinf

  # Find the current induction factor
  a = -v_induced / Uinf

  # Determine time scales from Oye model
  t1 = 1.1/(1-1.3*a) * R / Uinf
  t2 = (0.39-0.26*(r_R)**2)*t1

  # Solve the derivative of the intermediate velocity using Oyes model
  dv_int_dt = (v_qs1 + (v_qs2-v_qs1)/dt * 0.6 * t1 - v_int) / t1

  # Perform time integration
  v_int2 = v_int + dv_int_dt*dt

  # Determine derivative of the induced velocity
  dv_induced_dt = ((v_int+v_int2)/2-v_induced)/t2

  # Perform time integration
  v_induced2 = v_induced + dv_induced_dt*dt

  return v_induced2, v_int2

def larsen_madsen(Ct2,v_induced, Uinf, R, dt, glauert=False):
    # Determine the wake velocity
    v_wake = Uinf + v_induced

    # Calculate time scale of the model
    t = 0.5*R/v_wake

    # Determine the induction velocity at the next time step
    v_induction2 = - get_a_from_ct(-Ct2,glauert)*Uinf

    # Determine new induced velocity
    v_induced2 = v_induced*np.exp(-dt/t)+v_induction2*(1-np.exp(-dt/t))

    return v_induced2

# Perform unsteady BEM for different pitch angles
def unsteady(Uinf,input_var,model,k_reduced=[0],te=20,glauert=False,ct_cond=True, name="_"):
    # Introduce time
    dt = 0.05
    time = np.arange(0, te, dt)
    t1 = 5

    Uinf_lst = [Uinf, Uinf]

    CT_BEM = np.zeros((len(r_R),2))
    a_BEM = np.zeros((len(r_R),2))
    if ct_cond:
        for j,pitch in enumerate(input_var):
            twist = 14 * (1 - r_R) + pitch  # local twist angle at the interface of 2 annuli in degrees
            for i in range(len(r_R)):
                results_BEM[i, :] = BEM(TSR, r_Rint[i + 1] - r_Rint[i], r_R[i], chord[i], Area[i], twist[i],
                                        add_Prandtl_correction, r_Rtip, r_Rroot, tol, max_n_iterations, Uinf, Radius, Omega,
                                        alpha, CL, CD, blades)
            CT_BEM[:,j] = results_BEM[:,6]
            a_BEM[:,j] = results_BEM[:,0]
    else:
        for j, u_U in enumerate(input_var):
            twist = 14 * (1 - r_R) - 2
            Uinf_in = u_U*Uinf
            for i in range(len(r_R)):
                results_BEM[i, :] = BEM(TSR, r_Rint[i + 1] - r_Rint[i], r_R[i], chord[i], Area[i], twist[i],
                                        add_Prandtl_correction, r_Rtip, r_Rroot, tol, max_n_iterations, Uinf_in, Radius, Omega,
                                        alpha, CL, CD, blades)
            CT_BEM[:,j] = results_BEM[:,6]
            a_BEM[:,j] = results_BEM[:,0]
    # Define initial CT and induction from all annuli
    ct0 = CT_BEM[:,0]
    v_induction0 = -get_a_from_ct(-ct0,glauert) * Uinf_lst[0]

    # Define final CT and induction
    ct1 = CT_BEM[:, 1]
    d_ct = ct1 - ct0
    d_U = (Uinf_lst[1] - Uinf_lst[0])/Uinf

    v_induction1 = -get_a_from_ct(-ct1,glauert) * Uinf_lst[1]

    v_induction = np.zeros((len(time),len(ct0),len(k_reduced)))
    a = np.zeros(np.shape(v_induction))
    # iterate over all annuli
    ct = np.zeros((len(time), len(ct0), len(k_reduced)))
    ct_actual = np.zeros((len(time), len(ct0), len(k_reduced)))
    for j in range(len(ct0)):
        v_int = np.zeros(np.size(k_reduced))
        v_int += v_induction0[j]
        ct[:,j,:] += ct0[j]
        v_induction[0, j, :] += v_induction0[j]
        a[0, j, :] += v_induction[0, j, :]/Uinf_lst[0]
        ct_actual[0, j, :] = ct0[j]
        if len(k_reduced) == 1:
            ct[time >= t1, j, :] = ct1[j]

        for ix in range(len(k_reduced)):
            v_int[ix] = v_induction0[j]
        # iterate over all time
        Uinf = Uinf_lst[0]
        for i, t_value in enumerate(time[:-1]):
            # iterate over all reduced frequencies
            for ix, k_value in enumerate(k_reduced):
                if len(k_reduced) > 1:
                    ct[i+1,j,ix] = ct0[j] + d_ct[j] * np.sin(k_value * Uinf / (r_R[j]*Radius) * t_value)
                    Uinf = Uinf_lst[0]+d_U*np.sin(k_value * Uinf / (r_R[j]*Radius) * t_value)
                else:
                    if t_value >= t1:
                        Uinf = Uinf_lst[1]
            # model selection
                if model == "pit":
                    v_induction[i + 1, j, ix] = pittpeters(ct[i + 1,j,ix], v_induction[i,j,ix], Uinf, r_R[j]*Radius, dt)[0]
                elif model == "oye":
                    v_induction[i + 1, j, ix], v_int[ix] = oye(ct[i,j,ix], ct[i + 1,j,ix], v_induction[i,j,ix], v_int[ix], Uinf, r_R[j]*Radius, r_R[j], dt)
                elif model == "lar":
                    v_induction[i + 1, j, ix] = larsen_madsen(ct[i + 1,j,ix], v_induction[i,j,ix], Uinf, r_R[j]*Radius, dt)

                a[i + 1, j, ix] = -v_induction[i + 1, j, ix]/Uinf
                ct_actual[i + 1, j, ix] = -get_ct_from_a(a[i + 1, j, ix],glauert)

    # initiate plotting procedure
    if len(k_reduced) == 1:
        plotting_step(time,t1,v_induction,ct_actual,name)
    else:
        Uinf_final = Uinf_lst[1]
        plot_sinusoidal(v_induction, ct, ct0, d_ct, dt, k_reduced, Uinf_final, a, glauert,name)

# Retrieve Thrust Coefficient for all annuli for a given pitch
def annuli_iterator(pitch):
    twist = 14 * (1 - r_R) + pitch  # local twist angle at the interface of 2 annuli in degrees
    for i in range(len(r_R)):
        results_BEM[i, :] = BEM(TSR, r_Rint[i + 1] - r_Rint[i], r_R[i], chord[i], Area[i], twist[i],
                                add_Prandtl_correction, r_Rtip, r_Rroot, tol, max_n_iterations, Uinf, Radius, Omega,
                                alpha, CL, CD, blades)

    CT_total_BEM = sum(results_BEM[:, 6] * results_BEM[:, 2] * delta_rR * 2)
    return results_BEM, CT_total_BEM

# def find_nearest(value, y_array, x_array):
#     array = np.asarray(y_array)
#     idx = (np.abs(array - value)).argmin()
#     return x_array[idx]

def ct_condition_func(CT_boundaries):
    pitches = np.zeros((np.shape(CT_boundaries)))
    for k, CT_start in enumerate(CT_boundaries):
        pitch = np.arange(-7, 7, 1)  # Pitch angle of the entire turbine blade in degrees
        CT_interp = 10000
        while not abs(CT_interp - CT_start) < 0.001:
            CT_total_BEM = np.zeros(np.shape(pitch))
            for i, pitch_in in enumerate(pitch):
                _, CT_total_BEM[i] = annuli_iterator(pitch_in)

            # pitch_interp = find_nearest(CT_start, CT_total_BEM, pitch)
            f = sp.interp1d(CT_total_BEM, pitch)
            pitch_interp = f(CT_start)
            print(f"Pitch = {round(float(pitch_interp),2)} deg")
            results_BEM, CT_interp = annuli_iterator(pitch_interp)
            print(f"CT = {round(CT_interp,3)}")
            limit = abs(CT_start - CT_interp) * 100
            pitch = np.arange(pitch_interp - 2 * limit, pitch_interp + 2 * limit, limit / 2)  # limit = limit/10
        pitches[k] = pitch_interp

    return pitches

def plotting_step(time,t1,v_induction,ct,name):
    colors = pl.cm.cmap_d["bone"](np.linspace(0, 1, len(r_R)))

    tt, rr = np.meshgrid(time, r_R)

    fig, ax = plt.subplots()
    p = ax.pcolormesh(tt, rr, np.transpose(v_induction[:, :, 0]), cmap="viridis")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Radial Position [-]")

    fig.colorbar(p)
    fig.savefig(name+"_2d"+".pdf")


    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    p2 = ax2.plot_surface(tt, rr, np.transpose(v_induction[:, :, 0]), cmap="viridis")
    ax2.view_init(30, -30)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Radial Position [-]")
    ax2.xaxis.set_major_locator(MaxNLocator(3))
    ax2.yaxis.set_major_locator(MaxNLocator(3))

    ax2.set_zlabel("V_induction [m/s]")
    ax2.set_zticks([])
    fig2.colorbar(p2)
    fig2.savefig(name+"_3d"+".pdf")

    fig3, ax = plt.subplots(2)
    for i in range(len(r_R)):
        if i % (len(r_R) / 4) == 3:
            ax[0].plot((time - t1) * Radius / Uinf,
                     (v_induction[:, i] - v_induction[0, i]) / (v_induction[-1, i] - v_induction[0, i]),
                     color=colors[i], label=f"r_R = {round(r_R[i], 2)}")
            ax[1].plot((time - t1),ct[:, i], color=colors[i], label=f"r_R = {round(r_R[i], 2)}")

    ax[0].legend()
    ax[0].set_xlabel('tR/Uinf [-]') # 'tR/Uinf'
    ax[0].set_ylabel('Normalised v_induction [-]') # 'v_inductioax[]

    ax[1].legend()
    ax[1].set_xlabel('Time [s]') # 'tR/Uinf'
    ax[1].set_ylabel('Ct [-]') # 'v_induction'

    fig3.savefig(name+"_norm"+".pdf")

    plt.show()

def plot_sinusoidal(v_induction, ct, ct0, d_ct, dt, k_reduced, Uinf, a, glauert=False, name="_"):
    colors = pl.cm.cmap_d["bone"](np.linspace(0, 1, len(r_R)))

    nn = 100
    ct_ss = np.zeros((len(r_R),nn))
    a_ss = np.zeros(np.shape(ct_ss))
    for i in range(len(ct0)):
        ct_ss[i,:] = np.linspace((ct0[i] - d_ct[i]), (ct0[i] + d_ct[i]), nn)  # define an array of $C_T$
        a_ss[i,:] = -get_a_from_ct(-ct_ss[i,:],glauert)  # calculate steady solution of induction as a function of $C_T$

    n_plots = 2
    fig, ax = plt.subplots(n_plots)
    ax_i = 0
    for ix in range(len(r_R)):
        if ix % (len(r_R) / n_plots) == 3:
            ax[ax_i].plot(ct_ss[ix,:], a_ss[ix,:],label=f"Steady",color='blue')

            for j, k_value in enumerate(k_reduced):
                ind = -(np.floor(2 * np.pi / (k_value * Uinf / (r_R[ix]*Radius)) / dt)+1).astype(int)  # indices of the last full cycle to only plot 1 cycle
                print(ind)
                label1 = r'$\omega \frac{R}{U_\infty}=' + np.str(k_value) + '$'  # define label for the legend
                # plot unsteady solution
                ax[ax_i].plot(ct[ind:,ix,j], -a[ind:,ix,j], label=label1, linestyle=(0, (j + 1, j + 1)),
                                linewidth=(6 / (j + 2)),)

            ax[ax_i].legend(loc=(1.04,0))
            ax[ax_i].set_xlabel('C_t')
            ax[ax_i].set_ylabel('a')
            ax[ax_i].grid()
            ax[ax_i].set_title(f"r_R = {round(r_R[ix], 2)}")
            ax_i += 1
    fig.savefig(name+".pdf")
    plt.show()

if __name__ == "__main__":
    # Import lift and drag polars for the DU_airfoil, used for the wind turbine case
    data = np.genfromtxt("DU_airfoil.txt", delimiter=",")

    alpha = data[:,0] # Angle of attack
    CL = data[:,1] # Lift coefficient polar
    CD = data[:,2] # Drag coefficient polar
    CM = data[:,3] # Moment coefficient polar

    alpha_stall = 9.77 # angle of stall for the airfoil

    # Amount of iterations
    max_n_iterations = 100 

    # Allowed tolerance, below which convergence is considered to be achieved
    tol = 0.0000001 

    Uinf = 10 #Free stream velocity (from requirements)
    Radius = 50 # Radius in meters (from requirements)
    r_Rroot = 0.2 # Scaled location of root of blade
    r_Rtip = 1 # Scaled location of tip of blade
    blades = 3 # Number of blades
    rho = 1 # density of air
    N = 100 # Number of annuli

    delta_rR = (r_Rtip - r_Rroot)/N # Width of annuli (constant), in scaled radius
    r_Rint = np.linspace(0.2, 1, int(N+1)) # Radial interface locations
    r_R = (r_Rint[0:-1] + r_Rint[1:])/2 # Middle of an annulus in scaled radius
    chord = 3*(1-r_R)+1 # Chord in meters (from requirements)
    r = r_R*Radius # Middle of annulus in meters
    Area = np.pi*((r_Rint[1:]*Radius)**2-(r_Rint[0:-1]*Radius)**2) # Area of each streamtube

    results_BEM = np.zeros( (len(r_R),12))
    add_Prandtl_correction = True

    TSR = 10
    Omega = Uinf * TSR / Radius

    # Get real inflow conditions that would give the requested thrust coefficient.
    CT_boundaries = [0.9, 0.5]
    U_boundaries = [1, 1.5]
    models = ["pit","oye","lar"]
    # reduced frequencies!!!

    # Perform unsteady BEM
    glauert_cond = True

    # choose whether to use ct/U and step/sinusoidal and model no.
    ct_cond = True
    sinusoidal_cond = True
    n_model = 0  # 0,1,2 ==> ["pit","oye","lar"]

    if ct_cond:
        pitches = ct_condition_func(CT_boundaries)
        input_var = pitches
    else:
        input_var = U_boundaries

    # use reduced frequency?
    if sinusoidal_cond:
        k_reduced = np.arange(0.5, 2.1, .5)
        te = 200
    else:
        k_reduced = [0]
        te = 20


    if ct_cond:
        st1 = "CT"
        st3 = "_".join([str(elem) for elem in CT_boundaries])
    else:
        st1 = "U"
        st3 = "_".join([str(elem) for elem in U_boundaries])

    if sinusoidal_cond:
        st2 = "sinus"
    else:
        st2 = "step"

    name = f"figures/{st1}_{st2}_{st3}_{models[n_model]}"
    unsteady(Uinf, input_var, models[n_model], k_reduced=k_reduced, te=te, ct_cond=ct_cond,glauert=glauert_cond,name=name)


