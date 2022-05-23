# Import modules
import numpy as np
import matplotlib.pyplot as plt

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

def annuli_iterator(pitch, CT_start,limit):
    twist = 14 * (1 - r_R) + pitch  # local twist angle at the interface of 2 annuli in degrees
    for i in range(len(r_R)):
        results_BEM[i, :] = BEM(TSR, r_Rint[i + 1] - r_Rint[i], r_R[i], chord[i], Area[i], twist[i],
                                add_Prandtl_correction, r_Rtip, r_Rroot, tol, max_n_iterations, Uinf, Radius, Omega,
                                alpha, CL, CD, blades)

    CT_total_BEM = sum(results_BEM[:, 6] * results_BEM[:, 2] * delta_rR * 2)
    a_total_BEM = sum(results_BEM[:, 0] * results_BEM[:, 2] * delta_rR * 2)

    a_BEM = results_BEM[:, 0]
    ct_BEM = results_BEM[:, 6]
    r_R_BEM = results_BEM[:, 2]
    print(CT_total_BEM, pitch)

    return CT_total_BEM

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
    pitch = np.arange(-5,5,1) # Pitch angle of the entire turbine blade in degrees
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

    CT_start = 0.5
    limit = 0.1
    pitch_elem = np.zeros(np.shape(pitch))
    CT_total_BEM = np.ones(np.shape(pitch))
    CT_interp = 10000
    while not abs(CT_interp - CT_start) < 0.0001:
        for i, pitch_in in enumerate(pitch):
            CT_total_BEM[i] = annuli_iterator(pitch_in,CT_start,limit)

        print(CT_total_BEM, pitch)
        pitch_interp = find_nearest(CT_start, CT_total_BEM, pitch)
        print(f"interp = {pitch_interp}")

        CT_interp = annuli_iterator(pitch_interp,CT_start,limit)
        pitch = np.arange(pitch_interp - 2*limit, pitch_interp + 2*limit, limit/2)
        limit = limit/10


    plt.figure()
    plt.plot(r_R_BEM, a_BEM)
    plt.xlabel('r/R')
    plt.ylabel('a')
    plt.show()

    plt.figure()
    plt.plot(r_R_BEM, ct_BEM)
    plt.xlabel('r/R')
    plt.ylabel('C_t')
    plt.show()
