"""This file stores the necessary functions to code the program"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve


def get_ct(a, glauert=True):
    """This function defines the ct as a function the induction"""
    ct1 = 1.816

    is_scalar = isinstance(a, float)
    ct = 4*a*(1-a)

    if glauert and not is_scalar:
        ct[a >= (1-np.sqrt(ct1)/2)] = ct1-4*(np.sqrt(ct1)-1)*(1-a[a >= (1-np.sqrt(ct1)/2)])
    elif glauert and is_scalar:
        if a >= (1-np.sqrt(ct1)/2):
            ct = ct1-4*(np.sqrt(ct1)-1)*(1-a)

    return ct


def get_a(ct,yaw):
    """This function obtains a from ct, including glauert"""
    ct1 = 1.816
    ct2 = 2 * np.sqrt(ct1) - ct1

    is_scalar = isinstance(ct, float)
    if not is_scalar:
        a = np.zeros(np.shape(ct))
        a[ct < ct2] = 1 / 2 * (1 - np.sqrt(1-ct[ct < ct2]))
        a[ct >= ct2] = 1 + (ct[ct >= ct2] - ct1)/(4*np.sqrt(ct1) - 4)
    else:
        if ct >= ct2:
            a = 1 + (ct - ct1)/(4*np.sqrt(ct1) - 4)
        else:
            func_a = lambda a_in: 4*a_in*np.sqrt(1-a_in*(2*np.cos(yaw*np.pi/180)-a_in)) - ct
            a = fsolve(func_a, np.array(0.2))
            a = a[0]

            # a = 1 / 2 * (1 - np.sqrt(1 - ct))

    return a


def prandtl_correct(B, R, lam_tsr, a, r_start, centroid=None):
    """This function corrects for tip and root effects"""
    is_scalar = isinstance(a, float)

    CORREC   = True
    COSINE   = True
    if CORREC == True:
        if not is_scalar:
            if COSINE == True:
                step = np.pi/(len(a)-1)
                r    = np.arange(0,np.pi+step,step)
                r    = ((1+np.cos(r))/2.)*0.8+0.2
                mu   = r/R

            else:
                r    = np.linspace(r_start*R,R,len(a))
                mu   = r/R

        else:
            mu = centroid

        f_tip = 2/np.pi*np.arccos(np.exp(-B/2*(1-mu)/mu*np.sqrt(1+(lam_tsr*mu)**2/(1-a)**2)))
        f_root = 2/np.pi*np.arccos(np.exp(-B/2*(mu-r_start)/mu*np.sqrt(1+(lam_tsr*mu)**2/(1-a)**2)))

        if not is_scalar:
            f_tip[np.isnan(f_tip)] = 0
            f_root[np.isnan(f_root)] = 0
        else:
            if np.isnan(f_tip):
                f_tip = 0
            if np.isnan(f_root):
                f_root = 0

        f_total = f_tip*f_root

        if not is_scalar:
            f_total[f_total <= 0.0001] = 0.0001
        else:
            if f_total <= 0.0001:
                f_total = 0.0001
    else:
        if not is_scalar:
            r = np.linspace(r_start*R,R,len(a))
            mu = r/R
            print('True')
        else:
            mu = centroid
        f_total = 1
        f_tip = 1
        f_root = 1

    return f_total, mu, f_tip, f_root


def import_polars():
    """"This functions reads the CSV file and converts it to a dataframe"""
    polars = pd.read_csv("data/DU95W180.csv", sep=",")
    polars.alpha = polars.alpha*np.pi/180

    return polars


def twist_chord(centroid,pitch):
    """This function returns the twist and chord for a given annulus centroid"""
    twist = (14*(1-centroid)-pitch)*np.pi/180
    chord = 3*(1-centroid) + 1
    return twist, chord


def load_element(U_inf,a,a_tan,rot_speed,centroid,R,rho,pitch,polars,yaw):
    """This function calculates the normal and tangential loads on an annulus"""
    twist, chord = twist_chord(centroid,pitch)

    ## Add yaw corrections to this section on all places where deemed necessary.
    U_r = U_inf*np.cos(yaw*np.pi/180)*(1-a)
    U_tan = (1+a_tan)*rot_speed*centroid*R

    U_tot = np.sqrt(U_r**2 + U_tan**2)

    phi = np.arctan(U_r/U_tan)
    alpha = phi - twist
    cl = np.interp(alpha, polars.alpha, polars.cl)
    cd = np.interp(alpha, polars.alpha, polars.cd)
    L = 1/2*cl*chord*rho*U_tot**2
    D = 1/2*cd*chord*rho*U_tot**2
    f_norm = L*np.cos(phi) + D*np.sin(phi)
    f_tan = L*np.sin(phi) - D*np.cos(phi)
    gamma = 0.5*U_tot*cl*chord

    return f_norm, f_tan, alpha, phi, twist, gamma

## TODO hier
def pittpeters(ct, v_induction, U_inf, R, dt, glauert=True):
    # Determine induction and thrust coefficient
    a = -v_induction / U_inf
    ct_new = - get_ct(a, glauert)

    # Calculate the time derivative of the induced velocity
    dv_inductiondt = (ct - ct_new) / (16 / (3 * np.pi)) * (U_inf ** 2 / R)

    # Perform time integration
    v_induction_new = v_induction + dv_inductiondt * dt

    return v_induction_new, dv_inductiondt

## TODO hier
def get_v_induction_omega(U_inf=10, R=50, omega=1, yaw=0, ct_cond=[-0.5,-0.9], step=True):
    dt = 0.005
    time = np.arange(0, 20, dt)

    # omega = omega * U_inf / R
    ct0 = np.array([ct_cond[0]])
    if step:
        delta_ct = np.array([ct_cond[1] - ct_cond[0]])
    else:
        delta_ct = np.array([ct_cond[1]])

    v_induction = np.zeros([np.size(time)])
    ct = np.zeros([np.size(time)])

    ct[0] = ct0
    v_induction[0] = get_a(-ct0,yaw) * (-U_inf)

    if step:
        for i in range(len(time[:-1])):
            ct[i + 1] = ct0 + delta_ct
            v_induction[i + 1] = pittpeters(ct[i + 1], v_induction[i], U_inf, R, dt)[0]
    else:
        for i, timeval in enumerate(time[:-1]):
            ct[i + 1] = ct0 + delta_ct * np.sin(omega * U_inf / R * timeval)
            v_induction[i + 1] = pittpeters(ct[i + 1], v_induction[i], U_inf, R, dt)[0]

    return v_induction, ct


def perform_bem(B,U_inf,rot_speed,R,r_start,pitch,lam_tsr=7,n_annuli=10,rho=1,yaw=0,cosine=True,reduced_freq=1,ct_cond=[-0.5,-0.9], step=True):
    """This function performs the BEM iteration on all annuli"""
    polars = import_polars()

    if cosine == True:
        mu   = np.linspace(0, np.pi,n_annuli+1)
        mu   = 1 + np.cos(mu)
        mu   = (mu/2)*0.8+0.2
        mu   = np.flip(mu)
    else:
        mu = np.linspace(r_start, 1, n_annuli+1)

    inners = np.zeros(n_annuli)
    outers = np.zeros(n_annuli)
    centroids = np.zeros(n_annuli)
    areas = np.zeros(n_annuli)
    for i in range(n_annuli):
        inner = mu[i]
        outer = mu[i+1]

        centroids[i] = (inner + outer)/2
        areas[i] = np.pi*((outer*R)**2 - (inner*R)**2)
        inners[i] = inner
        outers[i] = outer

    resultant_error = 0.001

    all_a            = []
    all_a_tan        = []
    all_f_norm       = []
    all_f_tan        = []
    all_n_iterations = []
    all_alpha        = []
    all_phi          = []
    all_twist        = []
    all_f_tip        = []
    all_f_root       = []
    all_gamma        = []

    v_induction_interp, ct_interp = get_v_induction_omega(U_inf, R, omega=reduced_freq, ct_cond=ct_cond,step=step)
    # plt.plot(ct_interp,v_induction_interp)
    # plt.show()
    k = 1
    for inner, outer, area, centroid in zip(inners, outers, areas, centroids):
        #print(k)
        a = 0.5
        a_tan = 0.

        a_new = 0.33
        i = 1
        while np.abs(a-a_new) > resultant_error:
            f_norm, f_tan, alpha, phi, twist, gamma = load_element(U_inf,a,a_tan,rot_speed,centroid,R,rho,pitch,polars,yaw)
            load = f_norm*R*(outer - inner)*B
            ct = load/(1/2*rho*area*(U_inf*np.cos(yaw*np.pi/180))**2)

            ## TODO Here we include the unsteady inflow
            # a_new = get_a(ct, yaw)
            a_new = -np.interp(ct, ct_interp,v_induction_interp/U_inf)

            prandtl_total, mu, f_tip, f_root = prandtl_correct(B, R, lam_tsr, a_new, r_start, centroid)
            a_new = a_new/prandtl_total
            a = 0.75 * a + 0.25 * a_new

            a_tan = f_tan*B/(2*np.pi*U_inf*np.cos(yaw*np.pi/180)*(1-a)*rot_speed*2*(centroid*R)**2)
            a_tan = a_tan/prandtl_total
            if np.isnan(f_tan) == True:
                f_tan  = 0
                f_norm = 0
                a      = 0
                a_tan  = 0
            i += 1
            if i > 1000:
                print('Does not converge')
                TRIGGER = True
                break
            else:
                TRIGGER = False

        n_iterations = i-1

        if TRIGGER  == False:
            all_a.append(a)
            all_a_tan.append(a_tan)
            all_f_norm.append(f_norm)
            all_f_tan.append(f_tan)
            all_n_iterations.append(n_iterations)
            all_alpha.append(alpha)
            all_phi.append(phi)
            all_twist.append(twist)
            all_f_tip.append(f_tip)
            all_f_root.append(f_root)
            all_gamma.append(gamma)
        if TRIGGER  == True:
            all_a.append(0)
            all_a_tan.append(0)
            all_f_norm.append(0)
            all_f_tan.append(0)
            all_n_iterations.append(0)
            all_alpha.append(0)
            all_phi.append(0)
            all_twist.append(0)
            all_f_tip.append(0)
            all_f_root.append(0)
            all_gamma.append(0)

        k += 1

    results = {"a":all_a, "a_tan":all_a_tan, "centroid":centroids, "f_norm":all_f_norm, "f_tan":all_f_tan,
               "n_iterations":all_n_iterations, "alpha":all_alpha, "phi":all_phi, "twist":all_twist, "f_tip":all_f_tip,
               "f_root":all_f_root, "gamma":all_gamma, "areas":areas, "inners":inners, "outers":outers}
    df_results = pd.DataFrame(data=results)

    norm_annulus_size = outer - inner

    return df_results, norm_annulus_size, TRIGGER


