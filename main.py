"""This is the main file to control the output"""

import numpy as np
from functions import perform_bem
import matplotlib.pyplot as plt

def BEM(annuli,lam_tsr, yaw_angle, cosine, reduced_freq, ct_cond, step):
    """ This function deals with assignment 1, we can also just use a different file"""
    ## Define Parameters
    B = 3
    R = 50 #m
    lam_tsr = lam_tsr
    r_start = 0.2
    U_inf = 10 #m/s
    rot_speed = U_inf*lam_tsr/R
    pitch = 2 #deg


    ## Solve Bem
    results_df, norm_annulus_size, TRIGGER = perform_bem(B,U_inf,rot_speed,R,r_start,pitch,n_annuli=annuli,yaw=yaw_angle,cosine=cosine, reduced_freq=reduced_freq,ct_cond=ct_cond, step=step)
    ct_sum = np.sum(results_df.f_norm*norm_annulus_size*R*B / (0.5 * np.pi * U_inf**2 * R**2))
    cp_sum = np.sum(results_df.f_tan*results_df.centroid*norm_annulus_size*R*R*B*rot_speed/(0.5 * np.pi * U_inf**3 * R**2))

    return results_df, ct_sum, cp_sum, r_start, rot_speed, TRIGGER

if __name__ == "__main__":
    annuli = 50
    lam_tsr = 10
    yaw_angle = 0
    cosine = False
    step = True

    ## Pitt Peters
    reduced_freq = [1.5]


    ct_cond_step = [[-0.5,-0.9],[-0.9,-0.5],[-0.2,-1.1],[-1.1,-0.4]]
    ct_cond_sin = [[-0.5,-0.5],[-0.9,-0.3],[-0.2,-0.7]]

    if step:
        ct_conds = ct_cond_step
    else:
        ct_conds = ct_cond_sin

    for elem in reduced_freq:
        fig, ax = plt.subplots()
        for ct_cond in ct_conds:
            results_df, _, _, _, _, _ = BEM(annuli,lam_tsr, yaw_angle, cosine, elem, ct_cond, step)
            ax.plot(results_df.centroid, results_df.a,label=f"k = {elem}, ct = {ct_cond}")
            ax.legend()
        plt.show()
