def unsteady(Uinf,input_var,k_reduced=[0],te=20,glauert=False,ct_cond=True, name="_"):
    # Introduce time

    # do this for every U
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
    v_induction0 = a_BEM[:,0] * Uinf_lst[0]

    # Define final CT and induction
    ct1 = CT_BEM[:, 1]

    v_induction1 = get_a_from_ct(ct1,glauert) * Uinf_lst[1]

    v_induction = np.zeros((len(time),len(ct0),len(k_reduced)))
    a = np.zeros(np.shape(v_induction))
    # iterate over all annuli
    ct = np.zeros((len(time), len(ct0), len(k_reduced)))
    ct_actual = np.zeros((len(time), len(ct0), len(k_reduced)))
    for j in range(len(ct0)):
        ct[:,j,:] = ct0[j]
        v_induction[0, j, :] = v_induction0[j]
        a[0, j, :] = v_induction[0, j, :]/Uinf_lst[0]
        # a[0, j, :],ftotal = tip_root_correct(j,a[0, j, 0])
        ct_actual[0, j, :] = ct0[j]

        # iterate over all time
        Uinf = Uinf_lst[0]
        for i, t_value in enumerate(time[:-1]):
            # iterate over all reduced frequencies
            for ix, k_value in enumerate(k_reduced):
                v_induction[i + 1, j, ix] = larsen_madsen(ct[i + 1,j,ix], v_induction[i,j,ix], Uinf, r_R[j]*Radius, dt,glauert,j)

                a[i + 1, j, ix] = v_induction[i + 1, j, ix]/Uinf
                ct_actual[i + 1, j, ix] = get_ct_from_a(a[i + 1, j, ix],glauert)

    ct_actual[0, :, :] = ct_actual[1, :, :]
