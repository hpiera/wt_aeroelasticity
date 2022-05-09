def pittpeters(ct, v_induction, U_inf, R, dt, glauert=False):
  # Determine induction and thrust coefficient
  a = -v_induction / U_inf
  ct_new = - get_ct(a,glauert)

  # Calculate the time derivative of the induced velocity
  dv_inductiondt = (ct-ct_new) / (16 / (3*np.pi)) * (U_inf**2/R)

  # Perform time integration
  v_induction_new = v_induction + dv_inductiondt*dt

  return v_induction_new, dv_inductiondt

# Plot Pitt Peters dynamic inflow
def get_v_induction_omega(U_inf=10, R=50, omega=1):
  dt = 0.005
  time = np.arange(0,20,dt)

  omega = omega * U_inf/R
  ct0 = np.array([-0.5])
  delta_ct = np.array([-0.35])

  v_induction = np.zeros([np.size(time)])
  ct = np.zeros([np.size(time)])

  ct[0] = ct0
  v_induction[0] = get_a(-ct0)*(-U_inf)

  for i, timeval in enumerate(time[:-1]):
      ct[i+1] = ct0 + delta_ct*np.sin(omega*U_inf/R*timeval)
      v_induction[i+1] = pittpeters(ct[i+1], v_induction[i], U_inf, R, dt)[0]

  return v_induction, ct