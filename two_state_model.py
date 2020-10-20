def f2(u, t, p):
    """2 state ODE model of Virus and IFN"""
    k, big_k, r_ifn_v, d_v, p_v_ifn, d_ifn = p
    v, ifn = u
    v0 = 6.382
    ifn0 = 0.26
    dv = k*v*(1-v/big_k) - r_ifn_v*(ifn-ifn0)*v - d_v*v
    difn = p_v_ifn*v - d_ifn*(ifn - ifn0)
    return [dv, difn]
