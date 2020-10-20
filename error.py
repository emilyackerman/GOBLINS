import numpy as np
def errorfxn(tvec, tspan, result, data, std):
    """Return residual error"""
    vdata = data[:,0]; vstd = std[:,0] 
    ifndata = data[:,1]; ifnstd = std[:,1]
    
    v_interp = np.interp(tvec, tspan, result[:,0])
    # v_err = np.nansum((np.square(v_interp - vdata))/(np.count_nonzero(~np.isnan(vdata))*vstd))
    v_err = np.nansum((np.square(v_interp - vdata))/(2*np.square(vstd)))

    ifn_interp = np.interp(tvec, tspan, result[:,1])
    # ifn_err = np.nansum((np.square(ifn_interp - ifndata))/(np.count_nonzero(~np.isnan(vdata))*ifnstd))
    ifn_err = np.nansum((np.square(ifn_interp - ifndata))/(2*np.square(ifnstd)))
    return np.nansum(v_err) + np.nansum(ifn_err)