import numpy as np
import engine.pyhyd
import engine.vcdm
from flask import flash


def vcdm_calc(df, par_list):
    '''VCDM model calculation.

    Args:
        None

    Returns:
       results (dictinary):

    '''

    if len(df.columns) == 2:
        times_v, Q_v = df.values.T

    elif len(df.columns) > 3:
        times_v, Q_v = df.iloc[:, 0:2].values.T

    else:
        flash("Unkown data format.")

    # Extract parameters from input list
    L = par_list[0]
    D = par_list[1]
    k_s = par_list[2]

    removal_rate = par_list[3]
    regen_rate = par_list[4]
    alpha = par_list[5]
    bg = par_list[6]

    # Regen rate conversion to inverse seconds
    regen_rate = 1 / (regen_rate * 2.628e+6)

    # in Pa
    applied_shear_v = engine.pyhyd.shear_stress(D, Q_v, k_s)

    # i.e. 99 strength bands
    strength_v = np.linspace(applied_shear_v.min(), applied_shear_v.max(), 100)

    # i.e. all 99 strength bands initially set to fully regenerated i.e. 1
    init_cond_v = np.ones(len(strength_v) - 1)

    # Create a model
    # Run VCDM model to generate a material flux profile
    # Use Lagrangian transport model to give turbidity prediction
    model = engine. vcdm.VCDM(times_v, strength_v, init_cond_v,
                              applied_shear_v, removal_rate, regen_rate, alpha)

    model.sim(cython=False)

    turb_pred_v = model.advect(Q_v=Q_v, D=D, L=L, upstream_conc_v=None,
                               init_num_segs=100, conc_tol=0.001)

    phi = []
    percentile_value = []
    for percentile in (25, 50, 75):
        value = np.percentile(strength_v, percentile)
        phi.append(model.cond_m[:, (np.absolute(strength_v - value)).argmin()])
        percentile_value.append([percentile, value])

    results = {'time': times_v,
               'phi': phi,
               'percentile_value': percentile_value,
               'shear': applied_shear_v,
               'turbidity': turb_pred_v + bg}

    return results
