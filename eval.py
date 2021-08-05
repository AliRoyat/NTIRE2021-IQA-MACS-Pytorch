import numpy as np
from scipy import stats


def eval(predicted, gt):
    r_s = np.abs(stats.spearmanr(predicted, gt))[0]

    z = np.polyfit(predicted, gt, 3)
    fit_func = np.poly1d(z)
    fitted_MOS = fit_func(predicted)
    r_p = np.abs(stats.pearsonr(fitted_MOS, gt))[0]

    return round(r_s + r_p, 4), round(r_s, 4), round(r_p, 4)
