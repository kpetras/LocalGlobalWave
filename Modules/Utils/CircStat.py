
import cmath
import numpy as np
import scipy.stats as stats

def circular_distance_between_angles(angle1, angle2):
    return np.pi -np.abs(np.pi - np.abs(angle1 - angle2))

def circular_mean(angles, deg=True):
    '''Circular mean of angle data(default to degree)
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) if deg else mean, 7)

def circular_variance(angles, deg=True):
    '''Circular variance of angle data(default to degree)
    0 <= var <= 1
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r =abs(angles_complex.sum()) / len(angles)
    return round(1 - r, 7)

def circular_standard_deviation(angles, deg=True):
    '''Circular standard deviation of angle data(default to degree)
    0 <= std
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / len(angles)
    std = np.sqrt(-2 * np.log(r))
    return round(np.rad2deg(std) if deg else std, 7)

def circular_correlation(x, y, deg=True, test=False):
    '''Circular correlation coefficient of two angle data(default to degree)
    Set `test=True` to perform a significance test.
    '''
    convert = np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - circular_mean(x, deg)) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - circular_mean(y, deg)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    if test:
        l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
        test_stat = r * np.sqrt(l20 * l02 / l22)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        return tuple(round(v, 7) for v in (r, test_stat, p_value))
    return round(r, 7)

def circular_linear_correlation(alpha, x):
    n = np.size(alpha)
    rxs = np.corrcoef(x, np.sin(alpha))[0,1]
    rxc = np.corrcoef(x, np.cos(alpha))[0,1]
    rcs = np.corrcoef(np.sin(alpha), np.cos(alpha))[0,1]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))
    pval = 1-stats.chi2.cdf(n*rho**2,2)
    return rho, pval

def test():
    a = np.array([-30,  45,   0,  10, -15])
    b = np.array([200, 180, 170, 150, 210])
    assert circular_mean(a) == 1.543972
    assert circular_variance(a) == 0.0948982
    assert circular_standard_deviation(a) == 25.5860013
    assert circular_correlation(a, b) == -0.5305784
    assert circular_correlation(a, b, test=True) == (-0.5305784, -1.7436329, 0.0812231)