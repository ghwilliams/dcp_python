from scipy.integrate import quad as scipy_quad


def bracket_zero(func, x1, x2):
    NTRY = 50
    FACTOR = 1.6

    if x1 == x2:
        print('bracket_a_zero: Invalid range. Exiting.')
        exit(1)
    x1_t = x1
    x2_t = x2
    f1 = func(x1_t)
    f2 = func(x2_t)
    for i in range(1, NTRY):
        if f1 * f2 < 0:
            return True, x1_t, x2_t
        if abs(f1) < abs(f2):
            x1_t = x1_t + FACTOR * (x1_t - x2_t)
            f1 = func(x1_t)
        else:
            x2_t = x2_t + FACTOR * (x2_t - x1_t)
            f2 = func(x2_t)

    return False

def dcp_quad(f, xmin, xmax, args_list):
    L = xmax - xmin
    
    qlim = 1000
    r, error = scipy_quad(f, xmin, xmax, args_list,
                            epsabs=0, epsrel=1e-6, limit=qlim)   
    return r
