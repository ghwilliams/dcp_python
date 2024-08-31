import math
import sympy


class Law1994:
    def __init__(self, L0, theta):
        self.L0 = L0
        self.theta = theta

    def __call__(self, t):
        if t < 0:
            return self.L0

        return self.L0 + (self.L0 / (2 * math.pi)) * \
            (math.asin(math.sin(self.theta) * \
            math.cos(2 * math.pi * t / self.L0)) - self.theta)


class dcp_L1:
    def __init__(self, L0, e, q, s):
        self.L0 = L0
        self.e = e
        self.q = q
        self.s = s
        self.Tmax = 0

        ##
        # Defines some symbold necessary to define the law of motion
        L0_, e_, q_, s_, t_ = sympy.symbols('L0_ e_ q_ s_ t_')
        
        ##
        # Defines the law of motion expression
        L = L0_*(1 + e_*sympy.sin(q_*sympy.pi*t_/L0_)**s_)
        L = L.subs([(L0_, L0), (e_, e), (q_, q), (s_, s)])

        ##
        # Computes derivatives of L w.r.t time
        L_diff  = sympy.diff(L, t_)      # 1st derivative
        L_diff2 = sympy.diff(L, t_, 2)   # 2nd derivative
        L_diff3 = sympy.diff(L, t_, 3)   # 3rd derivative

        self.DL  = sympy.utilities.lambdify(t_, L_diff)
        self.D2L = sympy.utilities.lambdify(t_, L_diff2)
        self.D3L = sympy.utilities.lambdify(t_, L_diff3)
        

    def set_tmax(self, t):
        self.Tmax = t

    def __call__(self, t):
        if t < 0 or t > self.Tmax:
            return self.L0

        return self.L0 * (1. + self.e * math.sin(self.q * math.pi * t / self.L0) ** self.s)

    def DL(self, t):
        if t < 0 or t > self.Tmax:
            return 0

        return self.DL(t)


    def D2L(self, t):
        if t < 0 or t > self.Tmax:
            return 0
        
        return self.D2L(t)


    def D3L(self, t):
        if t < 0 or t > self.Tmax:
            return 0

        return self.D3L(t)


### alves etal: Relativistic bands in the spectrum of created particles
# in a oscillatory cavity.
#
# Equation 21
#
class L2:
    def __init__(self, L0, a):
        self.L0 = L0
        self.a = a

    def __call__(self, t):
        if t <= 0 or t > 2*self.L0:
            return self.L0

        return self.L0 + self.a * math.sin(2 * math.pi * t)
