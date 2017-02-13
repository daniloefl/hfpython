#!/usr/bin/env python

import sys
import numpy as np
import scipy.special

class bcolors:
    HEADER = '\033[4m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    ENDC = '\033[0m'

eV = 27.2113966413442 # Hartrees

def factorial(n):
    if n == 1:
        return 1
    elif n == 0:
        return 1
    elif n < 0:
        return 1
    return n * factorial(n-1)

#kappa(q_r, l_{\mu, k}, l_{\mu, o}, b_{\mu,k}, b_{\mu,o}, l) =
#  = int_{s_r} s_r^l_{\mu, k} s_r^l_{\mu, o} exp(- s_r^2 b_{\mu,k}) exp(- s_r^2 b_{\mu,o}) r<^l/r>^(l+1)   ds_r
# for s_r = exp(y) = exp(i*dx +xmin), with xmin = 1e-4 and dx = 1e-2, with N=1500
# ds_r = s_r dy = s_r dx
def kappa(qr, l1, l2, b1, b2, l):
  k = 0
  dx = 1e-2
  N = 1500
  xmin = 1e-4
  for ir in range(0, N):
    sr = np.exp(ir*dx + xmin)
    rl = sr
    rh = qr
    if sr > qr:
      rl = qr
      rh = sr
    k += sr**(l1 + l2)*np.exp(-sr**2 * (b1+b2)) * rl**l/rh**(l+1) * sr *dx
  return k

def SphHarmReal(m, l, theta, phi):
  if m < 0:
    return np.sqrt(2)*(-1.0)**m*scipy.special.sph_harm(m, l, theta, phi).imag
  elif m == 0:
    return scipy.special.sph_harm(m, l, theta, phi).real
  elif m > 0:
    return np.sqrt(2)*(-1.0)**m*scipy.special.sph_harm(m, l, theta, phi).real

def CG(j1, j2, m1, m2, j, m):
    if abs(m) > j:
        return 0
    if j1 + j2 < j:
        return 0
    if abs(j1-j2) > j:
        return 0
    if m1+m2 != m:
        return 0
    if j1 < j2:
        return (-1.0)**(j-j1-j2)*CG(j2,j1,m2,m1,j,m)
    if m < 0:
        return (-1.0)**(j-j1-j2)*CG(j1,j2,-m1,-m2,j,-m)
    if j2 == 0:
        if j == j1 and m == m1:
            return 1
        else:
            return 0
    elif j1 == 1 and j2 == 1:
        if m == 2:
            if m1 == 1 and m2 == 1 and j == 2:
                return 1
            else:
                return 0
        elif m == 1:
            if m1 == 1 and m2 == 0 and j == 2:
                return 0.5**0.5
            elif m1 == 1 and m2 == 0 and j == 1:
                return 0.5**0.5
            elif m1 == 0 and m2 == 1 and j == 1:
                return - (0.5**0.5)
            elif m1 == 1 and m2 == 0 and j == 2:
                return (0.5**0.5)
            else:
                return 0
        elif m == 0:
            if m1 == 1 and m2 == -1 and j == 2:
                return (1.0/6.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 1:
                return (1.0/2.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 0:
                return (1.0/3.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 2:
                return (2.0/3.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 1:
                return 0
            elif m1 == 0 and m2 == 0 and j == 0:
                return - (1.0/3.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 2:
                return (1.0/6.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 1:
                return - (1.0/2.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 0:
                return (1.0/3.0)**0.5
            else:
                return 0
        else:
            return 0
    elif j1 == 2 and j2 == 1:
        if m == 3:
            if m1 == 2 and m2 == 1 and j == 3:
                return 1
            else:
                return 0
        elif m == 2:
            if m1 == 2 and m2 == 0 and j == 3:
                return (1.0/3.0)**0.5
            elif m1 == 2 and m2 == 0 and j == 2:
                return (2.0/3.0)**0.5
            elif m1 == 1 and m2 == 1 and j == 3:
                return (2.0/3.0)**0.5
            elif m1 == 1 and m2 == 1 and j == 2:
                return - (1.0/3.0)**0.5
            else:
                return 0
        elif m == 1:
            if m1 == 2 and m2 == -1 and j == 3:
                return (1.0/15.0)**0.5
            elif m1 == 2 and m2 == -1 and j == 2:
                return (1.0/3.0)**0.5
            elif m1 == 2 and m2 == -1 and j == 1:
                return (3.0/5.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 3:
                return (8.0/15.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 2:
                return (1.0/16.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 1:
                return - (3.0/10.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 3:
                return (2.0/5.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 2:
                return -(1.0/2.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 1:
                return (1.0/10.0)**0.5
            else:
                return 0
        elif m == 0:
            if m1 == 1 and m2 == -1 and j == 3:
                return (1.0/5.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 2:
                return (1.0/2.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 1:
                return (3.0/10.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 3:
                return (3.0/5.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 2:
                return 0
            elif m1 == 0 and m2 == 0 and j == 1:
                return - (2.0/5.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 3:
                return (1.0/5.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 2:
                return - (1.0/2.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 1:
                return - (3.0/10.0)**0.5
            else:
                return 0
    elif j1 == 2 and j2 == 2:
        if m == 4:
            if m1 == 2 and m2 == 2 and j == 4:
                return 1
            else:
                return 0
        elif m == 3:
            if m1 == 2 and m2 == 1 and j == 4:
                return (1.0/2.0)**0.5
            elif m1 == 2 and m2 == 1 and j == 3:
                return (1.0/2.0)**0.5
            elif m1 == 1 and m2 == 2 and j == 4:
                return (1.0/2.0)**0.5
            elif m1 == 1 and m2 == 2 and j == 3:
                return - (1.0/2.0)**0.5
            else:
                return 0
        elif m == 2:
            if m1 == 2 and m2 == 0 and j == 4:
                return (3.0/14.0)**0.5
            elif m1 == 2 and m2 == 0 and j == 3:
                return (1.0/2.0)**0.5
            elif m1 == 2 and m2 == 0 and j == 2:
                return (2.0/7.0)**0.5
            elif m1 == 1 and m2 == 1 and j == 4:
                return (4.0/7.0)**0.5
            elif m1 == 1 and m2 == 1 and j == 3:
                return 0
            elif m1 == 1 and m2 == 1 and j == 2:
                return - (3.0/7.0)**0.5
            elif m1 == 0 and m2 == 2 and j == 4:
                return (3.0/14.0)**0.5
            elif m1 == 0 and m2 == 2 and j == 3:
                return -(1.0/2.0)**0.5
            elif m1 == 0 and m2 == 2 and j == 2:
                return (2.0/7.0)**0.5
            else:
                return 0
        elif m == 1:
            if m1 == 2 and m2 == -1 and j == 4:
                return (1.0/14.0)**0.5
            elif m1 == 2 and m2 == -1 and j == 3:
                return (3.0/10.0)**0.5
            elif m1 == 2 and m2 == -1 and j == 2:
                return (3.0/7.0)**0.5
            elif m1 == 2 and m2 == -1 and j == 1:
                return (1.0/5.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 4:
                return (3.0/7.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 3:
                return (1.0/5.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 2:
                return - (1.0/14.0)**0.5
            elif m1 == 1 and m2 == 0 and j == 1:
                return - (3.0/10.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 4:
                return (3.0/7.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 3:
                return - (1.0/5.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 2:
                return - (1.0/14.0)**0.5
            elif m1 == 0 and m2 == 1 and j == 1:
                return - (3.0/10.0)**0.5
            elif m1 == -1 and m2 == 2 and j == 4:
                return (1.0/14.0)**0.5
            elif m1 == -1 and m2 == 2 and j == 3:
                return - (3.0/10.0)**0.5
            elif m1 == -1 and m2 == 2 and j == 2:
                return (3.0/7.0)**0.5
            elif m1 == -1 and m2 == 2 and j == 1:
                return - (1.0/5.0)**0.5
            else:
                return 0
        elif m == 0:
            if m1 == 2 and m2 == -2 and j == 4:
                return (1.0/70.0)**0.5
            elif m1 == 2 and m2 == -2 and j == 3:
                return (1.0/10.0)**0.5
            elif m1 == 2 and m2 == -2 and j == 2:
                return (2.0/7.0)**0.5
            elif m1 == 2 and m2 == -2 and j == 1:
                return (2.0/5.0)**0.5
            elif m1 == 2 and m2 == -2 and j == 0:
                return (1.0/5.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 4:
                return (8.0/35.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 3:
                return (2.0/5.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 2:
                return (1.0/14.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 1:
                return -(1.0/10.0)**0.5
            elif m1 == 1 and m2 == -1 and j == 0:
                return -(1.0/5.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 4:
                return (18.0/35.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 3:
                return 0
            elif m1 == 0 and m2 == 0 and j == 2:
                return -(2.0/7.0)**0.5
            elif m1 == 0 and m2 == 0 and j == 1:
                return 0
            elif m1 == 0 and m2 == 0 and j == 0:
                return (1.0/5.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 4:
                return (8.0/35.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 3:
                return -(2.0/5.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 2:
                return (1.0/14.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 1:
                return (1.0/10.0)**0.5
            elif m1 == -1 and m2 == 1 and j == 0:
                return -(1.0/5.0)**0.5
            elif m1 == -2 and m2 == 2 and j == 4:
                return (1.0/70.0)**0.5
            elif m1 == -2 and m2 == 2 and j == 3:
                return -(1.0/10.0)**0.5
            elif m1 == -2 and m2 == 2 and j == 2:
                return (2.0/7.0)**0.5
            elif m1 == -2 and m2 == 2 and j == 1:
                return -(2.0/5.0)**0.5
            elif m1 == -2 and m2 == 2 and j == 0:
                return (1.0/5.0)**0.5
            else:
                return 0

    return 0


'''
Get the solid full angle in n-dimensions.
int(Omega) dOmega = (2 pi)^(n/2) / (n/2 - 1)! if d is even
int(Omega) dOmega = (2)^(n) pi^[1/2(n-1)] [1/2 / (n/2 - 1)! ] / (n-1)! if d is odd
'''

def Omega(n):
  return np.float64(2*(np.pi)**((n)/2.0)/scipy.special.gamma((n)/2.0))

'''
Calculates the matrix element of the position operator in the k-th power
in the GTO basis function set.

basis = |b,l,m>

Returns:
<b1,l1,m1| R^k | b2,l2,m2> = 
   = int(r from 0 to inf) int(Omega) exp(- r^2 b1) r^l1 Y*_{l1, m1}
                                     r^k
                                     exp(- r^2 b2) r^l2 Y_{l2, m2} dOmega dr
   = delta(l1,l2) delta(m1,m2) int(r) exp [ - r^2 (b1 + b2) ] r^[l1+l2+k] dr

n - 1 = l1+l2+k
int(r) exp [ - a r^2 ] r^[n-1] dr = ?

Changing to n-D space, r^2 = sum_i x_i^2, then: r dr = sum x_i dx_i.
dV = r^(n-1) dr d(Omega)

... = 1/[int(Omega) dOmega] int exp [ - a r^2 ] dV = 1/[int(Omega) dOmega] \prod int exp [ - a x^2 ] dx
    = (sqrt( pi / a))^n/[Omega(n)]

( int exp [ - x^2 / 2 sigma^2 ] dx = sqrt [ 2 pi sigma^2 ]
then, with a = 1/(2 sigma^2), sigma^2 = 1/(2a)
and: int exp [ - a x^2  ] dx = sqrt [ pi / a]

If n == 1, int(r) exp [ - r^2 a ] dr = sqrt( pi / a) / 2
And we know: int(r)_0^inf exp [ - r^2 a ] dr = sqrt( pi / a ) / 2

If n <= 0, we cannot use this, but we can use a recursive series:

Define v = -(n - 1) = -(l1+l2+k) > 0

int_r exp [ - a r^2 ] r^(-v) dr = r^(-v+1)/(-v+1) exp [ - a r^2 ] + 2 a r /(-v+1) int r^(-v+1) exp [ - a r^2 ] dr
f(-v, y) = int_y^inf exp [ - a r^2 ] r^(-v) dr
f(-v, y) = -y^(-v+1)/(-v+1) exp [ - a y^2 ] + 2 a y /(-v + 1) f(-v+1, y)
In the limit y -> 0, 1/0^(-v+1) -> inf ...


'''
def InnerProdGTO(b1, l1, m1, b2, l2, m2, k):
  if l1 != l2:
    return np.float64(0)
  if m1 != m2:
    return np.float64(0)
  n = l1+l2+k+1
  a = b1+b2
  if a == 0:
    return 1e10
  if n > 0:
    O = Omega(n)
    return np.float64(((np.sqrt(np.pi/a))**n)/O)
  print "Error trying to get inner product with negative exponent in r^n, where n = %d" % n
  sys.exit(-1)
  return 0
  # cannot use this logic for negative exponent in r

# returns integral_0^infinity of Integrand(x, params) dx
# params are a fixed set of parameters sent to the function Integrand
# x is the integration variable
def LegendreGaussQuadrature(Integrand, params):
  MapPos = [
(0.0486909570091397 , -0.0243502926634244),
(0.0486909570091397 , 0.0243502926634244),
(0.0485754674415034 , -0.0729931217877990),
(0.0485754674415034 , 0.0729931217877990),
(0.0483447622348030 , -0.1214628192961206),
(0.0483447622348030 , 0.1214628192961206),
(0.0479993885964583 , -0.1696444204239928),
(0.0479993885964583 , 0.1696444204239928),
(0.0475401657148303 , -0.2174236437400071),
(0.0475401657148303 , 0.2174236437400071),
(0.0469681828162100 , -0.2646871622087674),
(0.0469681828162100 , 0.2646871622087674),
(0.0462847965813144 , -0.3113228719902110),
(0.0462847965813144 , 0.3113228719902110),
(0.0454916279274181 , -0.3572201583376681),
(0.0454916279274181 , 0.3572201583376681),
(0.0445905581637566 , -0.4022701579639916),
(0.0445905581637566 , 0.4022701579639916),
(0.0435837245293235 , -0.4463660172534641),
(0.0435837245293235 , 0.4463660172534641),
(0.0424735151236536 , -0.4894031457070530),
(0.0424735151236536 , 0.4894031457070530),
(0.0412625632426235 , -0.5312794640198946),
(0.0412625632426235 , 0.5312794640198946),
(0.0399537411327203 , -0.5718956462026340),
(0.0399537411327203 , 0.5718956462026340),
(0.0385501531786156 , -0.6111553551723933),
(0.0385501531786156 , 0.6111553551723933),
(0.0370551285402400 , -0.6489654712546573),
(0.0370551285402400 , 0.6489654712546573),
(0.0354722132568824 , -0.6852363130542333),
(0.0354722132568824 , 0.6852363130542333),
(0.0338051618371416 , -0.7198818501716109),
(0.0338051618371416 , 0.7198818501716109),
(0.0320579283548516 , -0.7528199072605319),
(0.0320579283548516 , 0.7528199072605319),
(0.0302346570724025 , -0.7839723589433414),
(0.0302346570724025 , 0.7839723589433414),
(0.0283396726142595 , -0.8132653151227975),
(0.0283396726142595 , 0.8132653151227975),
(0.0263774697150547 , -0.8406292962525803),
(0.0263774697150547 , 0.8406292962525803),
(0.0243527025687109 , -0.8659993981540928),
(0.0243527025687109 , 0.8659993981540928),
(0.0222701738083833 , -0.8893154459951141),
(0.0222701738083833 , 0.8893154459951141),
(0.0201348231535302 , -0.9105221370785028),
(0.0201348231535302 , 0.9105221370785028),
(0.0179517157756973 , -0.9295691721319396),
(0.0179517157756973 , 0.9295691721319396),
(0.0157260304760247 , -0.9464113748584028),
(0.0157260304760247 , 0.9464113748584028),
(0.0134630478967186 , -0.9610087996520538),
(0.0134630478967186 , 0.9610087996520538),
(0.0111681394601311 , -0.9733268277899110),
(0.0111681394601311 , 0.9733268277899110),
(0.0088467598263639 , -0.9833362538846260),
(0.0088467598263639 , 0.9833362538846260),
(0.0065044579689784 , -0.9910133714767443),
(0.0065044579689784 , 0.9910133714767443),
(0.0041470332605625 , -0.9963401167719553),
(0.0041470332605625 , 0.9963401167719553),
(0.0017832807216964 , -0.9993050417357722),
(0.0017832807216964 , 0.9993050417357722)]
  R = 0
  for item in MapPos:
    w = item[0]
    a = item[1]
    # this would only integrate from -1 to 1
    #R += w*Integrand(x, params)*dy
    # we want to integrate from 0 to infinity
    # int_{0}^inf f(x) dx
    # x = tan(z)
    # z(x = 0) = atan(0) = 0
    # z(x = inf) = atan(inf) = pi/2
    # int_0^inf f(x) dx = int_0^{pi/2} f(tan(z)) dx/dz dz
    # dx/dz = sec^2(z)
    # So:
    # int_{0}^inf f(x) dx = int_0^{pi/2} f(x(z)) * sec^2(z) dz
    # = (np.pi/2 - 0)/2 int_{-1}^1 f( x[(np.pi/2 - 0)/2 a + (np.pi/2 + 0)/2] ) * sec^2(z(a)) da
    # a is between -1 and 1 and z = (np.pi/2 - 0)/2 a + (np.pi/2 + 0)/2
    z = np.pi/4.0*a + np.pi/4.0 # z is between 0 and pi/2
    x = np.tan(z)
    jacobian = 1.0/(np.cos(z)**2)
    R += w*np.pi/4.0*Integrand(x, params)*jacobian
  return R

