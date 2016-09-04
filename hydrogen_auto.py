
import numpy as np
import matplotlib.pyplot as plt

eV = 27.2113966413442 # Hartrees

def V(r, Z):
    pot = np.zeros(len(r))
    for i in range(0, len(r)):
        if r[i] != 0:
            pot[i] = -Z/r[i]
	else:
	    pot[i] = pot[i-1]
    return pot

def init(dx, N, xmin):
    r = np.zeros(N)
    for i in range(0, N):
        r[i] = np.exp(xmin + i*dx)
    return r

def F(i, f, y):
    return (12 - 10*f[i])*y[i] - f[i-1]*y[i-1] - f[i+1]*y[i+1]

def getAF(r, pot, E, l, dx, m):
    icl = -1
    a = np.zeros(len(r))
    f = np.zeros(len(r))
    for i in range(0, len(r)):
        a[i] = 2*m*r[i]**2*(E - pot[i]) - (l+0.5)**2
	f[i] = 1 + a[i]*dx**2/12.0
	if icl < 0 and i >= 1 and a[i]*a[i-1] < 0:
	    icl = i
    return [a, f, icl]

# only assuming tendency from Schr. eq.:
# psi = r^l
# psi -> r R(r) * angular part
# xi = r^l/r
# y = xi/sqrt(r)
# y = r^l/r/sqrt(r)
# generic approx. solution
# from ignoring r^2 term in:
# d^2y/dx^2 + [2mr^2(E-V) - (l+0.5)^2] y = 0
# when r -> 0, x-> - infinity
# -> d^2y/dx^2 = (l+0.5)^2 y
# -> y = exp((l+0.5) x)   # notice that x -> -infinity means exp(x) converges not exp(-x)
# -> y = exp((l+0.5) ln(Zr)) = (Zr)^(l+0.5)
def outwardSolution(r, l, Z, f):
    y = np.zeros(len(r))
    no = 0
    y[0] = ((Z*r[0])**(l+0.5))/n
    y[1] = ((Z*r[1])**(l+0.5))/n
    for i in range(1, len(r)-1):
	y[i+1] = ((12 - f[i]*10)*y[i] - f[i-1]*y[i-1])/f[i+1]
    for i in range(1, len(r)-1):
	if y[i]*y[i+1] < 0:
	    no += 1
    return [y, no]

def inwardSolution(r, l, Z, f, E):
    yp = np.zeros(len(r))
    nop = 0
    m = 1
    # for r->infinity, only d^2y/dx^2 + 2mEy = 0 terms contribute
    # (remember E is negative)
    # so approximate it at r->infinity as an exponential
    yp[len(r)-1] = np.exp(-np.sqrt(-2*m*E)*r[len(r)-1])
    yp[len(r)-2] = np.exp(-np.sqrt(-2*m*E)*r[len(r)-2])
    for i in reversed(range(1, len(r)-1)):
	yp[i-1] = ((12 - f[i]*10)*yp[i] - f[i+1]*yp[i+1])/f[i-1];
	if yp[i-1] > 10:
	  for j in reversed(range(i-1, len(r)-1)):
	    yp[j] /= yp[i-1]

    for i in reversed(range(1, len(r)-1)):
	if yp[i-1]*yp[i] < 0:
	    nop += 1
    return [yp, nop]

def matchInOut(y, yp, icl):
    # renormalise
    y_ren = np.zeros(len(y))
    if icl >= 0:
        rat = y[icl]/yp[icl]
	for i in range(0, icl):
	    y_ren[i] = y[i]
	for i in range(icl, len(y)):
	    y_ren[i] = rat*yp[i]
    return y_ren

# y is the rescaled wave function
# y = 1/(sqrt(r)) * xi
# xi = r*R(r), where R(r) is the radial solution
# the xi -> y transformation prevents a first derivative from appearing in the Schroedinger eq.

# a and f re the temporary variables in the Numerov method
# we want to solve deriv(deriv(y)) + [2m_e/hbar^2 r^2 (E - V)  - (l+0.5)^2 ] y = 0
# using Numerov's method:
# a = 2 m / hbar^2 * r^2 * (E - V)  - (l+0.5)^2
# f = 1 + a * dx^2/12
# y_{n+1} = ((12 - 10 f_n) y_n - f_{n-1} y_{n-1}) / f_{n+1}
# the derivative above is a function of x, where x = log(r)
# The original equation is:
# - hbar^2 / (2 m) deriv(deriv(xi)) + [ V + ( hbar^2 l (l+1) ) / (2 m r^2) - E ] xi (r) = 0
# where in the latter eq., the derivative is in relation to r.
# One other way of doing this is to keep the r(x) logarithm grid in WF, and try to solve
# the eq. in terms of r, so it would be:
# deriv(deriv(xi)) + 2 m / hbar^2 [ E - V - (hbar^2 l (l+1) ) / 2 m r^2 ] xi (r) = 0
# in which case, a = 2 m / hbar^2 [ E - V - (hbar^2 l (l+1) ) / 2 m r^2 ] and dx is replaced by r(i) - r(i-1)
def solve(r, dx, pot, n, l, E, Z):
    m = 1.0
    icl = -1
    [a, f, icl] = getAF(r, pot, E, l, dx, m)
    
    [y, no] = outwardSolution(r, l, Z, f)
    [yp, nop] = inwardSolution(r, l, Z, f, E)
    y_ren = matchInOut(y, yp, icl)
    # y_ren is continuous
    # y_ren was estimated outward until icl
    # y_ren was estimated inward from icl
    # y_ren has a discontinuity in its first derivative at icl
    # Schroedinger's Eq. is invalid at icl, but valid elsewhere
    # deriv(deriv(y)) + [2m_e/hbar^2 r^2 (E - V)  - (l+0.5)^2 ] y = 0
    # using Numerov's method:
    # a = 2 m / hbar^2 * r^2 * (E - V)  - (l+0.5)^2
    # f = 1 + a * dx^2/12
    # y_{n+1} = ((12 - 10 f_n) y_n - f_{n-1} y_{n-1}) / f_{n+1}
    # (12 - 10 f_n) y_n - f_{n-1} y_{n-1} - f_{n+1} y_{n+1} = 0
    # but this is not zero! This is some F(E).
    # We want to find the zero of a function F(E)
    # F(E) = F(E_current) + F'(E_current) (E-E_current)
    # for E_new, F(E_new) = 0
    # dE = E_new - E_current = - F(E_current)/(F'(E_current))
    # F(E_current) = (12 - 10 f_icl) y_icl - f_{icl-1} y_{icl-1} - f_{icl+1} y_{icl+1}
    Ficl = F(icl, f, y_ren) # get F at icl
    # calculate dF/dE, by varying E very slightly
    dE = -0.1e-1
    if dE == 0:
      dE = -0.1e-1
    # recalculate the solution with a slihtly varied E
    [ap, fp, iclp] = getAF(r, pot, E+dE, l, dx, m)
    [y_ep, no_ep] = outwardSolution(r, l, Z, fp)
    [yp_ep, nop_ep] = inwardSolution(r, l, Z, fp, E)
    y_ep_ren = matchInOut(y_ep, yp_ep, icl)
    # new solution has a discontinuity at icl again
    # dF/dE is defined as the difference over dE of the change in F
    Fp = F(icl, fp, y_ep_ren)
    print "F, Fp, dF ", Ficl, Fp, Fp - Ficl
    if Fp != Ficl:
        bestdE = -Ficl*dE/(Fp - Ficl)
    else:
        bestdE = dE
    if icl < 0:
        bestdE = 10 # arbitrary, but must be positive to make energy less negative
    return [y_ren, yp, icl, no, nop, bestdE]


def toPsi(x, y):
    n = 0
    psi = np.zeros(len(y))
    for i in range(0, len(y)):
        psi[i] = y[i]*r[i]**(-0.5) # undo y->R(r) transform
	ip = len(y)-1
	if i < len(y)-1:
	    ip = i+1
	dr = np.fabs(r[ip]-r[i])
        n += (r[i]*psi[i])**2*dr        # normalise it so that int |r R(r)|^2 dr == 1
    if n != 0:
        for i in range(0, len(y)):
            psi[i] /= np.sqrt(n)
    return psi

def nodes(n, l):
    return n - l - 1

n = 1
l = 0
Z = 1
E = -0.5
Emax = -1e-3
Emin = -20.0
dx = 1e-3
r = init(dx, 13000, np.log(1e-4))
pot = V(r, Z)
no = 0
nop = 0
for i in range(0,100):
    [y, yp, icl, no, nop, bestdE] = solve(r, dx, pot, n, l, E, Z)
    dE = 0
    if no > nodes(n, l):
	Emax = E-1e-15
	dE = (Emax + Emin)*0.5 - E
    elif no < nodes(n, l):
	Emin = E+1e-15
	dE = (Emax + Emin)*0.5 - E
    else:
        dE = 1e-1*bestdE
	if np.fabs(dE) > 0.5:
	    dE = 0.5*dE/np.fabs(dE)

    print "Iteration ", i, ", E = ", E, ", dE = ", dE, ", nodes = ", no, nop, ", expected nodes = ", nodes(n, l), ", crossing zero at = ", icl
    psi = toPsi(r, y)
    psip = toPsi(r, yp)
    exact_p = 2*np.exp(-r)   # solution for R(r) in Hydrogen, n = 1
    idx = np.where(r > 5)
    idx = idx[0][0]
    if i % 10 == 0:
      plt.clf()
      plt.plot(r[0:idx], psi[0:idx], 'r-', label='$R_{-}$')
      plt.plot(r[0:idx], psip[0:idx], 'b-', label='$R_{+}$')
      plt.plot(r[0:idx], exact_p[0:idx], 'g--', label='$R_{exact}$')
      plt.legend(('$R(r)$ from 0', '$R(r)$ from +$\\infty$', 'Exact $R(r)$'), frameon=False)
      plt.xlabel('$r$')
      plt.ylabel('$|R(r)|$')
      #plt.text(0.01*0.6, psip[0]*0.85, '$E = $ %s' % E)
      #plt.text(0.01*0.6, psip[0]*0.75, '$\\Delta E = $ %s' % dE)
      plt.title('')
      plt.draw()
      plt.show()
    E += dE
    if dE > 0 and E > Emax:
        E = Emax
    elif dE < 0 and E < Emin:
        E = Emin
    prev_dE = dE
    if np.fabs(dE) < 1e-8 or np.fabs(Emax - Emin) < 1e-5:
      print "Converged to energy ", E*eV, " eV"
      break

print "Last energy ", E*eV, " eV"
    


