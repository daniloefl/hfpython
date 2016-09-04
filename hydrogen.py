#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Coulomb potential
def V(r, Z):
    pot = np.zeros(len(r))
    for i in range(0, len(r)):
        if r[i] != 0:
            pot[i] = -Z/r[i]
	else:
	    pot[i] = pot[i-1]
    return pot

# initialise Grid
def init(dx, N, xmin):
    r = np.zeros(N)
    for i in range(0, N):
        r[i] = np.exp(xmin + i*dx)
    return r

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
# the problem is that then there is a 1/r dependenci in a(r), which causes instability
def solve(r, dx, pot, n, l, E, Z):
    m = 1.0
    a = np.zeros(len(r))
    f = np.zeros(len(r))
    y = np.zeros(len(r))
    yp = np.zeros(len(r))
    icl = -1
    no = 0
    nop = 0
    for i in range(0, len(r)):
        a[i] = 2*m*r[i]**2*(E - pot[i]) - (l+0.5)**2
	f[i] = 1 + a[i]*dx**2/12.0
	if icl < 0 and i >= 1 and a[i]*a[i-1] < 0:
	    icl = i
    
    y[0] = ((Z*r[0])**(l+0.5))/n
    y[1] = ((Z*r[1])**(l+0.5))/n
    #y[0] = 2*(r[0]**(l+1))*(1 - Z*r[0]/(2*l+2))/np.sqrt(r[0])
    #y[1] = 2*(r[1]**(l+1))*(1 - Z*r[1]/(2*l+2))/np.sqrt(r[1])
    for i in range(1, len(r)-1):
	y[i+1] = ((12 - f[i]*10)*y[i] - f[i-1]*y[i-1])/f[i+1];
	if y[i]*y[i+1] < 0:
	    no += 1

    yp[len(r)-1] = np.exp(-np.sqrt(-2*m*E)*r[len(r)-1])
    yp[len(r)-2] = np.exp(-np.sqrt(-2*m*E)*r[len(r)-2])
    #yp[len(r)-1] = y[len(r)-1]
    #yp[len(r)-2] = ((12 - f[len(r)-1]*10)*yp[len(r)-1])/f[len(r)-2]
    for i in reversed(range(1, len(r)-1)):
	yp[i-1] = ((12 - f[i]*10)*yp[i] - f[i+1]*yp[i+1])/f[i-1];
	#if yp[i-1] > 10:
	#  for j in reversed(range(i-1, len(r)-1)):
	#    yp[j] /= yp[i-1]

    for i in reversed(range(1, len(r)-1)):
	if yp[i-1]*yp[i] < 0:
	    nop += 1
    return [y, yp, icl, no, nop]

def toPsi(x, y):
    n = 0
    psi = np.zeros(len(y))
    for i in range(0, len(y)):
        psi[i] = y[i]*x[i]**(-0.5) # undo y->R(r) transform
	ip = len(y)-2
	if i < len(y)-1:
	    ip = i + 1
	dr = np.fabs(r[ip]-r[i])
        n += (x[i]*psi[i])**2*dr        # normalise it so that int |r R(r)|^2 dr == 1
    if n != 0:
        for i in range(0, len(y)):
            psi[i] /= np.sqrt(n)
    return psi

def nodes(n, l):
    return n - l - 1

n = 1
l = 0
Z = 1
E = -1
Emax = 0.0
Emin = -20.0
dx = 1e-3
r = init(dx, 14000, np.log(1e-4))
pot = V(r, Z)
no = 0
nop = 0
old_nodes = -1
old_E = 0
old_dE = 0
prev_dE = 0
while E < 0:
    E = input('Guess a value for the correct energy [0 to exit]: ')
    if E > 0:
        break
    [y, yp, icl, no, nop] = solve(r, dx, pot, n, l, E, Z)
    print "Information: E = ", E, ", nodes = ", no, nop, ", expected nodes = ", nodes(n, l), ", crossing zero at = ", icl
    psi = toPsi(r, y)
    psip = toPsi(r, yp)
    exact_p = 2*np.exp(-r) # exact R(r) solution for n = 1
    idx = np.where(r > 5)
    idx = idx[0][0]
    plt.clf()
    plt.plot(r[0:idx], psi[0:idx], 'r-', label='$R_{-}$')
    plt.plot(r[0:idx], psip[0:idx], 'b-', label='$R_{+}$')
    plt.plot(r[0:idx], exact_p[0:idx], 'g--', label='$R_{exact}$')
    plt.legend(('$R(r)$ from 0', '$R(r)$ from +$\\infty$', 'Exact $R(r)$'), frameon=False)
    plt.xlabel('$r$')
    plt.ylabel('$|R(r)|$')
    plt.title('')
    plt.draw()
    plt.show()

