#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------- global variables ----------

# atomic number
Z = 1  # stop here for H
Z = 2  # stop here for He
Z = 3  # stop here for Li
#Z = 4  # stop here for Be
#Z = 5  # stop here for B

# also change below to get the orbital configuration of the electrons
# consistent with this ...

# set this to true to show individual messages when scanning the energy
# can be annoying ...
debug = False
debug = True

# conversion from Hartree to eV
eV = 27.2113966413442 # 1 Hartree = 2 Rydberg, Bohr radius a_0 = 1, electron mass = 1, h/4pi = 1
nm = 0.052917721092 # Bohr radius

# size of the Grid in ln(rZ)
dx = 1e-3

# precision required when scanning energies
# change in energies in separate steps must be < eps for convergence
#eps = 1e-10
eps = 1e-11

# minimum value of r in the grid is rmin = exp(xmin)/Z
xmin = np.log(1e-4)

# number of Grid points
# if this is too little, there are convergence problems as the
# boundary condition wave function -> 0 as r->infinity fails
# if this is too high, when integrating the wave function from r = infinity
# down to r = 0, the wave function goes to infinity at r = 0 (actually NaN)
# choose this wisely
# 14000 is a good number, but you can reduce it to speed things up, if possible
# rmax = np.exp(xmin+(N-1)*dx)
# for the default values (xmin = log(1e-4), dx = 1e-3, N = 11000):
# rmax = 5.98 a_0
# (r is in Bohr radius units (a_0) ... here a_0 = 1)
# (6 Hydrogen atom radii seem reasonable, bu with N =14000, you can get 120 H radii)
# 13000 -> 44 H radii
N = 15000

# Coulomb potential
def Vcoulomb(r, Z):
    pot = np.zeros(len(r), dtype = np.longdouble)
    for i in range(0, len(r)):
        if r[i] != 0:
            pot[i] = -Z/r[i]
	else:
	    pot[i] = pot[i-1]
    return pot

# these functions depend on the transformation R -> xi = r R -> y = 1/sqrt(r) xi = sqrt(r) R
# they must change when changing the variable of integration (ie: the Grid)

# Grid is finer close to r = 0 and wider as r -> infinity
# r[i] = exp(xmin + i*dx)/C
# where i goes from 0 to N
# Z is the atomic number (to make the Grid finer for atoms that have higher Z)
# the reason for Z in this is that the Coulomb potential is stronger for
# atoms with higher Z, so more detail close to r = 0 is needed
# getAF() has been written with this Grid format in mind
# other Grids can be tried, but then getAF() needs to be changed, since
# the derivative in the Schr. equation now are taken as a function of
# x = ln(C*r) (that is: r = exp(x)/C)
# dx/dr = 1/r (that is: dr/dx = r)
def init(N, xmin, C):
    r = np.zeros(N, dtype = np.longdouble)
    for i in range(0, N):
        r[i] = np.exp(xmin + i*dx)/C
    return r

def norm(x, y):
    n = 0
    for i in range(0, len(y)):
        dx = 0
	ip = len(y)-2
	if i < len(y)-1:
	    ip = i + 1
	dx = np.fabs(x[ip]-x[i])
        n += (x[i]*y[i])**2*dx        # normalise it so that int |r R(r)|^2 dr == 1
    return np.sqrt(n)

# transform y back into R(r)
# and normalise it so that int R(r)^2 r^2 dr = 1
def toPsi(x, y):
    n = 0
    psi = np.zeros(len(y), dtype = np.longdouble)
    for i in range(0, len(y)):
        psi[i] = y[i]*x[i]**(-0.5) # undo y->R(r) transform
	ip = len(y)-2
	if i < len(y)-1:
	    ip = i + 1
	dx = np.fabs(x[ip]-x[i])
        n += (x[i]*psi[i])**2*dx        # normalise it so that int |r R(r)|^2 dr == 1
    if n != 0:
        for i in range(0, len(y)):
            psi[i] /= np.sqrt(n)
    return psi

# calculate auxiliary functions a and f
# a is the function in the Schr. eq as in:
# deriv(deriv(y)) + a * y = E * y
# f is necessary in the Numerov method
# icl is just the index where a changes sign (a reference for later)

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
# if V includes terms that do not depend on y (ie: the terms in Vex)
# then for V y = V_1 y + V_2,
# the equation is:
# deriv(deriv(y)) + [2m_e/hbar^2 r^2 (E - V_1)  - (l+0.5)^2 ] y - 2 m_e/hbar^2 r^2 V_2 = 0
# deriv(deriv(y)) + [2m_e/hbar^2 r^2 (E - V_1)  - (l+0.5)^2 ] y = 2 m_e/hbar^2 r^2 V_2
# the independent term 2 m_e/hbar^2 r^2 V_2 can be included in Numerov's method as:
# 
# y_{n+1} = ((12 - 10 f_n) y_n - f_{n-1} y_{n-1} + (s_{n-1} + 10 s_{n} + s_{n+1}) ) / f_{n+1}
# where s_{n} = dx^2/12 V_2_{n} 2 m_e/hbar^2 r^2
def getAF(r, pot, ind, E, l, m):
    icl = -1
    a = np.zeros(len(r), dtype = np.longdouble)
    f = np.zeros(len(r), dtype = np.longdouble)
    s = np.zeros(len(r), dtype = np.longdouble)
    for i in range(0, len(r)):
        a[i] = 2*m*r[i]**2*(E - pot[i]) - (l+0.5)**2
	f[i] = 1 + a[i]*dx**2/12.0
  	s[i] = (dx**2)/12.0*(ind[i]*np.sqrt(r[i]))*2*m*(r[i]**2)
        if icl < 0 and i >= 1 and a[i]*a[i-1] < 0:
            icl = i
    return [a, f, s, icl]

# for each orbital
class Orbital:
    n = 1          # principal quantum number
    l = 0          # orbital quantum number
    Emax = 0   # arbitrary highest energy to scan for (if zero is allowed, that solution diverges causing NaNs)
    Emin = -20     # arbitrary lowest energy to scan for
    E = -3         # energy eigenvalue in Hartree-Fock equation
    V = None       # effective potential felt by this electron
    no = 0         # number of zeros in wave function integrated outward (should be n - l - 1)
    nop = 0        # number of zeros in wave function integrated inward (should be n - l - 1)
    # copy of the grid r
    r = None
    Niter = 10000  # number of iterations to use to converge on Energy in Sturm-Liouville problem
    
    y = None       # wave function integrated outward
    yp = None      # wave function integrated inward
    yfinal = None  # wave function amalgamation between y and yp
    icl = -1       # index in r, where the potential changes sign

    Vd = None      # Hartree-Fock Coulomb potential
    Vex = None     # Hartree-Fock exchange potential
    Hex = None     # Hartree-Fock exchange energy
    psifinal = None # final, normalised R(r) function (full radial WF is r*R(r))

    spin = 0       # spin of this particle
    prev_dE = 0
    prev_no = 0

    def __init__(self, _n, _l, _Z, _r, _spin):
        self.n = _n
	self.l = _l
	self.Z = _Z
	## this is just a good first guess: the lowest energy is that when the electron is alone
	## with the nucleus (the other electrons' repulsion only increase it) and this is just
	## the energy in a Hydrogen atom if it had atomic number Z, which is Z^2/2 in Hartree atomic units
	self.Emax = 0  # don't let it become zero
	self.Emin = -self.Z**2-10
	self.E = -self.Z**2*0.5/(_n**2)

	self.r = _r
	self.V = Vcoulomb(self.r, self.Z)
	self.Vd = np.zeros(len(r), dtype = np.longdouble)
	self.Vex = np.zeros(len(r), dtype = np.longdouble)
	self.Hex = np.zeros(len(r), dtype = np.longdouble)
	self.spin = _spin
	pass

    # when counting the number of zeros in the solution
    # after the solution gets to an exponential decay, it can
    # have numerical fluctuations that cause extra "artificial zeroes"
    # we should avoid counting those zeroes as the number of zeroes is
    # used to distinguish between solutions to different principal numbers
    def getMaxIdxForNodeCount(self):
        # gets the last index for now
	return len(self.r)-1

    # this function is zero for neighbour points that
    # are solution of the Numerov equation
    def F(self, i, f, indepTerm, y):
        return (12 - 10*f[i])*y[i] - f[i-1]*y[i-1] - f[i+1]*y[i+1] + (indepTerm[i+1] + 10.0*indepTerm[i] + indepTerm[i-1])

    # assume the r->0 behaviour of the Hydrogen atom
    # and solve the Schr. equation from that
    # also count the number of zeroes of the solution
    def outwardSolution(self, m, indepTerm, E, f):
        y = np.zeros(len(self.r), dtype = np.longdouble)
        no = 0
	mu = 1 # TODO
	a = 1.0/mu

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
	N = 1
        y[0] = ((self.Z*self.r[0])**(np.sqrt((self.l+0.5)**2)))/N
        y[1] = ((self.Z*self.r[1])**(np.sqrt((self.l+0.5)**2)))/N
        for i in range(1, len(r)-1):
	    if f[i+1] != 0:
      	        y[i+1] = ((12 - f[i]*10)*y[i] - f[i-1]*y[i-1] + (indepTerm[i+1] + 10.0*indepTerm[i] + indepTerm[i-1]))/f[i+1]
	    else:
	        # f[i+1] = 0 -> y[i+1]*f[i+1] = 0
	        y[i+1] = y[i]

        for i in range(2, self.getMaxIdxForNodeCount()):
    	    if y[i]*y[i-1] < 0:
    	        no += 1
        return [y, no]
    
    # assume the r->infinity behaviour of the orbitals (ie: assume they tend to zero)
    # and solve the Schr. equation from that
    # also count number of zeroes of the solution
    def inwardSolution(self, m, indepTerm, E, f):
        yp = np.zeros(len(self.r), dtype = np.longdouble)
        nop = 0
  	N = 10.0
        yp[len(self.r)-1] = np.exp(-np.sqrt(-2*m*E)*self.r[len(self.r)-1])/N
        yp[len(self.r)-2] = np.exp(-np.sqrt(-2*m*E)*self.r[len(self.r)-2])/N
        for i in reversed(range(1, len(self.r)-1)):
	    if f[i-1] != 0:
                yp[i-1] = ((12 - f[i]*10)*yp[i] - f[i+1]*yp[i+1] + (indepTerm[i+1] + 10.0*indepTerm[i] + indepTerm[i-1]))/f[i-1];
            else:
                yp[i-1] = yp[i]
    
        for i in reversed(range(0, self.getMaxIdxForNodeCount()-2)):
      	    if yp[i]*yp[i+1] < 0:
    	        nop += 1
        return [yp, nop]
    
    # return a function that is identically y up until index icl
    # and that is yp afterwards, rescaling the yp part so that it matches
    # y in icl (ie: so that it is continuous)
    def matchInOut(self, y, yp, icl):
        # renormalise
        y_ren = np.zeros(len(y), dtype = np.longdouble)
	rat = 1
        if icl >= 0 and y[icl] != 0 and yp[icl] != 0:
            rat = y[icl]/yp[icl]
    	for i in range(0, icl):
    	    y_ren[i] = y[i]
    	for i in range(icl, len(y)):
    	    y_ren[i] = rat*yp[i]
        return y_ren

    # calculate dF/dE, by varying E very slightly
    # but we cannot change the number of zeroes of the function
    # so if this variation causes that, reduce the size of the shift
    # otherwise we would be comparing solutions with different number of zeroes
    # which correspond to different principal numbers/energies
    def solveDiff(self, r, pot, indepTerm, E, m):
        [a, f, s, icl] = getAF(r, pot, indepTerm, E, self.l, m)
        [y, no] = self.outwardSolution(m, s, E, f)
        [yp, nop] = self.inwardSolution(m, s, E, f)
        y_ren = self.matchInOut(y, yp, icl)

	no_ep = -1
	nop_ep = -1
        dE = -0.2*E
        if dE == 0:
            dE = -0.01
	while no_ep != no and nop_ep != nop:
	    dE = dE*0.5 # half it if the previous iteration changed number of zeroes
            # recalculate the solution with a slihtly varied E
            [ap, fp, sp, iclp] = getAF(r, pot, indepTerm, E+dE, self.l, m)
            [y_ep, no_ep] = self.outwardSolution(m, sp, E+dE, fp)
            [yp_ep, nop_ep] = self.inwardSolution(m, sp, E+dE, fp)
            y_ep_ren = self.matchInOut(y_ep, yp_ep, icl)
        Ficl = self.F(icl, f, s, y_ren) # get F at icl
        Fp = self.F(icl, fp, sp, y_ep_ren)
	return (Fp - Ficl)/dE

    def solveDiff2(self, r, pot, indepTerm, E, m):
        [a, f, s, icl] = getAF(r, pot, indepTerm, E, self.l, m)
        [y, no] = self.outwardSolution(m, s, E, f)
        [yp, nop] = self.inwardSolution(m, s, E, f)
        y_ren = self.matchInOut(y, yp, icl)

	no_ep = -1
	nop_ep = -1
        dE = -0.2*E
        if dE == 0:
            dE = -0.01
	while (no_ep != no or nop_ep != nop) or (no_ep2 != no or nop_ep2 != nop):
	    dE = dE*0.5 # half it if the previous iteration changed number of zeroes
            # recalculate the solution with a slihtly varied E
            [ap, fp, sp, iclp] = getAF(r, pot, indepTerm, E+dE, self.l, m)
            [y_ep, no_ep] = self.outwardSolution(m, sp, E+dE, fp)
            [yp_ep, nop_ep] = self.inwardSolution(m, sp, E+dE, fp)
            y_ep_ren = self.matchInOut(y_ep, yp_ep, icl)

            [ap2, fp2, sp2, iclp2] = getAF(r, pot, indepTerm, E+2*dE, self.l, m)
            [y_ep2, no_ep2] = self.outwardSolution(m, sp2, E+2*dE, fp2)
            [yp_ep2, nop_ep2] = self.inwardSolution(m, sp2, E+2*dE, fp2)
            y_ep_ren2 = self.matchInOut(y_ep2, yp_ep2, icl)
        Ficl = self.F(icl, f, s, y_ren) # get F at icl
        Fp = self.F(icl, fp, sp, y_ep_ren)
        Fp2 = self.F(icl, fp2, sp2, y_ep_ren2)
	Fdiff = (Fp - Ficl)/dE
	F2diff = (Fp2 - Fp)/dE
	Fdiff2 = (F2diff - Fdiff)/dE
	return Fdiff2
        

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
    # One other way of doing this is to keep the r(x) logarithm grid and try to solve
    # the eq. in terms of r, so it would be:
    # deriv(deriv(xi)) + 2 m / hbar^2 [ E - V - (hbar^2 l (l+1) ) / 2 m r^2 ] xi (r) = 0
    # in which case, a = 2 m / hbar^2 [ E - V - (hbar^2 l (l+1) ) / 2 m r^2 ] and dx is replaced by r(i) - r(i-1)
    def solve(self, r, pot, indepTerm, E):
        m = 1.0
        icl = -1
        [a, f, s, icl] = getAF(r, pot, indepTerm, E, self.l, m)
        
        [y, no] = self.outwardSolution(m, s, E, f)
        [yp, nop] = self.inwardSolution(m, s, E, f)
        y_ren = self.matchInOut(y, yp, icl)
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
        # F(E) = F(E_current) + F'(E_current) (E-E_current) + F''(E_current) (E-E_current)^2/2.0
        # for E_new, F(E_new) = 0
        # dE = E_new - E_current = - F(E_current)/(F'(E_current))
        # F(E_current) = (12 - 10 f_icl) y_icl - f_{icl-1} y_{icl-1} - f_{icl+1} y_{icl+1}
        Ficl = self.F(icl, f, s, y_ren) # get F at icl

	if np.isnan(yp[0]) or np.isinf(yp[0]):
	    # the inward solution is too unstable
	    # use the requirement that the outwad solution must match zero at infinity instead
	    # F(E) = y[infinity] - 0
            # get delta E using condition that y[infinity] from outward integration is zero
	    # this is the backup method in case comparing the functions at icl fails
	    # F(E) = y[infinity] - 0
            dE_inf = 0.1*E
            # recalculate the solution with a slihtly varied E
            [ap_inf, fp_inf, sp_inf, iclp_inf] = getAF(r, pot, indepTerm, E+dE_inf, self.l, m)
            [y_ep_inf, no_ep_inf] = self.outwardSolution(m, sp_inf, E+dE_inf, fp)
            [yp_ep_inf, nop_ep_inf] = self.inwardSolution(m, sp_inf, E+dE_inf, fp)
            y_ep_ren_inf = self.matchInOut(y_ep_inf, yp_ep_inf, icl)
            F_curr = y[len(r)-1]
	    F_diff = y_ep_inf[len(r)-1] - y[len(r)-1]
	    bestdE_inf = -F_curr*dE_inf/F_diff
	    bestdE = bestdE_inf
            y_ren = y
	    yp = y
	    nop = no
	else:
            # new solution has a discontinuity at icl again
            # dF/dE is defined as the difference over dE of the change in F
            Fdiff = self.solveDiff(r, pot, indepTerm, E, m)
            #Fdiff2 = self.solveDiff2(r, pot, indepTerm, E, m)
            if not np.isnan(Fdiff):
                bestdE = -Ficl/Fdiff
		# we could also go until the quadratic term
		# but it has been checked it makes a very small difference and the loss in performance is enormous ...
                # F(E_current) + F'(E_current) (E-E_current) + F''(E_current) (E-E_current)^2/2.0 = 0
		#ca = Fdiff2/2.0
		#cb = Fdiff
		#cc = Ficl
		#if cb**2 < 4*ca*cc:
		#  bestdE = -Ficl/Fdiff
		#else:
                #   bestdE1 = -cb/(2.0*ca) - np.sqrt(cb**2 - 4*ca*cc)/(2.0*ca)
                #   bestdE2 = -cb/(2.0*ca) + np.sqrt(cb**2 - 4*ca*cc)/(2.0*ca)
		#   bestdE = bestdE1
		#   if np.fabs(bestdE2) < np.fabs(bestdE1):
		#       bestdE = bestdE2
            else:
                bestdE = dE
        if icl < 0:
            bestdE = 0.5 # arbitrary, but must be positive to make energy less negative
        return [y, yp, y_ren, icl, no, nop, bestdE]
    
    
    # return the number of zeroes expected in the solution that has n and l
    # as quantum numbers
    def nodes(self):
        return self.n - self.l - 1
    
    # loop over energies to solve the Schr. equation and adapt the energy until a consistent
    # solution is found and a valid energy is available
    def solveWithCurrentPotential(self, label = 1):
        self.Emax = 0
	self.Emin = -self.Z**2-10
        Nscale = 1.0
	mismatchNodes = False
        for i in range(0, self.Niter):
	    # solve Schroedinger equation using self.E as energy guess
	    # solves it using Numerov's method assuming initial solution at r->0 (y) and
	    # assuming y = 0 for r->infinity (yp)
	    # then it rescales yp to y and returns y as the solution with boundary conditions at r=0,1
	    # up until icl and after icl, it uses the shape of yp, which has boundary conditions from r->infinity
	    # the solution returned in y is continuous, but there is no guarantee its derivative is continuous
	    # the error in the Numerov identity at icl in this amalgama-solution should be zero if both solutions
	    # are consistent.
            # as they are not, the discrepancy is used to return the "bestdE", which has the variation one
	    # should apply in the energy to get a better solution, to first order
            [self.y, self.yp, self.yfinal, self.icl, self.no, self.nop, bestdE] = self.solve(self.r, self.V + self.Vd + self.Vex, np.zeros(len(self.r)), self.E)

            dE = 0 # delta E to be used to shift energy

	    # first check the number of nodes

            # get list of crossings
	    nodesList = []
	    for idx in range(2, self.getMaxIdxForNodeCount()-2):
	      if self.yfinal[idx]*self.yfinal[idx-1] < 0:
	        nodesList.append(idx)
	    lnodes = len(nodesList)

	    # the number of zeros in y must be nodes(n, l) (==n - l - 1) for us to have the
	    # solution corresponding to the correct energy level given by n and l
	    # the number of zeroes is linear in n, so if it is too high, it means our current
	    # guess for the energy is too high
	    # if the number of zeroes is too low, it means our current guess for the energy
	    # is too low
	    if self.icl < 0:
	        dE = bestdE
	    elif lnodes > self.nodes():
	        # don't let us go above the current energy again, as it gives the wrong solution
		# and shift the energy to some value far far away
	        self.Emax = self.E
	        dE = (self.Emax + self.Emin)*0.5 - self.E
		#Nscale *= 2.0
            elif lnodes < self.nodes():
	        # don't let us go below the current energy again, as it gives the wrong solution
		# and shift the energy to some value far far away
	        self.Emin = self.E
	        dE = (self.Emax + self.Emin)*0.5 - self.E
		#Nscale *= 2.0
            else: # number of nodes is ok, but the energy needs to be adjusted
	        # in principle we could use the bestdE above as a shift
		# but it is too much sometimes, so let's soften it so we don't go too far away
                dE = 1.0/Nscale*bestdE
		# don't let it anyway give us a too big shift, otherwise this never converges
	        if np.fabs(dE) > 1.0/Nscale:
	            dE = 1.0/Nscale*dE/np.fabs(dE)
		if self.prev_dE*dE < 0: # we changed directions, so the step is to big and it is going from Emax to Emin
		    dE *= 0.5
		    Nscale *= 2

            if debug or i % 50 == 0:
                print "->  Iteration ", i, ", E = ", self.E*eV, " eV, dE = ", dE*eV, " eV, nodes = ", self.no, self.nop, lnodes, ", expected nodes = ", self.nodes(), ", crossing zero at = ", self.icl, ", with zero crossings at i = ", nodesList, ", r[zeros] = ", self.r[nodesList]

	    # now we have y for this energy
	    # if we want to plot it, we would need to
	    # undo the transformation done previously (y = sqrt(r)*psi) and normalise it so that
	    # int psi^2 r^2 dr = 1
            #psi = toPsi(self.r, self.y)
            #psip = toPsi(self.r, self.yp)
            self.psifinal = toPsi(self.r, self.yfinal)
	    # only plot it sometimes, as I don't have patience otherwise
	    if i % 10 == 0:
  	        plotWaveFunction(r, self.psifinal, self.V, self.Vd, self.Vex, self.E, self.n, self.l, 'lastwf.eps')

            # save previous dE
	    # can be useful to debug
	    self.prev_dE = dE
	    self.prev_no = lnodes
	    # limit the future scan range, depending on which direction we are going
	    # this only works if the dE calculation always points in the correct direction
	    # if next step is positive in energy, assume this is the minimum energy to scan
            if dE > 0:
	        self.Emin = self.E
            # if the next step is negative in energy, assume this is the maximum energy to scan
	    if dE < 0:
	        self.Emax = self.E
            # increment the energy now
            self.E += dE
	    # and cap it so it doesn't go crazy
            if dE > 0 and self.E > self.Emax:
                self.E = self.Emax
            elif dE < 0 and self.E < self.Emin:
                self.E = self.Emin
            #print "E, Emax, Emin = " , E, Emax, Emin
	    # if the delta E is too small, stop
            if (np.fabs(dE) < eps or np.fabs(self.Emax - self.Emin) < 1e-5):
                print "Converged to energy ", self.E*eV, " eV"
                break
        print "->  Final result for orbital, E = ", self.E*eV, " eV, dE = ", dE*eV, " eV, nodes = ", self.no, self.nop, len(nodesList), ", expected nodes = ", self.nodes(), ", crossing zero at = ", self.icl, ", with zero crossings at i = ", nodesList, ", r[zeros] = ", self.r[nodesList]

  	plotWaveFunction(r, self.psifinal, self.V, self.Vd, self.Vex, self.E, self.n, self.l, 'wf_'+str(label)+'.eps')
  	plotWaveFunction(r, self.psifinal, self.V, self.Vd, self.Vex, self.E, self.n, self.l, 'wf_'+str(label)+'_nolimit.eps', limit = False)

    # calculates the Vd = sum_orbitals integral psi_orb^2/r dr
    # calculates also Vex = sum orbitals integral psi_orb psi_this_orbital/r dr
    # returns Vd - Vex, which is the Hartree-Fock potential
    # For Helium, Vex = Vd/2, from the Virial Theorem, so we just need
    # to return Vd/2
    def loadHartreeFockPotential(self, orbitalList, orbKey):
        #thisVhf = np.zeros(len(self.r), dtype = np.longdouble)

        # must calculate
	# Vd = sum_{orb} int R(r')^2/(r-r') Y(theta', phi')^2 r'^2 sin theta' dr dtheta dphi
	# Vex = sum_{orb same spin} int R_this(r') R_other(r')/(r - r') Y(theta', phi')^2 r'^2 sin theta' dr dtheta dphi
	# ( r'^2 sin theta' dr dtheta dphi = dV)
        #
	# del^2 G(r) = delta(r -r0)
	#
	# G(r) = -1/(4pi*(r-r0))
	# if del^2 Vd = sigma, then Vd(r) = int sigma G(r) dV = -1/4pi int sigma(r')/(r'-r) dr'
	# so, if sigma is defined as - 4 * pi * R(r')^2 * Y(theta', phi')^2, then
	# del^2 Vd = - 4 * pi * R(r')^2 * Y(theta', phi')^2
	# Y for l = 0, m = 0 (Li and He!) is 1/sqrt(4*pi), so for s orbitals only:
	# del^2 Vd(r') = - R(r')^2
	# define curl E = 0 and E = - del Vd, so del^2 Vd = - R(r')^2 if div E = R(r')^2
	# define rho(r') = R(r')^2 as the charge density
	# int div E dV = int E . dS in a sphere containing rho(r) = E*4*pi*r'^2 = int R(r')^2 dV
	# E(r') = (in the r direction, outward) (Q = charge contained in sphere of radius r', centred at zero) / (4*pi*r'^2)
	# Vd(r') = - int_r'^infty E . dr
	# Vd(r'-h) = Vd(r') - int_r'-h^r' E . dr

	thisVd = np.zeros(len(r), dtype = np.longdouble)
	thisVex = np.zeros(len(r), dtype = np.longdouble)
	thisHex = np.zeros(len(r), dtype = np.longdouble)
        for orbitalName in orbitalList: # go through 1s, 2s, 2p, etc.
	    k = 0
	    for orbPsi in orbitalList[orbitalName]: # go through electrons in each of the orbital (ie: 1s1, 1s2)
	        k += 1
	        # now calculate Vhf = sum_orb integral psi_orb(r) psi_orb(r') 1/(r-r') dr'
		# for Vd, consider all orbitals, including this one
		# but this one is cancelled in Vex below
		if orbitalName == orbKey and self.spin == orbPsi.spin:
		    continue

		# So, in summary:
		# 0) calculate rho(r) = W(r)^2
		# 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
		# 2) calculate E(r) = Q(r)/(4*pi*r^2)
		# 3) calculate Vd(r) = sum_r'=inf^r E(r)*dr
                E = np.zeros(len(self.r), dtype = np.longdouble)
                rho = np.zeros(len(self.r), dtype = np.longdouble)
		for z in range(0, len(self.r)):
		    rho[z] = orbPsi.psifinal[z]**2
                Q = 0
		for z in range(0, len(self.r)):
		    dr = 0
		    if z >= 1:
		        dr = self.r[z] - self.r[z-1]
		    else:
		        dr = self.r[z]
		    Q += 4*np.pi*rho[z]*self.r[z]**2*dr
		    # this is E:
		    E[z] = Q/(4*np.pi*self.r[z]**2)
                Vd = np.zeros(len(self.r), dtype = np.longdouble)
		# now Vd will be integrated as sum r'=inf^r E(r) dr
		# in principle Vd = 0 for r = inf,
		# but we can choose any reference we want
		# in any case, the potential in r = r_max is due
		# to the charge contained
		# in r_max:
		Vd[len(self.r)-1] = 0#Q/self.r[len(self.r)-1]
		# now integrate backwards
		# Vd(r) = int_inf^r E(r') dr'
		# Vd(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
		# Vd(r-h) = Vd(r) + E(r)*dr
		for z in reversed(range(0, len(self.r)-1)):
                    Vd[z] = Vd[z+1] + E[z]*(self.r[z+1] - self.r[z])
		# for Helium, final Vhf = 0.5 Vd
		# not calculating Vex now: this makes it specific to Helium
		# the fact that Vex = 0.5 Vd is only true for Helium
		thisVd += Vd

        # now calculate Vex
        for orbitalName in orbitalList: # go through 1s, 2s, 2p, etc.
	    for orbPsi in orbitalList[orbitalName]: # go through electrons in each of the orbital (ie: 1s1, 1s2)
	        # now calculate Vex = sum_orb integral psi_this(r) psi_other(r') 1/(r-r') dr'
		# for Vex, only consider orbitals with same spin
		# skip myself: cancelled in Vd
		if orbPsi.spin != self.spin or (orbitalName == orbKey and orbPsi.spin == self.spin):
		    continue

                # calculate Vex(r) * W_other(r) = int W_this(r')*W_other(r')*1/(r-r') dV W_other(r)
		# notice that, differently from Vd, the potential is multiplying W_other, not W_this
		# so we calculate Vex(r) and then multiply it by W_other/W_this, so that we can add
		# this in a(x), which is multiplying W_this
		# in this way, the potential is added as (Vex(r)*W_other(r)/W_this(r))
		# and the multiplication by W_this(r) in the Schr. equation (it multiplies the potential)
		# will cancel the denominator out
		# Define a "charge density" rho(r) = W_this(r)W_other(r)
		# 0) calculate rho(r) = W_this(r)W_other(r)
		# 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
		# 2) calculate E(r) = Q(r)/(4*pi*r^2)
		# 3) calculate Vex(r) = sum_r'=inf^r E(r)*dr
                E = np.zeros(len(self.r), dtype = np.longdouble)
                rho = np.zeros(len(self.r), dtype = np.longdouble)
		for z in range(0, len(self.r)):
		    # orbPsi is "other"
		    # self is "this"
		    rho[z] = orbPsi.psifinal[z]*self.psifinal[z]
                Q = 0
                Q2 = 0
		for z in range(0, len(self.r)):
		    dr = 0
		    if z >= 1:
		        dr = self.r[z] - self.r[z-1]
		    else:
		        dr = self.r[z]
		    Q += 4*np.pi*rho[z]*self.r[z]**2*dr
		    # this is E:
		    E[z] = Q/(4*np.pi*self.r[z]**2)
                Vex = np.zeros(len(self.r), dtype = np.longdouble)
                Hex = np.zeros(len(self.r), dtype = np.longdouble)
		# now Vex will be integrated as sum r'=inf^r E(r) dr
		# in principle Vex = 0 for r = inf,
		# but we can choose any reference we want
		# in any case, the potential in r = r_max is due
		# to the charge contained
		# in r_max:
		Vex[len(self.r)-1] = Q/self.r[len(self.r)-1]
		Hex[len(self.r)-1] = Q/self.r[len(self.r)-1]
		# now integrate backwards
		# Vex(r) = int_inf^r E(r') dr'
		# Vex(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
		# Vex(r-h) = Vex(r) + E(r)*dr
		for z in reversed(range(0, len(self.r)-1)):
                    Vex[z] = Vex[z+1] + E[z]*(self.r[z+1] - self.r[z])
                    Hex[z] = Hex[z+1] + E[z]*(self.r[z+1] - self.r[z])
		for z in range(0, len(self.r)):
    		    Vex[z] *= orbPsi.psifinal[z]
		    #if np.fabs(orbPsi.psifinal[z]/self.psifinal[z]) > 5 and self.r[z] > 0.03/nm:
		    #    Vex[z] = 0
		    #else:
    		    #    Vex[z] *= orbPsi.psifinal[z]/self.psifinal[z]
  		    Hex[z] *= orbPsi.psifinal[z]*self.psifinal[z]
	        # and add it in
		thisVex -= Vex
                #thisVhf -= Vex
		thisHex -= Hex
	# this (alledgedly) helps in the convergence
	# should be just this otherwise:
	#self.Vhf = thisVhf
	print "Sum Vex", np.sum(thisVex)
	print "Sum Vd", np.sum(thisVd)
	self.Vd = 0.7*self.Vd + 0.3*thisVd
  	self.Vex = 0.7*self.Vex + 0.3*thisVex
  	self.Hex = 0.7*self.Hex + 0.3*thisHex
                

def plotPotential(r, V, Vhf, Vex, name):
    idx = np.where(r > 1.5)
    idx = idx[0][0]
    idxn = np.where(r > 0.05)
    idxn = idxn[0][0]
    plt.clf()
    Vtot = np.zeros(len(r), dtype = np.longdouble)
    plt.plot(r[idxn:idx]*nm, V[idxn:idx]*eV, 'r--', linewidth=2, label='Nucleus potential')
    plt.plot(r[idxn:idx]*nm, Vhf[idxn:idx]*eV, 'g--', linewidth=2, label='HF direct potential')
    plt.plot(r[idxn:idx]*nm, Vex[idxn:idx]*eV, 'g-.', linewidth=2, label='HF exchange potential')
    Vtot = V + Vhf + Vex
    plt.plot(r[idxn:idx]*nm, Vtot[idxn:idx]*eV, 'b-', linewidth=2, label='Total')
    plt.legend(('Nucleus potential', 'HF direct potential', 'HF exchange potential', 'Total'), frameon=False, loc = 'center right')
    plt.xlabel('$r$ [nm]')
    plt.ylabel('$V(r)$ [eV]')
    plt.title('')
    plt.draw()
    #plt.show()
    plt.savefig(name, transparent = True)
    plt.close()

from scipy.optimize import curve_fit
def zFitFunction(r, Z, C1, C2, C3):
    return -Z/r + (C1 + C2/r)*np.exp(-C3*r)

def fitPotential(r, V, Vhf, Vex, name):
    idx = np.where(r > 1)
    idx = idx[0][0]
    idxn = np.where(r > 0.1)
    idxn = idxn[0][0]
    plt.clf()
    Vtot = np.zeros(len(r), dtype = np.longdouble)
    plt.plot(r[idxn:idx]*nm, V[idxn:idx]*eV, 'r--', linewidth=2, label='Nucleus potential')
    plt.plot(r[idxn:idx]*nm, Vhf[idxn:idx]*eV, 'g--', linewidth=2, label='HF direct potential')
    plt.plot(r[idxn:idx]*nm, Vhf[idxn:idx]*eV, 'g-.', linewidth=2, label='HF exchange potential')
    Vtot = V + Vhf + Vex
    plt.plot(r[idxn:idx]*nm, Vtot[idxn:idx]*eV, 'b-', linewidth=2, label='Total')
    Vfit = None
    fitSuccessful = False
    try:
        fitParams = curve_fit(zFitFunction, r.astype(float), Vtot.astype(float))
        Zeff = fitParams[0][0]
        C1 = fitParams[0][1]
        C2 = fitParams[0][2]
        C3 = fitParams[0][3]
        print "------> Potential can be fit as -Z_{eff}/r + (C_{1} + C_{2}/r)exp(-C_{3} r), where Z_{eff}, C_{1}, C_{2}, C_{3} = ", Zeff, C1, C2, C3
        Vfit = np.zeros(len(r), dtype = np.longdouble)
        for z in range(0, len(r)):
            Vfit[z] = zFitFunction(r[z], Zeff, C1, C2, C3)
	fitSuccessful = True
    except:
        print "DANGER! DANGER! DANGER! Exception when doing effective potential fit: perhaps non-nuclei potential is zero (ie: single electron atom)? -- This is not essential, so skipping this ... just don't trust the resulting potential fir plots in ", name
    if fitSuccessful:
        plt.plot(r[idxn:idx]*nm, Vfit[idxn:idx]*eV, 'b-.', linewidth=3, label='Fit')
        plt.legend(('Nucleus potential', 'HF direct potential', 'HF exchange potential', 'Total', 'Fit'), frameon=False)
    else:
        plt.legend(('Coulomb potential', 'HF direct potential', 'HF exchange potential', 'Total'), frameon=False)
    plt.xlabel('$r$ [nm]')
    plt.ylabel('$V(r)$ [eV]')
    plt.title('')
    plt.draw()
    #plt.show()
    plt.savefig(name, transparent = True)
    plt.close()

# plot R(r)
# R_0 is the wave function with boundary conditions in r = 0
# R_infinity is the wave function with boundary conditions in r = infinity
# R(r) is R_0 until icl and R_infinity (scaled to make the result continuous) afterwards
# where icl is the point in which E-V crosses zero
# (note: at icl, d2y/dx2 = 0, which means this is an inflection point for
#  dy/dx, so the wave function has maximum/minimum slope at this point)
# Note: the full 3D wave function is:
#       Psi(r, theta, phi) = r R(r) Y(theta, phi)
# so there is an extra r there, since the Schr. eq. has
# been rewritten as a function of R(r)
# integral Y(theta, phi)^2 dOmega = 1
# integral |r R(r)|^2 dr = 1
# (so R(r) or |R(r)|^2 are not normalised to 1: |r R(r)|^2 is ...)
def plotWaveFunction(r, psi_final, V, Vd, Vex, E, n, l, name, limit = True):

    # for reference: this is the Hydrogen atom orbital
    # this is R(r)! that is the full solution is r R(r) Y(theta, phi)
    Htit = ''
    if n == 1:
        exact = 2*np.exp(-r)
	Htit = '1s'
    elif n == 2 and l == 0:
        exact = 1.0/(2*np.sqrt(2))*(2-r)*np.exp(-r/2.0)
	Htit = '2s'
    elif n == 2 and l == 1:
        exact = 1.0/(2*np.sqrt(6))*r*np.exp(-r/2.0)
	Htit = '1p'
    elif n == 3 and l == 0:
        exact = 2.0/(81*np.sqrt(3))*(27-18*r+2*r**2)*np.exp(-r/3.0)
	Htit = '3s'
    elif n == 3 and l == 1:
        exact = 4.0/(81*np.sqrt(6))*(6-r)*r*np.exp(-r/3.0)
	Htit = '3p'
    elif n == 3 and l == 2:
        exact = 4.0/(81*np.sqrt(30))*r**2*np.exp(-r/3.0)
	Htit = '3d'

    if limit:
        idx = np.where(r > 3)
        idx = idx[0][0]
        idxl = np.where(r > 0.2)
        idxl = idxl[0][0]
    else:
        idx = len(r)-1
        for i in range(1, len(r)):
            if psi_final[i-1]*psi_final[i] < 0:
	        idx = i+100
	        break
        idxl = 0
    if idx > len(r) - 1:
        idx = len(r)-1
    #plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(r[idxl:idx]*nm, psi_final[idxl:idx], 'r-', linewidth=2, label='$R(r)$')
    if n < 4:
        ax1.plot(r[idxl:idx]*nm, exact[idxl:idx], 'r:', linewidth=1, label='H '+Htit)
        ax1.legend(('$R(r)$', 'H '+Htit), frameon=False, loc = 'lower right')
    else:
        ax1.legend(('$R(r)$'), frameon=False, loc = 'lower right')
    ax1.set_xlabel('$r$ [nm]')
    ax1.set_ylabel('$R(r)$')
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    ax2 = ax1.twinx()
    ax2.plot(r[idxl:idx]*nm, V[idxl:idx]*eV, 'b--', linewidth=2, label='Nucleus pot.')
    ax2.plot(r[idxl:idx]*nm, Vd[idxl:idx]*eV, 'b-.', linewidth=2, label='HF direct pot.')
    ax2.plot(r[idxl:idx]*nm, Vex[idxl:idx]/psi_final[idxl:idx]*eV, 'b:', linewidth=2, label='HF exchange pot.')
    ax2.plot(r[idxl:idx]*nm, (V+Vd+Vex/psi_final)[idxl:idx]*eV, 'b-', linewidth=2, label='Total pot.')
    ax2.plot(r[idxl:idx]*nm, E*np.ones(idx-idxl)*eV, 'g--', linewidth=2, label='Energy')
    ax2.set_xlabel('$r$ [nm]')
    ax2.set_ylabel('Energy [eV]')
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
    ax2.legend(('Nucleus pot.', 'HF direct pot.', 'HF exchange pot.', 'Total pot.', 'Energy'), frameon = False, loc = 'center right')
    plt.title('')
    plt.draw()
    #plt.show()
    plt.savefig(name, transparent = True)
    plt.close()

def calculateTotalEnergy(orbitalList):
    E0 = 0
    JmK = 0
    for orbitalName in orbitalList: # go through 1s, 2s, 2p, etc.
        for orbPsi in orbitalList[orbitalName]: # go through electrons in each of the orbital (ie: 1s1, 1s2)
	    E0 += orbPsi.E # sums eigen values
	    for z in range(0, len(orbPsi.r)):
	        dr = 0
		if z < len(orbPsi.r)-1:
		    dr = orbPsi.r[z+1] - orbPsi.r[z]
		# should have 4*pi*Y^2, but for s orbitals Y^2 = 1/4pi
  	        JmK += (orbPsi.Vd[z]*orbPsi.psifinal[z]**2)*(orbPsi.r[z]**2)*dr - orbPsi.Hex[z]*(orbPsi.r[z]**2)*dr
    print "J-K", JmK
    E0 += -0.5*JmK
    return E0

# make Grid
# Grid is finer close to r = 0 and wider as r -> infinity
# r[i] = exp(xmin + i*dx)/Z
# wh_nolimitere i goes from 0 to N
# Z is the atomic number (to make the Grid finer for atoms that have higher Z)
# the reason for Z in this is that the Coulomb potential is stronger for
# atoms with higher Z, so more detail close to r = 0 is needed
# getAF() has been written with this Grid format in mind
# other Grids can be tried, but then getAF() needs to be changed, since
# the derivative in the Schr. equation now are taken as a function of
# x = ln(Z*r) (that is: r = exp(x)/Z)
# dx/dr = 1/r (that is: dr/dx = r)
r = init(N, xmin, C = 1.0)

# make orbital configuration
# for Helium: 1s^2
# spin is used to distinguish electrons in same orbital
# any other ID could be used: the objective is only to avoid
# double counting the energy when calculating the Hartree-Fock potential
# the electron's own energy should not be included
# n and l are needed to establish number of zeroes and initial conditions
# when solving the equation
# Z is used in Coulomb potential
orb = {}
orb['1s'] = []
orb['1s'].append(Orbital(_n = 1, _l = 0, _Z = Z, _r = r, _spin = 1))   # stop here for H
orb['1s'].append(Orbital(_n = 1, _l = 0, _Z = Z, _r = r, _spin = -1))  # stop here for He
orb['2s'] = []
orb['2s'].append(Orbital(_n = 2, _l = 0, _Z = Z, _r = r, _spin = 1))   # stop here for Li
#orb['2s'].append(Orbital(_n = 2, _l = 0, _Z = Z, _r = r, _spin = -1))  # stop here for Be
#orb['2p'] = []
#orb['2p'].append(Orbital(_n = 2, _l = 1, _Z = Z, _r = r, _spin = 1))   # stop here for B

E_gs_old = 0
hfIter = 0
while hfIter < 30:
    print '---> Hartree-Fock iteration', hfIter
    print '-->  (HF iteration '+str(hfIter)+') Will now solve atom Schr. equation using Coulomb potential and effective potential caused by other atoms'
    for orbitalName in orb:
	k = 0
        for orbPsi in orb[orbitalName]:
            print '-->  (HF iteration '+str(hfIter)+') Solving equation for orbital ', orbitalName, ' electron ', k
            orbPsi.solveWithCurrentPotential('hfIter'+str(hfIter)+'_'+orbitalName+str(k))
	    k += 1
	
    for orbitalName in orb:
	k = 0
        for orbPsi in orb[orbitalName]:
            print '-->  (HF iteration '+str(hfIter)+') Solving equation for orbital ', orbitalName, ' electron ', k
            print '->   (HF iteration '+str(hfIter)+') ', orbitalName, ', electron ', k, ': Hartree-Fock eigenvalue = ', orbPsi.E*eV, " eV"
	    k += 1

    print '---> (HF iteration '+str(hfIter)+') Solved the Schr. equation with effective potentials, now we use wave functions found to recalculate effective potentials of other electrons in electron x, for each x.'
    for orbitalName in orb:
        k = 0
        for orbPsi in orb[orbitalName]:
            print '-->  (HF iteration '+str(hfIter)+') Recalculating effective potentials for orbital ', orbitalName, ', electron ', k
            orbPsi.loadHartreeFockPotential(orb, orbitalName)
	    k += 1
    
    # plot potential
    keys = orb.keys()
    highestE = -999999999
    externOrb = ''
    externIdx = -1
    for k in keys:
        for item in range(0, len(orb[k])):
            if orb[k][item].E > highestE:
	        highestE = orb[k][item].E
		externOrb = k
		externIdx = item
    plotPotential(r, orb[externOrb][externIdx].V, orb[externOrb][externIdx].Vd, orb[externOrb][externIdx].Vex/orb[externOrb][externIdx].psifinal, 'potential_hfIter'+str(hfIter)+'.eps')
    fitPotential(r, orb[externOrb][externIdx].V, orb[externOrb][externIdx].Vd, orb[externOrb][externIdx].Vex/orb[externOrb][externIdx].psifinal, 'potentialFit_hfIter'+str(hfIter)+'.eps')

    # calculate ground state energy
    E_gs = calculateTotalEnergy(orb)
    print '-->  (HF iteration '+str(hfIter)+') Ground state energy = ', E_gs*eV, ' eV'
    
    hfIter += 1
    # stop when the new ground state energy is less than 0.1% of the old one
    if np.fabs((E_gs - E_gs_old)/E_gs) < 0.1e-2:
        break
    E_gs_old = E_gs

