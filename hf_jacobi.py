#!/usr/bin/env python

import sys
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

def F(i, f, y, s, no):
    return (12 - 10*f[i])*y[i] - f[i-1]*y[i-1] - f[i+1]*y[i+1] + (s[i+1] + 10.0*s[i] + s[i-1])

def getAF(r, pot, E, l, dx, m, potIndep):
    icl = -1
    a = np.zeros(len(r))
    f = np.zeros(len(r))
    s = np.zeros(len(r))
    for i in range(0, len(r)):
        a[i] = 2*m*r[i]**2*(E - pot[i]) - (l+0.5)**2
	f[i] = 1 + a[i]*dx**2/12.0
	s[i] = (dx**2)/12.0*2*m*r[i]**2*potIndep[i]
	if icl < 0 and i >= 1 and a[i]*a[i-1] < 0:
	    icl = i
    return [a, f, s, icl]

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
def outwardSolution(r, l, Z, f, s):
    y = np.zeros(len(r))
    no = 0
    y[0] = ((Z*r[0])**(l+0.5))
    y[1] = ((Z*r[1])**(l+0.5))
    for i in range(1, len(r)-1):
	y[i+1] = ((12 - f[i]*10)*y[i] - f[i-1]*y[i-1] + (s[i+1] + 10.0*s[i] + s[i-1]))/f[i+1]
    for i in range(1, len(r)-1):
        if np.isnan(y[i]) or np.isnan(y[i+1]) or np.fabs(y[i]) > sys.float_info.max or np.fabs(y[i+1]) > sys.float_info.max:
            no += 99
            continue
	if y[i]*y[i+1] < 0:
	    no += 1
    return [y, no]

def inwardSolution(r, l, Z, f, s, E):
    yp = np.zeros(len(r))
    nop = 0
    m = 1
    # for r->infinity, only d^2y/dx^2 + 2mEy = 0 terms contribute
    # (remember E is negative)
    # so approximate it at r->infinity as an exponential
    yp[len(r)-1] = np.exp(-np.sqrt(-2*m*E)*r[len(r)-1])
    yp[len(r)-2] = np.exp(-np.sqrt(-2*m*E)*r[len(r)-2])
    for i in reversed(range(1, len(r)-1)):
	yp[i-1] = ((12 - f[i]*10)*yp[i] - f[i+1]*yp[i+1] + (s[i+1] + 10.0*s[i] + s[i-1]))/f[i-1];

    for i in reversed(range(1, len(r)-1)):
        if np.isnan(yp[i]) or np.isnan(yp[i-1]) or np.fabs(yp[i]) > sys.float_info.max or np.fabs(yp[i-1]) > sys.float_info.max:
            no += 99
            continue
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
    no_ren = 0
    for i in range(1, int(len(y_ren))):
      if y[i]*y[i-1] < 0:
        no_ren += 1
    return [y_ren, no_ren]

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
def solve(r, dx, pot, n, l, E, Z, potIndep):
    m = 1.0
    icl = -1
    [a, f, s, icl] = getAF(r, pot, E, l, dx, m, potIndep)
    
    [y, no] = outwardSolution(r, l, Z, f, s)
    [yp, nop] = inwardSolution(r, l, Z, f, s, E)
    [y_ren, no_ren] = matchInOut(y, yp, icl)
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
    Ficl = F(icl, f, y_ren, s, no_ren - nodes(n, l)) # get F at icl
    # calculate dF/dE, by varying E very slightly
    dE = -1e-2*E
    if dE == 0:
      dE = -0.1e-1
    no_ren_ep = -1
    while no_ren_ep != no_ren:
        if no_ren_ep != -1:
            dE *= 0.5
        # recalculate the solution with a slihtly varied E
        [ap, fp, sp, iclp] = getAF(r, pot, E+dE, l, dx, m, potIndep)
        [y_ep, no_ep] = outwardSolution(r, l, Z, fp, sp)
        [yp_ep, nop_ep] = inwardSolution(r, l, Z, fp, sp, E+dE)
        [y_ep_ren, no_ren_ep] = matchInOut(y_ep, yp_ep, icl)
        # new solution has a discontinuity at icl again
        # dF/dE is defined as the difference over dE of the change in F
        Fp = F(icl, fp, y_ep_ren, sp, no_ren_ep - nodes(n, l))
        #print "F, Fp, dF ", Ficl, Fp, Fp - Ficl
        if Fp != Ficl:
            bestdE = -Ficl*dE/(Fp - Ficl)
            bestdE *= 1e-1
        else:
            bestdE = dE
        if icl < 0:
            bestdE = 0.5 # arbitrary, but must be positive to make energy less negative
    return [y_ren, icl, no_ren, bestdE]


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

## potential calculation

def getPotentialH(r, phiList):
    totalVd = np.zeros(len(r))
    for iOrb in phiList.keys():
	# So, in summary:
	# 0) calculate rho(r) = W(r)^2
	# 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
	# 2) calculate E(r) = Q(r)/(4*pi*r^2)
	# 3) calculate Vd(r) = sum_r'=inf^r E(r)*dr
        E = np.zeros(len(r))
        rho = np.zeros(len(r))
        for z in range(0, len(r)):
            rho[z] = phiList[iOrb].psi[z]**2
        Q = 0
        for z in range(0, len(r)):
            dr = 0
            if z >= 1:
                dr = r[z] - r[z-1]
            else:
                dr = r[z]
            Q += 4*np.pi*rho[z]*r[z]**2*dr
            # this is E:
            E[z] = Q/(4*np.pi*r[z]**2)
        # now Vd will be integrated as sum r'=inf^r E(r) dr
        # in principle Vd = 0 for r = inf,
        # but we can choose any reference we want
        # in any case, the potential in r = r_max is due
        # to the charge contained
        # in r_max:
        Vd = np.zeros(len(r))
        Vd[len(r)-1] = 0
        # now integrate backwards
        # Vd(r) = int_inf^r E(r') dr'
        # Vd(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
        # Vd(r-h) = Vd(r) + E(r)*dr
        for z in reversed(range(0, len(r)-1)):
            Vd[z] = Vd[z+1] + E[z]*(r[z+1] - r[z])
        totalVd += Vd
    return totalVd

def findEigenvalue(r, dx, pot_full, phi, Z, vxc_effective):
    Emax = 0
    Emin = -10.0
    Nscale = 1.0
    prev_dE = 0
    for i in range(0,2000):
        [y,  icl, no, bestdE] = solve(r, dx, pot_full, phi.n, phi.l, phi.E, Z, vxc_effective)
        phi.no = no
        dE = 0
        if phi.no > nodes(phi.n, phi.l):
            Emax = phi.E
            dE = (Emax + Emin)*0.5 - phi.E
            Nscale *= 2.0
        elif phi.no < nodes(phi.n, phi.l):
            Emin = phi.E
            dE = (Emax + Emin)*0.5 - phi.E
            Nscale *= 2.0
        else:
            dE = bestdE/Nscale
            if np.fabs(dE) > 1.0/Nscale:
                dE = 1.0/Nscale*dE/np.fabs(dE)
            if dE > 0:
                if phi.E > Emin:
                    Emin = phi.E
                    Nscale *= 2.0
            elif dE < 0:
                if phi.E < Emax:
                    Emax = phi.E
                    Nscale *= 2.0

        #if prev_dE*dE < 0:
        #    dE = -prev_dE+0.5*prev_dE
        #    if prev_dE < 0:
        #        Emax = phi.E-prev_dE
        #    elif prev_dE > 0:
        #        Emin = phi.E-prev_dE

        print "Iteration ", i, ", E = ", phi.E*eV, " eV, dE = ", dE*eV, " eV, nodes = ", no, ", expected nodes = ", nodes(phi.n, phi.l), ", crossing zero at = ", icl
        psi = toPsi(r, y)
        phi.psi = psi
        phi.E += dE
        if dE > 0 and phi.E > Emax:
            phi.E = Emax
            Nscale *= 2.0
        elif dE < 0 and phi.E < Emin:
            phi.E = Emin
            Nscale *= 2.0
        prev_dE = dE
        if no - nodes(phi.n, phi.l) == 0 and (np.fabs(dE) < 1e-7 or np.fabs(Emax - Emin) < 1e-5):
            print "Converged to energy ", phi.E*eV, " eV"
            break

# calculate exchange potential 
# returns the coefficient multiplying each orbital
def getPotentialX(r, phiList, iOrb):
    totalVx = {}
    for jOrb in phiList.keys():
        if ('+' in jOrb and '-' in iOrb) or ('-' in jOrb and '+' in iOrb):
              continue
        # calculate Vex(r) * W_other(r) = int W_this(r')*W_other(r')*1/(r-r') dV W_other(r)
	# notice that, differently from Vd, the potential is multiplying W_other, not W_this
	# Define a "charge density" rho(r) = W_this(r)W_other(r)
	# 0) calculate rho(r) = W_this(r)W_other(r)
	# 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
	# 2) calculate E(r) = Q(r)/(4*pi*r^2)
	# 3) calculate Vex(r) = sum_r'=inf^r E(r)*dr
        E = np.zeros(len(r))
        rho = np.zeros(len(r))
	for z in range(0, len(r)):
	    # jOrb is "other"
	    # iOrb is "this"
	    rho[z] = phiList[jOrb].psi[z]*phiList[iOrb].psi[z]
        Q = 0
        Q2 = 0
	for z in range(0, len(r)):
	    dr = 0
	    if z >= 1:
	        dr = r[z] - r[z-1]
	    else:
	        dr = r[z]
	    Q += 4*np.pi*rho[z]*r[z]**2*dr
	    # this is E:
	    E[z] = Q/(4*np.pi*r[z]**2)
        Vex = np.zeros(len(r))
	# now Vex will be integrated as sum r'=inf^r E(r) dr
	# in principle Vex = 0 for r = inf,
	# but we can choose any reference we want
	# in any case, the potential in r = r_max is due
	# to the charge contained
	# in r_max:
	Vex[len(r)-1] = 0
	# now integrate backwards
	# Vex(r) = int_inf^r E(r') dr'
	# Vex(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
	# Vex(r-h) = Vex(r) + E(r)*dr
	for z in reversed(range(0, len(r)-1)):
            Vex[z] = Vex[z+1] + E[z]*(r[z+1] - r[z])
        totalVx[jOrb] = Vex
    return totalVx

def calculateE0(r, listPhi, vd, vxc):
    E0 = 0
    JmK = 0
    for iOrb in listPhi.keys():
        E0 += listPhi[iOrb].E
	for z in range(0, len(r)):
	    dr = 0
	    if z < len(r)-1:
	        dr = r[z+1] - r[z]
	    # should have 4*pi*Y^2, but for s orbitals Y^2 = 1/4pi
            Hex = 0
            Vh = vd[z]
            for jOrb in vxc[iOrb].keys():
                if jOrb == iOrb:
                    Vh -= vxc[iOrb][jOrb][z]
                    continue
                Hex += -vxc[iOrb][jOrb][z]*listPhi[iOrb].psi[z]*listPhi[jOrb].psi[z]
  	    JmK += (Vh*listPhi[iOrb].psi[z]**2)*(r[z]**2)*dr - Hex*(r[z]**2)*dr
    E0 += -0.5*JmK
    return E0

class phi:
    n = 1
    l = 0
    E = -2.0
    no = 0
    psi = None
    def __init__(self, _n, _l, _E):
        self.n = _n
        self.l = _l
        self.E = _E

Z = 3

dx = 1e-2
r = init(dx, 1300, np.log(1e-4))

listPhi = {}
# create objects to hold energy and wave functions of each Hartree-Fock equation
# provide boundary conditions n, l in first arguments
# provide initial energy to use when starting to look for solutions
# propose to start with the Hydrogen-like (if Hydrogen had atomic number Z) energy level (0.5*Z^2/n^2)
listPhi['1s1+'] = phi(1, 0, -Z**2/(1.0**2)*0.5)
listPhi['1s1-'] = phi(1, 0, -Z**2/(1.0**2)*0.5)
listPhi['2s1+'] = phi(2, 0, -Z**2/(2.0**2)*0.5)
#listPhi['2s1-'] = phi(2, 0, -Z**2/2.0*0.5)

pot = V(r, Z)

for iOrb in listPhi.keys():
    listPhi[iOrb].psi = np.zeros(len(r))

E0_old = 0
E0 = 0
Nscf = 100
NJacobi = 20

for iSCF in range(0, Nscf):
    print "===> On SCF iteration %d" % iSCF

    vd = getPotentialH(r, listPhi)
    vxc = {}
    for iOrb in listPhi.keys():
        vxc[iOrb] = getPotentialX(r, listPhi, iOrb)
    for iJacobi in range(0, NJacobi):
        print "======> On Jacobi iteration %d (potential fixed here: only trying to solve linear system)" % iJacobi
        for iOrb in listPhi.keys():
            print "======> On orbital %s, trying to find its eigenvalue" % iOrb
            # calculate the extra term as \sum_j psi_j Vx_j
            # these are the linear terms due to the remainder of the potentials
            vxc_effective = np.zeros(len(r))
            pot_full_effective = pot + vd
            print "vd sum = %5f and vxc sum for same = %5f" % (np.sum(vd), np.sum(vxc[iOrb][iOrb]))
            for jOrb in vxc[iOrb].keys():
                if iOrb == jOrb:
                    pot_full_effective -= vxc[iOrb][jOrb]
                    continue
                vxc_effective += listPhi[jOrb].psi*vxc[iOrb][jOrb]
            # now scan energy to find eigenvalue
            findEigenvalue(r, dx, pot_full_effective, listPhi[iOrb], Z, vxc_effective)
            print "======> %s eigenvalue: %5f" % (iOrb, listPhi[iOrb].E)

    idx = np.where(r > 5)
    if len(idx[0]) != 0:
        idx = idx[0][0]
    else:
        idx = len(r)-1
    plt.clf()
    leg = []
    exact_p = 2*np.exp(-r)   # solution for R(r) in Hydrogen, n = 1
    col = ['r-', 'g-', 'b-', 'r-.', 'g-.', 'b-.']
    c = 0
    for iOrb in listPhi.keys():
        plt.plot(r[0:idx], listPhi[iOrb].psi[0:idx], col[c], label='$R_{%s}$'%iOrb)
        c += 1
        leg.append('%s (%3f eV)' % (iOrb, listPhi[iOrb].E*eV))
    plt.plot(r[0:idx], exact_p[0:idx], 'g--', label='$R_{exact}$')
    leg.append('Exact H (1s)')

    plt.legend(leg, frameon=False)
    plt.xlabel('$r$')
    plt.ylabel('$|R(r)|$')
    E0 = calculateE0(r, listPhi, vd, vxc)
    plt.title('Z=%d, SCF iter=%d, E_{0}=%4f eV'%(Z, iSCF, E0*eV))
    plt.draw()
    plt.savefig('pseudo_potentials.pdf', bbox_inches='tight')
    #plt.show()

    if np.fabs(1 - E0_old/E0) < 1e-3:
        print "===> Ground state energy changed by less than 0.11% (by ", 100.0*np.fabs(1 - E0_old/E0),"%). E0 = ", E0*eV, "eV"
        break
    else:
        print "===> SCF iteration %d, E0 = %5f eV, delta E0 = %5f eV " % (iSCF, E0*eV, (E0 - E0_old)*eV)
    E0_old = E0

