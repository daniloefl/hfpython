#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import scipy.sparse
import scipy.sparse.linalg

class bcolors:
    HEADER = '\033[4m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    ENDC = '\033[0m'

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
    r = np.zeros(N, dtype=np.float64)
    for i in range(0, N):
        r[i] = np.exp(xmin + i*dx)
    return r

# for each point x_i, we get psi_i
# for which the eq. is:
# (12 - 10 f_n) y_n - f_{n-1} y_{n-1} - f_{n+1} y_{n+1} + (s[i+1] + 10.0*s[i] + s[i-1]) = 0
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

def nodes(n, l):
    return n - l - 1

import random

def Y(l, m, theta, phi):
    if l == 0:
        return 0.5*np.sqrt(1.0/np.pi)
    elif l == 1:
        if m == -1:
            return np.sqrt(3.0/(4*np.pi))*np.sin(theta)*np.sin(phi)
        elif m == 0:
            return np.sqrt(3.0/(4*np.pi))*np.cos(theta)
        elif m == 1:
            return np.sqrt(3.0/(4*np.pi))*np.sin(theta)*np.cos(phi)
    # TO DO: add more
    return 0

def factorial(n):
    if n == 1:
        return 1
    elif n == 0:
        return 1
    elif n < 0:
        return 1
    return n * factorial(n-1)

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

## potential calculation
# calculate int rpsi1(r1)*Ylm(t1, p1)*rpsi1(r1)*Ylm(t1, p1)/|r1-r2| r^2 sin t1 dt1 dp1 dr1
# 1/|r1 - r2| = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1) rs^l/rb^(l+1) Ylm(t1, p1) Ylm(t2, p2)
# int rpsi(r1)^2 Ylm(t1, p1)^2/|r1 - r2| r1^2 dOmega1 dr1
# = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1)   (int rpsi(r1)^2 rs^l/rb^(l+1) r1^2 dr1)   (int Ylm(t1, p1)^2 Ylm(t1, p1) Ylm(t2, p2) dOmega1)
# int Ylm^3(t1, p1) dOmega1 = ... (A4.40)
#                           = (-1)^m sqrt( (2l+1)*(2l+1) / (4 pi (2l+1)) ) <ll00|l0> <llmm|l -m>
# = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1)   (int rpsi(r1)^2 rs^l/rb^(l+1) r1^2 dr1) (-1)^m sqrt( (2l+1)*(2l+1) / (4 pi (2l+1)) ) <ll00|l0> <llmm|l -m> Ylm(t2, p2)
def getIntegrandH(r, z1, t1, p1, z2, t2, p2, phi1):
    return (r[z1]**2)*(phi1.rpsi[z1]**2)*(Y(phi1.l, phi1.m, t1, p1)**2)*(np.sin(t1))/np.sqrt((r[z1]*np.sin(t1)*np.cos(p1) - r[z2]*np.sin(t2)*np.cos(p2))**2 + (r[z1]*np.sin(t1)*np.sin(p1) - r[z2]*np.sin(t2)*np.sin(p2))**2 + (r[z1]*np.cos(t1) - r[z2]*np.cos(t2))**2)

def getPotentialHMC(r, phiList, iOrb):
    Vd = np.zeros(len(r), dtype=np.float64)
    # integrate on r1[0, len(r)], p1[0, 2pi], t1[0, pi]
    for z2 in range(0, len(r)):
        if z2 % 50 == 0:
            print "Using MC integration to calculate Coulomb potential for %s: %d" % (iOrb, z2)
        Ntot = int(20000)
        for N in range(0, Ntot):
            x1 = np.pi*0.5*random.random()
            r1 = np.tan(x1)
            z1 = int((np.log(r1) - xmin)/dx)
            if z1 > len(r)-1:
                z1 = len(r)-1
            # r = tan(x) => dr/dx = sec^2(x)
            # int f(x) dr = int f(x) dr/dx dx
            t1 = random.random()*np.pi
            t2 = random.random()*np.pi
            p1 = random.random()*2*np.pi
            p2 = random.random()*2*np.pi
            #Vd[z2] += np.pi*2*np.pi*np.pi*0.5*1.0/(np.cos(x1)**2)*getIntegrandH(r, z1, t1, p1, z2, t2, p2, phiList[iOrb])*Y(phiList[xOrb].l, phiList[xOrb].m, t2, p2)*2*np.pi*np.pi*np.sin(t2)/(4*np.pi)/float(Ntot)
            Vd[z2] += np.pi*2*np.pi*np.pi*0.5*1.0/(np.cos(x1)**2)*getIntegrandH(r, z1, t1, p1, z2, t2, p2, phiList[iOrb])*2*np.pi*np.pi*np.sin(t2)/(4*np.pi)/float(Ntot)
    return Vd

## potential calculation
# calculate int rpsi1(r1)*Yl1m1(t1, p1)*rpsi2(r1)*Yl2m2(t1, p1)/|r1-r2| r1^2 sin t1 dt1 dp1 dr1
def getIntegrandXC(r, z1, t1, p1, z2, t2, p2, phi1, phi2):
    return r[z1]**2*phi1.rpsi[z1]*Y(phi1.l, phi1.m, t1, p1)*phi2.rpsi[z1]*Y(phi2.l, phi2.m, t1, p1)*np.sin(t1)/np.sqrt((r[z1]*np.sin(t1)*np.cos(p1) - r[z2]*np.sin(t2)*np.cos(p2))**2 + (r[z1]*np.sin(t1)*np.sin(p1) - r[z2]*np.sin(t2)*np.sin(p2))**2 + (r[z1]*np.cos(t1) - r[z2]*np.cos(t2))**2)

def getPotentialXCMC(r, phiList, iOrb, jOrb):
    Vex = np.zeros(len(r), dtype=np.float64)
    # integrate on r1[0, len(r)], p1[0, 2pi], t1[0, pi]
    for z2 in range(0, len(r)):
        if z2 % 50 == 0:
            print "Using MC integration to calculate XC potential for %s,%s: %d" % (iOrb, jOrb, z2)
        Ntot = int(20000)
        for N in range(0, Ntot):
            x1 = np.pi*0.5*random.random()
            r1 = np.tan(x1)
            z1 = int((np.log(r1) - xmin)/dx)
            if z1 > len(r)-1:
                z1 = len(r)-1
            # r = tan(x) => dr/dx = sec^2(x)
            # int f(x) dr = int f(x) dr/dx dx
            t1 = random.random()*np.pi
            t2 = random.random()*np.pi
            p1 = random.random()*2*np.pi
            p2 = random.random()*2*np.pi
            #Vex[z2] += np.pi*2*np.pi*np.pi*0.5*1.0/(np.cos(x1)**2)*getIntegrandXC(r, z1, t1, p1, z2, t2, p2, phiList[iOrb], phiList[jOrb])*Y(phiList[iOrb].l, phiList[iOrb].m, t2, p2)*2*np.pi*np.pi*np.sin(t2)/(4*np.pi)/float(Ntot)
            Vex[z2] += np.pi*2*np.pi*np.pi*0.5*1.0/(np.cos(x1)**2)*getIntegrandXC(r, z1, t1, p1, z2, t2, p2, phiList[iOrb], phiList[jOrb])*2*np.pi*np.pi*np.sin(t2)/(4*np.pi)/float(Ntot)
    return Vex

## potential calculation
def getPotentialH(r, phiList):
    totalVd = np.zeros(len(r), dtype=np.float64)
    for iOrb in phiList.keys():
        if listPhi[iOrb].virtual:
            continue
        # for p, d and f states, use the MC integration
        # we cannot factorize the spherical harmonics then
        if phiList[iOrb].l != 0:
            totalVd += getPotentialHMC(r, phiList, iOrb)
            continue
        # otherwise, we can use Gauss' law
        # to integrate rho^2(r)/|r-r'| dr, which is similar to a central Coulomb potential
        # for a charge density rho(r)^2, which is spherically symmetric
	# In summary:
	# 0) calculate rho(r) = W(r)^2
	# 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
	# 2) calculate E(r) = Q(r)/(4*pi*r^2)
	# 3) calculate Vd(r) = sum_r'=inf^r E(r)*dr
        E = np.zeros(len(r), dtype=np.float64)
        rho = np.zeros(len(r), dtype=np.float64)
        for z in range(0, len(r)):
            rho[z] = phiList[iOrb].rpsi[z]**2
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
        Vd = np.zeros(len(r), dtype=np.float64)
        Vd[len(r)-1] = 0
        # now integrate backwards
        # Vd(r) = int_inf^r E(r') dr'
        # Vd(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
        # Vd(r-h) = Vd(r) + E(r)*dr
        for z in reversed(range(0, len(r)-1)):
            Vd[z] = Vd[z+1] + E[z]*(r[z+1] - r[z])
        totalVd += Vd
    return totalVd

# calculate exchange potential 
# returns the coefficient multiplying each orbital
def getPotentialX(r, phiList, iOrb):
    totalVx = {}
    for jOrb in phiList.keys():
        if listPhi[jOrb].virtual:
            continue
        if ('+' in jOrb and '-' in iOrb) or ('-' in jOrb and '+' in iOrb):
              continue
        # for p, d and f states, use the MC integration
        # we cannot factorize the spherical harmonics then
        if phiList[iOrb].l != 0 or phiList[jOrb].l != 0:
            totalVx[jOrb] = getPotentialXCMC(r, phiList, iOrb, jOrb)
            continue
        # otherwise, we can use Gauss' law
        # to integrate rho^2(r)/|r-r'| dr, which is similar to a central Coulomb potential
        # for a charge density rho(r)^2, which is spherically symmetric

        # calculate Vex(r) * W_other(r) = int W_this(r')*W_other(r')*1/(r-r') dV W_other(r)
	# notice that, differently from Vd, the potential is multiplying W_other, not W_this
	# Define a "charge density" rho(r) = W_this(r)W_other(r)
	# 0) calculate rho(r) = W_this(r)W_other(r)
	# 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
	# 2) calculate E(r) = Q(r)/(4*pi*r^2)
	# 3) calculate Vex(r) = sum_r'=inf^r E(r)*dr
        E = np.zeros(len(r), dtype=np.float64)
        rho = np.zeros(len(r), dtype=np.float64)
	for z in range(0, len(r)):
	    # jOrb is "other"
	    # iOrb is "this"
	    rho[z] = phiList[jOrb].rpsi[z]*phiList[iOrb].rpsi[z]
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
        Vex = np.zeros(len(r), dtype=np.float64)
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

# calculate int |rpsi1(r1)*Yl1m1(t1, p1)|^2*|rpsi2(r2)*Yl2m2(t2, p2)|^2/|r1-r2| r1^2 sin t1 r2^2 sin t2 dt1 dp1 dr1 dt2 dp2 dr2
def getIntegrandJ(r, z1, t1, p1, z2, t2, p2, phi1, phi2):
    return (phi1.rpsi[z1]*Y(phi1.l, phi1.m, t1, p1))**2*(phi2.rpsi[z2]*Y(phi2.l, phi2.m, t2, p2))**2*(r[z1]**2*np.sin(t1))*(r[z2]**2*np.sin(t2))/np.sqrt((r[z1]*np.sin(t1)*np.cos(p1) - r[z2]*np.sin(t2)*np.cos(p2))**2 + (r[z1]*np.sin(t1)*np.sin(p1) - r[z2]*np.sin(t2)*np.sin(p2))**2 + (r[z1]*np.cos(t1) - r[z2]*np.cos(t2))**2)

def getJMC(r, phiList, iOrb, jOrb):
    J = 0
    J2 = 0
    print "Using MC integration to calculate J %s,%s" % (iOrb, jOrb)
    Ntot = int(400000)
    for N in range(0, Ntot):
        x1 = np.pi*0.5*random.random()
        r1 = np.tan(x1)
        z1 = int((np.log(r1) - xmin)/dx)
        if z1 > len(r)-1:
            z1 = len(r)-1
        # r = tan(x) => dr/dx = sec^2(x)
        # int f(x) dr = int f(x) dr/dx dx
        x2 = np.pi*0.5*random.random()
        r2 = np.tan(x2)
        z2 = int((np.log(r2) - xmin)/dx)
        if z2 > len(r)-1:
            z2 = len(r)-1
        t1 = random.random()*np.pi
        t2 = random.random()*np.pi
        p1 = random.random()*2*np.pi
        p2 = random.random()*2*np.pi
        Jn = (np.pi*2)**2*(np.pi)**2*(np.pi*0.5*1.0/(np.cos(x1)**2))*(np.pi*0.5*1.0/(np.cos(x2)**2))*getIntegrandJ(r, z1, t1, p1, z2, t2, p2, phiList[iOrb], phiList[jOrb])
        J += Jn
        J2 += Jn**2
    J = J/float(Ntot)
    dJ = (J2/float(Ntot-1) - J**2)/float(Ntot)
    return [J, dJ]

# calculate int rpsi1(r1)*Yl1m1(t1, p1)*rpsi1(r2)*Yl1m1(t2, p2)*rpsi2(r2)*Yl2m2(t2, p2)*rpsi2(r1)*Yl2m2(t1, p1)/|r1-r2| r1^2 sin t1 r2^2 sin t2 dt1 dp1 dr1 dt2 dp2 dr2
def getIntegrandK(r, z1, t1, p1, z2, t2, p2, phi1, phi2):
    return (phi1.rpsi[z1]*Y(phi1.l, phi1.m, t1, p1))*(phi1.rpsi[z2]*Y(phi1.l, phi1.m, t2, p2))*(phi2.rpsi[z2]*Y(phi2.l, phi2.m, t2, p2))*(phi2.rpsi[z1]*Y(phi2.l, phi2.m, t1, p1))*(r[z1]**2*np.sin(t1))*(r[z2]**2*np.sin(t2))/np.sqrt((r[z1]*np.sin(t1)*np.cos(p1) - r[z2]*np.sin(t2)*np.cos(p2))**2 + (r[z1]*np.sin(t1)*np.sin(p1) - r[z2]*np.sin(t2)*np.sin(p2))**2 + (r[z1]*np.cos(t1) - r[z2]*np.cos(t2))**2)

def getKMC(r, phiList, iOrb, jOrb):
    K = 0
    K2 = 0
    print "Using MC integration to calculate K %s,%s" % (iOrb, jOrb)
    Ntot = int(400000)
    for N in range(0, Ntot):
        x1 = np.pi*0.5*random.random()
        r1 = np.tan(x1)
        z1 = int((np.log(r1) - xmin)/dx)
        if z1 > len(r)-1:
            z1 = len(r)-1
        # r = tan(x) => dr/dx = sec^2(x)
        # int f(x) dr = int f(x) dr/dx dx
        x2 = np.pi*0.5*random.random()
        r2 = np.tan(x2)
        z2 = int((np.log(r2) - xmin)/dx)
        if z2 > len(r)-1:
            z2 = len(r)-1
        t1 = random.random()*np.pi
        t2 = random.random()*np.pi
        p1 = random.random()*2*np.pi
        p2 = random.random()*2*np.pi
        Kn = (np.pi*2)**2*(np.pi)**2*(np.pi*0.5*1.0/(np.cos(x1)**2))*(np.pi*0.5*1.0/(np.cos(x2)**2))*getIntegrandK(r, z1, t1, p1, z2, t2, p2, phiList[iOrb], phiList[jOrb])
        K += Kn
        K2 += Kn**2
    K = K/float(Ntot)
    dK = (K2/float(Ntot-1) - K**2)/float(Ntot)
    return [K, dK]

def calculateE0(r, listPhi, vd, vxc):
    E0 = 0
    dE0 = 0
    sumEV = 0
    J = 0
    K = 0
    JmK = 0
    for iOrb in listPhi.keys():
        if listPhi[iOrb].virtual:
            continue

        E0 += listPhi[iOrb].E
        sumEV += listPhi[iOrb].E
        # for the s orbitals, we can factor out the spherical harms.
        if listPhi[iOrb].l == 0:
            for z in range(0, len(r)):
                dr = 0
                if z < len(r)-1:
                    dr = r[z+1] - r[z]
                # should have 4*pi*Y^2, but for s orbitals Y^2 = 1/4pi and int dOmega = 4 pi
                J += (vd[z]*listPhi[iOrb].rpsi[z]**2)*(r[z]**2)*dr

        # for p, d, etc integrate J and K with MC
        for jOrb in listPhi.keys():
            if listPhi[iOrb].l != 0: # done above for s orbitals
                [Jn, dJn] = getJMC(r, listPhi, iOrb, jOrb)
                J += Jn
                dE0 += dJn**2

            if ('+' in jOrb and '-' in iOrb) or ('-' in jOrb and '+' in iOrb):
                continue

            if listPhi[iOrb].l == 0 and listPhi[jOrb].l == 0:
                Hex = 0
                for z in range(0, len(r)):
                    dr = 0
                    if z < len(r)-1:
                        dr = r[z+1] - r[z]
                    # should have 4*pi*Y^2, but for s orbitals Y^2 = 1/4pi and int dOmega = 4 pi
                    K += vxc[iOrb][jOrb][z]*listPhi[iOrb].rpsi[z]*listPhi[jOrb].rpsi[z]*(r[z]**2)*dr
            else:
                [Kn, dKn] = getKMC(r, listPhi, iOrb, jOrb)
                K += Kn
                dE0 += dKn**2
    E0 += -0.5*(J - K)
    dE0 = np.sqrt(dE0)
    return [E0, sumEV, J, K, dE0]


# average out in angle
## potential calculation
# term is:
# V = int_Oa int_ra rpsi1(ra) rpsi2(ra) Yl1m1(Oa) Yl2m2(Oa)/|ra-rb| ra^2 dOa dra dOb / (4 pi)
# with 1 = 2 -> iOrb
# 1/|ra - rb| = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1) r<^l/r>^(l+1) Y*lm(Oa) Ylm(Ob)
# V = \sum_l=0^inf \sum_m=-l^l ( int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra ) (int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa) (int Ylm(Ob) dOb / (4*pi))
# beta(rb, l) = int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra
# T1 = int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa
# T2 = 1/(4pi) int_Ob Ylm(Ob) dOb
#
#
# V = \sum_m \sum_l=0^inf beta(rb, l) T1(l, m) T2(l, m)
#
#
# T1 = int Yl1m1 Yl1m1 Y*lm = (-1)**m int Yl1m1 Yl1m1 Yl(-m)
# T1 = (-1)**m*(-1)**m*np.sqrt((2*l1+1)*(2*l1+1)/(4*np.pi*(2*l+1)))*CG(l1,l1,0,0,l,0)*CG(l1,l1,m1,m1,l,-(-m))
#
# T2 = 1.0/(4*np.pi) int Ylm dOb
#
def getPotentialHAna(r, phiList):
    totalVd = np.zeros(len(r), dtype=np.float64)
    for iOrb in phiList.keys():
        if phiList[iOrb].virtual:
            continue
        Vd = np.zeros(len(r), dtype=np.float64)
        l1 = phiList[iOrb].l
        m1 = phiList[iOrb].m
        n1 = phiList[iOrb].n
        
        #lx = phiList[xOrb].l
        #mx = phiList[xOrb].m

        nThisOrb = 0
        for yOrb in phiList.keys():
            if phiList[yOrb].n == n1 and phiList[yOrb].l == l1:
                nThisOrb += 1
        if nThisOrb == 2*l1+1: # filled orbital, can calculate it exactly
            nMTot = 2*l1+1
            for z in range(0, len(r)):
                r2 = r[z]
                beta = 0
                # integrate in r1 with r2 fixed
                for y in range(0, len(r)):
                    r1 = r[y]
                    dr = 0
                    if y >= 1:
                        dr = r[y] - r[y-1]
                    else:
                        dr = r[y]
                    rs = r1
                    rb = r2
                    if rb < rs:
                        rs = r2
                        rb = r1
                    beta += 1.0/float(nMTot)*(2*l1+1)*phiList[iOrb].rpsi[y]**2/rb*(r1**2)*dr
                Vd[z] += beta
        else: # not filled orbital, we can take an average over angles
            lmax = 2
            for l in range(0, lmax+1):
                for z in range(0, len(r)):
                    r2 = r[z]
                    beta = 0
                    # integrate in r1 with r2 fixed
                    for y in range(0, len(r)):
                        r1 = r[y]
                        dr = 0
                        if y >= 1:
                            dr = r[y] - r[y-1]
                        else:
                            dr = r[y]
                        rs = r1
                        rb = r2
                        if rb < rs:
                            rs = r2
                            rb = r1
                        beta += 4*np.pi/(2*l+1)*phiList[iOrb].rpsi[y]**2*rs**l/(rb**(l+1))*(r1**2)*dr
                    T = 0
                    for m in range(-l, l+1):
                        # T1 = int Y*l1m1 Yl1m1 Y*lm = (-1)**m int Y*l1m1 Yl1m1 Yl(-m)
                        # T1 = (-1)**m*(-1)**m int Yl1(-m1) Yl1m1 Yl(-m)
                        # T1 = (-1)**m*(-1)**m*(-1)**m*np.sqrt((2*l1+1)*(2*l1+1)/(4*np.pi*(2*l+1)))*CG(l1,l1,0,0,l,0)*CG(l1,l1,-m1,m1,l,-(-m))
                        T1 = (-1)**(m1)*np.sqrt((2*l1+1)*(2*l1+1)/(4*np.pi*(2*l+1)))*CG(l1, l1, 0, 0, l, 0)*CG(l1, l1, -m1, m1, l, -(-m))
                        # just average effect in angles of Ylm by itself
                        # average of Ylm is zero except for l = m = 0
                        T2 = 0
                        if l == 0 and m == 0:
                            T2 = 1.0/np.sqrt(4*np.pi)
                        T += T1*T2
                    Vd[z] += beta*T
        totalVd += Vd
    return totalVd


## potential calculation
# term is:
# V = int_Oa int_Ob int_ra rpsi1(ra) rpsi2(ra) Y*l1m1(Oa) Yl2m2(Oa)/|ra-rb| ra^2 dOa dra dOb / (4 pi)
# with 1 -> iOrb
# with 2 -> jOrb
# 1/|ra - rb| = \sum_l=0^inf \sum_m=-l^m=l 4 pi / (2l + 1) r<^l/r>^(l+1) Y*lm(Oa) Ylm(Ob)
# V = \sum_l=0^inf \sum_m=-l^l ( int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra ) (int_Oa Yl1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa) (int Ylm(Ob) dOb) / (4*pi)
# beta(rb, l) = int_ra 4 pi /(2l+1) rpsi1(ra) rpsi2(ra) r<^l/r>^(l+1) ra^2 dra
# T1 = int_Oa Y*l1m1(Oa) Yl2m2(Oa) Y*lm(Oa) dOa
# T2 = int_Ob Ylm(Ob) dOb
#
#
# V = \sum_m \sum_l=0^inf beta(rb, l) T1(l, m) T2(l, m) / (4*pi)
#
#
# T1 = int Yl1m1 Yl1m1 Y*lm = (-1)**m int Yl1m1 Yl1m1 Yl(-m)
# T1 = (-1)**m*(-1)**m*np.sqrt((2*l1+1)*(2*l1+1)/(4*np.pi*(2*l+1)))*CG(l1,l1,0,0,l,0)*CG(l1,l1,m1,m1,l,-(-m))
#
# T2 = 1.0/(4*np.pi) int Ylm
#
def getPotentialXAna(r, phiList, iOrb):
    totalVx = {}
    for jOrb in phiList.keys():
        if phiList[jOrb].virtual:
            continue

        if ('+' in jOrb and '-' in iOrb) or ('-' in jOrb and '+' in iOrb):
              continue
        totalVx[jOrb] = np.zeros(len(r), dtype=np.float64)
        Vex = np.zeros(len(r), dtype=np.float64)
        #lx = phiList[jOrb].l
        #mx = phiList[jOrb].m
        l1 = phiList[iOrb].l
        m1 = phiList[iOrb].m
        n1 = phiList[iOrb].n
        l2 = phiList[jOrb].l
        m2 = phiList[jOrb].m
        n2 = phiList[jOrb].n

        nThisOrb = 0
        for yOrb in phiList.keys():
            if phiList[yOrb].n == n2 and phiList[yOrb].l == l2:
                nThisOrb += 1
        if nThisOrb == 2*l2+1: # filled orbital, can calculate it exactly
            nMTot = 2*l2+1

            for z in range(0, len(r)):
                r2 = r[z]
                for l in range(abs(l1-l2), l1+l2+1):
                    beta = 0
                    # integrate in r1 with r2 fixed
                    for y in range(0, len(r)):
                        r1 = r[y]
                        dr = 0
                        if y >= 1:
                            dr = r[y] - r[y-1]
                        else:
                            dr = r[y]
                        rs = r1
                        rb = r2
                        if rb < rs:
                            rs = r2
                            rb = r1
                        beta += phiList[iOrb].rpsi[y]*phiList[jOrb].rpsi[y]*(rs**l)/(rb**(l+1))*r1**2*dr
                    Vex[z] += 1.0/float(nMTot)*(2*l2+1)/(2*l+1)*CG(l1, l2, 0, 0, l, 0)**2*beta
        else:
            lmax = 2
            for l in range(0, lmax+1):
                for z in range(0, len(r)):
                    r2 = r[z]
                    beta = 0
                    # integrate in r1 with r2 fixed
                    for y in range(0, len(r)):
                        r1 = r[y]
                        dr = 0
                        if y >= 1:
                            dr = r[y] - r[y-1]
                        else:
                            dr = r[y]
                        rs = r1
                        rb = r2
                        if rb < rs:
                            rs = r2
                            rb = r1
                        beta += 4*np.pi/(2*l+1)*phiList[iOrb].rpsi[y]*phiList[jOrb].rpsi[y]*rs**l/(rb**(l+1))*(r1**2)*dr
                    T = 0
                    for m in range(-l, l+1):
                        # T1 = int Y*l1m1 Yl2m2 Y*lm = (-1)**m int Y*l1m1 Yl2m2 Yl(-m)
                        # T1 = (-1)**m*(-1)**m int Yl1(-m1) Yl2m2 Yl(-m)
                        # T1 = (-1)**m*(-1)**m*(-1)**m*np.sqrt((2*l1+1)*(2*l2+1)/(4*np.pi*(2*l+1)))*CG(l1,l2,0,0,l,0)*CG(l1,l2,-m1,m2,l,-(-m))
                        T1 = (-1)**(m1)*np.sqrt((2*l1+1)*(2*l2+1)/(4*np.pi*(2*l+1)))*CG(l1, l2, 0, 0, l, 0)*CG(l1, l2, -m1, m2, l, -(-m))
                        # just average effect in angles of Ylm by itself
                        T2 = 0
                        if l == 0 and m == 0:
                            T2 = 1.0/np.sqrt(4*np.pi)
                        T += T1*T2
                    Vex[z] += beta*T
        totalVx[jOrb] += Vex
    return totalVx

def getLinSyst(listPhi, r, pot, vd, vxc):
        # prepare eq. F psi = 0
        # psi is a column vector with all orbital in each r value with one extra final entry, which is the energy
        # size of psi = len(listPhi)*len(r) + 1
        # the HF equations will be written in each point in the function F
        # the final equation in F will be (sum psi^2*r^2*dr = 1)
        Nr = len(r)
        N = len(listPhi)*Nr + len(listPhi) + 1
        idxE = len(listPhi)*Nr
        idxSE = len(listPhi)*Nr + len(listPhi)
        # F x is defined as:
        # (12 - 10 f_n) y_n - f_{n-1} y_{n-1} - f_{n+1} y_{n+1} + (s[i+1] + 10.0*s[i] + s[i-1]) = 0
        # we do not write F itself as it is non-linear due to the demand in the last eq.
        # we look for the eigenvalue, using Newton's method
        # given an initial vector psi = x0, we get the next one using:
        # F x = F x0 + J (x - x0)
        # F x = 0 => we should solve J delta X = - F (x0)
        # J is the jacobian => J_ij = dF_i/dx_j
    
        m = 1
    
        F0 = np.zeros(N, dtype=np.float64)
        #J = np.zeros((N, N), dtype=np.float64)
        J = scipy.sparse.lil_matrix((N, N), dtype=np.float64)
        for iOrb in sorted(listPhi.keys()):
            nOrb = phiToInt[iOrb]
            l = listPhi[iOrb].l
            n = listPhi[iOrb].n
            E = listPhi[iOrb].E
    
            # calculate the extra term as \sum_j psi_j Vx_j
            # these are the linear terms due to the remainder of the potentials
            pot_full_effective = pot + vd # this multiplies the current phi[iOrb]
            if iOrb in vxc[iOrb]:
                pot_full_effective -= vxc[iOrb][iOrb]
            potIndep = np.zeros(len(r), dtype = np.float64)
            for jOrb in vxc[iOrb].keys():
                if iOrb == jOrb:
                    continue
                potIndep += listPhi[jOrb].psi*vxc[iOrb][jOrb]
    
            # (12 - 10 f_n) y_n - f_{n-1} y_{n-1} - f_{n+1} y_{n+1} + (s[i+1] + 10.0*s[i] + s[i-1]) = 0
            for ir in range(0, len(r)):
                a = 2*m*r[ir]**2*(E-pot_full_effective[ir]) - (l+0.5)**2
                f = 1 + a*dx**2/12.0
                s_coeff = (dx**2)/12.0*2*m*r[ir]**2
                s = (dx**2)/12.0*2*m*r[ir]**2*potIndep[ir]
                F0[nOrb*Nr+ir] += (12 - 10*f)*listPhi[iOrb].psi[ir] + 10.0*s
                J[nOrb*Nr+ir, nOrb*Nr+ir] += (12 - 10*f)
                J[nOrb*Nr + ir, idxE + nOrb] += -10*(2*m*r[ir]**2)*(dx**2/12.0)*listPhi[iOrb].psi[ir]
                for jOrb in sorted(listPhi.keys()):
                    mOrb = phiToInt[jOrb]
                    if iOrb == jOrb:
                        continue
                    if jOrb in vxc[iOrb]:
                        J[nOrb*Nr+ir, mOrb*Nr+ir] += 10.0*s_coeff*vxc[iOrb][jOrb][ir]
                if ir > 0:
                    a = 2*m*r[ir-1]**2*(E-pot_full_effective[ir-1]) - (l+0.5)**2
                    f = 1 + a*dx**2/12.0
                    s = (dx**2)/12.0*2*m*r[ir-1]**2*potIndep[ir-1]
                    s_coeff = (dx**2)/12.0*2*m*r[ir-1]**2
                    F0[nOrb*Nr+ir] += -f*listPhi[iOrb].psi[ir-1] + s
                    J[nOrb*Nr+ir, nOrb*Nr+ir-1] += -f
                    J[nOrb*Nr + ir, idxE + nOrb] += -(2*m*r[ir-1]**2)*(dx**2/12.0)*listPhi[iOrb].psi[ir-1]
                    for jOrb in sorted(listPhi.keys()):
                        mOrb = phiToInt[jOrb]
                        if iOrb == jOrb:
                            continue
                        if jOrb in vxc[iOrb]:
                            J[nOrb*Nr+ir, mOrb*Nr+ir-1] += s_coeff*vxc[iOrb][jOrb][ir-1]
                if ir < len(r)-1:
                    a = 2*m*r[ir+1]**2*(E-pot_full_effective[ir+1]) - (l+0.5)**2
                    f = 1 + a*dx**2/12.0
                    s = (dx**2)/12.0*2*m*r[ir+1]**2*potIndep[ir+1]
                    s_coeff = (dx**2)/12.0*2*m*r[ir+1]**2
                    F0[nOrb*Nr+ir] += -f*listPhi[iOrb].psi[ir+1] + s
                    J[nOrb*Nr+ir, nOrb*Nr+ir+1] += -f
                    J[nOrb*Nr + ir, idxE + nOrb] += -(2*m*r[ir+1]**2)*(dx**2/12.0)*listPhi[iOrb].psi[ir+1]
                    for jOrb in sorted(listPhi.keys()):
                        mOrb = phiToInt[jOrb]
                        if iOrb == jOrb:
                            continue
                        if jOrb in vxc[iOrb]:
                            J[nOrb*Nr+ir, mOrb*Nr+ir+1] += s_coeff*vxc[iOrb][jOrb][ir+1]

        # (sum psi^2*r^2*dr = 1)
        for iOrb in listPhi:
            l = listPhi[iOrb].l
            n = listPhi[iOrb].n
            nOrb = phiToInt[iOrb]
            E = listPhi[iOrb].E
            for ir in range(0, len(r)):
    	        dr = 0
    	        if ir < len(r)-1:
                    dr = r[ir+1] - r[ir]
                F0[idxE + nOrb] += (listPhi[iOrb].psi[ir]*r[ir]**(-0.5))**2 * r[ir]**2 * dr
            F0[idxE + nOrb] += - 1.0
            F0[idxSE] += 0 # this is the lagrange multiplier eq.: lambda = sum E^2
            # n = int delta(psi(x)) |psi'(x)| dx = sum_roots int delta (x - x_i) |psi'(x)| / |psi'(x_i)| dx
            for ir in range(0, len(r)):
    	        dr = 0
    	        if ir < len(r)-1:
                    dr = r[ir+1] - r[ir]
                J[idxE + nOrb, nOrb*Nr + ir] += 2*listPhi[iOrb].psi[ir]*dr*r[ir]
            J[idxSE, idxE + nOrb] += -2*E
        J[idxSE, idxSE] += 1 # this is a lagrange multiplier: lambda = sum E^2 -> lambda - sum E^2 = 0
        nF0 = 0
        for i in range(0, len(F0)):
            nF0 += F0[i]**2

        return [J, F0, nF0, Nr, N, idxE]

class phi:
    n = 1
    l = 0
    m = 0
    E = -2.0
    no = 0
    psi = None
    rpsi = None
    Emax = 0.0
    Emin = -99.0
    wait = 2
    virtual = False
    def __init__(self, _n, _l, _m, _E, _virtual = False):
        self.n = _n
        self.l = _l
        self.m = _m
        self.E = _E
        self.Emax = 0.0
        self.Emin = -99.0
        self.wait = 2
        self.virtual = _virtual

    def toPsi(self, r, changeInPlace = False):
        n = 0
        for i in range(0, len(self.psi)):
            self.rpsi[i] = self.psi[i]*r[i]**(-0.5) # undo y->R(r) transform
	    ip = len(r)-1
	    if i < len(r)-1:
	        ip = i+1
	    dr = np.fabs(r[ip]-r[i])
            n += (r[i]*self.rpsi[i])**2*dr        # normalise it so that int |r R(r)|^2 dr == 1
        parity = 1
        if self.rpsi[0] < 0:
            parity = -1
        if n != 0:
            for i in range(0, len(self.psi)):
                self.rpsi[i] /= parity*np.sqrt(n)
        if changeInPlace:
            self.psi = self.rpsi[:]
    def toFile(self, r, name, fname):
        fout = open(fname, "w")
        fout.write("# name %s\n" %(name))
        fout.write("# n    %d\n" %(self.n))
        fout.write("# l    %d\n" %(self.l))
        fout.write("# m    %d\n" %(self.m))
        fout.write("# E    %.16f\n" %(self.E))
        for i in range(0, len(r)):
            fout.write("%.16f     %.16f\n" % (r[i], self.rpsi[i]))
        fout.close()

def writePotential(r, V, name, typ, forWF, actsOn, fname):
    fout = open(fname, "w")
    fout.write("# name   %s\n" %(name))
    fout.write("# type   %s\n" %(typ))
    fout.write("# forWF  %s\n" %(forWF))
    fout.write("# actsOn %s\n" %(actsOn))
    for i in range(0, len(r)):
        fout.write("%.16f     %.16f\n" % (r[i], V[i]))
    fout.close()

def savePlotInFile(fname, r, pot, legend, ylabel = '', yrange = [-5,5]):
    f = open(fname, 'w')
    f.write("# %s\n" % legend)
    f.write("set style line 1  lc rgb '#0060ad' lt 1 lw 2 pt 5   # blue\n")
    f.write("set style line 2  lc rgb '#dd181f' lt 1 lw 2 pt 7   # red\n")
    f.write("set style line 3  lc rgb '#00ff00' lt 1 lw 2 pt 9   # green\n")
    f.write("set style line 4  lc rgb '#ffffff' lt 1 lw 2 pt 5   # black\n")
    f.write("set style line 5  lc rgb 'orange'  lt 2 lw 2 pt 5   # orange\n")
    f.write("set style line 6  lc rgb 'skyblue' lt 2 lw 2 pt 5   # skyblue\n")
    f.write("set style line 7  lc rgb 'cyan'    lt 2 lw 2 pt 5   # cyan\n")
    f.write("set style line 8  lc rgb '#0060ad' lt 3 lw 2 pt 5   # blue\n")
    f.write("set style line 9  lc rgb '#dd181f' lt 3 lw 2 pt 7   # red\n")
    f.write("set style line 10 lc rgb '#00ff00' lt 3 lw 2 pt 9   # green\n")
    f.write("set terminal wxt size 800,600 enhanced font 'Verdana,12' persist\n")
    f.write("set grid\n")
    f.write("set xlabel '%s'\n" % ('r [a0]'))
    f.write("set ylabel '%s'\n" % (ylabel))
    f.write("set xrange [0:5]\n")
    f.write("set yrange [%f:%f]\n" %(yrange[0], yrange[1]))
    s = ""
    for i in range(1, len(pot)+1):
        pref = ""
        if i == 1:
            pref = "plot"
        nl = ", \\"
        if i == len(pot):
            nl = ""
        color = i
        s += '%s "-" using 1:2 title "%s" with lines ls %d %s\n' % (pref, legend[i-1], color, nl)
    f.write(s)
    for j in range(0, len(pot)):
        s = "# '%s' " % "r[a0]"
        s += " '%s' " % legend[j]
        s += "\n"
        f.write(s)
        for i in range(0, len(r)):
            s = "%10f " % r[i]
            s += " %10f " % pot[j][i]
            s += "\n"
            f.write(s)
        f.write("end\n")
    f.close()

Z = 5

xmin = np.log(1e-4)
dx = 1e-1/Z
r = init(dx, Z*150, xmin)

useMC = False

listPhi = {}
# create objects to hold energy and wave functions of each Hartree-Fock equation
# provide boundary conditions n, l in first arguments
# provide initial energy to use when starting to look for solutions
# propose to start with the Hydrogen-like (if Hydrogen had atomic number Z) energy level (0.5*Z^2/n^2)
listPhi['1s1+'] = phi(1, 0, 0, -Z**2/(1.0**2)*0.5)
listPhi['1s1-'] = phi(1, 0, 0, -Z**2/(1.0**2)*0.5)
listPhi['2s1+'] = phi(2, 0, 0, -Z**2/(2.0**2)*0.5)
listPhi['2s1-'] = phi(2, 0, 0, -Z**2/(2.0**2)*0.5)
listPhi['2p1+'] = phi(2, 1, 0, -Z**2/(2.0**2)*0.5)
listPhi['2p2+'] = phi(2, 1, 1, -Z**2/(2.0**2)*0.5, True)

Nwait = 4*len(listPhi)

phiToInt = {}
intToPhi = {}
nOrb = 0
for i in listPhi:
    phiToInt[i] = nOrb
    intToPhi[nOrb] = i
    nOrb += 1

pot = V(r, Z)

for iOrb in listPhi.keys():
    listPhi[iOrb].psi = np.zeros(len(r), dtype=np.float64)
    listPhi[iOrb].rpsi = np.zeros(len(r), dtype=np.float64)
    n = listPhi[iOrb].n
    l = listPhi[iOrb].l
    for ir in range(0, len(r)):
        listPhi[iOrb].psi[ir] = 1e-3

Nscf = 1000

vd_last = {}
vxc_last = {}
gamma_v = 0.3

abortIt = False
E0_old = 0
E0 = 0
for iSCF in range(0, Nscf):
    print bcolors.HEADER + "On HF SCF iteration %d" % iSCF + bcolors.ENDC

    for iOrb in sorted(listPhi.keys()):
        listPhi[iOrb].wait = 0

    if iSCF == 0:
        vxc = {}
        vd = np.zeros(len(r), dtype = np.float64)
        vd_last = vd
        for iOrb in sorted(listPhi.keys()):
            nOrb = phiToInt[iOrb]
            vxc[iOrb] = {}
            vxc_last[iOrb] = {}
            for jOrb in sorted(listPhi.keys()):
                vxc[iOrb][jOrb] = np.zeros(len(r), dtype = np.float64)
                vxc_last[iOrb][jOrb] = vxc[iOrb][jOrb]
    else:
        gamma_v_eff = gamma_v # *np.exp(-iSCF/20.0)
        if iSCF >= 20:
            gamma_v_eff = gamma_v # *np.exp(-1.0)
        vxc = {}
        vxc_new = {}
        if useMC:
            vd_new = getPotentialH(r, listPhi)
        else:
            vd_new = getPotentialHAna(r, listPhi)
        vd = vd_last*(1-gamma_v_eff) + vd_new*(gamma_v_eff)
        vd_last = vd
        for iOrb in sorted(listPhi.keys()):
            if useMC:
                vxc_new[iOrb] = getPotentialX(r, listPhi, iOrb)
            else:
                vxc_new[iOrb] = getPotentialXAna(r, listPhi, iOrb)
            vxc[iOrb] = {}
            for jOrb in vxc_new[iOrb]:
                vxc[iOrb][jOrb] = vxc_last[iOrb][jOrb]*(1-gamma_v_eff) + vxc_new[iOrb][jOrb]*(gamma_v_eff)
                vxc_last[iOrb][jOrb] = vxc[iOrb][jOrb]
    np.set_printoptions(threshold=np.inf)

    # Newton iterations
    # solve J dX = - F0
    minF0Sum = 1e50
    bestPhi = {}
    for iOrb in sorted(listPhi.keys()):
        listPhi[iOrb].Emin = -Z**2/listPhi[iOrb].n**2 
        listPhi[iOrb].Emax = 0

    listPhi_prev = {}
    scale_gamma = 1.0
    for iN in range(0, 2000):
        print bcolors.OKBLUE + "(SCF it. %d) On Newton-Raphson minimum search iteration %d (SCF potential fixed here)" % (iSCF, iN) + bcolors.ENDC

        [J, F0, nF0, Nr, N, idxE] = getLinSyst(listPhi, r, pot, vd, vxc)

        print bcolors.WARNING + "(SCF it. %d, NR it. %d) Current minimisation function value \sum F_i^2 = %.14f. Best minimum found in NR it. min \sum F_i^2 = %.14f" % (iSCF, iN, nF0, minF0Sum) + bcolors.ENDC
        finishNow = False
        if nF0 < minF0Sum:
            minF0Sum = nF0
            finishNow = True
            # save last state
            for iOrb in listPhi:
                listPhi_prev[iOrb] = listPhi[iOrb]
        elif iSCF > 1: # new step does not improve things ...
            w = 0
            for iOrb in listPhi:
                w += listPhi[iOrb].wait
            if w == 0:
                # this can happen when the direct and exchange potentials are not there as we are far off the solution
                # but after the first optimisation, we should take measures to avoid it
                # at that stage it happens often when we change the solution by too much and skip the minimum
                # so, let's go back and try to reduce the step
                # go back to the previous step and reduce gamma
                for iOrb in listPhi:
                    listPhi[iOrb] = listPhi_prev[iOrb]
                [J, F0, nF0, Nr, N, idxE] = getLinSyst(listPhi, r, pot, vd, vxc)
                print bcolors.WARNING + "(SCF it. %d, NR it. %d) New function is bigger than previous iteration. Going back and reducing the step to gamma = %.14f. Current minimisation function value \sum F_i^2 = %.14f. Best minimum found in NR it. min \sum F_i^2 = %.14f" % (iSCF, iN, gamma*scale_gamma, nF0, minF0Sum) + bcolors.ENDC
                # as the function value grew, let's end this ...
                #abortIt = True
                #break


        gamma = 0.3*scale_gamma

        no_old = {}
        for iOrb in listPhi:
            no_old[iOrb] = 0
            for i in range(1, int(len(r))):
                if listPhi[iOrb].rpsi[i]*listPhi[iOrb].rpsi[i-1] < 0 and r[i] > 0.01:
                    no_old[iOrb] += 1

        Jcsr = J.tocsr()
        dX = scipy.sparse.linalg.spsolve(Jcsr, F0)

        for iOrb in listPhi:
            nOrb = phiToInt[iOrb]
            n = listPhi[iOrb].n
            for ir in range(0, len(r)):
                listPhi[iOrb].psi[ir] += -gamma*dX[nOrb*Nr + ir]

        # multiply by 1/sqrt(r) to undo transformation that guarantees convergence at zero
        # and renormalise again (should already be guaranteed by last equations in J and F0, but
        # this should force this to be always true, even if we are slightly away from the true solution
        # result in listPhi[iOrb].rpsi
        for iOrb in listPhi:
            listPhi[iOrb].toPsi(r, False)

        no = {}
        for iOrb in listPhi:
            no[iOrb] = 0
            for i in range(1, int(len(r))):
                if listPhi[iOrb].rpsi[i]*listPhi[iOrb].rpsi[i-1] < 0 and r[i] > 0.1:
                    print "New (%s): zero crossing at %5f" %(iOrb, r[i])
                    no[iOrb] += 1

        for iOrb in listPhi:
            print "Old %s: E = %5f, nodes = %d, Emax = %5f, Emin = %5f, wait it. = %d" % (iOrb, listPhi[iOrb].E*eV, no_old[iOrb], listPhi[iOrb].Emax*eV, listPhi[iOrb].Emin*eV, listPhi[iOrb].wait)

        for iOrb in listPhi:
            nOrb = phiToInt[iOrb]
            n = listPhi[iOrb].n
            l = listPhi[iOrb].l
            dE = - gamma*dX[idxE+nOrb]

            #if no[iOrb] > nodes(listPhi[iOrb].n, listPhi[iOrb].l) and listPhi[iOrb].wait <= 0:
            #    listPhi[iOrb].Emax = listPhi[iOrb].E
            #    if nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0 and no[iOrb] != 0:
            #        dE = -np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2) - Z**2*0.5/(no[iOrb]**2))*0.1
            #    elif nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0:
            #        dE = -np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2))*0.1
            #    elif no[iOrb] != 0:
            #        dE = -np.fabs(Z**2*0.5/(no[iOrb]**2))*0.1
            #    else:
            #        dE = -0.001
            #    dE = (listPhi[iOrb].Emax + listPhi[iOrb].Emin)*0.5 - listPhi[iOrb].E
            #    listPhi[iOrb].E += dE
            #    for ir in range(0, len(r)):
            #        listPhi[iOrb].psi[ir] = 1
            #        listPhi[iOrb].rpsi[ir] = 0
            #    listPhi[iOrb].wait = Nwait
            #elif no[iOrb] < nodes(listPhi[iOrb].n, listPhi[iOrb].l) and listPhi[iOrb].wait <= 0:
            #    listPhi[iOrb].Emin = listPhi[iOrb].E
            #    if nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0 and no[iOrb] != 0:
            #        dE = np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2) - Z**2*0.5/(no[iOrb]**2))*0.1
            #    elif nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0:
            #        dE = np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2))*0.1
            #    elif no[iOrb] != 0:
            #        dE = np.fabs(Z**2*0.5/(no[iOrb]**2))*0.1
            #    else:
            #        dE = 0.001
            #    dE = (listPhi[iOrb].Emax + listPhi[iOrb].Emin)*0.5 - listPhi[iOrb].E
            #    listPhi[iOrb].E += dE
            #    for ir in range(0, len(r)):
            #        listPhi[iOrb].psi[ir] = 1
            #        listPhi[iOrb].rpsi[ir] = 0
            #    listPhi[iOrb].wait = Nwait
            #else:
            if True:
                if np.fabs(dE) > 0.1:
                    dE = 0.1*dE/np.fabs(dE)
                listPhi[iOrb].E += dE
                if dE > 0 and listPhi[iOrb].E > listPhi[iOrb].Emax:
                    listPhi[iOrb].E = listPhi[iOrb].Emax
                elif dE < 0 and listPhi[iOrb].E < listPhi[iOrb].Emin:
                    listPhi[iOrb].E = listPhi[iOrb].Emin

            listPhi[iOrb].wait -= 1
            if listPhi[iOrb].wait < 0:
                listPhi[iOrb].wait = 0
            print "New %s: E = %5f, nodes = %d, Emax = %5f, Emin = %5f, wait it. = %d" % (iOrb, listPhi[iOrb].E*eV, no[iOrb], listPhi[iOrb].Emax*eV, listPhi[iOrb].Emin*eV, listPhi[iOrb].wait)

        idxhigh = np.where(r > 10.0)
        if len(idxhigh[0]) != 0:
            idxhigh = idxhigh[0][0]
        else:
            idxhigh = len(r)-1
        idx = np.where(r > 5)
        if len(idx[0]) != 0:
            idx = idx[0][0]
        else:
            idx = len(r)-1
        idxlow = np.where(r > 1.0)
        if len(idxlow[0]) != 0:
            idxlow = idxlow[0][0]
        else:
            idxlow = 0
        plt.clf()
        plist = []
        leg = []
        exact_p = 2*np.exp(-r)   # solution for R(r) in Hydrogen, n = 1
        col = ['r-', 'g-', 'b-', 'r-.', 'g-.', 'b-.', 'r--', 'g--', 'b--']
        c = 0
        for iOrb in listPhi.keys():
            plt.plot(r[0:idx], listPhi[iOrb].rpsi[0:idx], col[c], label='$R_{%s}$'%iOrb)
            plist.append(listPhi[iOrb].rpsi)
            c += 1
            leg.append('%s (%3f eV)' % (iOrb, listPhi[iOrb].E*eV))
        plt.plot(r[0:idx], exact_p[0:idx], 'g--', label='$R_{exact}$')
        leg.append('Exact H (1s)')
    
        plt.legend(leg, frameon=False)
        plt.xlabel('$r$ [a0]')
        plt.ylabel('$|R(r)|$')
        plt.title('Z=%d, SCF iter=%d, E_{0}=%4f eV'%(Z, iSCF, E0*eV))
        plt.draw()
        plt.savefig('pseudo_potentials.pdf', bbox_inches='tight')
        ymin = np.amin(plist)
        ymax = np.amax(plist)
        savePlotInFile('pseudo_potentials.plt', r, plist, leg, 'R(r)', [ymin, ymax])

        # show potentials squared
        plt.clf()
        plist = []
        leg = []
        c = 0
        for iOrb in listPhi.keys():
            plt.plot(r[0:idxhigh], listPhi[iOrb].rpsi[0:idxhigh]**2*r[0:idxhigh]**2, col[c], label='$R_{%s}^2 r^2$'%iOrb)
            plist.append(listPhi[iOrb].rpsi**2*r**2)
            c += 1
            leg.append('%s (%3f eV)' % (iOrb, listPhi[iOrb].E*eV))
    
        plt.legend(leg, frameon=False)
        plt.xlabel('$r$ [a0]')
        plt.ylabel('$|R(r)|^2 r^2$')
        plt.title('Z=%d, SCF iter=%d, E_{0}=%4f eV'%(Z, iSCF, E0*eV))
        plt.draw()
        plt.savefig('pseudo_potentials2.pdf', bbox_inches='tight')
        ymin = np.amin(plist)
        ymax = np.amax(plist)
        savePlotInFile('pseudo_potentials2.plt', r, plist, leg, 'r^2 R(r)^2', [ymin, ymax])

        # now save the potential shapes
        for iOrb in listPhi.keys():
            leg = []
            plt.clf()
            c = 0
            ymin = pot[idxlow]
            l = [vd[0]]
            for item in vxc[iOrb]:
                l.append(vxc[iOrb][item][0])
            ymax = 1.3*np.amax(l)
            vlist = []
            plt.plot(r[0:idx], pot[0:idx], col[c], label='Vnuc')
            vlist.append(pot)
            leg.append('Vnuc')
            c += 1
            plt.plot(r[0:idx], vd[0:idx], col[c], label='Vd')
            vlist.append(vd)
            leg.append('Vd')
            c += 1
            for item in vxc[iOrb]:
                plt.plot(r[0:idx], vxc[iOrb][item][0:idx], col[c], label='Vxc wrt %s' % item)
                vlist.append(vxc[iOrb][item])
                leg.append('Vxc wrt %s' % item)
                c += 1
            plt.legend(leg, frameon=False)
            plt.xlabel('$r$ [a0]')
            plt.ylabel('Potential')
            plt.title('Z=%d, SCF iter=%d, %s %f eV'%(Z, iSCF, iOrb, listPhi[iOrb].E*eV))
            plt.ylim([ymin, ymax])
            plt.draw()
            plt.savefig('pot_%s.pdf' % iOrb, bbox_inches='tight')
            savePlotInFile('pot_%s.plt' % (iOrb), r, vlist, leg, 'Potential', [ymin, ymax])
            n = listPhi[iOrb].n
            l = listPhi[iOrb].l
            #if no[iOrb] != nodes(n, l):
            #    import sys
            #    sys.exit(0)
        print bcolors.WARNING + "(SCF it. %d, NR it. %d) Last ground state calculation: E0 = %.14f eV" % (iSCF, iN, E0*eV) + bcolors.ENDC
        if minF0Sum < 1e-12*float(len(listPhi)) and finishNow:
            print bcolors.WARNING + "(SCF it. %d, NR it. %d) Ending Newton-Raphson iterations due to very small target function: \sum F0^2 = %.14f." % (iSCF, iN, minF0Sum) + bcolors.ENDC
            break

    [E0, sumEV, J, K, dE0] = calculateE0(r, listPhi, vd, vxc)
    if (np.fabs(1 - E0_old/E0) < 5e-4 and iSCF > 5) or abortIt:
        print bcolors.WARNING + "(SCF it. %d) Ground state energy changed by less than 1e-4 (by %.14f). E0 = %.14f eV +/- %.14f. \sum e = %.14f eV. J = %.14f eV. K = %.14f eV." % (iSCF, np.fabs(1 - E0_old/E0), E0*eV, dE0*eV, sumEV*eV, J*eV, K*eV) + '' + bcolors.ENDC
        break
    else:
        print bcolors.WARNING + "(SCF it. %d ends) E0 = %.14f eV +/- %.14f, dE0/E0 = %.14f. \sum e = %.14f eV. J = %.14f eV. K = %.14f eV." % (iSCF, E0*eV, dE0*eV, (1 - E0_old/E0), sumEV*eV, J*eV, K*eV) + '' + bcolors.ENDC
    E0_old = E0

for item in listPhi:
    listPhi[item].toFile(r, item, "rpsi_"+name+".dat")

writePotential(r, pot, "nucleus", "nucleus", "all", "all", "pot_nuc.dat")
writePotential(r, vd,  "vd",      "hartree", "all", "all", "pot_vd.dat")
for item in vxc:
    for acted in vxc[item]:
        writePotential(r, vxc[item][acted],  "vxc", "exchange", item, acted, "pot_vxc_%s_%s.dat" % (item, acted))

