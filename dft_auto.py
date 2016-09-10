#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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
    r = np.zeros(N)
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

def getPotentialH(r, nu, nd):
    # 0) calculate rho(r) = n(r)
    # 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
    # 2) calculate E(r) = Q(r)/(4*pi*r^2)
    # 3) calculate Vd(r) = sum_r'=inf^r E(r)*dr
    Eu = np.zeros(len(r))
    Qu = 0
    Ed = np.zeros(len(r))
    Qd = 0
    E = np.zeros(len(r))
    Q = 0
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        Qu += 4*np.pi*2*nu[z]*r[z]**2*dr
        Qd += 4*np.pi*2*nd[z]*r[z]**2*dr
        Q += 4*np.pi*(nu[z] + nd[z])*r[z]**2*dr
        # this is E:
        Eu[z] = Qu/(4*np.pi*r[z]**2)
        Ed[z] = Qd/(4*np.pi*r[z]**2)
        E[z] = Q/(4*np.pi*r[z]**2)
    Vu = np.zeros(len(r))
    Vd = np.zeros(len(r))
    V = np.zeros(len(r))

    Vu[len(r)-1] = 0
    Vd[len(r)-1] = 0
    V[len(r)-1] = 0
    # now integrate backwards
    # Vd(r) = int_inf^r E(r') dr'
    # Vd(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
    # Vd(r-h) = Vd(r) + E(r)*dr
    for z in reversed(range(0, len(r)-1)):
        Vu[z] = Vu[z+1] + Eu[z]*(r[z+1] - r[z])
        Vd[z] = Vd[z+1] + Ed[z]*(r[z+1] - r[z])
        V[z] = V[z+1] + 0.5*(Eu[z] + Ed[z])*(r[z+1] - r[z])
    return [Vu, Vd, V]

def getPotentialXC(r, nu, nd):
    # 0) calculate rho(r) = n(r) * e_XC(n(r))
    # 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
    # 2) calculate E(r) = Q(r)/(4*pi*r^2)
    # 3) calculate Vd(r) = sum_r'=inf^r E(r)*dr

    # E_xc = -3/4 (3/pi)^(1/3) 4*pi int n(r)^(4/3) r^2 dr
    # e_xc = -3/4 (3/pi)^(1/3) n(r)^(4/3)
    # v_xc = e_xc + n(r) de_xc/dn
    #      = e_xc + n(r) (-3/4 (3/pi)^(1/3) (4/3) n(r)^(1/3) )

    excu = np.zeros(len(r))
    excd = np.zeros(len(r))
    exc = np.zeros(len(r))
    Vu = np.zeros(len(r))
    Vd = np.zeros(len(r))
    V = np.zeros(len(r))
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        exc[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(nu[z] + nd[z])**(1.0/3.0)
        excu[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*nu[z]**(1.0/3.0)
        excd[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*nd[z]**(1.0/3.0)
        # this is E:
        Vu[z] = excu[z] #+ nu[z]*(-3.0/4.0*(1.0/3.0)*(3.0/np.pi)**(1.0/3.0)*nu[z]**(-2.0/3.0))
        Vd[z] = excd[z] #+ nd[z]*(-3.0/4.0*(1.0/3.0)*(3.0/np.pi)**(1.0/3.0)*nd[z]**(-2.0/3.0))
        V[z] = exc[z] #+ (nu[z] + nd[z])*(-3.0/4.0*(1.0/3.0)*(3.0/np.pi)**(1.0/3.0)*(nu[z] + nd[z])**(-2.0/3.0))
        if nu[z] != 0:
            Vu[z] += nu[z]*(-3.0/4.0*(1.0/3.0)*(3.0/np.pi)**(1.0/3.0)*nu[z]**(-2.0/3.0))
        if nd[z] != 0:
            Vd[z] += nd[z]*(-3.0/4.0*(1.0/3.0)*(3.0/np.pi)**(1.0/3.0)*nd[z]**(-2.0/3.0))
        if nu[z] + nd[z] != 0:
            V[z] += (nu[z] + nd[z])*(-3.0/4.0*(1.0/3.0)*(3.0/np.pi)**(1.0/3.0)*(nu[z] + nd[z])**(-2.0/3.0))

    return [Vu, Vd, V, excu, excd, exc]

def calculateE0(r, listPhi, nu, nd):
    E0 = 0
    for iOrb in listPhi.keys():
        E0 += listPhi[iOrb].E

    # add - 1/2 int int dr dr' n(r) n(r')/(r-r') - int dr n(r) V_xc(r) + E_xc
    #  =  - 1/2 int dr n(r) vh(r) - int dr n(r) V_xc(r) + E_xc
    # in what follows everything is divided by 4 pi to compensate the normalisation of the
    # spherical harmonics (which we assume to be as in Hydrogen)
    [vu, vd, vh] = getPotentialH(r, nu, nd)

    Qh = 0
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        #Qh += 4*np.pi*(vu[z]*nu[z]+vd[z]*nd[z])*r[z]**2*dr/(4*np.pi)
        Qh += 4*np.pi*(vh[z]*(nu[z]+nd[z]))*r[z]**2*dr/(4*np.pi)

    [vxcu, vxcd, vxc, excu, excd, exc] = getPotentialXC(r, nu, nd)
    Qxc = 0
    Exc = 0
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        #Qxc += 4*np.pi*(vxcu[z]*nu[z] + vxcd[z]*nd[z])*r[z]**2*dr/(4*np.pi)
        Qxc += 4*np.pi*(vxc[z]*(nu[z] + nd[z]))*r[z]**2*dr/(4*np.pi)
        #Exc += 4*np.pi*(excu[z] + excd[z])*r[z]**2*dr/(4*np.pi)
        Exc += 4*np.pi*(exc[z])*r[z]**2*dr/(4*np.pi)

    print "sum epsilon = %4f eV" % (E0*eV)
    print "-0.5Qh      = %4f eV" % (-0.5*Qh*eV)
    print "-Qxc        = %4f eV" % (-Qxc*eV)
    print "Exc         = %4f eV" % (Exc*eV)
    E0 += -0.5*Qh - Qxc + Exc
    print "E0          = %4f eV" % (E0*eV)
    return E0

def getLinSyst(listPhi, r, pot, vd, vxcu, vxcd):
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
        J = np.zeros((N, N), dtype=np.float64)
        for iOrb in sorted(listPhi.keys()):
            nOrb = phiToInt[iOrb]
            l = listPhi[iOrb].l
            n = listPhi[iOrb].n
            E = listPhi[iOrb].E
    
            # calculate the extra term as \sum_j psi_j Vx_j
            # these are the linear terms due to the remainder of the potentials
            pot_full_effective = pot + vd # this multiplies the current phi[iOrb]
            if '+' in iOrb:
                pot_full_effective += vxcu
            elif '-' in iOrb:
                pot_full_effective += vxcd
            potIndep = np.zeros(len(r), dtype = np.float64)
            #if iOrb in vxc[iOrb]:
            #    pot_full_effective -= vxc[iOrb][iOrb]
            #for jOrb in vxc[iOrb].keys():
            #    if iOrb == jOrb:
            #        continue
            #    potIndep += listPhi[jOrb].psi*vxc[iOrb][jOrb]
    
            # (12 - 10 f_n) y_n - f_{n-1} y_{n-1} - f_{n+1} y_{n+1} + (s[i+1] + 10.0*s[i] + s[i-1]) = 0
            for ir in range(0, len(r)):
                a = 2*m*r[ir]**2*(E-pot_full_effective[ir]) - (l+0.5)**2
                f = 1 + a*dx**2/12.0
                s_coeff = (dx**2)/12.0*2*m*r[ir]**2
                s = (dx**2)/12.0*2*m*r[ir]**2*potIndep[ir]
                F0[nOrb*Nr+ir] += (12 - 10*f)*listPhi[iOrb].psi[ir] + 10.0*s
                if ir == 0:
                    F0[nOrb*Nr+ir] += -f*((Z*r[0])**(l+0.5))
                J[nOrb*Nr+ir, nOrb*Nr+ir] += (12 - 10*f)
                J[nOrb*Nr + ir, idxE + nOrb] += -10*(2*m*r[ir]**2)*(dx**2/12.0)*listPhi[iOrb].psi[ir]
                #for jOrb in sorted(listPhi.keys()):
                #    mOrb = phiToInt[jOrb]
                #    if iOrb == jOrb:
                #        continue
                #    if jOrb in vxc[iOrb]:
                #        J[nOrb*Nr+ir, mOrb*Nr+ir] += 10.0*s_coeff*vxc[iOrb][jOrb][ir]
                if ir > 0:
                    a = 2*m*r[ir-1]**2*(E-pot_full_effective[ir-1]) - (l+0.5)**2
                    f = 1 + a*dx**2/12.0
                    s = (dx**2)/12.0*2*m*r[ir-1]**2*potIndep[ir-1]
                    s_coeff = (dx**2)/12.0*2*m*r[ir-1]**2
                    F0[nOrb*Nr+ir] += -f*listPhi[iOrb].psi[ir-1] + s
                    J[nOrb*Nr+ir, nOrb*Nr+ir-1] += -f
                    J[nOrb*Nr + ir, idxE + nOrb] += -(2*m*r[ir-1]**2)*(dx**2/12.0)*listPhi[iOrb].psi[ir-1]
                    #for jOrb in sorted(listPhi.keys()):
                    #    mOrb = phiToInt[jOrb]
                    #    if iOrb == jOrb:
                    #        continue
                    #    if jOrb in vxc[iOrb]:
                    #        J[nOrb*Nr+ir, mOrb*Nr+ir-1] += s_coeff*vxc[iOrb][jOrb][ir-1]
                if ir < len(r)-1:
                    a = 2*m*r[ir+1]**2*(E-pot_full_effective[ir+1]) - (l+0.5)**2
                    f = 1 + a*dx**2/12.0
                    s = (dx**2)/12.0*2*m*r[ir+1]**2*potIndep[ir+1]
                    s_coeff = (dx**2)/12.0*2*m*r[ir+1]**2
                    F0[nOrb*Nr+ir] += -f*listPhi[iOrb].psi[ir+1] + s
                    J[nOrb*Nr+ir, nOrb*Nr+ir+1] += -f
                    J[nOrb*Nr + ir, idxE + nOrb] += -(2*m*r[ir+1]**2)*(dx**2/12.0)*listPhi[iOrb].psi[ir+1]
                    #for jOrb in sorted(listPhi.keys()):
                    #    mOrb = phiToInt[jOrb]
                    #    if iOrb == jOrb:
                    #        continue
                    #    if jOrb in vxc[iOrb]:
                    #        J[nOrb*Nr+ir, mOrb*Nr+ir+1] += s_coeff*vxc[iOrb][jOrb][ir+1]

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
    E = -2.0
    no = 0
    psi = None
    rpsi = None
    Emax = 0.0
    Emin = -99.0
    wait = 2
    def __init__(self, _n, _l, _E):
        self.n = _n
        self.l = _l
        self.E = _E
        self.Emax = 0.0
        self.Emin = -99.0
        self.wait = 2

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

def savePlotInFile(fname, r, pot, legend, ylabel = '', yrange = [-5,5]):
    f = open(fname, 'w')
    f.write("# %s\n" % legend)
    f.write("set style line 1  lc rgb '#0060ad' lt 1 lw 2 pt 5   # blue\n")
    f.write("set style line 2  lc rgb '#dd181f' lt 1 lw 2 pt 7   # red\n")
    f.write("set style line 3  lc rgb '#00ff00' lt 1 lw 2 pt 9   # green\n")
    f.write("set style line 4  lc rgb '#000000' lt 1 lw 2 pt 5   # black\n")
    f.write("set style line 5  lc rgb 'orange'  lt 1 lw 2 pt 5   # orange\n")
    f.write("set style line 6  lc rgb '#0060ad' lt 3 lw 2 pt 5   # blue\n")
    f.write("set style line 7  lc rgb '#dd181f' lt 3 lw 2 pt 7   # red\n")
    f.write("set style line 8  lc rgb '#00ff00' lt 3 lw 2 pt 9   # green\n")
    f.write("set style line 9  lc rgb 'skyblue' lt 2 lw 2 pt 5   # skyblue\n")
    f.write("set style line 10 lc rgb 'cyan'    lt 2 lw 2 pt 5   # cyan\n")
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

Z = 2
Einit = -2.0

dx = 1e-1/Z
r = init(dx, Z*150, np.log(1e-4))

listPhi = {}
listPhi['1s1+'] = phi(1, 0, -Z**2/(1.0**2)*0.5)
listPhi['1s1-'] = phi(1, 0, -Z**2/(1.0**2)*0.5)

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

gamma_v = 0.9

nu = np.zeros(len(r))
nd = np.zeros(len(r))

E0_old = 0
E0 = 0
Nscf = 100
for iSCF in range(0, Nscf):
    print bcolors.HEADER + "On HF SCF iteration %d" % iSCF + bcolors.ENDC

    for iOrb in sorted(listPhi.keys()):
        listPhi[iOrb].wait = Nwait

    if iSCF != 0:
        nu_old = nu
        nd_old = nd
        for idx in range(0, len(nu)):
            nu[idx] = 0
            nd[idx] = 0
        for iOrb in listPhi.keys():
            if '+' in iOrb:
                nu += listPhi[iOrb].rpsi**2
            else:
                nd += listPhi[iOrb].rpsi**2
        nu = nu_old*(gamma_v) + nu*(1-gamma_v)
        nd = nd_old*(gamma_v) + nu*(1-gamma_v)

    [vu, vd, vh] = getPotentialH(r, nu, nd)
    [vxcu, vxcd, vxc, excu, excd, exc] = getPotentialXC(r, nu, nd)

    np.set_printoptions(threshold=np.inf)

    # Newton iterations
    # solve J dX = - F0
    minF0Sum = 1e50
    bestPhi = {}
    E0 = 0
    for iOrb in sorted(listPhi.keys()):
        listPhi[iOrb].Emin = -Z**2*0.5
        listPhi[iOrb].Emax = 0

    for iN in range(0, 2000):
        print bcolors.OKBLUE + "(SCF it. %d) On Newton-Raphson minimum search iteration %d (SCF potential fixed here)" % (iSCF, iN) + bcolors.ENDC
        [J, F0, nF0, Nr, N, idxE] = getLinSyst(listPhi, r, pot, vd, vxcu, vxcd)
        gamma = 0.05
        for item in listPhi:
            if np.fabs(listPhi[item].E)*eV < 10:
                gamma = 0.01

        print bcolors.WARNING + "(SCF it. %d, NR it. %d) Current minimisation function value \sum F_i^2 = %5f. Best minimum found in NR it. min \sum F_i^2 = %5f" % (iSCF, iN, nF0, minF0Sum) + bcolors.ENDC
        finishNow = False
        if nF0 < minF0Sum:
            minF0Sum = nF0
            if iN > 40:
                finishNow = True
        #else:
        #    print "Getting out of loop as it went crazy!"
        #    break

        no_old = {}
        for iOrb in listPhi:
            no_old[iOrb] = 0
            for i in range(1, int(len(r))):
                if listPhi[iOrb].rpsi[i]*listPhi[iOrb].rpsi[i-1] < 0 and r[i] > 0.1:
                    no_old[iOrb] += 1

        dX = np.linalg.solve(J, F0)

        for iOrb in listPhi:
            nOrb = phiToInt[iOrb]
            n = listPhi[iOrb].n
            for ir in range(0, len(r)):
                #listPhi[iOrb].psi[ir] += -gamma*listPhi[iOrb].E/(-Z**2*0.5/(n**2))*dX[nOrb*Nr + ir]
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
            #dE = -gamma*listPhi[iOrb].E/(-Z**2*0.5/(n**2))*dX[idxE + nOrb]
            dE = -gamma*dX[idxE + nOrb]

            if no[iOrb] > nodes(listPhi[iOrb].n, listPhi[iOrb].l) and listPhi[iOrb].wait <= 0:
                listPhi[iOrb].Emax = listPhi[iOrb].E
                if nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0 and no[iOrb] != 0:
                    dE = -np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2) - Z**2*0.5/(no[iOrb]**2))*0.1
                elif nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0:
                    dE = -np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2))*0.1
                elif no[iOrb] != 0:
                    dE = -np.fabs(Z**2*0.5/(no[iOrb]**2))*0.1
                else:
                    dE = -0.001
                dE = (listPhi[iOrb].Emax + listPhi[iOrb].Emin)*0.5 - listPhi[iOrb].E
                listPhi[iOrb].E += dE
                for ir in range(0, len(r)):
                    listPhi[iOrb].psi[ir] = 1
                    listPhi[iOrb].rpsi[ir] = 0
                listPhi[iOrb].wait = Nwait
            elif no[iOrb] < nodes(listPhi[iOrb].n, listPhi[iOrb].l) and listPhi[iOrb].wait <= 0:
                listPhi[iOrb].Emin = listPhi[iOrb].E
                if nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0 and no[iOrb] != 0:
                    dE = np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2) - Z**2*0.5/(no[iOrb]**2))*0.1
                elif nodes(listPhi[iOrb].n, listPhi[iOrb].l) != 0:
                    dE = np.fabs(Z**2*0.5/(nodes(listPhi[iOrb].n, listPhi[iOrb].l)**2))*0.1
                elif no[iOrb] != 0:
                    dE = np.fabs(Z**2*0.5/(no[iOrb]**2))*0.1
                else:
                    dE = 0.001
                dE = (listPhi[iOrb].Emax + listPhi[iOrb].Emin)*0.5 - listPhi[iOrb].E
                listPhi[iOrb].E += dE
                for ir in range(0, len(r)):
                    listPhi[iOrb].psi[ir] = 1
                    listPhi[iOrb].rpsi[ir] = 0
                listPhi[iOrb].wait = Nwait
            else:
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
        E0 = calculateE0(r, listPhi, nu, nd)
        plt.title('Z=%d, SCF iter=%d, E_{0}=%4f eV'%(Z, iSCF, E0*eV))
        plt.draw()
        plt.savefig('pseudo_potentials.pdf', bbox_inches='tight')
        ymin = np.amin(plist)
        ymax = np.amax(plist)
        savePlotInFile('pseudo_potentials.plt', r, plist, leg, 'R(r)', [ymin, ymax])
        #plt.show()
        for iOrb in listPhi.keys():
            leg = []
            plt.clf()
            c = 0
            l = []
            l.append(pot)
            l.append(vd)
            l.append(vxcu)
            l.append(vxcd)
            l.append(nu)
            l.append(nd)
            ymax = 1.3*np.amax(l)
            ymin = np.amin([vd, vxcu, vxcd, nu, nd])
            if pot[idxlow] < ymin:
                ymin = pot[idxlow]
            ymin -= 1
            vlist = []
            plt.plot(r[0:idx], pot[0:idx], col[c], label='Vnuc')
            vlist.append(pot)
            leg.append('Vnuc')
            c += 1
            plt.plot(r[0:idx], vd[0:idx], col[c], label='Vd')
            vlist.append(vd)
            leg.append('Vd')
            c += 1
            plt.plot(r[0:idx], vxc[0:idx], col[c], label='Vxc up')
            vlist.append(vxcu)
            leg.append('Vxc up')
            c += 1
            plt.plot(r[0:idx], vxc[0:idx], col[c], label='Vxc dw')
            vlist.append(vxcu)
            leg.append('Vxc dw')
            c += 1
            plt.plot(r[0:idx], nu[0:idx], col[c], label='n_u')
            vlist.append(nu)
            leg.append('n_{u}')
            c += 1
            plt.plot(r[0:idx], nd[0:idx], col[c], label='n_d')
            vlist.append(nd)
            leg.append('n_{d}')
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
        print bcolors.WARNING + "(SCF it. %d, NR it. %d) Ground state calculation: E0 = %5f eV" % (iSCF, iN, E0*eV) + bcolors.ENDC
        if minF0Sum < 1e-4*float(len(listPhi)) and finishNow:
            break

    if np.fabs(1 - E0_old/E0) < 1e-3*Z and iSCF > 5:
        print bcolors.WARNING + "(SCF it. %d) Ground state energy changed by less than Z*1e-3 (by %5f). E0 = %5f eV." % (iSCF, np.fabs(1 - E0_old/E0), E0*eV) + '' + bcolors.ENDC
        break
    else:
        print bcolors.WARNING + "(SCF it. %d ends) E0 = %5f eV, dE0/E0 = %5f" % (iSCF, E0*eV, (E0 - E0_old)/E0) +'' + bcolors.ENDC
    E0_old = E0

