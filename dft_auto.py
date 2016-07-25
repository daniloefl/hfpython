
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
    y[0] = ((Z*r[0])**(l+0.5))
    y[1] = ((Z*r[1])**(l+0.5))
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
    #print "F, Fp, dF ", Ficl, Fp, Fp - Ficl
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

def getPotentialH(r, nu, nd):
    # 0) calculate rho(r) = n(r)
    # 1) calculate Q(r) = 4*pi*sum_r'=0^r rho(r)*r^2*dr
    # 2) calculate E(r) = Q(r)/(4*pi*r^2)
    # 3) calculate Vd(r) = sum_r'=inf^r E(r)*dr
    E = np.zeros(len(r))
    Q = 0
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        Q += 4*np.pi*(nu[z]+nd[z])*r[z]**2*dr
        # this is E:
        E[z] = Q/(4*np.pi*r[z]**2)
    Vd = np.zeros(len(r))

    Vd[len(r)-1] = 0
    # now integrate backwards
    # Vd(r) = int_inf^r E(r') dr'
    # Vd(r-h) = int_inf^r E(r') dr' + int_r^r-h E(r') dr'
    # Vd(r-h) = Vd(r) + E(r)*dr
    for z in reversed(range(0, len(r)-1)):
        Vd[z] = Vd[z+1] + E[z]*(r[z+1] - r[z])
    return Vd

def getPotentialXC(r, nu, nd):
    # E_xc = -3/4 (3/pi)^(1/3) 4*pi int n(r)^(4/3) r^2 dr
    # e_xc = -3/4 (3/pi)^(1/3) n(r)^(4/3)
    # v_xc = e_xc + n(r) de_xc/dn
    #      = e_xc + n(r) (-3/4 (3/pi)^(1/3) (4/3) n(r)^(1/3) )
    
    exc = np.zeros(len(r))
    excu = np.zeros(len(r))
    excd = np.zeros(len(r))
    for z in range(0, len(r)):
        exc[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(nu[z]+nd[z])**(4.0/3.0)
        excu[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(nu[z])**(4.0/3.0)
        excd[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(nd[z])**(4.0/3.0)

    dedn = np.zeros(len(r))
    dednu = np.zeros(len(r))
    dednd = np.zeros(len(r))
    for z in range(0, len(r)):
        dedn[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(4.0/3.0)*(nu[z] + nd[z])**(1.0/3.0)
        dednu[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(4.0/3.0)*(nu[z])**(1.0/3.0)
        dednd[z] = -3.0/4.0*(3.0/np.pi)**(1.0/3.0)*(4.0/3.0)*(nd[z])**(1.0/3.0)

    vxc = np.zeros(len(r))
    vxcu = np.zeros(len(r))
    vxcd = np.zeros(len(r))
    for z in range(0, len(r)):
        vxc[z] = dedn[z]
        vxcu[z] = dednu[z]
        vxcd[z] = dednd[z]
    return [vxcu, vxcd, exc]

def calculateE0(listPhi, r, nu, nd):
    E0 = 0
    for iOrb in listPhi.keys():
        E0 += listPhi[iOrb].E

    # add - 1/2 int int dr dr' n(r) n(r')/(r-r') - int dr n(r) V_xc(r) + E_xc
    vh = getPotentialH(r, nu, nd)

    Qh = 0
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        Qh += 4*np.pi*(vh[z]*(nu[z] + nd[z]))*r[z]**2*dr

    [vxcu, vxcd, exc] = getPotentialXC(r, nu, nd)
    Qxc = 0
    Exc = 0
    for z in range(0, len(r)):
        dr = 0
        if z >= 1:
            dr = r[z] - r[z-1]
        else:
            dr = r[z]
        Qxc += 4*np.pi*(vxcu[z]*nu[z] + vxcd[z]*nd[z])*r[z]**2*dr
        Exc += 4*np.pi*(exc[z])*r[z]**2*dr

    E0 += -0.5*Qh - Qxc + Exc
    return E0

class phi:
    n = 1
    l = 0
    E = -2.0
    _no = 0
    _nop = 0
    psi = None
    def __init__(self, _n, _l, _E, _r):
        self.n = _n
        self.l = _l
        self.E = _E
        self.r = _r

Z = 2
Einit = -2.0

dx = 1e-3
r = init(dx, 13000, np.log(1e-4))

listPhi = {}
listPhi['1s1+'] = phi(1, 0, Einit, r)
listPhi['1s1-'] = phi(1, 0, Einit, r)

pot = V(r, Z)

nu = np.zeros(len(r))
nd = np.zeros(len(r))

E0_old = 0
E0 = 0
Nscf = 100
for iSCF in range(0, Nscf):
    for iOrb in listPhi.keys():
        #pot_full = pot + 0.5*getPotentialH(r, n)
        vh = getPotentialH(r, nu, nd)
        [vxcu, vxcd, exc] = getPotentialXC(r, nu, nd)
        print "sum of Vh, Vxce, Vxcd = ", np.sum(vh), np.sum(vxcu), np.sum(vxcd)
        if '+' in iOrb:
            pot_full = pot + vh + vxcu
        else:
            pot_full = pot + vh + vxcd
        Emax = -1e-3
        Emin = -20.0
        for i in range(0,200):
            [y, yp, icl, no, nop, bestdE] = solve(r, dx, pot_full, listPhi[iOrb].n, listPhi[iOrb].l, listPhi[iOrb].E, Z)
            listPhi[iOrb].no = no
            listPhi[iOrb].nop = nop
            dE = 0
            if listPhi[iOrb].no > nodes(listPhi[iOrb].n, listPhi[iOrb].l):
	        Emax = listPhi[iOrb].E-1e-15
                dE = (Emax + Emin)*0.5 - listPhi[iOrb].E
            elif listPhi[iOrb].no < nodes(listPhi[iOrb].n, listPhi[iOrb].l):
                Emin = listPhi[iOrb].E+1e-15
	        dE = (Emax + Emin)*0.5 - listPhi[iOrb].E
            else:
                dE = 1e-1*bestdE
	        if np.fabs(dE) > 0.5:
	            dE = 0.5*dE/np.fabs(dE)

            print "(", iOrb ,") Iteration ", i, ", E = ", listPhi[iOrb].E, ", dE = ", dE, ", nodes = ", no, nop, ", expected nodes = ", nodes(listPhi[iOrb].n, listPhi[iOrb].l), ", crossing zero at = ", icl
            psi = toPsi(r, y)
            listPhi[iOrb].psi = psi
            psip = toPsi(r, yp)
            listPhi[iOrb].E += dE
            if dE > 0 and listPhi[iOrb].E > Emax:
                listPhi[iOrb].E = Emax
            elif dE < 0 and listPhi[iOrb].E < Emin:
                listPhi[iOrb].E = Emin
            prev_dE = dE
            if np.fabs(dE) < 1e-8 or np.fabs(Emax - Emin) < 1e-5:
                print "Converged to energy ", listPhi[iOrb].E*eV, " eV"
                break
    for idx in range(0, len(nu)):
        nu[idx] = 0
        nd[idx] = 0
    for iOrb in listPhi.keys():
        if '+' in iOrb:
            nu += listPhi[iOrb].psi**2
        else:
            nd += listPhi[iOrb].psi**2
    idx = np.where(r > 5)
    idx = idx[0][0]
    plt.clf()
    leg = []
    exact_p = 2*np.exp(-r)   # solution for R(r) in Hydrogen, n = 1
    col = ['r-', 'g-', 'b-', 'r-.', 'g-.', 'b-.']
    c = 0
    for iOrb in listPhi.keys():
        plt.plot(r[0:idx], listPhi[iOrb].psi[0:idx], col[c], label='$R_{%s}$'%iOrb)
        c += 1
        leg.append('%s (%5f eV)' % (iOrb, listPhi[iOrb].E*eV))
    plt.plot(r[0:idx], exact_p[0:idx], 'g--', label='$R_{exact}$')
    leg.append('Exact H')

    plt.legend(leg, frameon=False)
    plt.xlabel('$r$')
    plt.ylabel('$|R(r)|$')
    plt.title('Z=%d, SCF iter=%d'%(Z, iSCF))
    plt.draw()
    plt.savefig('pseudo_potentials.pdf', bbox_inches='tight')
    #plt.show()


    E0 = 0
    for iOrb in listPhi.keys():
        E0 += listPhi[iOrb].E
    if np.fabs(1 - E0_old/E0) < 1e-2:
        print "Ground state energy changed by less than 1% (precisely by ", 100.0*np.fabs(1 - E0_old/E0),"%). E0 = ", E0*eV, "eV"
        break
    E0_old = E0

