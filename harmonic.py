
import numpy as np
import matplotlib.pyplot as plt

eV = 27.2113966413442 # Hartrees

# harmonic oscillator
def V(x, k = 1.0):
    pot = np.zeros(len(x))
    for i in range(0, len(x)):
        pot[i] = k*x[i]**2/2.0
    return pot

# linear grid
def init(xmax, N):
    xmin = 0 # potential is symmetric, so assume everything is
    x = np.zeros(N)
    dx = (xmax - xmin)/N
    for i in range(0, N):
        x[i] = xmin + i*dx
    return x

def F(i, f, y):
    return (12 - 10*f[i])*y[i] - f[i-1]*y[i-1] - f[i+1]*y[i+1]

# to solve:
# -hbar/2 m d^2 y/ dx^2 + V y = E y
# d^2 y / dx^2 + 2 m (E - V) y = 0
# a = 2 m (E - V)
def getAF(x, pot, E, m, dx):
    icl = -1
    a = np.zeros(len(x))
    f = np.zeros(len(x))
    for i in range(0, len(x)):
        a[i] = 2*m*(E - pot[i])
	f[i] = 1 + a[i]*dx**2/12.0
	if icl < 0 and i >= 1 and a[i]*a[i-1] < 0:
	    icl = i
    return [a, f, icl]

# boundary conditions:
# x -> 0 => y = exp(+/-sqrt(-2 m E) x)
# energy is positive (V > 0), so solution is oscillatory
# y = cos(2 m E x) or y = sin(2 m E x)
def solution(x, m, f, E, n):
    y = np.zeros(len(x))
    m = 1
    no = 0
    if n % 2 == 0:
        y[0] = 1
        y[1] = ((12 - f[0]*10)*1)/(2*f[1])
    else:
        y[0] = 0
        y[1] = x[1] - x[0]
    for i in range(1, len(x)-1):
	y[i+1] = ((12 - f[i]*10)*y[i] - f[i-1]*y[i-1])/f[i+1]
    for i in range(1, len(x)-1):
	if y[i]*y[i+1] < 0:
	    no += 2
    if n % 2 == 1:
      no += 1
    return [y, no]

def inward(x, m, f, E, n):
    y = np.zeros(len(x))
    m = 1
    no = 0
    y[len(x)-1] = 0
    y[len(x)-2] = x[1] - x[0]
    for i in reversed(range(1, len(x)-1)):
	y[i-1] = ((12 - f[i]*10)*y[i] - f[i+1]*y[i+1])/f[i-1]
    for i in reversed(range(1, len(x)-1)):
	if y[i]*y[i-1] < 0:
	    no += 2
    if n % 2 == 1:
      no += 1
    return [y, no]

# return a function that is identically y up until index icl
# and that is yp afterwards, rescaling the yp part so that it matches
# y in icl (ie: so that it is continuous)
def matchInOut(y, yp, icl):
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

# f = 1 + a * dx^2/12
# y_{n+1} = ((12 - 10 f_n) y_n - f_{n-1} y_{n-1}) / f_{n+1}
def solve(x, pot, n, E):
    m = 1.0
    icl = -1
    dx = x[1] - x[0]
    [a, f, icl] = getAF(x, pot, E, m, dx)
    
    [y, no] = solution(x, m, f, E, n)
    [yp, nop] = inward(x, m, f, E, n)
    yren = matchInOut(y, yp, icl)

    return [yren, f, icl]


def toPsi(x, y, even):
    n = 1
    dx = np.fabs(x[1]-x[0])
    psi = np.zeros(2*len(y))
    xr = np.zeros(2*len(y))
    if even:
        isEven = 1
    else:
        isEven = -1
    for i in range(0, len(y)):
        psi[i] = isEven*y[len(y)-i-1]
	xr[i] = -x[len(y)-i-1]
        n += psi[i]**2*dx        # normalise it so that int |Psi(x)|^2 dx == 1
    for i in range(len(y), 2*len(y)):
        psi[i] = y[i-len(y)]
        xr[i] = x[i-len(y)]
        n += psi[i]**2*dx        # normalise it so that int |Psi(x)|^2 dx == 1
    if n != 0:
        for i in range(0, len(psi)):
            psi[i] /= np.sqrt(n)
    return [xr, psi]

def reflect(x, y):
    n = 1
    psi = np.zeros(2*len(y))
    xr = np.zeros(2*len(y))
    for i in range(0, len(y)):
        psi[i] = y[len(y)-i-1]
	xr[i] = -x[len(y)-i-1]
    for i in range(len(y), 2*len(y)):
        psi[i] = y[i-len(y)]
        xr[i] = x[i-len(y)]
    return [xr, psi]

n = 0
Emax = 0
Emin = 0

E = 20
x = init(10.0, 400)
pot = V(x, k = 1.0)
for i in range(0, len(pot)):
  if pot[i] < Emin:
    Emin = pot[i]
  if pot[i] > Emax:
    Emax = pot[i]
nodes = 0
if n % 2 == 0:
    even = True
else:
    even = False
for i in range(0,10):
    dE = 0.01*E
    [y, f, icl] = solve(x, pot, n, E)
    [y_dE, f, icl_dE] = solve(x, pot, n, E+dE)
    # F(new) = 0 = F(old) + F'(old)*dE
    # dE = -F(old)/F'(old)
    F_old = F(icl, f, y)
    F_new = F(icl, f, y_dE)
    dE = - F_old*dE/(F_new - F_old)

    [xr_full, psi_full] = toPsi(x, y, even)
    nodes = 0
    nodesList = []
    for it in range(0, len(xr_full)-1):
        if psi_full[it]*psi_full[it+1] < 0:
	    nodes += 1
	    nodesList.append(i)
    if not even:
        nodes += 1
    print "Iteration ", i, ", E = ", E, ", nodes = ", nodes, ", expected nodes = ", n, ", crossing zero at = ", icl, nodesList


    idx = np.where(x > 4)
    idx=  idx[0][0]
    [xr, psi] = toPsi(x[0:idx], y[0:idx], even)
    psi2 = psi*psi
    [xV_full, V_full] = reflect(x[0:idx], pot[0:idx])
    if True:
      #plt.clf()
      fig, ax1 = plt.subplots()
      ax1.plot(xr, psi, 'r-', linewidth=2, label='$\Psi(x)$')
      ax1.plot(xr, psi2, 'r--', linewidth=2, label='$\Psi^2(x)$')
      ax1.set_xlabel('$x$')
      ax1.set_ylabel('$\Psi(x)$ or $\Psi^2(x)$', color='r')
      for tl in ax1.get_yticklabels():
          tl.set_color('r')
      ax2 = ax1.twinx()
      ax2.plot(xr, V_full, 'b--', linewidth=2, label='$V(x)$')
      ax2.plot(xr, E*np.ones(len(V_full)), 'b:', linewidth=2, label='$E$')
      ax2.set_xlabel('$x$')
      ax2.set_ylabel('Energy', color='b')
      for tl in ax2.get_yticklabels():
          tl.set_color('b')
      ax2.legend(('$V(x)$', '$E$'), frameon = False, loc = 'upper right')
      ax1.legend(('Wave function', 'Probability'), frameon = False, loc = 'upper left')
      plt.title('')
      #plt.draw()
      plt.show()
    if nodes != n:
        if nodes > n:
            Emax = E
        elif nodes < n:
            Emin = E
        E = (Emax + Emin)*0.5
    else:
        if dE > 0:
	    Emin = E
        elif dE < 0:
	    Emax = E
        E += dE
    if np.fabs(Emax - Emin) < 0.01:
        break
print "Last energy ", E*eV, " eV"
    


