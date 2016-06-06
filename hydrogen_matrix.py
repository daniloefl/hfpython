
import numpy as np
import matplotlib.pyplot as plt

eV = 27.2113966413442 # Hartrees

# a[i] = 2*m*r[i]**2*(E - pot[i]) - (l+0.5)**2
# f[i] = 1 + a[i]*dx**2/12.0
# y[i+1]*f[i+1] - (12 - f[i]*10)*y[i] + f[i-1]*y[i-1] = 0 (1)
# y[i+1]*f[i+1] - (12 - f[i]*10)*y[i] + f[i-1]*y[i-1] = -f_e[i+1]*y[i+1] + ((12 - f_e[i]*10)*y[i] - f_e[i-1]*y[i-1] (1)
# f_e = 1 + (2*m*r[i]**2*E)*dx**2/12
# -Z/r (2)
# boundary conditions at r->infinity
# for j = N, any i: y[j+1] = 0
# ==> F_iN = 0
# boundary conditions at r = 0:
# y[0] = ((Z*r[0])**(l+0.5)) -> F_00 = 
def fock(r, N, E, l, Z = 1, dx = 1e-2):
  F = np.zeros((N*len(r), N*len(r)))
  icl = np.zeros(N, dtype=np.int)
  m = 1
  for yi in range(0, N):
    ai_prev = 1
    for i in range(0, len(r)):
      ai = (2*m*r[i]**2*(E[yi] - (-Z/r[i])) - (l[yi]+0.5)**2)
      fi = 1 + dx**2/12.0*ai
      if i > 1 and icl[yi] == 0 and ai*ai_prev < 0:
          icl[yi] = i
      ai_prev = ai
      for yj in range(0, N):
        for j in range(0, len(r)):
          if yi == yj: # same wave function (include 1)
	    if i == j:
	      F[yi*len(r) + i, yj*len(r) + j] += -(12 - fi*10)
	    elif i == j-1 or i == j+1: # to multiply previous or next point
	      F[yi*len(r) + i, yj*len(r) + j] += fi
	  else: # exchange terms come here
	    F[yi*len(r) + i, yj*len(r) + j] += 0
  return [F, icl]

def indepOutward(r, N, E, l, Z, dx):
  F = np.zeros((N*len(r), 1))
  m = 1
  for yi in range(0, N):
    for i in range(0, len(r)):
      if i == 0:
        fi = 1 + dx**2/12.0*(2*m*r[i]**2*(E[yi] - (-Z/r[i])) - (l[yi]+0.5)**2)
        F[yi*len(r) + i] += -fi*((Z*r[0])**(l[yi]+0.5))
  return F

def indepInward(r, N, E, l, Z, dx):
  F = np.zeros((N*len(r), 1))
  m = 1
  for yi in range(0, N):
    for i in range(0, len(r)):
      if i == len(r)-1:
        fi = 1 + dx**2/12.0*(2*m*r[i]**2*(E[yi] - (-Z/r[i])) - (l[yi]+0.5)**2)
        F[yi*len(r) + i] += -fi*np.exp(-np.sqrt(-2*m*E[yi])*r[len(r)-1])
  return F

def init(dx, N, xmin):
  r = np.zeros(N)
  for i in range(0, N):
    r[i] = np.exp(xmin + i*dx)
  return r


def F(i, E, r, l, Z, dx, y):
  m = 1
  ai = (2*m*r[i]**2*(E - (-Z/r[i])) - (l+0.5)**2)
  fi = 1 + dx**2/12.0*ai
  aip1 = (2*m*r[i+1]**2*(E - (-Z/r[i+1])) - (l+0.5)**2)
  fip1 = 1 + dx**2/12.0*aip1
  aim1 = (2*m*r[i-1]**2*(E - (-Z/r[i-1])) - (l+0.5)**2)
  fim1 = 1 + dx**2/12.0*aim1
  return (12 - 10*fi)*y[i] - fim1*y[i-1] - fip1*y[i+1]

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

def matchInOut(yp, y, icl):
    # renormalise
    y_ren = np.zeros(len(y))
    if icl >= 0:
        rat = y[icl]/yp[icl]
	for i in range(0, icl):
	    y_ren[i] = y[i]
	for i in range(icl, len(y)):
	    y_ren[i] = rat*yp[i]
    return y_ren

def solve(r, N, E, l, Z, dx):
    [Fo, icl] = fock(r, N, E, l, Z, dx)
    B_out = indepOutward(r, N, E, l, Z, dx)
    Finv = np.linalg.inv(Fo)
    xi_out = np.matmul(Finv, B_out)

    B_in = indepInward(r, N, E, l, Z, dx)
    xi_in = np.matmul(Finv, B_in)

    nodes = np.zeros(N)
    xi_in2 = []
    xi_out2 = []
    xi_ren = []
    for j in range(0, N):
      xi_in2.append(xi_in[j*len(r):j*len(r)+len(r)])
      xi_out2.append(xi_out[j*len(r):j*len(r)+len(r)])

      xi_ren.append(matchInOut(xi_out2[j], xi_in2[j], icl[j]))
      for z in range(1, len(r)):
        if xi_ren[j][z]*xi_ren[j][z-1] < 0:
	  nodes[j] += 1

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
    Ficl = np.zeros(N)
    dE = np.ones(N)*(-1e-1)
    bestdE = np.zeros(N)

    [Fo_shift, icl_shift] = fock(r, N, E+dE, l, Z, dx)
    B_out_shift = indepOutward(r, N, E+dE, l, Z, dx)
    Finv_shift = np.linalg.inv(Fo_shift)
    xi_out_shift = np.matmul(Finv_shift, B_out_shift)

    B_in_shift = indepInward(r, N, E+dE, l, Z, dx)
    xi_in_shift = np.matmul(Finv_shift, B_in_shift)

    xi_in2_shift = []
    xi_out2_shift = []
    xi_ren_shift = []
    for j in range(0, N):
      xi_in2_shift.append(xi_in_shift[j*len(r):j*len(r)+len(r)])
      xi_out2_shift.append(xi_out_shift[j*len(r):j*len(r)+len(r)])

      xi_ren_shift.append(matchInOut(xi_out2_shift[j], xi_in2_shift[j], icl[j]))

      Ficl = F(icl[j], E[j], r, l[j], Z, dx, xi_ren[j]) # get F at icl
      # calculate dF/dE, by varying E very slightly
      # recalculate the solution with a slihtly varied E
      # new solution has a discontinuity at icl again
      # dF/dE is defined as the difference over dE of the change in F
      Ficl_shift = F(icl[j], E[j]+dE[j], r, l[0], Z, dx, xi_ren_shift[j]) # get F at icl
      print Ficl_shift - Ficl
      if Ficl_shift != Ficl:
          bestdE[j] = -Ficl*dE[j]/(Ficl_shift - Ficl)
      else:
        bestdE[j] = dE[j]
      if icl < 0:
        bestdE[j] = 10 # arbitrary, but must be positive to make energy less negative
    return [xi_ren, nodes, bestdE]

Z = 1
N = 1 # number of orbitals

dx = 1e-2
r = init(dx, 1300, np.log(1e-4))
E = np.ones(N)*(-0.5)
l = np.zeros(N)
n = np.ones(N)

Emin = np.ones(N)*(-20)
Emax = np.zeros(N)

for i in range(0,100):
  [xi, nodes_total, bestdE] = solve(r, N, E, l, Z, dx)
  print bestdE, nodes_total
  dE = np.zeros(N)
  for j in range(0, N): #for each orbital
    if nodes_total[j] > nodes(n[j], l[j]):
      Emax[j] = E[j]-1e-15
      dE[j] = (Emax[j] + Emin[j])*0.5 - E[j]
    elif nodes_total[j] < nodes(n[j], l[j]):
      Emin[j] = E[j]+1e-15
      dE[j] = (Emax[j] + Emin[j])*0.5 - E[j]
    else:
      dE[j] = 1e-1*bestdE[j]
      if np.fabs(dE[j]) > 0.5:
        dE[j] = 0.5*dE[j]/np.fabs(dE[j])

  print E, dE
  exact_p = 2*np.exp(-r) # exact R(r) solution for n = 1
  idx = np.where(r > 3)
  idx = idx[0][0]
  idxm = 0
  plt.clf()
  col = ['r-', 'r-.', 'r:', 'g--', 'g-.', 'g:']
  for j in range(0, N):
      plt.plot(r[idxm:idx], toPsi(r, xi[j])[idxm:idx], col[j], label='xi%d'%j)
  plt.plot(r[idxm:idx], exact_p[idxm:idx], 'g--', label='$R_{exact}$')
  plt.xlabel('$r$')
  plt.ylabel('$|R(r)|$')
  plt.title('')
  plt.draw()
  plt.show()

  converged = True
  for j in range(0, N):
    E[j] += dE[j]
    if dE[j] > 0 and E[j] > Emax[j]:
      E[j] = Emax[j]
    elif dE[j] < 0 and E[j] < Emin[j]:
      E[j] = Emin[j]

    if np.fabs(dE[j]) > 1e-8 and np.fabs(Emax[j] - Emin[j]) > 1e-5:
      converged = False
  if converged:
    print "Converged", E
    break
print E

