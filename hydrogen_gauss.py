
import numpy as np
import matplotlib.pyplot as plt

# to solve
# - hbar^2 / (2 m) deriv(deriv(xi)) + [ V + ( hbar^2 l (l+1) ) / (2 m r^2) - E ] xi (r) = 0
# xi = r*R(r), where R(r) is the radial solution
# - r^2/2m deriv(deriv(xi)) - [ V - l (l+1)/r^2 ] r^2 xi = E r^2 xi
# xi = sum_i c_i exp(-a_i r^2) = sum_i c_i f_i

# deriv(xi) = - sum_i c_i a_i r exp(-a_i r^2)

# deriv(deriv(xi)) = sum_i c_i a_i (a_i r - 1) exp(-a_i r^2)
#                  = sum_i c_i f_i a_i (a_i r - 1)

# int_0^inf exp(-a x^2) dx = 0.5 (pi/a)^0.5
# int_0^inf x exp(-a x^2) dx = 1/(2a)
# int_0^inf x^2 exp(-a x^2) dx = 1!! sqrt(pi)/(4 a^1.5) 
# int_0^inf x^3 exp(-a x^2) dx = 1/(2 a^2)
# int_0^inf x^4 exp(-a x^2) dx = 3!! sqrt(pi) / ( 8 a^2.5)
# int_0^inf x^5 exp(-a x^2) dx = 2 / ( 2 a^3 )
# int f_i f_j dr = int exp(-(a_i+a_j) r^2) dr = 0.5 (pi/(a_i+a_j))^0.5

# - 1/2m int r^2 xi deriv(deriv(xi)) dr = -1/2m sum_ij c_i c_j int a_i (a_i r^3 - r^2) exp (- (a_i + a_j) r^2) dr
#                                       = -1/2m sum_ij c_i c_j { a_i^2 / [ 2 (a_i + a_j) ] - a_i sqrt(pi) / [ 4 (a_i + a_j)^1.5 ] }
#
# + int xi^2 Z r dr = Z sum_ij c_i c_j int exp(-(a_i + a_j) r^2) r dr = Z sum_ij c_i c_j / [ 2 (a_i + a_j) ]
#
# + l (l+1) int xi^2 dr = l (l+1)/2 sum_ij c_i c_j sqrt[pi/(a_i+a_j)]
#
# int r^2 xi^2 dr = sum_ij c_i c_j sqrt(pi)/[ 4 (a_i + a_j)^1.5 ]
#
# sum_i { -1/2 { a_i^2 / [ 2 (a_i + a_j) ] - a_i sqrt(pi) / [ 4 (a_i + a_j)^1.5 ] } + Z / [ 2 (a_i + a_j) ] + l (l+1)/2 sqrt[pi/(a_i+a_j)] } c_i = 0
#
#

def init(N = 4, dx = 0.01):
    a = np.zeros(N)
    for i in range(0, N):
        a[i] = dx*i +dx
    return a


# Coulomb potential
def Coulomb(Z, a):
    V = np.zeros((len(a), len(a)))
    for i in range(0, len(a)):
        for j in range(0, len(a)):
	    V[i, j] = Z/(2*(a[i] + a[j]))
    return V

# l
def Vcent(l, a):
    V = np.zeros((len(a), len(a)))
    for i in range(0, len(a)):
        for j in range(0, len(a)):
	    V[i, j] = l*(l+1)/2.0*np.sqrt(np.pi/(a[i] + a[j]))
    return V

# S
def overlap(a):
    V = np.zeros((len(a), len(a)))
    for i in range(0, len(a)):
        for j in range(0, len(a)):
	    V[i, j] = np.sqrt(np.pi)/(4*(a[i] + a[j])**1.5)
    return V

# Kinetic energy
def kinetic(a):
    V = np.zeros((len(a), len(a)))
    for i in range(0, len(a)):
        for j in range(0, len(a)):
	    V[i, j] = -0.5*(a[i]**2/(2*(a[i] + a[j])) - a[i]*np.sqrt(np.pi)/(4*(a[i] + a[j])**1.5))
    return V

N = 8
dx = 0.02
a = init(N, dx)
M = kinetic(a) + Vcent(l = 0, a = a) + Coulomb(Z = 1, a = a)
S = overlap(a)
Sinv = np.linalg.inv(S)
w, v = np.linalg.eig(np.matmul(Sinv, M))

# initialise Grid
def initGrid(dx, N, xmin):
    r = np.zeros(N)
    for i in range(0, N):
        r[i] = np.exp(xmin + i*dx)
    return r

def intoGrid(r, a, v):
    xi = np.zeros(len(r))
    for i in range(0, len(v)):
        xi += v[i]*np.exp(-a[i]*r**2)
    n = 0
    for k in range(0, len(r)):
	ip = len(r)-2
	if i < len(r)-1:
	    ip = i + 1
	dr = np.fabs(r[ip]-r[i])
        n += r[k]*xi[k]**2*dr
    coeff = 1.0
    if xi[0] < 0:
        coeff = -1.0
    for k in range(0, len(r)):
        xi[k] /= coeff*np.sqrt(n)
    return xi

r = initGrid(1e-3, 14000, np.log(1e-4))
xi = []
st = ['r-', 'r:', 'r--', 'r-.', 'b-', 'b:', 'b--', 'b-.']
for j in range(0, N):
    xi.append(intoGrid(r, a, v[:, j]))
    print "Eigenvalue ", j , " energy ", w[j]
    
exact_p = 2*np.exp(-r) # exact R(r) solution for n = 1
idx = np.where(r > 10)
idx = idx[0][0]
plt.clf()
for j in range(0, N):
    plt.plot(r[0:idx], xi[j][0:idx], st[j], label=str(j))
plt.plot(r[0:idx], exact_p[0:idx], 'g--', label='$R_{exact}$')
plt.xlabel('$r$')
plt.ylabel('$|R(r)|$')
plt.title('')
plt.draw()
plt.show()

