#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

# initialise grid
def init(N, xmin, dx):
    x = np.zeros(N, dtype = np.longdouble)
    for i in range(0, N):
        x[i] = xmin + i*dx
    return x

# solve:
# d^y/dx^2 + g(x) y = h(x)
def numerov(x, f, indepTerm, y0 = 0, y1 = 1):
    y = np.zeros(len(x), dtype = np.longdouble)
    y[0] = y0
    y[1] = y1
    for i in range(1, len(x)-1):
        y[i+1] = ((12 - f[i]*10)*y[i] - f[i-1]*y[i-1] + (indepTerm[i+1] + 10.0*indepTerm[i] + indepTerm[i-1]))/f[i+1]
    # for the example below, f = 1, indepTerm = 2*dx^2/12
    # y[i+1] = 2y[i] - y[i-1] + (2+20+2)*dx^2/12
    # y[i+1] = 2y[i] - y[i-1] + 2*dx^2
    # y[2] = 2*dx^2 - 0 + 2*dx^2 = 4*dx^2
    # y[3] = 2*(4*dx^2) - dx^2 + 2*dx^2 = 9*dx^2
    # y[4] = 2*(9*dx^2) - 4*dx^2 + 2*dx^2 = 16*dx^2
    # ie: y[n] = n^2*dx^2
    return y

def getAF(x, g, ind, dx):
    f = np.zeros(len(x), dtype = np.longdouble)
    s = np.zeros(len(x), dtype = np.longdouble)
    for i in range(0, len(x)):
        f[i] = 1 + g[i]*dx**2/12.0
        s[i] = (dx**2)/12.0*(ind[i])
    return [f, s]

dx = 1e-3
N = 2000
xmin = 0
x = init(N, xmin, dx)

# load linear term
g = np.zeros(len(x), dtype = np.longdouble)
# keeping it at zero for tests

# now load independent term
ind = np.zeros(len(x), dtype = np.longdouble)
for i in range(0, len(x)):
    ind[i] = 2

# solve d^2y/dx^2 = 2
# solution should be: y = A x^2 + B x + C
# A, B and C set by initial conditions
# set initial conditions:
y0 = 0
y1 = dx**2
# with those, A = 1, B = C = 0

# load temporary variables:
[f, s] = getAF(x, g, ind, dx)

# calculate solution
y = numerov(x, f, s, y0, y1)

fig, ax1 = plt.subplots()
ax1.plot(x, y, 'r-', linewidth=2)
ax1.plot(x, g, 'r-.', linewidth=2)
ax1.plot(x, ind, 'r--', linewidth=2)
ax1.plot(x, x**2, 'b:', linewidth=2)
ax1.legend(('$y(x)$', '$g(x)$', '$h(x)$', '$x^2$'), frameon=False, loc = 'lower right')
ax1.set_xlabel('$x$')
ax1.set_ylabel('Function')
for tl in ax1.get_yticklabels():
    tl.set_color('r')
plt.title("$y''(x) + g(x)y(x) = h(x)$, with $g(x) = 0$, $h(x) = 2$")
plt.show()

plt.close()

