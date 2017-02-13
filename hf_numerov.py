#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import scipy.special
import scipy.sparse
import scipy.sparse.linalg

from utils import *

Z = 3
rmin = 1e-4
dy = 1e-2/Z
N = 1400*Z

def initR(rmin, dy, N):
  r = np.zeros(N)
  for i in range(0, N):
    r[i] = rmin*np.exp(dy*i)
  return r

r = initR(rmin, dy, N)

def y(r, rmin):
  return np.log(r/rmin)

class Orbital:
  def __init__(self, idx = 1, n = 1, spin = +1, N = 1500):
    self.idx = idx
    self.spin = spin
    self.n = n
    self.N = N
    self.l = []
    self.m = []
    self.E = -Z**2*0.5/n**2 #-5.0
    self.a = []
  def addY(self, l, m):
    self.l.append(l)
    self.m.append(m)
    self.a.append(np.zeros(self.N, dtype = np.float64))
  def name(self):
    l = self.l[0]
    ls = ""
    if l == 0:
      ls = "s"
    elif l == 1:
      ls = "p"
    elif l == 2:
      ls = "d"
    elif l == 3:
      ls = "f"
    return "%d%s" % (self.n, ls)

  def __str__(self):
    s = "Orbital index %d, spin %d, eigenvalue %.10f." % (self.idx, self.spin, self.E)
    s += " l = "
    s += str(self.l)
    s += " m = "
    s += str(self.m)
    return s
  def valueR(self, r, yidx = -1):
    v = np.zeros(len(r), dtype = np.float64)
    for i in range(0, len(self.a)): # each Y() term
      if yidx >= 0 and yidx == i:
        for idx in range(0, len(r)):
          v[idx] += self.a[i][idx]/np.sqrt(r[idx]/r[0])
      elif yidx < 0:
        for idx in range(0, len(r)):
          v[idx] += self.a[i][idx]/np.sqrt(r[idx]/r[0])
    return v

listOrb = []
listOrb.append(Orbital(0, n=1, spin = +1, N = N))
listOrb[-1].addY(0,0)
listOrb.append(Orbital(1, n=1, spin = -1, N = N))
listOrb[-1].addY(0,0)
listOrb.append(Orbital(2, n=2, spin = +1, N = N))
listOrb[-1].addY(0,0)

old_listOrb = listOrb[:]

No = len(listOrb)

'''
The objective is to solve, for each orbital u_l(q):
[ -1/2 del^2 - Z/r + \sum_\mu V^d_\mu(q) - \sum_\mu V^ex_\mu(q) ] u_l(q) = E_l u_l(q)

where:

V^d_\mu(q) u_\lambda(q) = [int u*_\mu(s) u_\mu(s)/ |s - q| ds ] u_\lambda(q)
V^ex_\mu(q) u_\lambda(q) = \delta_{spin \lambda,spin \mu} [int u*_\mu(s) u_\lambda(s) / |s - q| ds ] u_\mu(q)

Define: r = r_min*exp(idx*dy) for idx = 0 .. N-1
Algebraically: r = r_min*exp(y) -> y = ln(r/r_min)

df(y)/dr = df(y)/dy * dy/dr = df(y)/dy * 1/r

u_\lambda(q) = xi_{spin \lambda} U(q) Y_{l_{\lambda, k}, m_{\lambda, k}}

del^2 u_\lambda(q) = xi_{spin \lambda} del^2 [ U(q) Y_{l_{\lambda, k}, m_{\lambda, k}} ]

del^2 [U(r)] = 1/r^2 d/dr [ r^2 d/dr [ U(r) ]]
             = 1/r^2 d/dr [ r^2 1/r dU/dy ] = 1/r^2 d/dr [ r dU/dy ]
             = 1/r^2 dU/dy + 1/r^2 d^2U/dy^2

del^2 [Y_{l_{\lambda, k}, m_{\lambda, k}} ] = 1/(r^2 sin^2 phi) d^2 Y/d theta^2 + 1/(r^2 sin phi) d/dphi [ sin phi dY/dphi ]
                                            = - l_{\lambda, k} (l_{\lambda, k} + 1)/r^2 Y_{l_{\lambda, k}, m_{\lambda, k}}

-1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/r^2 d^2U/dy^2 -1/2 1/r^2 dU/dy + 1/2 l_{\lambda, k} (l_{\lambda, k} + 1)/r^2 U
         }

Would like to have eq. in form d^2S/dy^2 + g(y)S = 0, so define S such that:
dS^2/dy = d^2U/dy^2 + dU/dy + f(y) U for some f(y)
S =  b(y) U
dS/dy = db/dy U + b dU/dy
d^2S/dy^2 = d^2b/dy^2 U + 2 db/dy dU/dy  + b d^2U/dy^2
We want: b = 2 db/dy for all y, so b = e^(1/2 y) = sqrt(r/rmin) is a solution to the latter.

exp(-1/2 y) d^2S/dy^2 = d^2U/dy^2 + dU/dy + 1/4 U
exp(-1/2 y) d^2S/dy^2 -1/4 U = d^2U/dy^2 + dU/dy


-1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/r^2 [ exp(-1/2 y) d^2S/dy  -1/4 U ] + 1/2 l_{\lambda, k} (l_{\lambda, k} + 1)/r^2 U
         }

-1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/r^2 exp(-1/2 y) d^2S/dy + 1/8 1/r^2 U + 1/2 l_{\lambda, k} (l_{\lambda, k} + 1)/r^2 U
         }

-1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/r^2 exp(-1/2 y) d^2S/dy + 1/2 [ l_{\lambda, k}^2 + l_{\lambda, k} + 1/4 ] /r^2 U
         }

-1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/r^2 exp(-1/2 y) d^2S/dy + 1/2 [ l_{\lambda, k} + 1/2 ]^2 /r^2 exp(-1/2 y) S
         }

-1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/r^2 exp(-1/2 y) { d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S }
         }

  * Nucleus potential:
V = -Z/q_r u_\lambda(q) = -Z/rmin xi_{spin \lambda} [ exp(-3/2 y) S ] Y_{l_{\lambda, k}, m_{\lambda, k}}
V = -Z/q_r u_\lambda(q) = -Z/rmin exp(-3/2 y) xi_{spin \lambda} S(y) Y_{l_{\lambda, k}, m_{\lambda, k}}

-> Final eq:
--> For each orbital and each (l,m):
    -1/2 1/r^2 exp(-1/2 y) { d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S } + (-Z/rmin exp(-3/2 y) S(y)) - E exp(-1/2 y) S = 0
    -1/2 1/rmin^2 exp(-4/2 y) exp(-1/2 y) { d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S } + (-Z/rmin exp(-3/2 y) S(y)) - E exp(-1/2 y) S = 0
    -1/2 1/rmin^2 exp(-5/2 y) { d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S } + (-Z/rmin exp(-3/2 y) S(y)) - E exp(-1/2 y) S = 0
    exp(-5/2 y) { d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S } + (-2 rmin^2) (-Z/rmin exp(-3/2 y) S(y)) - E (-2 rmin^2) exp(-1/2 y) S = 0
    d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S + (-2 rmin^2) exp(5/2 y) (-Z/rmin exp(-3/2 y) S(y)) - E (-2 rmin^2) exp(5/2 y) exp(-1/2 y) S = 0
    d^2S/dy - [ l_{\lambda, k} + 1/2 ]^2 S + (2 Z rmin) exp(y) S(y) + 2 E rmin^2 exp(2 y) S = 0
    d^2S/dy  + (2 E rmin^2 exp(2 y) + 2 Z rmin exp(y) - [ l_{\lambda, k} + 1/2 ]^2) S = 0

If there was another term V2 U2(y) in the equation, this term would become:
U2 = exp(-1/2 y) S2
V2 U2 = V2 exp (-1/2 y) S2

And it is multiplied by (-2 rmin^2 exp(5/2 y) ) as above, so the original equation is:
(T + V) U  + V2 U2 = E U

And it becomes:

d^2S/dy  - ( - 2 E rmin^2 exp(2 y) - 2 Z rmin exp(y) + [ l_{\lambda, k} + 1/2 ]^2) S - (+ 2 V2 rmin^2 exp(2y) ) S2 = 0

-> In Hartree-Fock, we would have:
d^2a/dy - c11(y) a - c12(y) b + ... = 0
d^2b/dy - c21(y) a - c22(y) b + ... = 0

Vectorially, if A = [a ; b ; ...] and C = [c11, c12, ... ; c21, c22, ...; ... ]:
d^2A/dy^2 - C(y) A = 0

Using Numerov's method:

f_{ij}(k) = \delta_{ij} - dy^2/12 c_{ij}(k))
g_{i}(k) = \sum_j f_{ij}(k) a_{j}(k)
g_{i}(k+1) = 12 a_{i}(k) - 10 g_{i}(k) - g_{i}(k-1)
a_{i}(k+1) = (F(k+1)^(-1) G(k+1))_{i}

...

We define

Summary:

Define: r = r_min*exp(idx*dy) for idx = 0 .. N-1
y = ln(r/r_min)
S = exp(1/2 y) U(y) = sqrt(r/rmin) U(y)
1/r^2 = 1/rmin^2 exp(-2 y)

  * Orbital:
u_\lambda(q) = xi_{spin \lambda} [ U(q) ] Y_{l_{\lambda, k}, m_{\lambda, k}}

  * Kinetic energy:
T = -1/2 del^2 u_\lambda(q) = xi_{spin \lambda} Y_{l_{\lambda, k}, m_{\lambda, k}} {
                -1/2 1/rmin^2 exp(-5/2 y) { d^2S/dy - [ l_{\lambda, k}^2 + 1/2 ]^2 S }
         }

  * Nucleus potential:
V = -Z/q_r u_\lambda(q) = -Z/rmin xi_{spin \lambda} [ exp(-3/2 y) S ] Y_{l_{\lambda, k}, m_{\lambda, k}}

d^2S/dy  - ( - 2 E r^2 - 2 Z r + [ l_{\lambda, k} + 1/2 ]^2) S - (+ 2 V2 r^2 ) S2 = 0

-> In Hartree-Fock, we would have:
d^2a/dy - c11(y) a - c12(y) b + ... = 0
d^2b/dy - c21(y) a - c22(y) b + ... = 0

Vectorially, if A = [a ; b ; ...] and C = [c11, c12, ... ; c21, c22, ...; ... ]:
d^2A/dy^2 - C(y) A = 0

Using Numerov's method:

f_{ij}(k) = \delta_{ij} - dy^2/12 c_{ij}(k))
g_{i}(k) = \sum_j f_{ij}(k) a_{j}(k)
g_{i}(k+1) = 12 a_{i}(k) - 10 g_{i}(k) - g_{i}(k-1)
a_{i}(k+1) = (F(k+1)^(-1) G(k+1))_{i}

-----

1/|s_r - q_r| = \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) Y*lm(\Omega(s)) Ylm(\Omega(q))

int Yl1m1 Yl2m2 Yl3m3 d\Omega = (-1)^m3 \sqrt{ (2 l1 + 1) (2 l2 + 1) / (4 \pi (2 l3 + 1) ) } CG(l1,l2,0,0,l3,0) CG(l1,l2,m1,m2,l3,-m3)

int Y*l1m1 Yl2m2 d\Omega = delta_{l1,l2} delta_{m1,m2}

int Y_{l_{\mu, k}, -m_{\mu, k}}(\Omega(s)) Y_{l_{\mu, o}, m_{\mu, o}}(\Omega(s)) Y_{l,-m}(\Omega(s)) d\Omega(s)  = (-1)^(-m) \sqrt{ (2 l_{\mu, k} + 1) (2 l_{\mu, o} + 1) / (4 \pi (2 l + 1) ) } CG(l_{\mu, k},l_{\mu, o},0,0,l,0) CG(l_{\mu, k},l_{\mu, o},-m_{\mu, k},m_{\mu, o},l,m)

-----

k and m are indices for terms within an orbital

V^d_\mu(q) = \sum_{k,m} 
               int_{\Omega(s)} int_{s_r} 
                     sqrt(r/rmin)^(-2) U_k(s) U_m(s)
                        Y*_{l_{\mu, k}, m_{\mu, k}} Y_{l_{\mu, m}, m_{\mu, m}}
                             1/|s_r - q_r| d\Omega(s) s_r^2 ds_r

V^d_\mu(q) = \sum_{k,m} 
               int_{\Omega(s)} int_{s_r} 
                     (r/rmin)^(-1) U_k(s) U_m(s)
                        Y*_{l_{\mu, k}, m_{\mu, k}} Y_{l_{\mu, m}, m_{\mu, m}}
                             1/|s_r - q_r|
                                   d\Omega(s) s^2 ds_r

Using 1/|s_r - q_r| = \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) Y*lm(\Omega(s)) Ylm(\Omega(q))

V^d_\mu(q) = \sum_{k,m} 
               int_{\Omega(s)} int_{s_r} 
                     (r/rmin)^(-1) U_k(s) U_m(s)
                        Y*_{l_{\mu, k}, m_{\mu, k}} Y_{l_{\mu, m}, m_{\mu, m}}
                          \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) Y*lm(\Omega(s)) Ylm(\Omega(q))
                                   d\Omega(s) s^2 ds_r

V^d_\mu(q) = \sum_{k,m} 
               int_{\Omega(s)} int_{s_r} 
                    \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(q) U_m(q)
                        Y*_{l_{\mu, k}, m_{\mu, k}} (Omega(s)) Y_{l_{\mu, m}, m_{\mu, m}} (Omega(s)) Y*lm(\Omega(s))
                                   d\Omega(s) s^2 ds_r
                                        Ylm(\Omega(q))

Do Omega(s) integral using 
int Yl1m1 Yl2m2 Yl3m3 d\Omega = (-1)^m3 \sqrt{ (2 l1 + 1) (2 l2 + 1) / (4 \pi (2 l3 + 1) ) } CG(l1,l2,0,0,l3,0) CG(l1,l2,m1,m2,l3,-m3)

int_{\Omega(s)} Y*_{l_{\mu, k}, m_{\mu, k}} (Omega(s)) Y_{l_{\mu, m}, m_{\mu, m}} (Omega(s)) Y*lm(\Omega(s)) d\Omega(s) = ...
    = (-1)^(-m_k - m) int_{\Omega(s)} Y_{l_{\mu, k}, -m_{\mu, k}} (Omega(s)) Y_{l_{\mu, m}, m_{\mu, m}} (Omega(s)) Y_{l,-m}(\Omega(s)) d\Omega(s)
    = (-1)^(-m_k - m) (-1)^(m) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)

Then:

V^d_\mu(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                            Ylm(\Omega(q))

V^d_\mu(q) u(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                            \sum_n U_n(q_r) Y_{l_n,m_n}(\Omega(q)) Ylm(\Omega(q))

int(Omega(q)) V^d_\mu(q) u(q) Y*_{l_a,m_a} dOmega(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                            \sum_n U_n(q_r) int(Omega(q)) Y*_{l_a,m_a} Y_{l_n,m_n}(\Omega(q)) Ylm(\Omega(q)) dOmega(q)

int(Omega(q)) V^d_\mu(q) u(q) Y*_{l_a,m_a} dOmega(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                            \sum_n U_n(q_r) int(Omega(q)) (-1)^m_a Y_{l_a,-m_a} Y_{l_n,m_n}(\Omega(q)) Ylm(\Omega(q)) dOmega(q)

Use this:
int Yl1m1 Yl2m2 Yl3m3 d\Omega = (-1)^m3 \sqrt{ (2 l1 + 1) (2 l2 + 1) / (4 \pi (2 l3 + 1) ) } CG(l1,l2,0,0,l3,0) CG(l1,l2,m1,m2,l3,-m3)

int(Omega(q)) V^d_\mu(q) u(q) Y*_{l_a,m_a} dOmega(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                          (-1)^m_a \sqrt{ (2 l_a + 1) (2 l_n + 1) / (4 \pi (2 l + 1) ) } CG(l_a,l_n,0,0,l,0) CG(l_a,l_n,-m_a,m_n,l,-m)
                              \sum_n U_n(q_r) 




----------

V^ex_\mu(q) u_\lambda(q) = \delta_{spin \lambda,spin \mu} [int u*_\mu(s) u_\lambda(s) / |s - q| ds ] u_\mu(q)

V^ex_\mu(q) u_\lambda(q) = \delta_{spin \lambda,spin \mu}
            \sum_{k,o}
               int_{\Omega(s)} int_{s_r} 
                  U(r) V(r)
                        Y*_{l_{\mu, k}, m_{\mu, k}} Y_{l_{\lambda, o}, m_{\lambda, o}}
                             1/|s_r - q_r| d\Omega(s) ds_r u_\mu(q)


k and o are indices for terms within orbitals idx and nidx respectively

V^ex_\mu(q) u_\lambda(q) = \delta_{spin \lambda,spin \mu} \sum_{k,o} 
               int_{\Omega(s)} int_{s_r} 
                     sqrt(r/rmin)^(-2) U_k(s) U_o(s)
                        Y*_{l_{\mu, k}, m_{\mu, k}} Y_{l_{\lambda, o}, m_{\lambda, o}}
                             1/|s_r - q_r| d\Omega(s) s_r^2 ds_r

int(Omega(q)) V^ex_\mu(q) u(q) Y*_{l_a,m_a} dOmega(q) = \sum_{k,o} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_o(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_o + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_o,0,0,l,0) CG(l_k,l_o,-m_k,m_o,l,m)
                        s^2 ds_r
                            \sum_n U_n(q_r) int(Omega(q)) (-1)^m_a Y_{l_a,-m_a} Y_{l_n,m_n}(\Omega(q)) Ylm(\Omega(q)) dOmega(q)

int(Omega(q)) V^d_\mu(q) u(q) Y*_{l_a,m_a} dOmega(q) = \delta_{spin \lambda,spin \mu} \sum_{k,o} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_o(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_o,0,0,l,0) CG(l_k,l_o,-m_k,m_o,l,m)
                        s^2 ds_r
                          (-1)^m_a \sqrt{ (2 l_a + 1) (2 l_n + 1) / (4 \pi (2 l + 1) ) } CG(l_a,l_n,0,0,l,0) CG(l_a,l_n,-m_a,m_n,l,-m)
                              \sum_n U_n(q_r) 


'''

def makeStruct(x):
  global No
  global listOrb

  # x = (a00 a01 a02 a03 a10 a11 a12 a13 ... 
  mlistOrb = []
  for orbidx in range(0, No):
    mlistOrb.append(Orbital(orbidx, n = listOrb[orbidx].n, spin = listOrb[orbidx].spin, N = listOrb[orbidx].N))

  i = 0
  for orbidx in range(0, No): # for each orbital
    mlistOrb[orbidx].E = x[i]
    i += 1
  for orbidx in range(0, No): # for each orbital
    for yi in range(0, len(listOrb[orbidx].l)):
      mlistOrb[orbidx].addY(listOrb[orbidx].l[yi], listOrb[orbidx].m[yi])

  return mlistOrb

def makeX(mlistOrb):
  global No
  xN = No

  x = np.zeros(xN, dtype = np.float64)
  i = 0
  for orbidx in range(0, No):
    x[i] = mlistOrb[orbidx].E
    i += 1

  return x

# la and ma are the q. numbers of the projection sph. harm. --> they are related to the numbers of the equation used (row)
# ln and mn are the q. numbers of the U(r) term by which this operator is multiplied -> they are related to the column
def getVd(mlistOrb, idx, N, r, dy, la, ma, ln, mn):
  '''
V^d_\mu(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                            Ylm(\Omega(q))

----

int(Omega(q)) V^d_\mu(q) u(q) Y*_{l_a,m_a} dOmega(q) = \sum_{k,m} 
               int_{s_r} 
                 \sum_l=0^inf \sum_m=-l^m=l 4 \pi / (2 l + 1) r<^l/r>^(l+1) (r/rmin)^(-1) U_k(s) U_m(s)
                    (-1)^(-m_k) \sqrt{ (2 l_k + 1) (2 l_m + 1) / (4 \pi (2 l + 1) ) } CG(l_k,l_m,0,0,l,0) CG(l_k,l_m,-m_k,m_m,l,m)
                        s^2 ds_r
                          \sum_n U_n(q_r) 
                             (-1)^m_a \sqrt{ (2 l_a + 1) (2 l_n + 1) / (4 \pi (2 l + 1) ) } CG(l_a,l_n,0,0,l,0) CG(l_a,l_n,-m_a,m_n,l,-m)


  '''

  Vd = np.zeros(N, dtype = np.float64)
  if abs(ma) > la:
    return Vd
  if abs(mn) > ln:
    return Vd

  # \sum_{k,m} (sums over terms of a single orbital of the form u = U_1(r) Y_l1m1 + U_2(r) Y_l2m2 + ...)
  # we have to calculate u^2, so we get all cross terms
  for yidx1 in range(len(mlistOrb[idx].a)):
    mk = mlistOrb[idx].m[yidx1]
    lk = mlistOrb[idx].l[yidx1]
    Uk = mlistOrb[idx].a[yidx1]
    for yidx2 in range(len(mlistOrb[idx].a)):
      mm = mlistOrb[idx].m[yidx2]
      lm = mlistOrb[idx].l[yidx2]
      Um = mlistOrb[idx].a[yidx2]
      # now for each q value of V^d_\mu(q)
      for i in range(N):
        q = r[i]

        Vd[i] += LegendreGaussQuadrature(VdToIntegrate, [dy, q, r, rmin, Uk, Um, lk, mk, lm, mm, la, ma, ln, mn])

        # integrate over s_r
        #for sidx in range(N):
        #  s = r[sidx]
        #  Vd[i] += VdToIntegrate(s, [dy, q, r, r[0], Uk, Um, lk, mk, lm, mm, la, ma, ln, mn])*s*dy

        #  lmax = 2 # only sum up to this l
        #  rl = np.min([q, s])
        #  rg = np.max([q, s])
        #  for l in range(0, lmax):
        #    for m in range(-l,l+1):
        #      Vd[i] += 4*np.pi/(2.0*l+1.0)*(r[0]/s)*(rl**l)/(rg**(l+1))*Uk[sidx]*Um[sidx]*((-1)**(-mk))*np.sqrt((2.0*lk+1)*(2.0*lm+1)/(4*np.pi*(2.0*l+1)))*CG(lk,lm,0,0,l,0)*CG(lk,lm,-mk,mm,l,m)*((-1)**ma)*np.sqrt((2*la+1)*(2*ln+1)/(4*np.pi*(2*l+1)))*CG(la,ln,0,0,l,0)*CG(la,ln,-ma,mn,l,-m)*(s**2)*s*dy
  return Vd

# Vd(q) = int F(q,s) dy
# this returns F(q,s)
def VdToIntegrate(x, params):
  [dy, q, r, rmin, Ukv, Umv, lk, mk, lm, mm, la, ma, ln, mn] = params
  s = x
  sidx = int(np.log(s/rmin)/dy)
  if sidx > N-1:
    sidx = N-1
  Uk = Ukv[sidx]
  Um = Umv[sidx]
  lmax = 2 # only sum up to this l
  rl = np.min([q, s])
  rg = np.max([q, s])
  R = 0
  for l in range(0, lmax):
    for m in range(-l,l+1):
      R += 4*np.pi/(2.0*l+1.0)*(rmin/s)*(rl**l)/(rg**(l+1))*Uk*Um*((-1)**(-mk))*np.sqrt((2.0*lk+1)*(2.0*lm+1)/(4*np.pi*(2.0*l+1)))*CG(lk,lm,0,0,l,0)*CG(lk,lm,-mk,mm,l,m)*((-1)**ma)*np.sqrt((2*la+1)*(2*ln+1)/(4*np.pi*(2*l+1)))*CG(la,ln,0,0,l,0)*CG(la,ln,-ma,mn,l,-m)*(s**2)
  return R

# la and ma are the q. numbers of the projection sph. harm. --> they are related to the numbers of the equation used (row)
# ln and mn are the q. numbers of the U(r) term by which this operator is multiplied -> they are related to the column
def getVex(mlistOrb, idx, nidx, N, r, dy, la, ma, ln, mn):
  Vex = np.zeros(N, dtype = np.float64)
  if abs(ma) > la:
    return Vex
  if abs(mn) > ln:
    return Vex

  # \sum_{k,m} (sums over terms of a single orbital of the form u = U_1(r) Y_l1m1 + U_2(r) Y_l2m2 + ...)
  # we have to calculate u^2, so we get all cross terms
  for yidx1 in range(len(mlistOrb[idx].a)):
    mk = mlistOrb[idx].m[yidx1]
    lk = mlistOrb[idx].l[yidx1]
    Uk = mlistOrb[idx].a[yidx1]
    for yidx2 in range(len(mlistOrb[nidx].a)):
      mm = mlistOrb[nidx].m[yidx2]
      lm = mlistOrb[nidx].l[yidx2]
      Um = mlistOrb[nidx].a[yidx2]
      if mlistOrb[idx].spin != mlistOrb[nidx].spin:
        continue
      # now for each q value of V^x_\mu(q)
      for i in range(N):
        q = r[i]
        Vex[i] += -LegendreGaussQuadrature(VdToIntegrate, [dy, q, r, rmin, Uk, Um, lk, mk, lm, mm, la, ma, ln, mn])
        # integrate over s_r
        #for sidx in range(N):
        #  s = r[sidx]
        #  Vex[i] += VdToIntegrate(s, [dy, q, r, r[0], Uk, Um, lk, mk, lm, mm, la, ma, ln, mn])*s*dy

  return Vex

def applyNumerovOnlyE(mlistOrb, N, r, rmin, dy, calculateE0):
  global Z
  global Vd
  '''
d^2S/dy  - ( - 2 E rmin^2 exp(2 y) - 2 Z rmin exp(y) + [ l_{\lambda, k} + 1/2 ]^2) S - (+ 2 V2 rmin^2 exp(2y) ) S2 = 0

Vectorially, if A = [a ; b ; ...] and C = [c11, c12, ... ; c21, c22, ...; ... ]:
d^2A/dy^2 - C(y) A = 0

Using Numerov's method:

f_{ij}(k) = \delta_{ij} - dy^2/12 c_{ij}(k))
g_{i}(k) = \sum_j f_{ij}(k) a_{j}(k)
g_{i}(k+1) = 12 a_{i}(k) - 10 g_{i}(k) - g_{i}(k-1)
a_{i}(k+1) = (F(k+1)^(-1) G(k+1))_{i}
  '''
  Nor = 0
  for i in range(len(mlistOrb)):
    Nor += len(mlistOrb[i].l)
  outward = np.zeros((Nor, N), dtype = np.float64)
  inward = np.zeros((Nor, N), dtype = np.float64)
  matched = np.zeros((Nor, N), dtype = np.float64)
  icl = np.zeros(Nor, dtype = np.int)
  for i in range(Nor): icl[i] = -1

  idx = 0
  for i in range(len(mlistOrb)):
    for j in range(len(mlistOrb[i].l)):
      l = mlistOrb[i].l[j]
      E = mlistOrb[i].E
      n = mlistOrb[i].n
      #outward[idx, 0] = ((Z*r[0])**(l+0.5))/n
      #outward[idx, 1] = ((Z*r[1])**(l+0.5))/n
      #outward[idx, 0] = 2*(r[0]**(l+1))*(1 - Z*r[0]/(2*(l+1)))/np.sqrt(r[0])/np.sqrt(r[0])
      #outward[idx, 1] = 2*(r[1]**(l+1))*(1 - Z*r[1]/(2*(l+1)))/np.sqrt(r[1])/np.sqrt(r[0])
      outward[idx, 0] = ((r[0]/r[0])**(l+0.5))
      outward[idx, 1] = ((r[1]/r[0])**(l+0.5))
      inward[idx, N-1] = np.exp(-np.sqrt(-2*E)*r[N-1])/(r[N-1]/r[0])
      inward[idx, N-2] = np.exp(-np.sqrt(-2*E)*r[N-2])/(r[N-2]/r[0])
      idx += 1

  F = np.zeros((Nor, Nor, N), dtype = np.float64)
  diag = np.zeros(N, dtype = np.float64)

  for k in range(0, N): # for each r value
    y = dy*k
    idx = 0
    jdx = 0
    for i0 in range(len(mlistOrb)):
      for i1 in range(len(mlistOrb[i0].l)):
        li = mlistOrb[i0].l[i1]
        mi = mlistOrb[i0].m[i1]
        Ei = mlistOrb[i0].E
        jdx = 0
        for j0 in range(len(mlistOrb)):
          for j1 in range(len(mlistOrb[j0].l)):
            lj = mlistOrb[j0].l[j1]
            mj = mlistOrb[j0].m[j1]
            Ej = mlistOrb[j0].E
            if idx == jdx:
              F[idx, jdx, k] += (-2.0*Ei*rmin**2*np.exp(2*y))
              F[idx, jdx, k] += (-2.0*Z*rmin*np.exp(y))
              F[idx, jdx, k] += (li + 0.5)**2
              if icl[idx] < 0 and k >= 1 and F[idx, jdx, k]*F[idx, jdx, k-1] < 0:
                icl[idx] = k
            F[idx, jdx, k] += (2.0*Vex[i0][i1][j0][j1][k]*rmin**2*np.exp(2*y))
            jdx += 1
        idx += 1
    # given an equation and projection
    jdx = 0
    for j0 in range(len(mlistOrb)):
      for j1 in range(len(mlistOrb[j0].l)):
        # add the int u u* dr terms
        for i0 in range(len(mlistOrb)):
          for i1 in range(len(mlistOrb[i0].l)):
            F[jdx, jdx, k] += (2.0*Vd[i0][i1][j0][j1][k]*rmin**2*np.exp(2*y))
        jdx += 1

  for idx in range(Nor):
    for jdx in range(Nor):
      for k in range(0, N): # for each r value
        if idx == jdx:
          F[idx, jdx, k] = 1 - dy*dy/12.0 * F[idx, jdx, k]
        else:
          F[idx, jdx, k] = - dy*dy/12.0 * F[idx, jdx, k]

  for k in range(1, N-1): # for each r value
    y = dy*k

    Fk = F[0:Nor, 0:Nor, k]
    Fkm1 = F[0:Nor, 0:Nor, k-1]
    Fkp1 = F[0:Nor, 0:Nor, k+1]

    Ak = outward[0:Nor,k]
    Akm1 = outward[0:Nor, k-1]
    Akp1 = np.matmul(np.linalg.inv(Fkp1), (np.matmul((12.0*np.identity(Nor) - 10.0*Fk), Ak) - np.matmul(Fkm1, Akm1)))
    for idx in range(Nor):
      outward[idx, k+1] = Akp1[idx]
      if np.isnan(outward[idx, k+1]) or np.isinf(outward[idx,k+1]):
        outward[idx,k+1] = outward[idx,k]

  for k in reversed(range(1, N-1)): # for each r value
    y = dy*k

    Fk = F[0:Nor, 0:Nor, k]
    Fkm1 = F[0:Nor, 0:Nor, k-1]
    Fkp1 = F[0:Nor, 0:Nor, k+1]

    Ak = inward[0:Nor, k]
    Akp1 = inward[0:Nor, k+1]
    Akm1 = np.matmul(np.linalg.inv(Fkm1), (np.matmul((12.0*np.identity(Nor) - 10.0*Fk), Ak) - np.matmul(Fkp1, Akp1)))
    for idx in range(Nor):
      inward[idx, k-1] = Akm1[idx]
      if np.isnan(inward[idx, k-1]) or np.isinf(inward[idx,k-1]):
        inward[idx,k-1] = inward[idx,k]

  ratio_xing = np.zeros(Nor, dtype = np.float64)
  for idx in range(Nor):
    if icl[idx] < 2:
      for k in range(N):
        matched[idx, k] = outward[idx, k]
    else:
      ratio_xing[idx] = outward[idx, icl[idx]]/inward[idx, icl[idx]]
      for k in range(0, icl[idx]):
        matched[idx, k] = outward[idx, k]
      for k in range(icl[idx], N):
        matched[idx, k] = inward[idx, k]*ratio_xing[idx]

  norm = np.zeros(Nor, dtype = np.float64)
  no = np.zeros(Nor, dtype = np.float64)
  idx = 0
  for i0 in range(len(mlistOrb)):
    for i1 in range(len(mlistOrb[i0].l)):
      li = mlistOrb[i0].l[i1]
      mi = mlistOrb[i0].m[i1]
      Ei = mlistOrb[i0].E  
      for j in range(len(r)):
        norm[idx] += (matched[idx, j]/np.sqrt(r[j]/r[0]))**2 * r[j]**2 * r[j]*dy
      idx += 1
  for idx in range(Nor):
    if norm[idx] != 0:
      for k in range(N):
        matched[idx, k] = (matched[idx,k])/np.sqrt(norm[idx])
        if k > 1 and matched[idx,k]*matched[idx,k-1] < 0:
          no[idx] += 1
  idx = 0
  for i0 in range(len(mlistOrb)):
    for i1 in range(len(mlistOrb[i0].l)):
      no[idx] -= mlistOrb[i0].n - mlistOrb[i0].l[i1] - 1
      idx += 1
  

  for k in range(N): # for each r value
    idx = 0
    for i0 in range(len(mlistOrb)):
      for i1 in range(len(mlistOrb[i0].l)):
        li = mlistOrb[i0].l[i1]
        mi = mlistOrb[i0].m[i1]
        Ei = mlistOrb[i0].E
        mlistOrb[i0].a[i1][k] = matched[idx, k]
        idx += 1

  diffs = np.zeros(Nor, dtype = np.float64)
  for i in range(Nor):
    micl = np.zeros(Nor, dtype = np.float64)
    miclm1 = np.zeros(Nor, dtype = np.float64)
    miclp1 = np.zeros(Nor, dtype = np.float64)
    Ficlp1 = np.zeros((Nor, Nor), dtype = np.float64)
    Ficlm1 = np.zeros((Nor, Nor), dtype = np.float64)
    Ficl = np.zeros((Nor, Nor), dtype = np.float64)

    for a in range(Nor):
      for b in range(Nor):
        Ficl[a,b] = F[a, b, icl[i]]
        Ficlm1[a,b] = F[a, b, icl[i]-1]
        Ficlp1[a,b] = F[a, b, icl[i]+1]
      micl[a] = matched[a, icl[i]]
      miclm1[a] = matched[a, icl[i]-1]
      miclp1[a] = matched[a, icl[i]+1]

  diffs = np.matmul((12.0*np.identity(Nor) - 10.0*Ficl), micl) - np.matmul(Ficlm1, miclm1) - np.matmul(Ficlp1, miclp1)

  idx = 0
  for i0 in range(len(mlistOrb)):
    for i1 in range(len(mlistOrb[i0].l)):
      if icl[idx] < 2:
        diffs[idx] = mlistOrb[i0].a[0][-1]
      idx += 1

  E0 = 0
  if calculateE0:
    for i in old_listOrb:
      E0 += i.E
    # FIXME: check J and K
    J = 0
    K = 0
    for i0 in range(len(old_listOrb)):
      for i1 in range(len(old_listOrb[i0].l)):
        for j0 in range(len(old_listOrb)):
          for j1 in range(len(old_listOrb[j0].l)):
            J += LegendreGaussQuadrature(ToIntegrateJ, [Vd, i0, i1, j0, j1, old_listOrb, r, dy])
            K += LegendreGaussQuadrature(ToIntegrateK, [Vex, i0, i1, j0, j1, old_listOrb, r, dy])
            #for k in range(N):
            #  J += Vd[i0][i1][j0][j1][k]*(old_listOrb[j0].a[j1][k]/np.sqrt(r[k]/r[0]))**2*r[k]**3*dy
            #  K += -Vex[i0][i1][j0][j1][k]*(old_listOrb[j0].a[j1][k]/np.sqrt(r[k]/r[0]))*(old_listOrb[i0].a[i1][k]/np.sqrt(r[k]/r[0]))*r[k]**3*dy
    E0 -= 0.5*(J-K)

  return [no, diffs, mlistOrb, E0]

def ToMinimize(x):
  global r
  global rmin
  global dy
  global N
  global old_listOrb
  mlistOrb = makeStruct(x)

  [no, diffs, flistOrb, E0] = applyNumerovOnlyE(mlistOrb, N, r, rmin, dy, calculateE0 = False)

  R = np.float64(0)
  for i in range(len(diffs)):
    R += (diffs[i])**2
  for i in no:
    R += i**2
  #R += 1.0/E0**2

  return R

import scipy.optimize

x0 = makeX(listOrb)

bounds = []
for i in range(len(x0)):
  bounds.append((None, 0))

Vd = {}
for i in range(len(listOrb)):
  Vd[i] = {}
  for j in range(len(listOrb[i].l)):
    Vd[i][j] = {}
    for col_i in range(len(listOrb)):
      Vd[i][j][col_i] = {}
      for col_j in range(len(listOrb[col_i].l)):
        Vd[i][j][col_i][col_j] = np.zeros(N, dtype=np.float64)

Vex = {}
for i in range(len(listOrb)):
  Vex[i] = {}
  for j in range(len(listOrb[i].l)):
    Vex[i][j] = {}
    for col_i in range(len(listOrb)):
      Vex[i][j][col_i] = {}
      for col_j in range(len(listOrb[col_i].l)):
        Vex[i][j][col_i][col_j] = np.zeros(N, dtype=np.float64)

epsv = 1e-5

def ToIntegrateJ(x, params):
  [Vd, i0, i1, j0, j1, old_listOrb, r, dy] = params
  k = int(np.log(x/r[0])/dy)
  if k > N-1:
    k = N-1
  return Vd[i0][i1][j0][j1][k]*(old_listOrb[j0].a[j1][k]/np.sqrt(r[k]/r[0]))**2*r[k]**2

def ToIntegrateK(x, params):
  [Vex, i0, i1, j0, j1, old_listOrb, r, dy] = params
  k = int(np.log(x/r[0])/dy)
  if k > N-1:
    k = N-1
  return -Vex[i0][i1][j0][j1][k]*(old_listOrb[j0].a[j1][k]/np.sqrt(r[k]/r[0]))*(old_listOrb[i0].a[i1][k]/np.sqrt(r[k]/r[0]))*r[k]**2

def plotWF(r, old_listOrb, Z, N):
  style = ['b-', 'g-', 'b-']
  import matplotlib.pyplot as plt
  rMax = np.where(r > 4)
  rMax = rMax[0][0]
  rMax = N-1
  exact = 2*np.sqrt(Z**3)*np.exp(-Z*r)
  leg = []
  for i in range(len(old_listOrb)):
    plt.plot(r[0:rMax], old_listOrb[i].valueR(r)[0:rMax], style[i])
    leg.append("Z=%d, %s (%.5f eV)" % (Z, old_listOrb[i].name(), old_listOrb[i].E*eV))
  plt.plot(r[0:rMax], exact[0:rMax], 'r--')
  leg.append("Z=%d, 1s" % Z)
  plt.xlabel("$r$ [a0]")
  plt.ylabel("$|R(r)|$")
  plt.legend(leg, frameon=False)
  plt.show()

def plotPot(r, Vd, Vex, Z, N, row):
  styleV = 'r--'
  styleD = ['b-', 'g-', 'r-']
  styleEx = ['b-.', 'g-.', 'r-.']
  import matplotlib.pyplot as plt
  rMax = np.where(r > 4)
  rMax = rMax[0][0]
  rMin = 0
  rMinV = np.where(r > 0.5)
  rMinV = rMinV[0][0]
  y0 = []
  V = -Z/r
  leg = []
  plt.plot(r[rMin:rMax], V[rMin:rMax], styleV)
  leg.append("Z=%d, coulomb pot." % Z)
  for i in range(len(old_listOrb)):
    plt.plot(r[rMin:rMax], Vd[row][0][i][0][rMin:rMax], styleD[i])
    leg.append("Z=%d, direct pot. [%d, %d]" % (Z, row, i))
    y0.append(Vd[row][0][i][0][rMin])
    plt.plot(r[rMin:rMax], Vex[row][0][i][0][rMin:rMax], styleEx[i])
    leg.append("Z=%d, exchange pot. [%d, %d]" % (Z, row, i))
    y0.append(Vex[row][0][i][0][rMin])
  y0.append(V[rMinV])
  plt.xlabel("$r$ [a0]")
  plt.ylabel("$|R(r)|$")
  plt.ylim((np.min(y0), np.max(y0)))
  plt.legend(leg, frameon=False)
  plt.show()

while True: # SCF loop

  E0 = 0
  mlistOrb = makeStruct(x0)
  [no, diffs, flistOrb, E0] = applyNumerovOnlyE(mlistOrb, N, r, rmin, dy, calculateE0 = True)
  print "Total energy: %.10f" % E0

  # get correct energy with fixed potentials
  while True:
    #epsv = np.max([epsv*0.01, 1e-18])
    res = scipy.optimize.minimize(ToMinimize, x0, method='L-BFGS-B', bounds = bounds, options={'gtol': 0, 'disp': True, 'maxiter': 10, 'ftol':0, 'eps': epsv, 'maxfun': 100000, 'maxcor': 10000})
    x0 = res.x

    mlistOrb = makeStruct(x0)
    [no, diffs, flistOrb, E0] = applyNumerovOnlyE(mlistOrb, N, r, rmin, dy, calculateE0 = False)
    for i in flistOrb:
      print str(i)
    old_listOrb = flistOrb[:]

    print "After E minimization only: x0, no", x0, no
    plotWF(r, old_listOrb, Z, N)
    plotPot(r, Vd, Vex, Z, N, 2)

    rerun = False
    for i in range(len(no)):
      if no[i] > 0:
        x0[i] = np.min([0, x0[i] + 0.5*0.5*Z**2*np.fabs(1.0/mlistOrb[i].n**2 - 1.0/(mlistOrb[i].n+1)**2)])
        rerun = True
      elif no[i] < 0:
        x0[i] = np.min([0, x0[i] - 0.5*0.5*Z**2*np.fabs(1.0/mlistOrb[i].n**2 - 1.0/(mlistOrb[i].n+1)**2)])
        rerun = True
    print "After forcefully skipping energy levels to look for another minimum: x0, no", x0, no
    if rerun:
      continue
    diffs_sum = np.sum(diffs**2)
    if diffs_sum < 1e-7:
      break

  plotWF(r, old_listOrb, Z, N)
  for i in range(len(old_listOrb)):
    plotPot(r, Vd, Vex, Z, N, i)

  alpha = 0.25
  print "Updating Vd"
  for i0 in range(len(old_listOrb)):
    for i1 in range(len(old_listOrb[i0].l)):
      li = old_listOrb[i0].l[i1]
      mi = old_listOrb[i0].m[i1]
      Ei = old_listOrb[i0].E
      for j0 in range(len(old_listOrb)):
        for j1 in range(len(old_listOrb[j0].l)):
          lj = old_listOrb[j0].l[j1]
          mj = old_listOrb[j0].m[j1]
          Ej = old_listOrb[j0].E
          Vd[i0][i1][j0][j1] = (1-alpha)*Vd[i0][i1][j0][j1] + alpha*getVd(old_listOrb, i0, N, r, dy, li, mi, lj, mj)

  print "Updating Vex"
  for i0 in range(len(old_listOrb)):
    for i1 in range(len(old_listOrb[i0].l)):
      li = old_listOrb[i0].l[i1]
      mi = old_listOrb[i0].m[i1]
      Ei = old_listOrb[i0].E
      for j0 in range(len(old_listOrb)):
        for j1 in range(len(old_listOrb[j0].l)):
          lj = old_listOrb[j0].l[j1]
          mj = old_listOrb[j0].m[j1]
          Ej = old_listOrb[j0].E
          Vex[i0][i1][j0][j1] = (1-alpha)*Vex[i0][i1][j0][j1] + alpha*getVex(old_listOrb, i0, j0, N, r, dy, li, mi, lj, mj)
  print "vd20",Vd[2][0][0][0]
  print "vd21",Vd[2][0][1][0]
  print "vd22",Vd[2][0][2][0]
  print "vex20",Vex[2][0][0][0]
  print "vex21",Vex[2][0][1][0]
  print "vex22",Vex[2][0][2][0]



