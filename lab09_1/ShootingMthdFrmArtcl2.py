"""
Script for solving the one dimensional Schroedinger equation numerically.
Numerical integration method used is the fourth order Runge Kutta.
Counts the nodes of the wave function and determins the harmonic.
Then refines the solution until proper energy is found.
Potentials:
Infinite Potential Well
V(x_ < 0) = inf, V(x_ = 0, 1) = 0, V(x_ > 1) = inf
Analytic solution:
sin(k * pi * x)
Harmonic Oscillator:
V(x_) = x_**2
Analytic solution:
(1 / (sqrt((2**n) * n!) H (x)) * exp (-x**2 / 2)
"""

import numpy as np
import scipy
from scipy import integrate
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def Schroed(y, r, V, E):
    """Return one dim Schroedinger eqation with Potential V."""
    psi, phi = y
    dphidx = [phi, (V - E) * psi]
    return np.asarray(dphidx)

def rk4(f, psi0, x, V, E):
    """Fourth-order Runge-Kutta method to solve phi' = f(psi, x) with psi(x[0]) = psi0.
    Integrates function f with inital values psi0 and potenital V numerically.
    Output is possible multidimensional (in psi) array with len(x)."""
    n = len(x)
    psi = np.array([psi0] * n)
    for i in range(n - 1):
        h =  x[i+1] - x[i]
        k1 = h * f(psi[i], x[i], V[i], E)
        k2 = h * f(psi[i] + 0.5 * k1, x[i] + 0.5 * h, V[i], E)
        k3 = h * f(psi[i] + 0.5 * k2, x[i] + 0.5 * h, V[i], E)
        k4 = h * f(psi[i] + k3, x[i+1], V[i], E)
        psi[i+1] = psi[i] + (k1 + 2.0*(k2 + k3) + k4) / 6.0
    return psi

def findZeros(rightbound_vals):
    """ 
    Find zero crossing due to sign change in rightbound_vals array.
    Return array with array indices before sign change occurs.
    """
    return np.where(np.diff(np.signbit(rightbound_vals)))[0]

def normalize(output_wavefunc):
    """A function to roughly normalize the wave function to 1."""
    normal = max(output_wavefunc)
    return output_wavefunc * (1 / (normal))

def countNodes(wavefunc):
    """Count nodes of wavefunc by finding Minima and Maxima in wavefunc."""
    maxarray = argrelextrema(wavefunc, np.greater)[0]
    minarray = argrelextrema(wavefunc, np.less)[0]
    nodecounter = len(maxarray) + len(minarray)
    return nodecounter

def RefineEnergy(Ebot, Etop, Nodes, psi0, x, V):
    tolerance = 1e-12
    ET = Etop
    EB = Ebot
    psi = [1]
    while (abs(EB - ET) > tolerance or abs(psi[-1]) > 1e-3):
        initE = (ET + EB) / 2.0
        psi = rk4(Schroed, psi0, x, V, initE)[:, 0]
        nodes_ist = len(findZeros(psi)) - 1
        if nodes_ist > Nodes + 1:
            ET = initE
            continue
        if nodes_ist < Nodes - 1:
            EB = initE
            continue
        if (nodes_ist % 2 == 0):
            if ((psi[len(psi) - 1] <= 0.0)):
                ET = initE
            else:
                EB = initE
        elif nodes_ist > 0:
            if ((psi[len(psi) - 1] <= 0.0)):
                EB = initE
            else:
                ET = initE
        elif nodes_ist < 0:
            EB = initE
    return EB, ET

def ShootingInfinitePotentialWell(E_interval, nodes):
    """
    Implementation of Shooting method for Infinite PotWell
    INPUT : E_interval array with top and bottom value, len(E_interval) = 2
    nodes : Number wavefunction nodes => determins quantum state.
    OUTPUT : refined energy value
    numerical wavefunction as array.
    """
    psi_0 = 0.0
    phi_0 = 1.0
    psi_init = np.asarray ([psi_0, phi_0])
    h_mesh = 1.0 / 100.0 # stepsize for range arrays
    x_arr_ipw = np.arange(0.0, 1.0 + h_mesh, h_mesh) # set up mesh
    V_ipw = np.zeros(len(x_arr_ipw)) # set up potential
    EBref , ETref = RefineEnergy(E_interval[0], E_interval[1], nodes, psi_init, x_arr_ipw, V_ipw)
    psi = rk4(Schroed, psi_init, x_arr_ipw, V_ipw, EBref)[:, 0]
    return EBref, normalize(psi), x_arr_ipw

def IPW_ana(x, k):
    """Return analytical wavefunc of respective state(k) of IPW ."""
    return np.asarray(np.sin(k * np.pi * x))

def ShootingQuantumHarmonicOscillator(E_interval, nodes):
    """Shooting QHO."""
    psi_0 = 0.0
    phi_0 = 1.0
    psi_init = np.asarray([psi_0, phi_0])
    h_mesh = 1.0 / 100.0 # stepsize for range arrays
    x_arr_qho = np.arange(-5.0, 5.0 + h_mesh, h_mesh) # set up mesh
    V_qho = x_arr_qho**2 # set up potential
    EBref, ETref = RefineEnergy(E_interval[0], E_interval[1], nodes, psi_init, x_arr_qho, V_qho)
    psiB = rk4(Schroed, psi_init, x_arr_qho, V_qho, EBref)[:, 0]
    psiT = rk4(Schroed, psi_init, x_arr_qho, V_qho, ETref)[:, 0]
    return EBref, ETref, normalize(psiB), normalize(psiT), x_arr_qho

def QHO_ana(x, nodes):
    """Return analytic solution for QHO for up to 5 nodes."""
    if (nodes == 1):
        return np.exp(-( x )**2 / 2)
    elif (nodes == 2):
        return np.sqrt(2.0) * (x) * np.exp(-(x)**2 / 2) * (-1)
    elif (nodes == 3):
        return (1.0 / np.sqrt(2.0)) * (2.0 * (x)**2 - 1.0) * np.exp(-( x )**2 / 2)
    elif (nodes == 4):
        return (1.0 / np.sqrt(3.0)) * (2.0 * (x)**3 - 3.0 * x) * np.exp(-(x)**2 / 2) * ( -1)
    elif (nodes == 5):
        return (1.0 / np.sqrt(24.0)) * (4.0 * (x)**4 - 12.0 * x**2 + 3.) * np.exp(-(x)**2 / 2)
    else :
        print("No analytic wave function found. Please try again.")
        print("Output will be zero array.")
        return np.zeros(len(x))
    
# Start
E_qho = [0.1, 100.0]
E_ipw = [1.0, 500.0]
nodes_arr = np.arange(1, 6, 1)
L = 0.0
N = 1.0
print("Welcome!")
print("Maximum quantum state is currently limited to the 4 th excited state.")
print()
print()

print("Infinte Potential Well Shooting")
figipw = plt.figure()
for ii in nodes_arr:
    Energy, psi_ipw, x_ipw = ShootingInfinitePotentialWell(E_ipw, ii)
    psi_ana = normalize(IPW_ana(ii, x_ipw))
    print("Found quantum state at energy = %s [Hartree]" %(Energy, ))
    plt.cla() # clear axis
    plt.clf() # clear figure
    plt.plot(x_ipw, psi_ipw, label=r'$\Psi(x)_{num}$')
    plt.plot(x_ipw, psi_ana, label=r'$\Psi(x)_{ana}$')
    plt.title('Eigenstate: %s' %(ii, ))
    plt.legend()
    plt.grid ()
    figipw.savefig('plots/ipw_shoottest_state_' + str(ii) + '.png')

print()
print("Quantum Harmonic Oscillator Shooting:")
figqho = plt.figure()

for ii in nodes_arr:
    EB, ET, psibot, psitop, x_qho = ShootingQuantumHarmonicOscillator(E_qho, ii)
    psi_ana = QHO_ana(x_qho, ii)
    print("Found quantum state at energy = %s [Hartree]" %(ET, ))
    plt.cla() # clear axis
    plt.clf() # clear figure
    plt.plot(x_qho, psitop, label=r'$\Psi(x)_{num}$')
    plt.plot(x_qho, normalize(psi_ana), 'r--', label=r'$\Psi(x)_{ana}$')
    plt.title('Eigenstate : %s' %(ii, ))
    plt.legend()
    plt.grid()
    figqho.savefig('plots/qho_shoottest_state_' + str(ii) + '.png')

print()
print()
print("Please find plots of wavefunctions in 'plots' - folder.")
print("\nGoodbye.")

