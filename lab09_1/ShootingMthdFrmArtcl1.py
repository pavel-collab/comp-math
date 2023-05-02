"""
Script for solving the one dimensional Schroedinger equation numerically.
Numerical integration method used is the fourth order Runge Kutta.
Counts the nodes of the wave function and determins the harmonic.
Then refines the solution until proper energy is found.
Potentials:
Infinite Potential Well
V(x < 0) = inf, V(x = [0,1]) = 0, V(x > 1) = inf
Harmonic Oscillator :
V(x) = x**2
Radial Hydrogen Atom Coulomb attraction:
V(r) = 2/r - (L(L +1))/(r**2)
"""

import numpy as np
import scipy
from scipy import integrate
from scipy.optimize import newton
import matplotlib.pyplot as plt

def Schroed (y, r, V, E):
    """ Return one dim Schroedinger eqation with Potential V . """
    psi, phi = y
    dphidx = [phi, (V - E) * psi]
    return np.asarray(dphidx)

def rk4 (f, psi0, x, V, E):
    """ Fourth - order Runge - Kutta method to solve phi' = f(psi, x) with psi (x[0])= psi0 .
    Integrates function f with inital values psi0 and potenital V numerically .
    Output is possible multidimensional (in psi) array with len (x).
    """
    n = len(x)
    psi = np.array([psi0] * n)
    for i in range (n - 1):
        h = x[i+1] - x[i]
        k1 = h*f(psi[i], x[i], V[i], E)
        k2 = h*f(psi[i] + 0.5*k1, x[i] + 0.5*h, V[i], E)
        k3 = h*f(psi[i] + 0.5*k2, x[i] + 0.5*h, V[i], E)
        k4 = h*f(psi[i] + k3, x[i+1], V[i], E)
        psi[i+1] = psi[i] + (k1 + 2.0*(k2 + k3) + k4) / 6.0
    return psi

def shoot(func, psi0, x, V, E_arr):
    """ Shooting method : find zeroes of function func for energies in E_arr.
    func: Schroedinger equation to solve.
    psi0: initial conditions on left side, can be array.
    V: Potential to solve SE with.
    E_arr: array of energy values: find possible zeroes.
    """
    psi_rightb = []
    for EN in E_arr :
        psi = rk4(func, psi0, x, V, EN)
        psi_rightb.append(psi[len(psi) - 1][0])
    return np.asarray(psi_rightb)

def shoot1(E, func, psi0, x, V):
    """ Helper function for optimizing resuts ."""
    psi = rk4(func, psi0, x, V, E)
    return psi [ len ( psi ) -1][0]


def shoot_ode(E, psi_init, x, L):
    '''Helper function for optimizing resuts.'''
    sol = integrate.odeint(Schrod_deriv, psi_init, x, args=(L, E))
    return sol[len(sol)-1][0]

def findZeros(rightbound_vals):
    """ 
    Find zero crossing due to sign change in rightbound_vals array.
    Return array with array indices before sign change occurs.
    """
    return np.where(np.diff(np.signbit(rightbound_vals)))[0]

def optimizeEnergy(func, psi0, x, V, E_arr):
    """ Optimize energy value for function using brentq."""
    shoot_try = shoot(func, psi0, x, V, E_arr)
    crossings = findZeros(shoot_try)
    energy_list = []
    for cross in crossings:
        energy_list.append(newton(shoot1, E_arr[cross], args=(func, psi0, x, V)))
    return np.asarray(energy_list)

def normalize(output_wavefunc):
    """ A function to roughly normalize the wave function . """
    normal = max(output_wavefunc)
    return output_wavefunc*(1/normal)

def shoot_potwell(psi_init, h_):
    """ 
    Shooting method for infinte potential well.
    500 mesh points.
    Returns the numerical and analytical solution as arrays.
    """
    x_arr_ipw = np.arange(0.0, 1.0 + h_, h_)
    V_ipw = np.zeros(len(x_arr_ipw))
    E_arr = np.arange(1.0, 100.0, 5.0)
    eigE = optimizeEnergy(Schroed, psi_init, x_arr_ipw, V_ipw, E_arr)
    ipw_out_list = []
    for EE in eigE:
        out = rk4(Schroed, psi_init, x_arr_ipw, V_ipw, EE)
        ipw_out_list.append(normalize(out[:, 0]))
    out_arr = np.asarray(ipw_out_list)
    # analytical solution for IPW
    k = np.arange(1.0, 4.0, 1.0)
    ipw_sol_ana = []
    for kk in k:
        ipw_sol_ana.append(np.sin(kk * np.pi * x_arr_ipw))
    ipw_sol_ana_arr = np.asarray(ipw_sol_ana)
    return x_arr_ipw, out_arr, ipw_sol_ana_arr

def shoot_QuantumHarmonicOscillator ( psi_init , h_ ):
    """ 
    Shooting method for quantum harmonic oscillator.
    500 mesh points.
    Returns the numerical and analytical solution as arrays.
    """
    x_arr_qho = np.arange (-5.0, 5.0+h_, h_)
    V_qho = x_arr_qho**2
    E_arr = np.arange(1.0, 15.0, 1.0)
    eigEn = optimizeEnergy(Schroed, psi_init, x_arr_qho, V_qho, E_arr)
    qho_out_list = []
    for EN in eigEn:
        out = rk4(Schroed, psi_init, x_arr_qho, V_qho, EN)
        qho_out_list.append(normalize(out[:, 0]))
    qho_out_arr = np.asarray(qho_out_list)
    # analytical solution for QHO
    qho_sol_ana_0 = np.exp(-(x_arr_qho )**2 / 2)
    qho_sol_ana_1 = np.sqrt(2.0) * (x_arr_qho ) * np.exp(-( x_arr_qho )**2 / 2) * ( -1)
    qho_sol_ana_2 = (1.0 / np.sqrt(2.0)) * (2.0*(x_arr_qho)**2 - 1.0) * np.exp(-( x_arr_qho )**2 / 2)
    qho_sol_list = []
    qho_sol_list.append(qho_sol_ana_0)
    qho_sol_list.append(qho_sol_ana_1)
    qho_sol_list.append(qho_sol_ana_2)
    return x_arr_qho, qho_out_arr, np.asarray(qho_sol_list)

def Schrod_deriv(y, r, L, E):
    """ Odeint calls routine to solve Schroedinger equation of the Hydrogen atom."""
    du2 = y[0] * ((L * (L+1))/(r**2) - 2./r - E)
    return [y[1], du2]

def shoot_hydrogen(psi_init, h_, L):
    """ """
    x_arr_hydro = np.arange(0.0001, 35.0+h_, h_)
    E_arr = np.arange(-1., 0., 0.001)
    rightb = []
    for EE in E_arr:
        psi = integrate.odeint(Schrod_deriv, psi_init, x_arr_hydro, args=(L, EE))[:, 0]
        rightb.append(psi[len(psi)-1])
    rightb_arr = np.asarray(rightb)
    crossings = findZeros(rightb_arr)
    energy_l = []
    for cross in crossings:
        energy_l.append(newton(shoot_ode, E_arr[cross], args=(psi_init, x_arr_hydro, L)))
    psi_out = []
    for En in energy_l:
        psi_out.append(integrate.odeint(Schrod_deriv, psi_init, x_arr_hydro, args=(L, En))[:, 0])
    return x_arr_hydro, np.asarray(psi_out)

def HYDRO_ana (x , N , L ):
    """Return analytical solution for Hydrogen SE."""
    # analytical solution hydrogen for N =1
    if (((N - L - 1) == 0) and (L == 0)):
        # return 2.0 * np.exp(-x/2) * x
        return x * np.exp(-x)
    elif (((N - L - 1) == 1) and (L == 0)):
        return (np.sqrt(2.) * (-x + 2.) * np.exp(-x / 2.) / 4.) * x
    elif (((N - L - 1) == 2)):
        return (2. * np.sqrt(3.) * (2. * x**2./9. - 2.* x + 3.) * np.exp(-x /3.)/27.) * x
    elif (((N - L -1) == 0) and (L == 1)):
        return (np.sqrt(6.) * x * np.exp(-x/2.)/12.) * x
    else :
        print("No analytic wave function found. Please try again.")
        print("Output will be zero array.")
        return np.zeros(len(x))
    
def plot_wavefunction(fig, title_string, x_arr, num_arr, ana_arr, axis_list):
    """Output plots for w av ef un ct io ns."""
    # clear plot
    plt.cla() # clear axis
    plt.clf() # clear figure
    plt.plot(x_arr, num_arr, 'b:', linewidth=4, label =r'$\Psi(\hat{x})_{num}$')
    plt.plot (x_arr, normalize(ana_arr), 'r-', label =r"$\Psi(\hat{ x })_{ana}$")
    plt.ylabel(r"$\Psi(\hat{ x })$", fontsize=16)
    plt.xlabel(r'$\hat{x}$', fontsize=16)
    plt.legend(loc='best', fontsize='small')
    plt.axis(axis_list)
    plt.title(title_string)
    plt.grid()
    fig.savefig("plots/wavefunc_" + title_string + ".png")

# Initial conditions for pot . well and harmonic osc
psi_0 = 0.0
phi_0 = 1.0
psi_init = np.asarray([psi_0, phi_0])
h_ = 1.0 / 200.0 # stepsize for range arrays

fig = plt.figure()

ipw_x, ipw_num, ipw_ana = shoot_potwell(psi_init, h_,)
qho_x, qho_num, qho_ana = shoot_QuantumHarmonicOscillator(psi_init, h_)
hydro_x, hydro_num = shoot_hydrogen(psi_init, h_, 0)
hydro_x2p , hydro_num2p = shoot_hydrogen(psi_init, h_, 1)
hydro_ana1s = HYDRO_ana(hydro_x, 1, 0)
hydro_ana2s = HYDRO_ana(hydro_x, 2, 0)
hydro_ana3s = HYDRO_ana(hydro_x, 3, 0)
# print hydro_num
hydro_ana2p = HYDRO_ana(hydro_x, 2, 1)

print("IPW shooting")
plot_wavefunction(fig, "Infinte Potential Well -- Ground State", ipw_x, ipw_num[0, :], ipw_ana[0, :], [-0.1, 1.1, -0.2, 1.2])
plot_wavefunction(fig, "Infinte Potential Well -- First Excited State", ipw_x, ipw_num[1, :], ipw_ana[1, :], [-0.1, 1.1, -1.2, 1.2])
plot_wavefunction(fig, "Infinte Potential Well -- Second Excited State", ipw_x, ipw_num[2, :], ipw_ana[2, :], [-0.1, 1.1, -1.2, 1.2])
print("QHO shooting")
plot_wavefunction(fig, "Quantum Hamonic Oscillator -- Ground State", qho_x, qho_num[0, :], qho_ana[0, :], [-5.2, 5.2, -1.2, 1.2])
plot_wavefunction(fig, "Quantum Hamonic Oscillator -- First Excited State", qho_x, qho_num[1, :], qho_ana[1, :], [-5.2, 5.2, -1.2, 1.2])
plot_wavefunction (fig, "Quantum Hamonic Oscillator -- Second Excited State", qho_x, qho_num[2, :], qho_ana[2, :], [-5.2, 5.2, -1.2, 1.2])
print("Hydrogen Atom shooting")

plot_wavefunction(fig, "Hydrogen Atom -- 1s State", hydro_x, normalize(hydro_num[0, :]), hydro_ana1s, [-0.1, 30., -0.1, 1.2])
plot_wavefunction(fig, "Hydrogen Atom -- 2s State", hydro_x, normalize(hydro_num[1, :]), hydro_ana2s, [-0.1, 30., -2.2, 1.2])
plot_wavefunction(fig, "Hydrogen Atom -- 2p State", hydro_x2p, normalize(hydro_num2p[0, :]), hydro_ana2p, [-0.1, 30., -0.1, 1.2])
plot_wavefunction(fig, "Hydrogen Atom -- 3s State", hydro_x, normalize(hydro_num[2, :]), hydro_ana3s, [-0.1, 30., -1.2, 1.2])