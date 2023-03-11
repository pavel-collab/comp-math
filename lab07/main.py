import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import ode

y1, y2 =  smp.Symbol('y[0]'), smp.Symbol('y[1]')
a, b, c = smp.Symbol('a'), smp.Symbol('b'), smp.Symbol('c')

f1 = a*(-(y1**3/3 - y1) + y2)
f2 = -y1 - b*y2 + c

Y = smp.Matrix([f1, f2])
X = smp.Matrix([y1, y2])

# print('f1 =', f1)
# print('f2 =', f2)
# print('jacobian = ', Y.jacobian(X))

def f(t, y, a, b, c):
  return [a*(y[0] + y[1] - y[0]**3/3),
          c - y[0] - b*y[1]]

def jac(t, y, a, b, c):
  return [[a*(1 - y[0]**2),  a],
          [             -1, -b]]

y01, y02 = 2., 0.
a, b, c = 1000., 0.5, 1.0

r = ode(f, jac).set_integrator('vode', method='bdf', with_jacobian=True)
r.set_initial_value((y01, y02)).set_f_params(a, b, c).set_jac_params(a, b, c)

t1, dt, t, y1, y2 = 100, 0.0001, [0.], [y01], [y02]
while r.successful() and r.t < t1:
    r.integrate(r.t + dt)
    t.append(r.t)
    y1.append(r.y[0])
    y2.append(r.y[1])

fig = plt.figure(figsize=(13, 18))

splt1 = plt.subplot(3, 1, 1)
plt.title("$y^{\prime}_1 = a (- (\\frac{y_1^3}{3} - y_1 ) + y_2),\, y^{\prime}_2 = - y_1 - b y_2 + c,\, a = %.2f,\, b = %.2f,\, c = %.2f,\, y1(0) = %.2f, \,y2(0)=%.2f$" % (a, b, c, y01, y02))

plt.ylabel(r'$y_1$', {'fontsize': 18})
plt.xlabel(r'$t$', {'fontsize': 18})
plt.plot(t, y1)
plt.grid(True)

splt2 = plt.subplot(3, 1, 2)
plt.ylabel(r'$y_2$', {'fontsize': 18})
plt.xlabel(r'$t$', {'fontsize': 18})
plt.grid(True)
plt.plot(t, y2)

splt3 = plt.subplot(3, 1, 3)
plt.ylabel(r'$y_2$', {'fontsize': 18})
plt.xlabel(r'$y_1$', {'fontsize': 18})
plt.grid(True)
plt.plot(y1, y2)

plt.show()