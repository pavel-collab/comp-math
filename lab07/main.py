import numpy as np
import matplotlib.pyplot as plt
import SlaeAPI

A = 1e3
# A = 1e6
B = 0.5
C = 1

def f1(y1, y2):
    return A * (-(y1**3 / 3 - y1) + y2)

def f2(y1, y2):
    return -y1 -B*y2 + C

def F(y):
    y1 = y[0]
    y2 = y[1]
    return np.array([
        f1(y1, y2),
        f2(y1, y2)
    ])

# коэффициенты для трехстадийного метода Розенброка
a  = 0.435866521508459
p1 = 0.435866521508459
p2 = 0.4782408332745185
p3 = 0.0858926452170225 
beta21 = 0.435866521508459
beta31 = 0.435866521508459
beta32 = -2.116053335949811

def J(yn, h):
    y1n = yn[0]
    y2n = yn[1]

    df1dx1 = (f1(y1n + h, y2n) - f1(y1n - h, y2n)) / (2*h)
    df1dx2 = (f1(y1n, y2n + h) - f1(y1n, y2n - h)) / (2*h)
    df2dx1 = (f2(y1n + h, y2n) - f2(y1n - h, y2n)) / (2*h)
    df2dx2 = (f2(y1n, y2n + h) - f2(y1n, y2n - h)) / (2*h)

    res = np.array([
        [df1dx1, df1dx2],
        [df2dx1, df2dx2]
    ])
    return res

def Dn(yn, h):
    E = np.eye(2)
    return E + a * h * J(yn, h)

def Rozenbrok3Mthd(f, h, t_start, t_end, initial_solution: tuple):
    t = np.linspace(t_start, t_end, int((t_end-t_start)/h+1))
    solution = np.zeros((int((t_end-t_start)/h+1), 2))
    solution[0] = initial_solution
    
    for i in range(len(t)-1):
        # составляем СЛАУ
        slae1 = SlaeAPI.Slae(Dn(solution[i], h), h*f(solution[i]))
        #TODO np.linalg.solve(A, f) try
        k1 = slae1.Gauss_mthd()
        slae2 = SlaeAPI.Slae(Dn(solution[i], h), h*f(solution[i] + beta21*k1))
        k2 = slae2.Gauss_mthd()
        slae3 = SlaeAPI.Slae(Dn(solution[i], h), h*f(solution[i] + beta31*k1 + beta32*k2))
        k3 = slae3.Gauss_mthd()
        # получаем y_{n+1}
        solution[i+1] = solution[i] + p1*k1 + p2*k2 + p3*k3
    return solution

def main():
    h = 0.001
    init_sol = np.array([2, 0])
    t_start = 0
    t_end = 100

    sol = Rozenbrok3Mthd(F, h, t_start, t_end, init_sol)
    y1, y2 = sol.T
    t = np.linspace(t_start, t_end, int((t_end-t_start)/h+1))

    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t, y2)
    plt.title('Решение уравнения')
    plt.xlabel('t')
    plt.ylabel('y2')

    plt.subplot(1, 2, 2)
    plt.plot(y1, y2)
    plt.title('Фазовая траектория')
    plt.xlabel('y1')
    plt.ylabel('y2')
    fig.savefig("./image.jpg")

if __name__ == '__main__':
    main()