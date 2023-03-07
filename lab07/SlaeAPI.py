import numpy as np
import math
import copy
import numba

def fst_vec_norm(x: np.ndarray):
    return max(abs(x))

def scd_vec_norm(x: np.ndarray):
    return sum(abs(x))

def trd_vec_norm(x: np.ndarray):
    return math.sqrt(np.dot(x, x))

def fst_m_norm(A: np.ndarray):
    assert(A.shape[0] == A.shape[1])
    return max([sum(abs(A[i])) for i in range(A.shape[0])])

def scd_m_norm(A: np.ndarray):
    assert(A.shape[0] == A.shape[1])
    return max([sum(abs(A.T[i])) for i in range(A.T.shape[0])])

# поскольку работаем в R, эрмитово сопряжение эквивалентвно транспонированию
def trd_m_norm(A: np.ndarray):
    B = np.dot(A.T, A)
    num, _ =  np.linalg.eigh(B)
    print(num)
    return math.sqrt(max(num))


# this is a round parametr. We need it only for visualisation system (overloar __str__)
# it doesn't influence on the computations
round_n = 3

class Slae:
    def __init__(self, matrix: np.ndarray, values: np.ndarray):

        # assert(matrix.shape[0] == matrix.shape[1])
        # assert(matrix.shape[0] == values.shape[0])
        
        self.A = matrix
        self.f = values

    #================================================Приватные методы и декараторы================================================#

    @property
    def dimention(self):
        return self.A.shape[0]

    def __CheckSymmetric(self, tol=1e-16):
        return not False in (np.abs(self.A-self.A.T) < tol)

    def __IsLU_compatible(self):
        N = self.dimention
        A = self.A.astype(float, copy=True)

        for i in range(1, N+1):
            M = A[:i, :i]

            if np.linalg.det(M) == 0:
                return False

        return True

    def __SylvesterCriterion(self):
        N = self.dimention
        A = self.A.astype(float, copy=True)

        for i in range(1, N+1):
            M = A[:i, :i]

            if np.linalg.det(M) < 0:
                return False

        return True

    def __LU_decomposition(self):
        N = self.dimention
        A = self.A.astype(float, copy=True)

        """Decompose matrix of coefficients to L and U matrices.
        L and U triangular matrices will be represented in a single nxn matrix.
        :param a: numpy matrix of coefficients
        :return: numpy LU matrix
        """
        # create emtpy LU-matrix
        lu_matrix = np.matrix(np.zeros([N, N]))

        for k in range(N):
            # calculate all residual k-row elements
            for j in range(k, N):
                lu_matrix[k, j] = A[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
            # calculate all residual k-column elemetns
            for i in range(k + 1, N):
                lu_matrix[i, k] = (A[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

        """Get triangular L matrix from a single LU-matrix
        :param m: numpy LU-matrix
        :return: numpy triangular L matrix
        """
        L = lu_matrix.copy()
        for i in range(L.shape[0]):
                L[i, i] = 1
                L[i, i+1 :] = 0

        """Get triangular U matrix from a single LU-matrix
        :param m: numpy LU-matrix
        :return: numpy triangular U matrix
        """
        U = lu_matrix.copy()
        for i in range(1, U.shape[0]):
            U[i, :i] = 0
        
        return L, U
    #==================================================Численные методы==================================================#

    def Gauss_mthd(self):

        '''
        Here is some explonation why do we do what we do.

        Firstly, in the begining some matrices could have type int, but then we do some operations that can change their type.
        So, we change their type right in the top of method, to avoid errors.

        Secondly, when we write
        a = b
        python use links default. This mean -- if we modify object a, the object b is modified too.
        So, to save invariant of self.A we copy it in every method.
        '''
        A = self.A.astype(float, copy=True)
        f = self.f.astype(float, copy=True)
        N = self.dimention

        # straight
        for k in range(N):
            for m in range(k+1, N):

                alpha = A[m][k] / A[k][k]

                f[m] = f[m] - f[k] * alpha 
                for i in range(k, N):
                    A[m][i] = A[m][i] - A[k][i] * alpha

        # reverce
        solution = np.full((N, ), 0.0)
        
        # as indexes in python start from 0 and finish n-1, the last equation has index n-1
        solution[N-1] = f[N-1] / A[N-1][N-1]

        # the second from tail equation has index n-2
        # as function range() returns semi-open interval, the second parametr is -1 instead of 0
        for k in range(N-2, 0-1, -1):
            solution[k] = 1 / A[k][k] * (f[k] - np.dot(A[k], solution))

        return solution


    def LU_mthd(self):

        if self.__IsLU_compatible() == False:
            print('[-] Error. Sorry, this problem could not be solved by LU method')
            return None

        A = self.A.astype(float, copy=True)
        f = self.f.astype(float, copy=True)
        N = self.dimention

        L, U = self.__LU_decomposition()

        solution_level1 = np.full((N, ), 0.0)
        solution_level2 = np.full((N, ), 0.0)

        solution_level1[0] = f[0] / L[0, 0]

        for i in range(1, N):
            solution_level1[i] = 1 / L[i, i] * (f[i] - np.dot(L[i], solution_level1))

        solution_level2[N-1] = solution_level1[N-1] / U[N-1, N-1]

        
        for k in range(N-2, 0-1, -1):
            solution_level2[k] = 1 / U[k, k] * (solution_level1[k] - np.dot(U[k], solution_level2))

        return solution_level2


    def Cholesky_mthd(self):
        
        if self.__CheckSymmetric() or self.__SylvesterCriterion:
            print('[-] Error. Sorry, this problem could not be solved by Cholesky method')
            return None

        A = self.A.astype(float, copy=True)
        f = self.f.astype(float, copy=True)
        N = self.dimention

        L = np.zeros([N, N])

        for j in range(0, N):
            LSum = 0.0
            for k in range(0, j):
                LSum += L[j, k] * L[j, k]

            L[j, j] = np.sqrt(A[j, j] - LSum)

            for i in range(j + 1, N):
                LSum = 0.0
                for k in range(0, j):
                    LSum += L[i, k] * L[j, k]
                L[i][j] = (1.0 / L[j, j] * (A[i, j] - LSum))
        
        solution_level1 = np.full((N, ), 0.0)
        solution_level2 = np.full((N, ), 0.0)

        solution_level1[0] = f[0] / L[0, 0]

        U = L.T

        for i in range(1, N):
            solution_level1[i] = 1 / L[i, i] * (self.f[i] - np.dot(L[i], solution_level1))

        solution_level2[N-1] = solution_level1[N-1] / U[N-1, N-1]

        
        for k in range(N-2, 0-1, -1):
            solution_level2[k] = 1 / U[k, k] * (solution_level1[k] - np.dot(U[k], solution_level2))

        return solution_level2
        
    
    def UpperRelaxation(self, w=1.5, UR=True):
        '''
        What is the UR?
        As you can see, the Seidel is a particular case of Upper Relaxation. So, we use UpperRelaxation() with w=1
        to Seidel_mthd. But in the classic Upper Relaxation w must be in (1, 2), so we need to add verefication
        in UpperRelaxation function.
        To avoid errors in Seidel_mthd (with w=1) we add one more function field UR that is True if we in the classic
        Upper Relaxation method (we want to vereficate 1 < w < 2) and False in other method that use UpperRelaxation as base.
        '''    
        if (not (1 < w < 2)) and UR:
            print('[-] Error. In the Upper relaxation method wight w must be in 1 < w < 2\n'
                  'Change the wight and try again.')
            return None

        # accuracy
        eps = 1e-6  

        A = self.A.astype(float, copy=True)
        f = self.f.astype(float, copy=True)     
        N = self.dimention

        D = np.eye(N) * np.diag(A)
        U = np.triu(A) - D
        L = np.tril(A) - D

        B = np.dot(np.linalg.inv(L*w + D), D*(w - 1) + U*w)
        F = np.linalg.inv(L*w + D)

        solution_prev = np.full((N, ), 0.0)
        solution_cur = np.full((N, ), 0.0)

        while(trd_vec_norm(f - np.dot(A, solution_cur)) > eps):
            solution_prev = solution_cur
            solution_cur = - np.dot(B, solution_prev) + np.dot(F*w, f)

        return solution_cur

    def Seidel_mthd(self):
        res = self.UpperRelaxation(w=1, UR=False)
        return res

    #==================================================Другие функции класса==================================================#

    ## overloading output
    def __str__(self):
        n = self.dimention

        res = ''
        for i in range(n):
            string = ''
            for j in range(n):
                string = string + str(round(self.A[i][j], round_n)) + ' x{}'.format(j + 1)
                # string = string + str(self.A[i][j]) + ' x{}'.format(j + 1)
                if j != n - 1:
                    string = string + ' + '
                else:
                    string = string + ' = ' + str(round(self.f[i], round_n))
                    # string = string + ' = ' + str(self.f[i])
            string = string + '\n'
            res = res + string

        return res