###################### Build the Heisenberg Hamiltonian ##########################
# This class constructs the Heisenberg Hamiltonian given an array of interaction #
# strengths J, an external magnetic field constants h, and N (d=2)               #
##################################################################################

import cmath
import numpy as np
from MPS_OBC import MPS_OBC
from MPO import MPO
from scipy.linalg import expm
import time

class HeisenbergHamiltonian(object) :

    #Define all the relevant tensors
    A = np.zeros((4, 2, 2), dtype='complex')
    # I
    A[0][0][0] = 1.0
    A[0][1][1] = 1.0
    # sigma_x
    A[1][0][1] = 1.0
    A[1][1][0] = 1.0
    # sigma_y
    A[2][0][1] = -1j
    A[2][1][0] = 1j
    # sigma_z
    A[3][0][0] = 1.0
    A[3][1][1] = -1.0

    B = []
    B.append(np.zeros((4, 1, 5), dtype = 'complex'))
    B.append(np.zeros((4, 5, 5), dtype = 'complex'))
    B.append(np.zeros((4, 5, 1), dtype = 'complex'))

    #constructor
    def __init__(self, N, Jx, Jy, Jz, h) :
        self.N = N
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.h = h

        self.buildMPO()

    def buildMPO(self) :
        tensors = []

        #arrays that contain the appropriate factors
        kx = np.lib.scimath.sqrt(-0.5 * self.Jx)
        ky = np.lib.scimath.sqrt(-0.5 * self.Jy)
        kz = np.lib.scimath.sqrt(-0.5 * self.Jz)

        #construct B tensors
        HeisenbergHamiltonian.B[0][0][0][0] = 1.0
        HeisenbergHamiltonian.B[0][1][0][1] = kx[0]
        HeisenbergHamiltonian.B[0][2][0][2] = ky[0]
        HeisenbergHamiltonian.B[0][3][0][3] = kz[0]
        HeisenbergHamiltonian.B[0][3][0][4] = -0.5 * h[0]

        tensors.append(np.tensordot(HeisenbergHamiltonian.A, 
            HeisenbergHamiltonian.B[0], ([0], [0])))

        HeisenbergHamiltonian.B[1][0][0][0] = 1.0
        HeisenbergHamiltonian.B[1][0][4][4] = 1.0

        for i in range(1, self.N - 1) :
            HeisenbergHamiltonian.B[1][1][0][1] = kx[i]
            HeisenbergHamiltonian.B[1][1][1][4] = kx[i - 1]
            HeisenbergHamiltonian.B[1][2][0][2] = ky[i]
            HeisenbergHamiltonian.B[1][2][2][4] = ky[i - 1]
            HeisenbergHamiltonian.B[1][3][0][3] = kz[i]
            HeisenbergHamiltonian.B[1][3][3][4] = kz[i - 1]
            HeisenbergHamiltonian.B[1][3][0][4] = -0.5 * h[i]
            tensors.append(np.tensordot(HeisenbergHamiltonian.A, 
                HeisenbergHamiltonian.B[1], ([0], [0])))

        HeisenbergHamiltonian.B[2][0][4][0] = 1.0
        HeisenbergHamiltonian.B[2][1][1][0] = kx[self.N - 2]
        HeisenbergHamiltonian.B[2][2][2][0] = ky[self.N - 2]
        HeisenbergHamiltonian.B[2][3][3][0] = kz[self.N - 2]
        HeisenbergHamiltonian.B[2][3][0][0] = -0.5 * h[self.N - 1]

        tensors.append(np.tensordot(HeisenbergHamiltonian.A,
            HeisenbergHamiltonian.B[2], ([0], [0])))

        self.mpo = MPO()
        self.mpo.buildMPO(tensors)

    #return <psi|H|psi>
    def getEnergy(self, psi) :
        return self.mpo.overlapLeft(psi, psi).real

    #compute H exactly as a matrix
    def exactMatrix(self) :
        sxsx = np.kron(HeisenbergHamiltonian.A[1], HeisenbergHamiltonian.A[1])
        sysy = np.kron(HeisenbergHamiltonian.A[2], HeisenbergHamiltonian.A[2])
        szsz = np.kron(HeisenbergHamiltonian.A[3], HeisenbergHamiltonian.A[3])
        szid = np.kron(HeisenbergHamiltonian.A[3], HeisenbergHamiltonian.A[0])

        H = np.zeros((2 ** self.N, 2 ** self.N), dtype = 'complex')

        for i in range(0, self.N - 1) :
            left_id = np.eye(2 ** i)
            mid = (-0.5 * self.Jx[i] * sxsx + -0.5 * self.Jy[i] * sysy 
                + -0.5 * self.Jz[i] * szsz + -0.5 * self.h[i] * szid)
            right_id = np.eye(2 ** (self.N - i - 2))
            H += np.kron(np.kron(left_id, mid), right_id)

        H += np.kron(np.eye(2 ** (self.N - 1)), 
            -0.5 * self.h[self.N - 1] * HeisenbergHamiltonian.A[3])

        return H

    #compute explicitly the even / odd operator
    def EvenTimeEvolutionOperator(self, delta_t, imaginaryTime = False) :
        k = -delta_t if imaginaryTime else -1j * delta_t

        even_tensors = []
        sxsx = np.kron(HeisenbergHamiltonian.A[1], HeisenbergHamiltonian.A[1])
        sysy = np.kron(HeisenbergHamiltonian.A[2], HeisenbergHamiltonian.A[2])
        szsz = np.kron(HeisenbergHamiltonian.A[3], HeisenbergHamiltonian.A[3])
        szid = np.kron(HeisenbergHamiltonian.A[3], HeisenbergHamiltonian.A[0])
        idsz = np.kron(HeisenbergHamiltonian.A[0], HeisenbergHamiltonian.A[3])

        for i in range(0, self.N, 2) :
            if i < self.N - 1 :
                mid = (-0.5 * self.Jx[i] * sxsx + -0.5 * self.Jy[i] * sysy 
                    + -0.5 * self.Jz[i] * szsz + -0.25 * self.h[i] * szid 
                    + -0.25 * self.h[i + 1] * idsz)
                M = expm(k * mid)
                M = M.reshape((2, 2, 2, 2))
                M = np.transpose(M, (0, 2, 1, 3))
                M = M.reshape((4, 4))
                Q, R = np.linalg.qr(M)
                s = len(Q[0])
                Q = Q.reshape((2, 2, 1, s))
                R = np.transpose(R, (1, 0)).reshape((2, 2, s, 1))
                even_tensors.append(Q)
                even_tensors.append(R)
            else : #Odd number of sites
                mid = -0.25 * self.h[i] * HeisenbergHamiltonian.A[3]
                even_tensors.append(expm(k * mid).reshape((2, 2, 1, 1)))

        Ueven = MPO()
        Ueven.buildMPO(even_tensors)
        return Ueven

    def OddTimeEvolutionOperator(self, delta_t, imaginaryTime = False) :        
        k = -delta_t if imaginaryTime else -1j * delta_t

        odd_tensors = []

        mid = -0.25 * self.h[0] * HeisenbergHamiltonian.A[3]
        odd_tensors.append(expm(k * mid).reshape((2, 2, 1, 1)))

        for i in range(1, self.N, 2) :
            if i < self.N - 1 :
                mid = (-0.5 * self.Jx[i] * sxsx + -0.5 * self.Jy[i] * sysy 
                    + -0.5 * self.Jz[i] * szsz + -0.25 * self.h[i] * szid 
                    + -0.25 * self.h[i + 1] * idsz)
                M = expm(k * mid)
                M = M.reshape((2, 2, 2, 2))
                M = np.transpose(M, (0, 2, 1, 3))
                M = M.reshape((4, 4))
                Q, R = np.linalg.qr(M)
                s = len(Q[0])
                Q = Q.reshape((2, 2, 1, s))
                R = np.transpose(R, (1, 0)).reshape((2, 2, s, 1))
                odd_tensors.append(Q)
                odd_tensors.append(R)
            else : #even number of sites
                mid = -0.25 * self.h[i] * HeisenbergHamiltonian.A[3]
                odd_tensors.append(expm(k * mid).reshape((2, 2, 1, 1)))

        Uodd = MPO()
        Uodd.buildMPO(odd_tensors)

        return Uodd

N = 5
Jx = np.random.randn(N - 1) 
Jy = np.random.randn(N - 1)
Jz = np.random.randn(N - 1)
h = np.random.randn(N)

A = HeisenbergHamiltonian(N, Jx, Jy, Jz, h)
H1 = A.mpo.matrix()
H2 = A.exactMatrix()

w1 = np.linalg.eigvalsh(H1)
w2 = np.linalg.eigvalsh(H2)

w1, w2 = np.sort(w1), np.sort(w2)
print(w1)
print(w2)

w3 = w1 - w2
print(np.dot(np.conjugate(w3), w3))