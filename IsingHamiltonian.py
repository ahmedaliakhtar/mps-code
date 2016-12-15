###################### Build the Ising Hamiltonian ###############################
# This class constructs the Ising Hamiltonian given an array of interaction      #
# strengths J, an array of external magnetic field constants h, and N (d=2)      #
##################################################################################
import cmath
import numpy as np
from MPS_OBC import MPS_OBC
from MPO import MPO
from scipy.linalg import expm
import time

class IsingHamiltonian(object) :
    
    #Define all the relevant tensors
    A = np.zeros((3, 2, 2), dtype='complex')
    A[0][0][0] = 1.0
    A[0][1][1] = 1.0
    A[1][0][0] = 1.0
    A[1][1][1] = -1.0
    A[2][0][1] = 1.0
    A[2][1][0] = 1.0

    B = []
    B.append(np.zeros((3, 1, 3), dtype='complex'))
    B.append(np.zeros((3, 3, 3), dtype='complex'))
    B.append(np.zeros((3, 3, 1), dtype='complex'))

    #Define the left boundary tensor
    B[0][0][0][0] = 1.0
    B[0][1][0][1] = 1.0
    B[0][2][0][2] = 1.0

    #Define the middle tensors
    B[1][0][0][0] = 1.0
    B[1][0][2][2] = 1.0
    B[1][1][0][1] = 1.0
    B[1][1][1][2] = 1.0
    B[1][2][0][2] = 1.0

    #Define the end tensor
    B[2][0][2][0] = 1.0
    B[2][1][1][0] = 1.0
    B[2][2][0][0] = 1.0

    #create empty MPO object
    def __init__(self, N, J, h) :
        self.N = N
        self.J = np.copy(J)
        self.h = np.copy(h)
        self.buildMPO()

    def buildMPO(self) :
        tensors = []
        IsingHamiltonian.B[0][1][0][1] = np.lib.scimath.sqrt(self.J[0])
        IsingHamiltonian.B[0][2][0][2] = self.h[0]
        tensors.append(np.tensordot(IsingHamiltonian.A, IsingHamiltonian.B[0], ([0], [0])))

        for i in range(1, self.N - 1) :
            IsingHamiltonian.B[1][1][0][1] = np.lib.scimath.sqrt(self.J[i])
            IsingHamiltonian.B[1][1][1][2] = np.lib.scimath.sqrt(self.J[i - 1])
            IsingHamiltonian.B[1][2][0][2] = self.h[i]
            tensors.append(np.tensordot(IsingHamiltonian.A, IsingHamiltonian.B[1], ([0], [0])))

        IsingHamiltonian.B[2][1][1][0] = np.lib.scimath.sqrt(self.J[self.N - 2])
        IsingHamiltonian.B[2][2][0][0] = self.h[self.N - 1]
        tensors.append(np.tensordot(IsingHamiltonian.A, IsingHamiltonian.B[2], ([0], [0])))
        self.mpo = MPO()

        self.mpo.buildMPO(tensors)

        #reset values
        IsingHamiltonian.B[0][1][0][1] = 1.0
        IsingHamiltonian.B[0][2][0][2] = 1.0
        IsingHamiltonian.B[1][1][0][1] = 1.0
        IsingHamiltonian.B[1][1][1][2] = 1.0
        IsingHamiltonian.B[1][2][0][2] = 1.0
        IsingHamiltonian.B[2][1][1][0] = 1.0
        IsingHamiltonian.B[2][2][0][0] = 1.0

    def getEnergy(self, psi) :
        return self.mpo.overlapLeft(psi, psi).real

    def ImaginaryTimeEvolution(self, t, delta_t, psi, updateTimeStep = True) :

        #compute relevant MPOs
        Ueven, Uodd = self.TrotterSecondOrder(-1j * delta_t)
        energy, time = [], []

        energy.append(self.mpo.overlapLeft(psi, psi).real)
        time.append(0.0)

        #perform time evolution
        while time[len(time) - 1] < t :
            psi = MPO.compressMPOProduct(Ueven, psi, psi)
            psi = MPO.compressMPOProduct(Uodd, psi, psi)
            psi = MPO.compressMPOProduct(Ueven, psi, psi)

            psi.makeLeftNormalized()
            energy.append(self.mpo.overlapLeft(psi, psi).real)
            time.append(time[len(time) - 1] + delta_t)
            if updateTimeStep and (energy[len(time) - 2] - energy[len(time) - 1]) / delta_t < 1e-3 : 
                if delta_t < 1e-8 :
                    break
                else :
                    delta_t = delta_t / 10.0
                    Ueven, Uodd = self.TrotterSecondOrder(-1j * delta_t)

        return time, energy, psi

    #one evolution method to rule them all, because I have too many methods
    #that do similar things. Does Imaginary time evolution on default.
    def Evolution(self, t, delta_t, psi, updateTimeStep = True, imaginary = True, 
        writeToFile = True, computeEnergy = True, computeSingleCutEntropy = False, 
        cut_i = -1, computeAllSingleCutEntropies = False, computeDoubleCutEntropy = False, 
        cut_a = -1, cut_b = -1, filename = 'temp', thresh = 1e-5, fac = 2.0, min_dt = 1e-8) :

        def appendTensors() :
            if computeEnergy :
                energy.append(self.getEnergy(psi))
            if computeAllSingleCutEntropies :
                Smax, SList = psi.maximumEntropy(returnAllCuts = True)
                singleEE.append(SList)
            else :
                if computeSingleCutEntropy :
                    if cut_i < 0 or cut_i >= self.N - 1:
                        singleEE.append(psi.maximumEntropy())
                    else :
                        singleEE.append(psi.singleCutEntropy(cut_i))
            if computeDoubleCutEntropy :
                doubleEE.append(psi.doubleCutEntropy(cut_a, cut_b))

        #what to pass the TEO generator
        kt = delta_t
        if imaginary :
            kt = -1j * kt

        U = self.SecondOrderTrotterDecomp(kt)
        time, energy, singleEE, doubleEE = [], [], [], []

        time.append(0.0)
        appendTensors()

        #perform time evolution
        while time[len(time) - 1] < t :
            psi = MPO.compressMPOProduct(U, psi, psi)
            if imaginary :    
                psi.makeLeftNormalized()
            appendTensors()
            time.append(time[len(time) - 1] + delta_t)
            if updateTimeStep and (energy[len(energy) - 2] - energy[len(energy) - 1]) / delta_t < thresh :
                if delta_t < min_dt :
                    break
                else :
                    delta_t = delta_t / fac
                    U = self.SecondOrderTrotterDecomp(kt)

        if writeToFile :
            pointfile = open(filename + '_points.txt', 'w')
            pointfile.write('Time(t)')
            if computeEnergy :
                pointfile.write('\tEnergy')
            if computeSingleCutEntropy or computeAllSingleCutEntropies :
                pointfile.write('\t(Single_cut-)Entropy')
            if computeDoubleCutEntropy :
                pointfile.write('\t(Double_cut-)Entropy')
            pointfile.write('\n')
            for i in range(0, len(time)) :
                pointfile.write(str(time[i]))
                if computeEnergy :
                    pointfile.write('\t' + str(energy[i]))
                if computeAllSingleCutEntropies :
                    for el in singleEE[i] :
                        pointfile.write('\t' + str(el))
                else :
                    if computeSingleCutEntropy :
                        pointfile.write('\t' + str(singleEE[i]))
                if computeDoubleCutEntropy :
                    pointfile.write('\t' + str(doubleEE[i]))
                pointfile.write('\n')

            pointfile.close()
            
            psi.writeStateToFile(filename + '_final_state.txt')

        return psi

    #Construct the unnormalized Gibbs state as an MPO. D is the bond dimension.
    def GibbsState(self, beta, D, db = 0.001, returnMPS = False) :
        #reshape the identity into a vector
        tensors = []
        A = np.eye(2, dtype = 'complex').reshape((4, 1, 1))
        for i in range(0, self.N) :
            tensors.append(A)
        iv = MPS_OBC()
        iv.buildMPS(tensors)

        #increase bond dimension to D
        psi = MPS_OBC.increaseBondDim(iv, delta = D - 1)

        #construct a tensor which changes (dim 2 x dim 2) to (dim 4)
        dimchanger = MPO.dimChanger(2, 2)
        
        #construct the new TEO
        tensors1, tensors2 = [], []

        U = self.SecondOrderTrotterDecomp(-1j * db)

        for i in range(0, self.N) :
            A = np.tensordot(dimchanger, dimchanger, ([2], [2]))
            tensors1.append(np.tensordot(A, U.mpo[i], ([1, 3], [0, 1])))
            A = np.tensordot(dimchanger, dimchanger, ([1], [1]))
            tensors2.append(np.tensordot(A, U.mpo[i], ([1, 3], [1, 0])))

        TEO1 = MPO()
        TEO1.buildMPO(tensors1)

        TEO2 = MPO()
        TEO2.buildMPO(tensors2)

        #compute the Gibbs state
        for i in range(0, int(beta / db / 2)) :
            psi = MPO.compressMPOProduct(TEO1, psi, psi)
            psi = MPO.compressMPOProduct(TEO1, psi, psi)

        #return the final state
        if returnMPS :
            return psi

        tensors = []
        for i in range(0, self.N) :
            tensors.append(psi.mps[i].reshape((2, 2, psi.Dims[i], psi.Dims[i + 1])))

        gibbs = MPO()
        gibbs.buildMPO(tensors)

        return gibbs

    def NormalizedGibbsMPO(self, beta, D, db = 0.001, returnMPS = False) :
        g1 = self.GibbsState(beta / 2.0, D, db = db, returnMPS = True)
        g1.makeLeftNormalized()

        #reshape into an MPO
        tensors1 = []
        for i in range(0, self.N) :
            tensors1.append(g1.mps[i].reshape((2, 2, g1.Dims[i], g1.Dims[i + 1])))

        rho1 = MPO()
        rho1.buildMPO(tensors1)
        
        rho2 = rho1.adjoint()
        
        return MPO.product(rho2, rho1)

    #Compute the time evolution operators by writing explicitly Ueven, Uodd
    def TrotterSecondOrder(self, delta_t) :
        k = -1j * delta_t / 2.0

        eye = np.copy(IsingHamiltonian.A[0])
        sigma_z = np.copy(IsingHamiltonian.A[1])
        sigma_x = np.copy(IsingHamiltonian.A[2])

        even_tensors = []

        for i in range(0, self.N, 2) :
            if i < self.N - 1 :
                M = expm(k * self.J[i] * np.kron(sigma_z, sigma_z)
                    + k * self.h[i] / 2.0 * np.kron(sigma_x, eye)
                    + k * self.h[i + 1] / 2.0 * np.kron(eye, sigma_x))
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
                M = expm(k * self.h[i] / 2.0 * sigma_x)
                M = M.reshape((2, 2, 1, 1))
                even_tensors.append(M)

        odd_tensors = []
        k = -1j * delta_t

        M = expm(k * self.h[0] / 2.0 * sigma_x)
        M = M.reshape((2, 2, 1, 1))
        odd_tensors.append(M)

        for i in range(1, self.N, 2) :
            if i < self.N - 1 :
                M = expm(k * self.J[i] * np.kron(sigma_z, sigma_z)
                    + k * self.h[i] / 2.0 * np.kron(sigma_x, eye)
                    + k * self.h[i + 1] / 2.0 * np.kron(eye, sigma_x))
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
                M = expm(k * self.h[i] / 2.0 * sigma_x)
                M = M.reshape((2, 2, 1, 1))
                odd_tensors.append(M)

        Ueven = MPO()
        Ueven.buildMPO(even_tensors)
        Uodd = MPO()
        Uodd.buildMPO(odd_tensors)

        return Ueven, Uodd

###############################Extraneous Methods###############################           

    #To check the MPO representation we construct the exact representation
    def exactHamiltonian(self) :
        H = np.zeros((2 ** self.N, 2 ** self.N))
        for i in range(0, self.N - 1) :
            O1 = np.eye(2 ** i)
            O2 = np.diag([1.0, -1.0, -1.0, 1.0])
            O3 = np.eye(2 ** (self.N - i - 2))
            M = np.kron(O1, O2)
            M = np.kron(M, O3)
            H = H + M * self.J[i]

        for i in range(0, self.N) :
            O1 = np.eye(2 ** i)
            O2 = IsingHamiltonian.A[2]
            O3 = np.eye(2 ** (self.N - i - 1))
            M = np.kron(O1, O2)
            M = np.kron(M, O3)
            H = H + M * self.h[i]

        return H
    #Compute the imaginary time evolution of |psi0> under Ising Hamiltonian
    def ImaginaryTimeEvolution2(self, t, delta_t, psi, updateTimeStep = True, 
        thresh = 1e-5, fac = 2.0, min_dt = 1e-8, writeGSToFile = True) :
        #compute relevant MPOs
        U = self.SecondOrderTrotterDecomp(-1j * delta_t)
        energy, time_list = [], []

        energy.append(self.mpo.overlapLeft(psi, psi).real)
        time_list.append(0.0)

        #perform time evolution
        while time_list[len(time_list) - 1] < t :
            psi = MPO.compressMPOProduct(U, psi, psi)
            psi.makeLeftNormalized()
            energy.append(self.mpo.overlapLeft(psi, psi).real)
            time_list.append(time_list[len(time_list) - 1] + delta_t)
            if updateTimeStep and (energy[len(energy) - 2] - energy[len(energy) - 1]) / delta_t < thresh :
                if delta_t < min_dt :
                    break
                else :
                    delta_t = delta_t / fac
                    U = self.SecondOrderTrotterDecomp(-1j * delta_t)
        if writeGSToFile :
            psi.writeStateToFile('Plots/N=' + str(self.N) + '_J=' + str(self.J[0]) + '_h=' + str(self.h[0])
                                    + '_GS.txt')

        return time_list, energy, psi
    #Compute the imaginary time evolution of |psi0> under Ising Hamiltonian
    def ImaginaryTimeEvolution3(self, t, delta_t, psi, updateTimeStep = True) :
        #compute relevant MPOs
        U = self.SecondOrderTrotterDecomp2(-1j * delta_t)
        energy, time = [], []

        energy.append(self.mpo.overlapLeft(psi, psi).real)
        time.append(0.0)

        #perform time evolution
        while time[len(time) - 1] < t :
            psi = MPO.compressMPOProduct(U, psi, psi)
            psi.makeLeftNormalized()

            energy.append(self.mpo.overlapLeft(psi, psi).real)
            time.append(time[len(time) - 1] + delta_t)
            if updateTimeStep and (energy[len(time) - 2] - energy[len(time) - 1]) / delta_t < 1e-3 :
                if delta_t < 1e-8 :
                    break
                else :
                    delta_t = delta_t / 10.0
                    U = self.SecondOrderTrotterDecomp2(-1j * delta_t)

        return time, energy, psi

    #Compute the time evolution operators
    def SecondOrderTrotterDecomp(self, delta_t) :
        #first compute the component of the ising model related to the external b field
        k = -1j * delta_t / 2.0

        Ux_tensors = []
        for i in range(0, self.N) :
            X = np.zeros((2, 2, 1, 1), dtype = 'complex')
            X[0][0][0][0] = np.cosh(self.h[i] * k)
            X[0][1][0][0] = np.sinh(self.h[i] * k)
            X[1][0][0][0] = np.sinh(self.h[i] * k)
            X[1][1][0][0] = np.cosh(self.h[i] * k)
            Ux_tensors.append(X)

        Ux = MPO()
        Ux.buildMPO(Ux_tensors)

        #Now construct Hze and Hzo (the even and odd operators)
        #First the even terms of the interaction term in the Hamiltonian
        Uze_tensors = []
        Uzo_tensors = []
        identity = np.eye(2, dtype = 'complex').reshape((2, 2, 1, 1))

        #For the second order approximation
        k = -1j * delta_t

        #probably wanna be careful here
        for i in range(0, self.N, 2) : 
            if i < self.N - 1 :
                U = np.zeros((2, 2, 1, 2), dtype = 'complex')
                V = np.zeros((2, 2, 2, 1), dtype = 'complex')
                U[0][0][0][0] = np.cosh(self.J[i] * k)
                U[0][0][0][1] = 1
                U[1][1][0][0] = np.cosh(self.J[i] * k)
                U[1][1][0][1] = -1
                V[0][0][0][0] = 1
                V[0][0][1][0] = np.sinh(self.J[i] * k)
                V[1][1][0][0] = 1
                V[1][1][1][0] = -np.sinh(self.J[i] * k)
                Uze_tensors.append(U)
                Uze_tensors.append(V)
            else  : # we have an odd number of sites, last site should be I
                Uze_tensors.append(np.copy(identity))

        Uzo_tensors.append(np.copy(identity))
        for i in range(1, self.N, 2) :
            if i < self.N - 1 :
                U = np.zeros((2, 2, 1, 2), dtype = 'complex')
                V = np.zeros((2, 2, 2, 1), dtype = 'complex')
                U[0][0][0][0] = np.cosh(self.J[i] * k)
                U[0][0][0][1] = 1
                U[1][1][0][0] = np.cosh(self.J[i] * k)
                U[1][1][0][1] = -1
                V[0][0][0][0] = 1
                V[0][0][1][0] = np.sinh(self.J[i] * k)
                V[1][1][0][0] = 1
                V[1][1][1][0] = -np.sinh(self.J[i] * k)
                Uzo_tensors.append(U)
                Uzo_tensors.append(V)
            else :
                Uzo_tensors.append(np.copy(identity))
        
        Uze = MPO()
        Uze.buildMPO(Uze_tensors)
        Uzo = MPO()
        Uzo.buildMPO(Uzo_tensors)

        Uz = MPO.product(Uze, Uzo)
        U = MPO.product(Ux, MPO.product(Uz, Ux))

        return U

    #Compute the time evolution operators
    def SecondOrderTrotterDecomp2(self, delta_t) :
        #first compute the component of the ising model related to the external b field
        k = -1j * delta_t / 2.0

        Ux_tensors = []
        for i in range(0, self.N) :
            X = np.zeros((2, 2, 1, 1), dtype = 'complex')
            X[0][0][0][0] = np.cosh(self.h[i] * k)
            X[0][1][0][0] = np.sinh(self.h[i] * k)
            X[1][0][0][0] = np.sinh(self.h[i] * k)
            X[1][1][0][0] = np.cosh(self.h[i] * k)
            Ux_tensors.append(X)

        Ux = MPO()
        Ux.buildMPO(Ux_tensors)

        #Now construct Hze and Hzo (the even and odd operators)
        #First the even terms of the interaction term in the Hamiltonian
        Uze_tensors = []
        Uzo_tensors = []
        identity = np.eye(2, dtype = 'complex').reshape((2, 2, 1, 1))

        #For the second order approximation
        k = -1j * delta_t

        #probably wanna be careful here
        for i in range(0, self.N, 2) : 
            if i < self.N - 1 :
                C = []
                C.append(np.zeros((2, 1, 2), dtype = 'complex'))
                C.append(np.zeros((2, 2, 1), dtype = 'complex'))
                C[0][0][0][0] = cmath.sqrt(np.cosh(k * self.J[i]))
                C[0][1][0][1] = cmath.sqrt(np.sinh(k * self.J[i]))
                C[1][0][0][0] = cmath.sqrt(np.cosh(k * self.J[i]))
                C[1][1][1][0] = cmath.sqrt(np.sinh(k * self.J[i]))
                Uze_tensors.append(np.tensordot(IsingHamiltonian.A[0:2], C[0], ([0], [0])))
                Uze_tensors.append(np.tensordot(IsingHamiltonian.A[0:2], C[1], ([0], [0])))
            else  : # we have an odd number of sites, last site should be I
                Uze_tensors.append(np.copy(identity))

        Uzo_tensors.append(np.copy(identity))

        for i in range(1, self.N, 2) :
            if i < self.N - 1 :
                C = []
                C.append(np.zeros((2, 1, 2), dtype = 'complex'))
                C.append(np.zeros((2, 2, 1), dtype = 'complex'))
                C[0][0][0][0] = cmath.sqrt(np.cosh(k * self.J[i]))
                C[0][1][0][1] = cmath.sqrt(np.sinh(k * self.J[i]))
                C[1][0][0][0] = cmath.sqrt(np.cosh(k * self.J[i]))
                C[1][1][1][0] = cmath.sqrt(np.sinh(k * self.J[i]))
                Uzo_tensors.append(np.tensordot(IsingHamiltonian.A[0:2], C[0], ([0], [0])))
                Uzo_tensors.append(np.tensordot(IsingHamiltonian.A[0:2], C[1], ([0], [0])))
            else : #if we have an even number of sites
                Uzo_tensors.append(np.copy(identity))
        
        Uze = MPO()
        Uze.buildMPO(Uze_tensors)
        Uzo = MPO()
        Uzo.buildMPO(Uzo_tensors)

        Uz = MPO.product(Uze, Uzo)
        U = MPO.product(Ux, MPO.product(Uz, Ux))

        return U

    def testUnitaryGuysExactly(self, delta_t) :
        ozoz = np.kron(IsingHamiltonian.A[1], IsingHamiltonian.A[1])
        k = -1j * delta_t 

        Ueven = np.eye(1)
        for i in range(0, self.N, 2) :
            if i < self.N - 1 :
                Ueven = np.kron(Ueven, expm(k * self.J[i] * ozoz))
            else :
                Ueven = np.kron(Ueven, np.eye(2))

        Uodd = np.eye(2)
        for i in range(1, self.N, 2) :
            if i < self.N - 1 :
                Uodd = np.kron(Uodd, expm(k * self.J[i] * ozoz))
            else :
                Uodd = np.kron(Uodd, np.eye(2))

        Ux = np.eye(1)
        k = -1j * delta_t / 2.0
        for i in range(0, self.N) :
            Ux = np.kron(Ux, expm(k * self.h[i] * IsingHamiltonian.A[2]))

        return Ueven, Uodd, Ux

    def exactEvenOddOperators(self) :
        eye = np.copy(IsingHamiltonian.A[0])
        sigma_z = np.copy(IsingHamiltonian.A[1])
        sigma_x = np.copy(IsingHamiltonian.A[2])

        Heven = np.zeros((2 ** self.N, 2 ** self.N))
        Hodd = np.zeros((2 ** self.N, 2 ** self.N))

        for i in range(0, self.N, 2) :
            if i < self.N - 1 :
                O1 = np.eye(2 ** i)
                O2 = (self.J[i] * np.kron(sigma_z, sigma_z) 
                    + self.h[i] / 2.0 * np.kron(sigma_x, eye)
                    + self.h[i + 1] / 2.0 * np.kron(eye, sigma_x))
                O3 = np.eye(2 ** (self.N - i - 2))
                Heven = Heven + np.kron(O1, np.kron(O2, O3))
            else :
                O1 = np.eye(2 ** i)
                O2 = self.h[i] / 2.0 * sigma_x
                Heven = Heven + np.kron(O1, O2)

        Hodd = Hodd + np.kron(self.h[0] / 2.0 * sigma_x, np.eye(2 ** (self.N - 1)))
        for i in range(1, self.N, 2) :
            if i < self.N - 1 :
                O1 = np.eye(2 ** i)
                O2 = (self.J[i] * np.kron(sigma_z, sigma_z) 
                    + self.h[i] / 2.0 * np.kron(sigma_x, eye)
                    + self.h[i + 1] / 2.0 * np.kron(eye, sigma_x))
                O3 = np.eye(2 ** (self.N - i - 2))
                Hodd = Hodd + np.kron(O1, np.kron(O2, O3))
            else : 
                O1 = np.eye(2 ** i)
                O2 = self.h[i] / 2.0 * sigma_x
                Hodd = Hodd + np.kron(O1, O2)

        return Heven, Hodd
#Extraneous Testing

# N = 7
# Jfac = 1
# hfac = 1
# delta_t = 4
# J = np.ones(N) * Jfac
# h = np.ones(N) * hfac
# H = IsingHamiltonian(N, J, h)
# Ueven, Uodd = H.TrotterSecondOrder(-1j * delta_t)
# Heven, Hodd = H.exactEvenOddOperators()
# Htot = H.exactHamiltonian()

# #first check if Heven + Hodd = H
# print(sum(sum(np.absolute(Htot - Heven - Hodd))))

# #now check if Ueven is the correct exponential of Heven by comparing spectra
# w1, v1 = np.linalg.eig(expm(-delta_t / 2.0 * Heven))
# w2, v2 = np.linalg.eig(Ueven.matrix())
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# #same for Hodd
# w1, v1 = np.linalg.eig(expm(-delta_t * Hodd))
# w2, v2 = np.linalg.eig(Uodd.matrix())
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# #compare the product
# U1 = np.dot(Ueven.matrix(), np.dot(Uodd.matrix(), Ueven.matrix()))
# U2 = expm(-delta_t * Htot)
# w1, v1 = np.linalg.eig(U1)
# w2, v2 = np.linalg.eig(U2)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# # Do the same thing for the other implementations
# Uze, Uzo, Ux = H.SecondOrderTrotterDecomp(-1j * delta_t)
# Uze_exact, Uzo_exact, Ux_exact = H.testUnitaryGuysExactly(-1j * delta_t)

# w1, v1 = np.linalg.eig(Uze.matrix())
# w2, v2 = np.linalg.eig(Uze_exact)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# w1, v1 = np.linalg.eig(Uzo.matrix())
# w2, v2 = np.linalg.eig(Uzo_exact)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# w1, v1 = np.linalg.eig(Ux.matrix())
# w2, v2 = np.linalg.eig(Ux_exact)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# Uze, Uzo, Ux = H.SecondOrderTrotterDecomp2(-1j * delta_t)

# w1, v1 = np.linalg.eig(Uze.matrix())
# w2, v2 = np.linalg.eig(Uze_exact)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# w1, v1 = np.linalg.eig(Uzo.matrix())
# w2, v2 = np.linalg.eig(Uzo_exact)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))

# w1, v1 = np.linalg.eig(Ux.matrix())
# w2, v2 = np.linalg.eig(Ux_exact)
# print(sum(np.absolute(np.sort(w1) - np.sort(w2))))
N = 10
Beta = 5.0
Jfac = 1.0
hfac = 1.0

J = Jfac * np.ones(N)
h = hfac * np.ones(N)

H = IsingHamiltonian(N, J, h)
#print(H.mpo.trace())
Hmat = H.mpo.matrix()
#print(np.trace(Hmat))

gibbs1 = H.NormalizedGibbsMPO(Beta, 10).matrix()
gibbs2 = expm(-Beta * Hmat)
gibbs2 = gibbs2 / np.trace(gibbs2)

s1, s2 = np.sort(np.linalg.eigvals(gibbs1)), np.sort(np.linalg.eigvals(gibbs2))
print(s1)
print(s2)
s3 = s1 - s2
#print(s3)

print(np.dot(np.conjugate(s3), s3) / (2 ** N))