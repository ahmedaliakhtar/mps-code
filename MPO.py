###################### Data structure for an MPO ###############################
#   Takes as input several parameters: N (number of sites),                    #
#   D (bond dimension), d (physical dimension), and a file containing matrices #
################################################################################

import numpy as np
import sys
from MPS_OBC import MPS_OBC
import time

class MPO(object) :

    #create empty MPO object
    def __init__(self) :
        self.mpo = None
        self.N = None
        self.d = None
        self.D = None
        self.Dims = None

    #Put data from the file into a tensor that contains all the matrices
    def buildMPOFromFile(self, N, d, file) :
        self.N = N
        self.d = d

        #make an array of dimensions
        self.Dims = np.zeros(N + 1)

        mpo = []

        f = open(file)
        lines = f.readlines()

        #Create empty tensors of the right dimensions
        for i in range(0, N) :
            line = lines[i * d * d].split()
            if line.count(':') == 0 :
                mpo.append(np.zeros((d, d, 1, len(line)), dtype = 'complex'))
                self.Dims[i] = 1
                self.Dims[i + 1] = len(line)
            else :
                c = line.index(':')
                r = line.count(':') + 1
                mpo.append(np.zeros((d, d, r, c), dtype = 'complex'))
                self.Dims[i] = r
                self.Dims[i + 1] = c

        #Fill the tensors
        for site in range(0, N) :
            for state_i in range(0, d) :
                for state_j in range(0, d) :
                    place = (site * d * d) + (d * state_i) + state_j
                    elements = lines[place].split()

                    r = int(self.Dims[site])
                    c = int(self.Dims[site + 1])

                    #remove the colons
                    for s in range(1, r) :
                        elements.remove(':')

                    #enter matrix elements
                    for i in range(0, r) :
                        for j in range(0, c) :
                            mpo[site][state_i][state_j][i][j] = complex(elements[i * c + j])

        self.mpo = mpo 
        self.D = max(self.Dims)

    #Write MPO to file
    def writeMPOToFile(self, filename) :
        file = open(filename, 'w')
        for tensor in self.mpo :
            for di in range(0, len(tensor)) :
                for dj in range(0, len(tensor[0])) :
                    for r in range(0, len(tensor[0][0])) :
                        for c in range(0, len(tensor[0][0][0])) :
                            file.write(str(tensor[di][dj][r][c]) + ' ')
                        if r < len(tensor[0][0]) - 1 :
                            file.write(' : ')
                    file.write('\n')
        file.close()

    #generate a random MPO with length N, physical dimension d, and bond dimension D
    def buildRandomMPO(self, N, d, D) :
        self.N = N
        self.d = d
        self.D = D
        #make array of dimensions so that you know what sized matrices to generate
        self.Dims = np.ones(N + 1)

        for i in range(1, N) :
            if i < int(N / 2.0) :
                self.Dims[i] = min(d * self.Dims[i - 1], D)
            else :
                if i == int(N / 2.0) or i == int(N / 2.0 + 0.5) :
                    self.Dims[i] = D
                else :
                    self.Dims[i] = self.Dims[N - i]

        self.mpo = []
        for n in range(0, N) :
            r = int(self.Dims[n])
            c = int(self.Dims[n + 1])
            tensor = np.zeros((d, d, r, c), dtype='complex')
            for dim1 in range(0, d) :
                for dim2 in range(0, d) :
                    for i in range(0, r) :
                        for j in range(0, c) :
                            tensor[dim1][dim2][i][j] = complex(np.random.random_sample(), np.random.random_sample())

            self.mpo.append(tensor)

    #generate MPO given a list of tensors
    def buildMPO(self, mpo) :
        self.N = len(mpo)
        self.d = len(mpo[0])

        #create list of dimensions, and determine D
        self.Dims = np.ones(self.N + 1)
        for n in range(0, self.N) :
            self.Dims[n] = len(mpo[n][0][0])
            self.Dims[n + 1] = len(mpo[n][0][0][0])

        self.D = max(self.Dims)

        self.mpo = []
        for thing in mpo :
            self.mpo.append(np.copy(thing))

    #generate the identity MPO
    @staticmethod
    def eye(N, d) :
        mpo = []
        I = np.zeros((d, d, 1, 1), dtype = 'complex')
        for i in range(0, d) :
            I[i][i][0][0] = 1

        for n in range(0, N) :
            mpo.append(np.copy(I))

        ide = MPO()
        ide.buildMPO(mpo)
        return ide

    #returns a copy of the current instance
    def copy(self) :
        tensors = []
        for tensor in self.mpo :
            tensors.append(tensor)

        copie = MPO()
        copie.buildMPO(tensors)
        return copie

    #return the adjoint of the current mpo
    def adjoint(self) :
        new_tensors = []
        for n in range(0, self.N) :
            new_tensors.append(np.conjugate(np.transpose(self.mpo[n], (1, 0, 2, 3))))

        new_mpo = MPO()
        new_mpo.buildMPO(new_tensors)
        return new_mpo

    #Return O|Psi> where |Psi> is an MPS (refer to Schollwock for procedure)
    def actOn(self, Psi) :
        #compute the action of O on |Psi> by contracting tensors
        #define a new set of tensors
        new_mps = []
        for n in range(0, self.N) :
            new_mps.append(np.tensordot(self.mpo[n], Psi.mps[n], ([1], [0])))

        #reshape tensors
        for n in range(0, self.N) :
            new_mps[n] = np.transpose(new_mps[n], (0, 1, 3, 2, 4))
            new_mps[n] = new_mps[n].reshape((self.d, self.Dims[n] * Psi.Dims[n], self.Dims[n + 1] * Psi.Dims[n + 1]))

        #Create arbitrary MPS. We will rewrite its values.
        new_Psi = MPS_OBC()
        new_Psi.buildMPS(new_mps)
        return new_Psi

    #computes <v1|O|v2> going from left to right
    def overlapLeft(self, v1, v2) : 
        E0 = np.tensordot(np.conjugate(v1.mps[0]), self.mpo[0], ([0], [0]))
        E0 = np.tensordot(E0, v2.mps[0], ([2], [0]))
        E0 = np.transpose(E0, (0, 2, 4, 1, 3, 5))
        E0 = E0[0][0][0]


        for n in range(1, self.N) :
            E0 = np.tensordot(E0, np.conjugate(v1.mps[n]), ([0], [1]))
            E0 = np.tensordot(E0, v2.mps[n], ([1], [1]))
            E0 = np.tensordot(E0, self.mpo[n], ([0, 1, 3], [2, 0, 1]))
            E0 = np.transpose(E0, (0, 2, 1))

        return E0[0][0][0]

    #computes <v1|O|v2> going right to left
    def overlapRight(self, v1, v2) :
        E0 = np.tensordot(np.conjugate(v1.mps[self.N - 1]), self.mpo[self.N - 1], ([0], [0]))
        E0 = np.tensordot(E0, v2.mps[self.N - 1], ([2], [0]))
        E0 = np.transpose(E0, (1, 3, 5, 0, 2, 4))
        E0 = E0[0][0][0]

        for n in reversed(range(0, self.N - 1)) :
            E0 = np.tensordot(E0, np.conjugate(v1.mps[n]), ([0], [2]))
            E0 = np.tensordot(E0, v2.mps[n], ([1], [2]))
            E0 = np.tensordot(E0, self.mpo[n], ([0, 1, 3], [3, 0, 1]))
            E0 = np.transpose(E0, (0, 2, 1))

        return E0[0][0][0]        

    #returns a compressed MPS of the product MPO|MPS> given a guess MPS |PSI>
    @staticmethod
    def compressMPOProduct(mpo, mps, psi, error = 1e-12) :
        N = mpo.N
        d = mpo.d
        #create copies of MPS in memory
        mps_copy = MPS_OBC()
        psi_copy = MPS_OBC()
        mps_copy.buildMPS(mps.mps)
        psi_copy.buildMPS(psi.mps)

        mps = mps_copy
        psi = psi_copy

        #first, generate the right end tensors and update the tensor at the first site
        psi.makeMixedCanonicalForm(0)
        E0L = [0]
        E0R = [0]
        for n in reversed(range(1, N)) :
            E0R.append(MPO.rightEnd(n, psi, mpo, mps, E0R[len(E0R) - 1]))

        psi.mps[0] = MPO.updatedTensor(0, psi, mpo, mps, 0, E0R[len(E0R) - 1])
        e = MPO.epsilon(mpo, mps, psi)

        for j in range(0, 10) :
            #sweep to right
            for i in range(1, N) :
                psi.makeSiteLeftNormalized(i - 1)
                E0L.append(MPO.leftEnd(i - 1, psi, mpo, mps, E0L[len(E0L) - 1]))
                psi.mps[i] = MPO.updatedTensor(i, psi, mpo, mps, E0L[len(E0L) - 1], E0R[len(E0R) - 1 - i])

            E0R = [0]
            #sweep to left
            for i in reversed(range(0, N - 1)) :
                psi.makeSiteRightNormalized(i + 1)
                E0R.append(MPO.rightEnd(i + 1, psi, mpo, mps, E0R[len(E0R) - 1]))
                psi.mps[i] = MPO.updatedTensor(i, psi, mpo, mps, E0L[i], E0R[len(E0R) - 1])

            E0L = [0]

            ee = MPO.epsilon(mpo, mps, psi) 
            if np.absolute(e) < error or np.absolute(e - ee) / e < 1e-5 :
                break
            else :
                e = ee

        return psi

    #Contract from the right up until and including site i
    @staticmethod
    def rightEnd(i, psi, mpo, mps, tensor) :
        if i == mpo.N - 1 :
            tensor = np.tensordot(np.conjugate(psi.mps[i]), mpo.mpo[i], ([0], [0]))
            tensor = np.tensordot(tensor, mps.mps[i], ([2], [0]))
            tensor = np.transpose(tensor, (1, 3, 5, 0, 2, 4))
            return tensor[0][0][0]
        else :
            tensor = np.tensordot(tensor, np.conjugate(psi.mps[i]), ([0], [2]))
            tensor = np.tensordot(tensor, mps.mps[i], ([1], [2]))
            tensor = np.tensordot(tensor, mpo.mpo[i], ([0, 1, 3], [3, 0, 1]))
            return np.transpose(tensor, (0, 2, 1))

    @staticmethod
    def leftEnd(i, psi, mpo, mps, tensor) :
        if i == 0 :
            tensor = np.tensordot(np.conjugate(psi.mps[0]), mpo.mpo[0], ([0], [0]))
            tensor = np.tensordot(tensor, mps.mps[0], ([2], [0]))
            tensor = np.transpose(tensor, (0, 2, 4, 1, 3, 5))
            return tensor[0][0][0]
        else :
            tensor = np.tensordot(tensor, np.conjugate(psi.mps[i]), ([0], [1]))
            tensor = np.tensordot(tensor, mps.mps[i], ([1], [1]))
            tensor = np.tensordot(tensor, mpo.mpo[i], ([0, 1, 3], [2, 0, 1]))
            return np.transpose(tensor, (0, 2, 1))

    #sub-module for updating tensor A[i]
    @staticmethod
    def updatedTensor(i, psi, mpo, mps, E0L, E0R) :
        #contract at the final site
        if i == 0 :
            E0R = np.tensordot(E0R, mps.mps[0], ([2], [2]))
            M = np.tensordot(E0R, mpo.mpo[0], ([1, 2], [3, 1]))
            M = np.transpose(M, (1, 2, 3, 0))[0]
            
            return M

        if i == mpo.N - 1 :
            E0L = np.tensordot(E0L, mps.mps[mpo.N - 1], ([2], [1]))
            M = np.tensordot(E0L, mpo.mpo[mpo.N - 1], ([1, 2], [2, 1]))
            M = np.transpose(M, (1, 2, 0, 3))[0]

            return M

        else :
            E0 = np.tensordot(E0L, mps.mps[i], ([2], [1]))
            E0 = np.tensordot(E0, E0R, ([3], [2]))
            E0 = np.tensordot(E0, mpo.mpo[i], ([1, 2, 4], [2, 1, 3]))

            M = np.transpose(E0, (2, 0, 1))
            return M

    #compute ||(MPO|MPS>) - |PSI>||^2 as a check 
    @staticmethod
    def epsilon(mpo, mps, psi) :
        return MPO.product(mpo.adjoint(), mpo).overlapLeft(mps, mps) - 2 * mpo.overlapLeft(psi, mps).real + psi.innerProd(psi)

    #Return a new MPO that is OTHER * OTHER2
    @staticmethod
    def product(other, other2) :
        #define a new set of tensors
        new_mpo = []
        for n in range(0, other2.N) :
            new_mpo.append(np.tensordot(other.mpo[n], other2.mpo[n], ([1], [0])))
            
        #reshape tensors
        for n in range(0, other2.N) :
            new_mpo[n] = np.transpose(new_mpo[n], (0, 3, 1, 4, 2, 5))
            new_mpo[n] = new_mpo[n].reshape((other2.d, other2.d, other2.Dims[n] * other.Dims[n], other2.Dims[n + 1] * other.Dims[n + 1]))

        prod = MPO()
        prod.buildMPO(new_mpo)
        return prod

    #return the trace of the MPO
    def trace(self) :
        eye = np.eye(2, dtype = 'complex')
        trace = np.tensordot(self.mpo[0], eye, ([0, 1], [0, 1]))
        for i in range(1, self.N) :
            trace = np.tensordot(trace, self.mpo[i], ([1], [2]))
            trace = np.tensordot(trace, eye, ([1, 2], [0, 1]))

        return np.trace(trace)

    #Print matrices
    def printMatrices(self) :
        output = ''
        for thing in self.mpo :
            for i in range(0, len(thing)) :
                for j in range(len(thing[i])) :
                    output += '%d, %d \n' % (i, j)
                    for row in range(0, len(thing[i][j])) :
                        for col in range(0, len(thing[i][j][row])) :
                            output += str(thing[i][j][row][col]) + ' '
                        output += '\n'
                    output += '\n'
            output += '----------------------------------\n'

        print(output)
    #Define a matrix
    def matrix(self) :
        terms = []
        column_spins = np.zeros(self.N)
        row_spins = np.zeros(self.N)
        for row in range(0, self.d ** self.N) :
            for col in range(0, self.d ** self.N) :
                result = np.eye(1)
                for i in range(0, self.N) :
                    result = np.dot(result, self.mpo[i][row_spins[i]][column_spins[i]])
                terms.append(result)
                self.addOne(column_spins)
            self.addOne(row_spins)
        m = np.array(terms)
        return m.reshape((self.d ** self.N, self.d ** self.N))

    def addOne(self, spins) :
        for j in reversed(range(0, self.N)) :
            if spins[j] == self.d - 1 :
                spins[j] = 0
            else :
                spins[j] = spins[j] + 1
                break;

    #construct a tensor that converts two indices to one
    @staticmethod
    def dimChanger(a, b) :
        dimchanger = np.zeros((a * b, a, b), dtype = 'complex')
        for alpha in range(0, a) :
            for beta in range(0, b) :
                dimchanger[alpha * b + beta][alpha][beta] = 1.0
        return dimchanger

    #fold the MPO into an MPS
    @staticmethod
    def asMPS(self) :
        tensors = []
        dd = MPO.dimChanger(self.d, self.d)
        for A in self.mpo :
            tensors.append(np.tensordot(dd, A, ([1, 2], [0, 1])))

        psi = MPS_OBC()
        psi.buildMPS(tensors)
        return psi

    @staticmethod
    #construct the density matrix on [a...b] by tracing out the parts outside [a...b]
    def constructDensityMPO(psi, a, b) :
        #begin by left normalizing up to a, right normalizing up to b
        for i in range(0, a) :
            psi.makeSiteLeftNormalized(i)

        for i in reversed(range(b + 1, psi.N)) :
            psi.makeSiteRightNormalized(i)

        #compute the reduced density matrix
        tensors = []

        for i in range(a, b + 1) :
            r, c = int(psi.Dims[i]), int(psi.Dims[i + 1])
            if i == a :
                dc = MPO.dimChanger(c, c)
                A = np.tensordot(psi.mps[a], np.conjugate(psi.mps[a]), ([1], [1]))
                A = np.tensordot(A, dc, ([1, 3], [1, 2])).reshape((psi.d, psi.d, 1, 
                    c ** 2))
                tensors.append(A)
            if i == b :
                dc = MPO.dimChanger(r, r)
                A = np.tensordot(psi.mps[i], np.conjugate(psi.mps[i]), ([2], [2]))
                A = np.tensordot(A, dc, ([1, 3], [1, 2])).reshape((psi.d, psi.d, 
                    r ** 2, 1))
                tensors.append(A)
            if not (i == a or i == b) :
                dcl = MPO.dimChanger(r, r)
                dcr = MPO.dimChanger(c, c)
                A = np.tensordot(psi.mps[i], dcl, ([1], [1]))
                A = np.tensordot(A, np.conjugate(psi.mps[i]), ([3], [1]))
                A = np.tensordot(A, dcr, ([1, 4], [1, 2]))
                tensors.append(np.transpose(A, (0, 2, 1, 3)))

        rho = MPO()
        rho.buildMPO(tensors)

        return rho

# N, d, D, a, b = 10, 3, 5, 0, 5
# psi = MPS_OBC()
# psi.buildRandomMPS(N, d, D)
# psi.makeLeftNormalized()
# print(psi.doubleCutEntropy(a, b))

# rho = MPO.constructDensityMPO(psi, a, b).matrix()
# ee = 0
# evals = np.linalg.eigvalsh(rho)

# for eig in evals :
#     if eig > 0 :
#         ee = ee - eig * np.log(eig) / np.log(2)

# print(ee)