###################### Data structure for an MPS ###############################
#   Takes as input several parameters: N (number of sites),                    #
#   D (bond dimension), d (physical dimension), and a file containing matrices #
################################################################################

import numpy as np
import sys

class MPS_OBC(object) :

    def __init__(self) :
        self.mps = None
        self.N = None
        self.D = None
        self.d = None
        self.Dims = None

    #Put data from the file into a tensor that contains all the matrices
    def buildMPSFromFile(self, N, d, file) :
        self.N = N
        self.d = d

        #make an array of dimensions
        self.Dims = np.zeros(N + 1)

        mps = []

        f = open(file)
        lines = f.readlines()

        #Create empty tensors of the right dimensions
        for i in range(0, N) :
            line = lines[i * d].split()
            if line.count(':') == 0 :
                mps.append(np.zeros((d, 1, len(line)), dtype = 'complex'))
                self.Dims[i] = 1
                self.Dims[i + 1] = len(line)
            else :
                c = line.index(':')
                r = line.count(':') + 1
                mps.append(np.zeros((d, r, c), dtype = 'complex'))
                self.Dims[i] = r
                self.Dims[i + 1] = c

        #Fill the tensors
        for site in range(0, N) :
            for state in range(0, d) :
                place = site * d + state
                elements = lines[place].split()

                r = int(self.Dims[site])
                c = int(self.Dims[site + 1])

                #remove the colons
                for s in range(1, r) :
                    elements.remove(':')

                #enter matrix elements
                for i in range(0, r) :
                    for j in range(0, c) :
                        mps[site][state][i][j] = complex(elements[i * c + j])

        self.mps = mps #Load the matrices from a file
        self.D = max(self.Dims)

    #Write MPS to file
    def writeStateToFile(self, filename) :
        file = open(filename, 'w')
        for tensor in self.mps :
            for d in range(0, len(tensor)) :
                for r in range(0, len(tensor[0])) :
                    for c in range(0, len(tensor[0][0])) :
                        file.write(str(tensor[d][r][c]) + ' ')
                    if r < len(tensor[0]) - 1 :
                        file.write(' : ')
                file.write('\n')
        file.close()

    #Generate a random MPS with OBC
    def buildRandomMPS(self, N, d, D) :
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

        self.mps = []
        for n in range(0, N) :
            r = int(self.Dims[n])
            c = int(self.Dims[n + 1])
            tensor = np.zeros((d, r, c), dtype='complex')
            for dim in range(0, d) :
                for i in range(0, r) :
                    for j in range(0, c) :
                        tensor[dim][i][j] = complex(np.random.randn(), np.random.randn())

            self.mps.append(tensor)

    #provide a list of tensors to initialize the MPS
    def buildMPS(self, mps) :
        self.N = len(mps)
        self.d = len(mps[0])

        #create list of dimensions, and determine D
        self.Dims = np.ones(self.N + 1)
        for n in range(0, self.N) :
            self.Dims[n] = len(mps[n][0])
            self.Dims[n + 1] = len(mps[n][0][0])

        self.D = max(self.Dims)
        self.mps = []
        for thing in mps :
            self.mps.append(np.copy(thing))

    #Compute inner product <SELF|OTHER> (without construction of transfer matrices)
    def innerProd(self, other) :
        E0 = np.tensordot(np.conjugate(self.mps[0]), other.mps[0], ([0], [0]))        
        for i in range(1, self.N) :
            E0 = np.tensordot(E0, np.conjugate(self.mps[i]), ([1], [1]))
            E0 = np.tensordot(E0, other.mps[i], ([2, 3], [1, 0]))
            E0 = np.transpose(E0, (0, 2, 1, 3))

        return E0[0][0][0][0]

    #Alternative methods for inner products
    #Compute inner product <SELF|OTHER> (by construction of transfer matrices)
    def innerProd2(self, other) :
        #define a list of transfer matrices
        E = []

        for i in range(0, self.N) :
            E.append(np.tensordot(np.conjugate(self.mps[i]), other.mps[i], ([0], [0])))

        #make each transfer tensor a matrix
        for i in range(0, self.N) :
            E[i] = np.transpose(E[i], (0, 2, 1, 3))
            E[i] = E[i].reshape((self.Dims[i] * other.Dims[i], self.Dims[i + 1] * other.Dims[i + 1]))

        #multiply all the matrices
        M = E[0]

        for i in range(1, len(E)) :
            M = np.dot(M, E[i])

        #return their trace
        return np.trace(M)

    #create a copy of the current instance
    def copy(self) :
        tensors = []
        for tensor in self.mps :
            tensors.append(tensor)

        copie = MPS_OBC()
        copie.buildMPS(tensors)
        return copie

    #Compute inner product INEFFICIENTLY to check your answer <SELF|OTHER>
    def innerProdSlow(self, other) :
        #define a recursive function to sum over all the possible matrix products
        spins = np.zeros(self.N)
        sum = 0
        for i in range(0, self.d ** self.N) :
            A = np.conjugate(self.mps[0][spins[0]])
            for j in range(1, self.N) :
                A = np.dot(A, np.conjugate(self.mps[j][spins[j]]))
            B = other.mps[0][spins[0]]
            for j in range(1, other.N) :
                B = np.dot(B, other.mps[j][spins[j]])

            sum = sum + np.trace(A) * np.trace(B)

            #append spins
            for j in range(0, self.N) :
                if spins[j] == self.d - 1 :
                    spins[j] = 0
                else :
                    if spins[j] < self.d - 1 :
                        spins[j] = spins[j] + 1
                        break;
        return sum

    #printing matrices
    def printMatrices(self) :
            output = ''
            for thing in self.mps :
                for i in range(0, len(thing)) :
                    for row in range(0, len(thing[i])) :
                        for col in range(0, len(thing[i][row])) :
                            output += str(thing[i][row][col]) + ' '
                        output += '\n'
                    output += '\n'
                output += '----------------------------------\n'

            print(output)

    #Make the MPS product left-normalized. 
    def makeLeftNormalized(self) :
        #Going left to right, reassign each tensor to a unitary equivalent
        for i in range(0, self.N) :
            
            M = self.mps[i]
            
            #obtain the dimensions of M
            r = self.Dims[i]
            c = self.Dims[i + 1]

            #M[i]^s_i,j --> M[i]_(s, i), j 
            M = M.reshape((self.d * r, c))

            #Compute QR decomposition of M, where Q is orthogonal and R upper-triangular
            Q, R = np.linalg.qr(M)
            #Q is d*r x c', R is c' x c
            new_c = len(Q[0])

            if i < self.N - 1 :
                #reassign M to Q
                self.mps[i] = Q.reshape(self.d, r, new_c)
                self.mps[i + 1] = np.transpose(np.tensordot(self.mps[i + 1], R, ([1], [1])), (0, 2, 1))
            else :
                #We are at the last matrix, which should be a vector
                #Q would be the normalized vector, and R its magnitude
                self.mps[i] = Q.reshape(self.d, r, new_c)
                self.norm = np.sqrt(np.conjugate(np.trace(R)) * np.trace(R))
                #retain phase contribution from R to preserve direction of vector
                self.mps[i] = self.mps[i] * (np.trace(R) / self.norm)

            self.Dims[i + 1] = new_c

    #Left normalize a particular site (i < N - 1)
    def makeSiteLeftNormalized(self, i) :
        M = self.mps[i]
        
        #obtain the dimensions of M
        r = self.Dims[i]
        c = self.Dims[i + 1]

        #M[i]^s_i,j --> M[i]_(s, i), j 
        M = M.reshape((self.d * r, c))

        #Compute QR decomposition of M, where Q is orthogonal and R upper-triangular
        Q, R = np.linalg.qr(M)
        #Q is d*r x c', R is c' x c
        new_c = len(Q[0])

        #reassign M to Q
        self.mps[i] = Q.reshape(self.d, r, new_c)
        self.mps[i + 1] = np.transpose(np.tensordot(self.mps[i + 1], R, ([1], [1])), (0, 2, 1))

        self.Dims[i + 1] = new_c

    #Make the MPS product right-normalized.
    def makeRightNormalized(self) :
        #Going right to left, reassign each tensor to a unitary equivalent
        for i in reversed(range(0, self.N)) :

            M = self.mps[i]

            #obtain dimensions of M
            r = self.Dims[i]
            c = self.Dims[i + 1]

            #M[i]^s_i,j --> M[i]_i, (s, j)
            #First, M[i]^s_i, j --> M[i]^s_j, i
            M = np.transpose(M, (0, 2, 1))
            #Next, M[i]^s_j, i --> M[i]_(s, j), i
            M = M.reshape((self.d * c, r))
            #Finally, M[i]_(s, j), i --> M[i]_i, (s, j)
            M = np.transpose(M, (1, 0))

            #Compute SVD of M
            U, s, V = np.linalg.svd(M, full_matrices = False)
            S = np.diag(s)
            X = np.dot(U, S)

            new_r = len(s)

            #V is now new_r x d*c, X is r x new_r

            if i > 0 :
                #reassign M to V after appropriate reshaping
                self.mps[i] = np.transpose(np.transpose(V, (1, 0)).reshape((self.d, c, new_r)), (0, 2, 1))
                self.mps[i - 1] = np.tensordot(self.mps[i - 1], X, ([2], [0])) 
            else :
                self.mps[i] = np.transpose(np.transpose(V, (1, 0)).reshape((self.d, c, new_r)), (0, 2, 1))
                self.norm = np.sqrt(np.conjugate(np.trace(X)) * np.trace(X))
                #retain any phase thing
                self.mps[i] = self.mps[i] * (np.trace(X) / self.norm)

            self.Dims[i] = new_r

    #Make site i right normalized (i > 0)
    def makeSiteRightNormalized(self, i) :
        M = self.mps[i]

        #obtain dimensions of M
        r = self.Dims[i]
        c = self.Dims[i + 1]

        #M[i]^s_i,j --> M[i]_i, (s, j)
        #First, M[i]^s_i, j --> M[i]^s_j, i
        M = np.transpose(M, (0, 2, 1))
        #Next, M[i]^s_j, i --> M[i]_(s, j), i
        M = M.reshape((self.d * c, r))
        #Finally, M[i]_(s, j), i --> M[i]_i, (s, j)
        M = np.transpose(M, (1, 0))

        #Compute SVD of M
        U, s, V = np.linalg.svd(M, full_matrices = False)
        S = np.diag(s)
        X = np.dot(U, S)

        new_r = len(s)

        #V is now new_r x d*c, X is r x new_r

        #reassign M to V after appropriate reshaping
        self.mps[i] = np.transpose(np.transpose(V, (1, 0)).reshape((self.d, c, new_r)), (0, 2, 1))
        self.mps[i - 1] = np.tensordot(self.mps[i - 1], X, ([2], [0])) 

        self.Dims[i] = new_r

    #make all the states to the left of l left-normalized and vice versa (0<= l <= N - 1)
    def makeMixedCanonicalForm(self, l) :
        for i in range(0, l) :
            self.makeSiteLeftNormalized(i)

        for i in reversed(range(l + 1, self.N)) :
            self.makeSiteRightNormalized(i)

    #Produce a compressed state with dimension D. Return the norm and the new state vector.
    @staticmethod
    def SVDCompressedState(psi, D) :
        #first normalize the state
        vec = MPS_OBC()
        vec.buildMPS(psi.mps)
        vec.makeLeftNormalized()
        norm = vec.getOriginalNorm()

        #Going right to left, reassign each tensor to a unitary, possibly contracted equivalent
        for i in reversed(range(0, vec.N)) :

            M = vec.mps[i]

            #obtain dimensions of M
            r = vec.Dims[i]
            c = vec.Dims[i + 1]

            #M[i]^s_i,j --> M[i]_i, (s, j)
            #First, M[i]^s_i, j --> M[i]^s_j, i
            M = np.transpose(M, (0, 2, 1))
            #Next, M[i]^s_j, i --> M[i]_(s, j), i
            M = M.reshape((vec.d * c, r))
            #Finally, M[i]_(s, j), i --> M[i]_i, (s, j)
            M = np.transpose(M, (1, 0))

            #Compute SVD of M
            U, s, V = np.linalg.svd(M, full_matrices = False)

            #contract matrices if they're too big
            if (len(s) > D) :
                s = np.array(s[0 : D])
                V = np.array(V[0 : D, :])
                U = np.array(U[:, 0 : D])
            
            S = np.diag(s)
            X = np.dot(U, S)

            new_r = len(s)

            #V is now new_r x d*c, X is r x new_r

            if i > 0 :
                #reassign M to V after appropriate reshaping
                vec.mps[i] = np.transpose(np.transpose(V, (1, 0)).reshape((vec.d, c, new_r)), (0, 2, 1))
                vec.mps[i - 1] = np.tensordot(vec.mps[i - 1], X, ([2], [0])) 
            else :
                vec.mps[i] = np.transpose(np.transpose(V, (1, 0)).reshape((vec.d, c, new_r)), (0, 2, 1))
                vec.norm = np.sqrt(np.conjugate(np.trace(X)) * np.trace(X))
                #retain any phase thing
                vec.mps[i] = vec.mps[i] * (np.trace(X) / vec.norm)

            vec.Dims[i] = new_r

        return vec, norm

    #Produce an equivalent state with bond dimension D + delta
    @staticmethod
    def increaseBondDim(psi, delta = 1) :
        #First define new dimensions
        dd = np.ones(psi.N + 1) * delta
        dd[0] = 0
        dd[psi.N] = 0

        Dims = psi.Dims + dd
        tensors = []

        #Define empty tensors with new dimensions
        for i in range(0, psi.N) :
            tensors.append(np.zeros((psi.d, Dims[i], Dims[i + 1]), dtype = 'complex'))
            for sigma in range(0, psi.d) :
                for row in range(0, int(psi.Dims[i])) :
                    for col in range(0, int(psi.Dims[i + 1])) :
                        tensors[i][sigma][row][col] = psi.mps[i][sigma][row][col]

        mps = MPS_OBC()
        mps.buildMPS(tensors)

        return mps

    #The state MUST first be normalized to calculate the proper entropy

    #Entropy across cut [0...a]X[a+1...N-1]. By assumption 0 <= a < N - 1.
    def singleCutEntropy(self, a) :
        #First left normalize up until site a
        for i in range(0, a) :
            self.makeSiteLeftNormalized(i)

        #Left normalize site a
        M = self.mps[a]
        r, c = self.Dims[a], self.Dims[a + 1]
        Q, R = np.linalg.qr(M.reshape((self.d * r, c)))
        
        #Right normalize up until a + 1
        for i in reversed(range(a + 2, self.N)) :
            self.makeSiteRightNormalized(i)

        #Right normalize site a + 1
        M = self.mps[a + 1]
        r, c = self.Dims[a + 1], self.Dims[a + 2]
        M = np.transpose(M, (0, 2, 1)).reshape((self.d * c, r))
        M = np.transpose(M, (1, 0))
        U, s, V = np.linalg.svd(M, full_matrices = False)
        S = np.diag(s)
        X = np.dot(U, S)
        
        #Find the singular values of the bond matrix
        s = np.linalg.svd(np.dot(R, X), full_matrices = False, compute_uv = False)
        entropy = 0 
        for val in s :
            l = np.conjugate(val) * val
            entropy += -l * np.log(l) / np.log(2)

        #Leave the state awkwardly semi-normalized

        return entropy

    #Compute the maximum entropy over all single cuts of the system
    def maximumEntropy(self, returnAllCuts = False) :
        Smax = 0
        EE = []
        M = self.mps[0]
        r, c = self.Dims[0], self.Dims[1]
        Q, R = np.linalg.qr(M.reshape((self.d * r, c)))
        
        for i in reversed(range(2, self.N)) :
            self.makeSiteRightNormalized(i)

        M = self.mps[1]
        r, c = self.Dims[1], self.Dims[2]
        M = np.transpose(M, (0, 2, 1)).reshape((self.d * c, r))
        M = np.transpose(M, (1, 0))
        U, s, V = np.linalg.svd(M, full_matrices = False)
        S = np.diag(s)
        X = np.dot(U, S)
        
        #Find the singular values of the bond matrix
        s = np.linalg.svd(np.dot(R, X), full_matrices = False, compute_uv = False)
        entropy = 0 
        for val in s :
            l = np.conjugate(val) * val
            entropy += -l * np.log(l) / np.log(2)

        if returnAllCuts :
            EE.append(entropy)

        if entropy > Smax :
            Smax = entropy

        for a in range(1, self.N - 1) :
            self.makeSiteLeftNormalized(a - 1)

            M = self.mps[a]
            r, c = self.Dims[a], self.Dims[a + 1]
            Q, R = np.linalg.qr(M.reshape((self.d * r, c)))

            #Right normalize site a + 1
            M = self.mps[a + 1]
            r, c = self.Dims[a + 1], self.Dims[a + 2]
            M = np.transpose(M, (0, 2, 1)).reshape((self.d * c, r))
            M = np.transpose(M, (1, 0))
            U, s, V = np.linalg.svd(M, full_matrices = False)
            S = np.diag(s)
            X = np.dot(U, S)            

            #Find the singular values of the bond matrix
            s = np.linalg.svd(np.dot(R, X), full_matrices = False, compute_uv = False)
            entropy = 0 
            for val in s :
                l = np.conjugate(val) * val
                entropy += -l * np.log(l) / np.log(2)

            if returnAllCuts :
                EE.append(entropy)

            if entropy > Smax :
                Smax = entropy

        if returnAllCuts :
            return Smax, EE
        else :
            return Smax
        

    #Compute the entropy for the subsystem [a ... b]
    def doubleCutEntropy(self, a, b) :
        #begin by left normalizing up to a, right normalizing up to b
        for i in range(0, a) :
            self.makeSiteLeftNormalized(i)

        for i in reversed(range(b + 1, self.N)) :
            self.makeSiteRightNormalized(i)

        #compute the reduced density matrix
        T = []
        for i in range(a, b + 1) :
            A = np.tensordot(self.mps[i], np.conjugate(self.mps[i]), ([0], [0]))
            A = np.transpose(A, (0, 2, 1, 3)).reshape((self.Dims[i] ** 2, self.Dims[i + 1] ** 2))
            T.append(A)

        rho = T[0]
        for i in range(1, len(T)) :
            rho = np.dot(rho, T[i])

        #reshape rho
        rho = rho.reshape((self.Dims[a], self.Dims[a], self.Dims[b + 1], self.Dims[b + 1]))
        rho = np.transpose(rho, (0, 2, 1, 3))
        rho = rho.reshape((self.Dims[a] * self.Dims[b + 1], self.Dims[a] * self.Dims[b + 1]))

        s = np.linalg.eigvalsh(rho)
        entropy = 0 
        for val in s :
            if not val <= 0 :
                entropy += -val * np.log(val) / np.log(2)

        return entropy

    #return the original norm of the vector
    def getOriginalNorm(self) :
        return self.norm

    #scale the MPS
    def scale(self, alpha) :
        k = alpha ** (1.0 / self.N)
        for i in range(0, self.N) :
            self.mps[i] = self.mps[i] * k

    #return the vector equivalent of the MPS
    def vector(self) :
        terms = []
        self.buildTerms(0, np.eye(1), terms)
        return np.array(terms).reshape(self.d ** self.N)

    #recursive build the vector
    def buildTerms(self, i, result, terms) :
        if i == self.N :
            terms.append(result)
        else :
            for dim in range(0, self.d) :
                self.buildTerms(i + 1, np.dot(result, self.mps[i][dim]), terms)


# vec = MPS_OBC()
# vec.buildRandomMPS(10, 5, 10)
# vec.makeLeftNormalized()
# vec.writeStateToFile('mps_test.txt')
# vec2 = MPS_OBC()
# vec2.buildMPSFromFile(10, 5, 'mps_test.txt')
# print(vec.innerProd(vec2))