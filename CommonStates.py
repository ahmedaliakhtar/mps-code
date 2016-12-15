import numpy as np
from MPS_OBC import MPS_OBC

#This class computes many common states as mps

class CommonStates(object) :

    #returns the state |p...p>
    @staticmethod
    def simpleProductState(N, d, p = 0) :
        tensors = []
        A = np.zeros((d, 1, 1), dtype = 'complex')
        A[p][0][0] = 1

        for i in range(0, N) :
            tensors.append(A)

        mps = MPS_OBC()
        mps.buildMPS(tensors)

        return mps

    #returns a product state |values[0] values[1] ... >
    @staticmethod
    def productState(d, values) :
        N = len(values)
        tensors = []

        for i in range(0, N) :
            A = np.zeros((d, 1, 1), dtype = 'complex')
            A[values[i]][0][0] = 1
            tensors.append(A)

        mps = MPS_OBC()
        mps.buildMPS(tensors)

        return mps        

    #returns the state |0...0> + |1...1> ... 
    @staticmethod
    def maximallyEntangledState(N, d) :
        B = []
        B.append(np.eye(d, dtype = 'complex').reshape((d, 1, d)))

        B.append(np.zeros((d, d, d), dtype = 'complex'))
        for i in range(0, d) :
            B[1][i][i][i] = 1
        B.append(np.eye(d, dtype = 'complex').reshape((d, d, 1)))

        tensors = []

        tensors.append(B[0])

        for i in range(0, N - 2) :
            tensors.append(B[1])

        tensors.append(B[2])

        mps = MPS_OBC()
        mps.buildMPS(tensors)
        mps.makeLeftNormalized()

        return mps

    #returns an equal superposition of all states
    def equalSuperpositionState(N, d, D = 1) :
        if D == 1 :
            tensors = []
            A = np.ones((d, 1, 1), dtype = 'complex')

            for i in range(0, N) :
                tensors.append(A)

            mps = MPS_OBC()
            mps.buildMPS(tensors)
            mps.makeLeftNormalized()

            return mps
        else :
            tensors = []
            A1 = np.zeros((d, 1, D), dtype = 'complex')
            for i in range(0, d) :
                A1[i][0][0] = 1.0
            A2 = np.zeros((d, D, D), dtype = 'complex')
            for i in range(0, d) :
                for j in range(0, D) :
                    A2[i][j][j] = 1.0 
            A3 = np.zeros((d, D, 1), dtype = 'complex')
            for i in range(0, d) :
                A3[i][0][0] = 1.0

            tensors.append(A1)
            for i in range(0, N - 2) :
                tensors.append(A2)
            tensors.append(A3)

            mps = MPS_OBC()
            mps.buildMPS(tensors)
            mps.makeLeftNormalized()

            return mps             