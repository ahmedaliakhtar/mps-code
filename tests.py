from MPS_OBC import MPS_OBC
from MPO import MPO
from IsingHamiltonian import IsingHamiltonian
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
import time
from CommonStates import CommonStates

c = np.sqrt(1.0 / 2.0)
k = np.sqrt(c)

def MPSTests() :
    vec = MPS_OBC()
    vec2 = MPS_OBC()

    vec.buildRandomMPS(8, 4, 10)
    vec2.buildRandomMPS(8, 4, 20)
    print(vec.Dims)
    #print('exact inner product = ', vec.innerProdSlow(vec2))
    print('<v1|v2> using efficient contractions: ', vec.innerProd(vec2))
    print('norm = ', vec.innerProd(vec))
    vec.makeRightNormalized()
    print('norm (in right-canonical form) = ', vec.innerProd(vec))
    print('original norm = ', vec.getOriginalNorm() ** 2)
    print('check that normalized form produces correct vector : ', vec.innerProd(vec2) * vec.getOriginalNorm())

    #compare to compressed vec2
    print(vec2.innerProd(vec2))
    vec2_copy, norm = MPS_OBC.SVDCompressedState(vec2, 20)
    vec2_copy.scale(norm)
    print(vec2_copy.innerProd(vec2_copy))

    print(vec2_copy.innerProd(vec2_copy) + vec2.innerProd(vec2) - vec2.innerProd(vec2_copy) - vec2_copy.innerProd(vec2))


def SingleSiteTestMPO() :
    mps_tensor = []
    mps_tensor.append(np.zeros((2, 1, 1)))
    mps_tensor[0][0][0][0] = c
    mps_tensor[0][1][0][0] = c

    mps = MPS_OBC()
    mps.buildMPS(mps_tensor)

    mpo_tensor = []
    mpo_tensor.append(np.zeros((2, 2, 1, 1)))
    mpo_tensor[0][0][0][0][0] = c
    mpo_tensor[0][0][1][0][0] = c
    mpo_tensor[0][1][0][0][0] = c
    mpo_tensor[0][1][1][0][0] = -c

    mpo = MPO()
    mpo.buildMPO(mpo_tensor)

    result = mpo.actOn(mps)

    result.printMatrices()

    mpo = MPO.product(mpo, mpo)

    result = mpo.actOn(mps)

    result.printMatrices()

def DoubleSiteTestMPO() :
    mps_tensor = []
    mps_tensor.append(np.zeros((2, 1, 2)))
    mps_tensor.append(np.zeros((2, 2, 1)))
    mps_tensor[0][0][0][0] = k
    mps_tensor[0][1][0][1] = k
    mps_tensor[1][0][0][0] = k
    mps_tensor[1][1][1][0] = k

    mps = MPS_OBC()
    mps.buildMPS(mps_tensor)
    
    mpo_tensor = []
    mpo_tensor.append(np.zeros((2, 2, 1, 2)))
    mpo_tensor.append(np.zeros((2, 2, 2, 1)))

    mpo_tensor[0][0][0][0][0] = k
    mpo_tensor[0][0][1][0][0] = k
    mpo_tensor[0][1][0][0][0] = k
    mpo_tensor[0][1][1][0][0] = -k

    mpo_tensor[1][0][0][0][0] = k
    mpo_tensor[1][0][0][1][0] = k
    mpo_tensor[1][0][1][1][0] = k
    mpo_tensor[1][1][0][1][0] = k
    mpo_tensor[1][1][1][0][0] = k

    #check to see if you're constructing the right matrix
    O = np.zeros((4, 4))
    for i in range(0, 2) :
        for j in range(0, 2) :
            for z in range(0, 2) :
                for w in range(0, 2) :
                    O[2 * i + j][2 * z + w] = np.dot(mpo_tensor[0][i][z], mpo_tensor[1][j][w])

    print(O)

    mpo = MPO()

    mpo.buildMPO(mpo_tensor)

    print(mpo.matrix())

    result = mpo.actOn(mps)
    result.printMatrices()
    print(result.innerProd(result))
    print(mps.innerProd(result))

    mpo_squared = MPO.product(mpo, mpo)
    #mpo_squared.printMatrices()
    #print(mpo_squared.N)
    #print(mpo_squared.d)
    #print(mpo_squared.Dims)

    result = mpo_squared.actOn(mps)

    print(result.innerProd(result))
    print(mps.innerProd(result))

def MPOTests() :
    #check to see if adjoint is consistent

    vec = MPS_OBC()
    vec2 = MPS_OBC()

    vec.buildRandomMPS(5, 3, 10)
    vec2.buildRandomMPS(5, 3, 20)

    O1 = MPO()
    O2 = MPO()

    O1.buildRandomMPO(5, 3, 20)
    O2.buildRandomMPO(5, 3, 30)

    vec.makeLeftNormalized()
    vec2.makeLeftNormalized()

    print(vec.innerProd(O1.actOn(vec2)))
    print(vec2.innerProd(O1.adjoint().actOn(vec)))

    #check to see if you're doing products correctly

    O3 = MPO.product(O1, O2)

    vec3 = O1.actOn(O2.actOn(vec))
    vec4 = O3.actOn(vec)

    print(vec2.innerProd(vec3))

    print(vec2.innerProd(vec4))

    O4 = MPO.product(O2.adjoint(), O1.adjoint())
    O5 = MPO.product(O1, O2).adjoint()
    print(vec.innerProd(O4.actOn(vec2)))
    print(vec.innerProd(O5.actOn(vec2)))

def MPOTests2() :
    v1 = MPS_OBC()
    v2 = MPS_OBC()

    v1.buildRandomMPS(5, 3, 10)
    v2.buildRandomMPS(5, 3, 20)

    O = MPO()

    O.buildRandomMPO(5, 3, 20)

    print(v1.innerProd(O.actOn(v2)))
    print(O.overlapLeft(v1, v2))
    print(O.overlapRight(v1, v2))

def CompressionTests() :
    mps_tensor = []
    mps_tensor.append(np.zeros((2, 1, 2)))
    mps_tensor.append(np.zeros((2, 2, 1)))
    mps_tensor[0][0][0][0] = k
    mps_tensor[0][1][0][1] = k
    mps_tensor[1][0][0][0] = k
    mps_tensor[1][1][1][0] = k

    mps = MPS_OBC()
    mps.buildMPS(mps_tensor)
    mps_tensor.append(np.zeros((2, 1, 2)))
    mps_tensor.append(np.zeros((2, 2, 1)))

    psi_tensor = []
    psi_tensor.append(np.zeros((2, 1, 2)))
    psi_tensor.append(np.zeros((2, 2, 1)))
    psi_tensor[0][0][0][0] = 0.5 + np.random.randn()
    psi_tensor[0][0][0][1] = 1.5 + np.random.randn()
    psi_tensor[0][1][0][0] = 0.5 + np.random.randn()
    psi_tensor[0][1][0][1] = -0.5 + np.random.randn()
    psi_tensor[1][0][0][0] = 1.0 + np.random.randn()
    psi_tensor[1][0][1][0] = 0.0 + np.random.randn()
    psi_tensor[1][1][0][0] = -0.5 + np.random.randn()
    psi_tensor[1][1][1][0] = 0.5 + np.random.randn()

    psi = MPS_OBC()
    psi.buildMPS(psi_tensor)

    mpo_tensor = []
    mpo_tensor.append(np.zeros((2, 2, 1, 2)))
    mpo_tensor.append(np.zeros((2, 2, 2, 1)))

    mpo_tensor[0][0][0][0][0] = k
    mpo_tensor[0][0][1][0][0] = k
    mpo_tensor[0][1][0][0][0] = k
    mpo_tensor[0][1][1][0][0] = -k

    mpo_tensor[1][0][0][0][0] = k
    mpo_tensor[1][0][0][1][0] = k
    mpo_tensor[1][0][1][1][0] = k
    mpo_tensor[1][1][0][1][0] = k
    mpo_tensor[1][1][1][0][0] = k

    mpo = MPO()
    mpo.buildMPO(mpo_tensor)

    print(MPO.epsilon(mpo, mps, psi))

    psi = MPO.compressMPOProduct(mpo, mps, psi)

    print(MPO.epsilon(mpo, mps, psi))

def CompressionTests2() :
    mps = MPS_OBC()

    mpo = MPO()
    mpo.buildRandomMPO(6, 2, 5)

    mps.buildRandomMPS(6, 2, 5)

    mps2 = mpo.actOn(mps)

    mps2.makeLeftNormalized()
    norm = mps2.getOriginalNorm()

    mps.scale(1.0 / norm)

    psi = MPS_OBC()
    psi.buildRandomMPS(6, 2, 5)
    psi.makeLeftNormalized()

    print(MPO.epsilon(mpo, mps, psi))

    psi = MPO.compressMPOProduct(mpo, mps, psi)

    print(MPO.epsilon(mpo, mps, psi))

def CompressionTests3() :
    mps_tensor = []
    mps_tensor.append(np.zeros((2, 1, 1), dtype = 'complex'))
    mps_tensor.append(np.zeros((2, 1, 1), dtype = 'complex'))
    mps_tensor.append(np.zeros((2, 1, 1), dtype = 'complex'))
    mps_tensor[0][0][0][0] = 0
    mps_tensor[0][1][0][0] = 1
    mps_tensor[1][0][0][0] = 0
    mps_tensor[1][1][0][0] = 1
    mps_tensor[2][0][0][0] = 0
    mps_tensor[2][1][0][0] = 1

    mps = MPS_OBC() 
    mps.buildMPS(mps_tensor)

    mpo_tensor = []

    mpo_tensor.append(np.zeros((2, 2, 1, 2), dtype = 'complex'))
    mpo_tensor.append(np.zeros((2, 2, 2, 2), dtype = 'complex'))
    mpo_tensor.append(np.zeros((2, 2, 2, 1), dtype = 'complex'))

    mpo_tensor[0][1][1][0][0] = 1
    mpo_tensor[0][1][1][0][1] = -1
    mpo_tensor[1][1][1][0][1] = 1
    mpo_tensor[1][1][1][1][0] = 1
    mpo_tensor[2][1][1][0][0] = 1
    mpo_tensor[2][1][1][1][0] = -1

    mpo = MPO()
    mpo.buildMPO(mpo_tensor)

    psi_tensor = []
    psi_tensor.append(np.zeros((2, 1, 2), dtype = 'complex'))
    psi_tensor.append(np.zeros((2, 2, 2), dtype = 'complex'))
    psi_tensor.append(np.zeros((2, 2, 1), dtype = 'complex'))
    psi_tensor[0][0][0][0] = 0 + np.random.randn() 
    psi_tensor[0][1][0][0] = 1 + np.random.randn()
    psi_tensor[1][0][0][0] = 0 + np.random.randn()
    psi_tensor[1][1][0][0] = 1 + np.random.randn()
    psi_tensor[2][0][0][0] = 0 + np.random.randn()
    psi_tensor[2][1][0][0] = 1 + np.random.randn()

    psi = MPS_OBC()
    psi.buildMPS(psi_tensor)

    print(MPO.epsilon(mpo, mps, psi))

    psi = MPO.compressMPOProduct(mpo, mps, psi)

    print(MPO.epsilon(mpo, mps, psi))

def CompressionTests4() :
    mps = MPS_OBC()

    mpo = MPO()

    mpo.buildRandomMPO(5, 3, 5)

    mps.buildRandomMPS(5, 3, 5)

    mps.makeLeftNormalized()

    psi, norm = MPS_OBC.SVDCompressedState(mpo.actOn(mps), 25)

    print(MPO.epsilon(mpo, mps, psi))

    psi = MPO.compressMPOProduct(mpo, mps, psi)

    print(MPO.epsilon(mpo, mps, psi))

def CheckGauge() :
    N = 6
    for i in range(0, N) :
        psi = MPS_OBC()
        psi.buildRandomMPS(N, 3, 10)

        print(psi.innerProd(psi))
        psi.makeMixedCanonicalForm(i)

        print(np.tensordot(np.conjugate(psi.mps[i]), psi.mps[i], ([0, 1, 2], [0, 1, 2])))

def CheckOverlapMethods() :
    N = 6
    for i in range(0, N) :
        O = MPO()
        O.buildRandomMPO(N, 3, 10)

        v1 = MPS_OBC()
        v1.buildRandomMPS(N, 3, 10)
        v1.makeLeftNormalized()

        v2 = MPS_OBC()
        v2.buildRandomMPS(N, 3, 10)
        v2.makeRightNormalized()

        x = O.overlapRight(v2, v1)

        M = np.conjugate(MPO.updatedTensor(i, v1, O, v2))
        y = np.tensordot(M, v1.mps[i], ([0, 1, 2], [0, 1, 2]))

        print(x - y)

def CheckMultiplication() :
    v1 = MPS_OBC()
    mpo = MPO()

    v1.buildRandomMPS(5, 3, 30)
    mpo.buildRandomMPO(5, 3, 10)
    M = mpo.matrix()
    v = v1.vector()
    v = np.dot(M, v)

    v2 = mpo.actOn(v1)
    v2 = v2.vector()

    print(np.dot(v - v2, v - v2))

def CheckIsingEigenvalues() :
    J = np.ones(5) * 0.5
    h = np.ones(5) * 0.001
    H = IsingHamiltonian(5, J, h)
    w1, v1 = np.linalg.eig(H.mpo.matrix())
    w2, v2 = np.linalg.eig(H.exactHamiltonian())
    print(np.sort(w1) - np.sort(w2))
    print(np.sort(w1))


def IsingImaginaryTimeEvolution() :
    N = 5
    Jfac = 1
    hfac = 1
    t = 1
    delta_t = .01
    D = 10

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac
    H = IsingHamiltonian(N, J, h)

    #give the ising hamiltonian an eigenstate
    phi = MPS_OBC()
    T = np.zeros((2, 1, 1))
    T[1][0][0] = 1.0
    tensors = []
    for i in range(0, N) :
        tensors.append(T)
    phi.buildMPS(tensors)

    x = np.linspace(0.0, t, num = int(t / delta_t) + 1)

    points = H.ImaginaryTimeEvolution(t, delta_t, phi)

    plt.plot(x, points, 'ro')
    plt.xlabel('Imaginary time (it)')
    plt.ylabel('Energy (h-bar = 1)')
    plt.title('Energy vs. Imaginary time N = %d J = %f h = %f' % (N, Jfac, hfac))
    plt.show()

def IsingImaginaryTimeEvolution2() :
    N = 5
    Jfac = 0.5
    hfac = 0.5
    t = 2
    delta_t = .005
    D = 4

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac
    H = IsingHamiltonian(N, J, h)

    psi = MPS_OBC()
    psi.buildRandomMPS(N, 2, D)
    psi.makeLeftNormalized()

    time, energy, psi = H.ImaginaryTimeEvolution(t, delta_t, psi)

    plt.figure(str(N) + ' ' + str(Jfac) + ' ' + str(hfac))
    plt.plot(time, energy, 'ro')
    plt.xlabel(r'Imaginary time ($it$)')
    plt.ylabel(r'Energy ($\hbar = 1$)')
    plt.text((max(time) + min(time)) / 2.0, (max(energy) + min(energy)) / 2.0, 
        r'$E_{min} = $' + str(energy[len(energy) - 1]))
    plt.title('Energy vs. Imaginary Time in the Ising Model \n' + r'$N =$ ' + str(N) + r', $J =$ ' 
        + str(Jfac) + r', $h =$ ' + str(hfac) + r', $\Delta t = $' + str(delta_t)
        + r', $D =$ ' + str(D))
    plt.show()

def IsingImaginaryTimeEvolution3() :
    N = 5
    Jfac = 0.5
    hfac = 1.0
    t = 1.5
    delta_t = .01
    D = 4

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac
    H = IsingHamiltonian(N, J, h)
    Hmatrix = H.exactHamiltonian()
    w, v = np.linalg.eig(Hmatrix)
    sorted_w = np.sort(w)
    sorted_v = v[:, w.argsort()]

    psi = MPS_OBC()
    psi.buildRandomMPS(N, 2, D)
    psi.makeLeftNormalized()

    time, energy, psi = H.ImaginaryTimeEvolution(t, delta_t, psi)

    v_t = psi.vector()

    print(np.dot(sorted_v[0], v_t))

def IsingImaginaryTimeEvolution4() :
    N = int(sys.argv[1])
    Jfac = float(sys.argv[2])
    hfac = float(sys.argv[3])
    exact_energy = float(sys.argv[4])

    t = 0.4422
    delta_t = .0002
    D = 20#2 ** int(N / 2)

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac
    H = IsingHamiltonian(N, J, h)

    psi = CommonStates.equalSuperpositionState(N, 2, D= D)

    psi_copy, psi_copy2 = MPS_OBC(), MPS_OBC()
    psi_copy.buildMPS(psi.mps)
    psi_copy2.buildMPS(psi.mps)

    # time, energy, gs = H.ImaginaryTimeEvolution(t, delta_t, psi)
    time2, energy2, gs2 = H.ImaginaryTimeEvolution2(t, delta_t, psi_copy, 
        updateTimeStep = False, writeGSToFile = False)

    f = open('data.txt', 'w')
    for i in range(0, len(time2)) :
        f.write(str(time2[i]) + '\t' + str(energy2[i]))
    f.close()
    # time3, energy3, gs3 = H.ImaginaryTimeEvolution3(t, delta_t, psi_copy2)

    # y1 = np.absolute(np.ones(len(energy)) * exact_energy - energy) / np.absolute(exact_energy)
    y2 = np.absolute(np.ones(len(energy2)) * exact_energy - energy2) / np.absolute(exact_energy)
    # y3 = np.absolute(np.ones(len(energy3)) * exact_energy - energy3) / np.absolute(exact_energy)

    f = open('results_N10_D20_dt0002.txt')
    lines = f.readlines()
    lines[0:5] = []
    time4, energy4 = [], []
    
    for line in lines :
        values = line.split()
        time4.append(float(values[4]))
        energy4.append(float(values[5]))

    y4 = np.absolute(np.ones(len(energy4)) * exact_energy - energy4) / np.absolute(exact_energy)

    plt.figure('N=' + str(N) + ' J=' + str(Jfac) + ' h=' + str(hfac))
    # plt.semilogy(time, y1, 'ro')
    plt.semilogy(time2, y2, 'b^')
    # plt.semilogy(time3, y3, 'g--')
    plt.semilogy(time4, y4, 'cs')

    plt.xlabel(r'Imaginary time ($it$)')
    plt.ylabel('Relative Error in Energy')
    # plt.text((max(max(time), max(time2)) + min(min(time), min(time2))) / 2.0, 
    #     (max(max(y1), max(y2)) + min(min(y1), min(y2))) / 2.0, r'$E_{gs} = $' + str(exact_energy))
    plt.title('Energy vs. Imaginary Time in the Ising Model \n' + r'$N =$ ' + str(N) + r', $J =$ ' 
        + str(Jfac) + r', $h =$ ' + str(hfac) + r', $\Delta t = $' + str(delta_t)
        + r', $D =$ ' + str(D))
    plt.show()

    print(energy2[len(energy2) - 1])

def CheckStates() :
    N, d, D = 5, 2, 2
    ess = CommonStates.equalSuperpositionState(N, d, D= D)
    print(ess.innerProd(ess))
    for i in range(0, d ** N) :
        num = np.base_repr(i, base = d)
        values = np.zeros(N)
        for j in range(0, len(num)) :
            values[N - j - 1] = float(num[len(num) - 1 - j])

        mps2 = CommonStates.productState(d, values)

        if (mps2.innerProd(ess) > 0) :
            print(str(values) + ' ' + str(mps2.innerProd(ess)))

def CheckEntropy() :
    N, d, D = 50, 4, 10
    mps = MPS_OBC()
    mps.buildRandomMPS(N, d, D)
    mps.makeLeftNormalized()
    E = []
    for i in range(0, N - 1) :
        E.append(mps.singleCutEntropy(i))
        print(mps.doubleCutEntropy(0, i) - E[i])

    print(max(E))
    print(mps.maximumEntropy())
    # productmps = CommonStates.simpleProductState(N, d, p = int(d / 2))

    # for i in range(0, N - 1) :
    #     print(productmps.singleCutEntropy(i))
    #     print(productmps.doubleCutEntropy(0, i))

    # print(productmps.doubleCutEntropy(1, N - 2))
    # print(entangled_mps.doubleCutEntropy(2, 30))

def CheckEntropy2() :
    N, a, b = 15, 5, 10
    x, y = [], []
    for d in range(2, 15) :
        entangled_mps = CommonStates.maximallyEntangledState(N, d)
        x.append(entangled_mps.doubleCutEntropy(a, b))
        y.append(np.log(d) / np.log(2))

    t = np.linspace(1, len(x), len(x))
    plt.plot(t, x, 'ro')
    plt.plot(t, y, 'b+')
    plt.show()

def CheckEntropy3() :
    N, d, a, b = 10, 2, 3, 5
    x, y = [], []

    for D in range(2, 15) :
        random_mps = MPS_OBC()
        random_mps.buildRandomMPS(N, d, D)
        x.append(random_mps.doubleCutEntropy(a, b))
        y.append(np.log(D) / np.log(2))

    t = np.linspace(1, len(x), len(x))
    plt.plot(t, x, 'ro')
    plt.plot(t, y, 'b+')
    plt.show()

def GenerateIsingGroundState() :
    N, D = 10, 5
    Jfac = 0
    hfac = 0.5
    t = 5.0
    delta_t = 0.005

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac

    H = IsingHamiltonian(N, J, h)

    psi = MPS_OBC()
    psi.buildMPSFromFile(N, 2, 'temp_final_state.txt')

    psi = H.Evolution(t, delta_t, psi, min_dt = 1e-12, thresh = 1e-6)

    print(H.getEnergy(psi))

def IsingEntropyEvolution() :
    N = 10
    Jfac = 10
    hfac = 10
    t = 0.4422
    delta_t = 0.0002
    D = 20

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac

    H = IsingHamiltonian(N, J, h)

    psi = CommonStates.equalSuperpositionState(N, 2, D= D)

    psi = H.Evolution(t, delta_t, psi, imaginary = True, computeEnergy = True, 
        computeAllSingleCutEntropies = True, filename = 'temp3', updateTimeStep = False)

    # plt.figure(str(N) + ' ' + str(Jfac) + ' ' + str(hfac) + ' Entropy')
    # plt.plot(time, entropy, 'ro')
    # plt.xlabel(r'Real time ($t$)')
    # plt.ylabel(r'Entanglement Entropy ($\hbar = 1$)')
    # plt.title('Entropy vs. Time in the Ising Model \n' + r'$N =$ ' + str(N) + r', $J =$ ' 
    #     + str(Jfac) + r', $h =$ ' + str(hfac) + r', $\Delta t = $' + str(delta_t)
    #     + r', $D =$ ' + str(D) + ', Sites [' + str(a) + ',' + str(b) + ']')
    plt.show()

def QuenchedEvolution() :
    N, D = 10, 5
    Jfac = 1.0
    hfac = 1.0
    t = 5.0
    delta_t = 0.005
    # a, b = 4, 5
    #Load ground state from file
    psi = CommonStates.equalSuperpositionState(N, 2, D= D)

    #Perform a local quench on sites a and b (flip 0 with 1)
    # psi.mps[a][[0, 1]] = psi.mps[a][[1, 0]]
    # psi.mps[b][[0, 1]] = psi.mps[b][[1, 0]]

    #evolve the entropy with time

    J = np.ones(N) * Jfac
    h = np.ones(N) * hfac

    H = IsingHamiltonian(N, J, h)

    psi = H.Evolution(t, delta_t, psi, imaginary = False, computeEnergy = False, 
        computeSingleCutEntropy =  True, filename = 'temp2', updateTimeStep = False, cut_i = 4)

    print(H.getEnergy(psi))
    # plt.figure(str(N) + ' ' + str(Jfac) + ' ' + str(hfac) + ' Entropy')
    # plt.plot(time, entropy, 'ro')
    # plt.xlabel(r'Real time ($t$)')
    # plt.ylabel(r'Entanglement Entropy ($\hbar = 1$)')
    # plt.title('Entropy vs. Time in the Ising Model \n' + r'$N =$ ' + str(N) + r', $J =$ ' 
    #     + str(Jfac) + r', $h =$ ' + str(hfac) + r', $\Delta t = $' + str(delta_t)
    #     + r', $D =$ ' + str(D) + ', Sites [' + str(a) + ',' + str(b) + ']')
    # plt.show()

def CheckThermalState() :
    N, D = 5, 5
    Jfac = 1.0
    hfac = 1.0
    Beta = 4.0

    H = IsingHamiltonian(N, J, h)

    rho_half = H.GibbsState(Beta, D, returnMPS = True)

     


CheckEntropy()