import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

N=3
t_coupling = 1
d = 2
dt = 0.02
steps = 150




Ham = -t_coupling/2 * (np.kron(np.kron(Sx, np.eye(d)), Sx) + np.kron(np.kron(Sy, np.eye(d)), Sy))

U = expm(-1j*dt*Ham)
U_dagger = expm(1j*dt*Ham)


psi_temp = np.array([1,0], dtype=complex)
psi_temp = np.kron(psi_temp, np.array([np.sqrt(1/2), np.sqrt(1/2)], dtype=complex))
psi_temp = np.kron(psi_temp, np.array([0,1], dtype=complex))

psi_2_temp = np.array([np.sqrt(1/2), np.sqrt(1/2)], dtype=complex)
psi_2_temp = np.kron(psi_2_temp, np.array([np.sqrt(1/2), np.sqrt(1/2)], dtype=complex))
psi_2_temp = np.kron(psi_2_temp, np.array([np.sqrt(1/2), np.sqrt(1/2)], dtype=complex))


#psi_temp = np.array([np.sqrt(4.5/5), np.sqrt(0.5/5)])
#psi_temp = np.kron(psi_temp, np.array([np.sqrt(1/2), np.sqrt(1/2)]))
#psi_temp = np.kron(psi_temp, np.array([np.sqrt(0.5/5), np.sqrt(4.5/5)]))

psi = np.ones((1,d**3), dtype=complex)
psi *= psi_temp
psi = psi.T

psi_2 = np.ones((1,d**3), dtype=complex)
psi_2 *= psi_2_temp
psi_2 = psi_2.T



#Creating density matrix
rho = 1/2* (np.matmul(psi, np.conj(psi.T)) + np.matmul(psi_2, np.conj(psi_2.T)))


theta = np.zeros((20, 20, 4, 4, 4), dtype=complex)
temp = rho.reshape((2,2,2,2,2,2))
temp = temp.transpose((0,3,1,4,2,5))
temp = temp.reshape((4,4,4))
theta[0,0] = temp
#np.save("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_Hubbard\\temp.npy",theta)
#a = np.load("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_Hubbard\\temp.npy")


Sz_exp = np.zeros((3, d**3, d**3))
Sz_exp[0] = np.kron(Sz, np.eye(d**2))
Sz_exp[1] = np.kron(np.kron(np.eye(d), Sz), np.eye(d))
Sz_exp[2] = np.kron(np.eye(d**2), Sz)

def calc_expvals(State):
    result = np.zeros(3, dtype=complex)
    for i in range(3):
        result[i] = np.matmul(np.conj(State.T), np.matmul(Sz_exp[i], State))
    result = np.round(result, decimals=15)
    return np.real(result)

def calc_dens_expvals(State):
    result = np.zeros(3, dtype=complex)
    for i in range(3):
        result[i] = np.trace(np.matmul(Sz_exp[i], State))
    result = np.round(result, decimals=15)
    return np.real(result)


print(calc_dens_expvals(np.matmul(rho, rho)))


rho_1 = np.ones((d**3, d**3), dtype=complex)
rho_2 = np.ones((d**3, d**3), dtype=complex)
rho_1 *= rho
rho_2 *= rho


expvals_psi = np.zeros((3,steps+1))
expvals_rho_1 = np.zeros((3,steps+1))
expvals_rho_2 = np.zeros((3,steps+1))

expvals_psi[:,0] = calc_expvals(psi)
expvals_rho_1[:,0] = calc_dens_expvals(rho_1)
expvals_rho_2[:,0] = calc_dens_expvals(rho_2)


for i in range(steps):
    #psi = np.matmul(U, psi)
    #expvals_psi[:,i+1] = calc_expvals(psi)
    
    #rho_1 = np.matmul(U, np.matmul(rho_1,U_dagger))    
    #expvals_rho_1[:,i+1] = calc_dens_expvals(rho_1)
    
    rho_2 = np.matmul(np.matmul(U, rho_2), U_dagger)
    expvals_rho_2[:,i+1] = calc_dens_expvals(rho_2)
    
    


#for i in range(3):
#    plt.plot(expvals_psi[i])
#plt.title("State vector")
#plt.show()

#for i in range(3):
#    plt.plot(expvals_rho_1[i])
#plt.title("Rho_1")
#plt.show()

x_range = np.linspace(0,dt*steps, steps+1)

for i in range(3):
    plt.plot(x_range, expvals_rho_2[i], label=f"Site {i}")
#plt.title("Rho_2")
plt.ylim((-1.05, 1.05))
plt.title("Exact solution")
plt.ylabel("<Sz>")
plt.xlabel("Time")
plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.grid()
plt.show()

print(expvals_psi[:,-1])
print(expvals_rho_1[:,-1])
print(expvals_rho_2[:,-1])





theta = np.transpose(np.eye(16).reshape((2,2,2,2,2,2,2,2)), (0,4,1,5,2,6,3,7))
#theta = np.transpose(np.eye(8).reshape((2,2,2,2,2,2)), (0,3,1,4,2,5))
theta = theta.reshape((4,4,4,4))
np.save("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_Hubbard\\I_twosite.npy",theta)


a=0.176776695 # = 2^(-5/2)
# bijbehorende lambda 2.82842712 = 2^(3/2)

#structuur Gammas
# 2^(-1/2)
# 2^(-4/2) én 2^(-5/2)
# 2^(-1/2)
# structuur Lambdas
# 2^(3/2)
# 2^(3/2)


b = 0.144337567
print("here")


print(a)

print(1/2**(1/2))
print(1/2**(4/2))
print(1/2**(5/2))
print(2**(3/2))



test_expvals = np.load("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_Hubbard\\expvals_hope_good.npy")

test_expvals[2] *= -1
test_expvals[0,40:118] *= -1
test_expvals[2,40:118] *= -1


diff = test_expvals - expvals_rho_2
diff = np.round(diff, decimals=13)
print(diff[np.nonzero(diff)])


for i in range(3):
    plt.plot(test_expvals[i])
    




test = np.kron(np.array([1,0,0,1]), np.array([1,0,0,1]))
test = np.kron(test, test)

test = test.reshape((4,64))
U1, S, V = np.linalg.svd(test, full_matrices=False)
U1 = np.matmul(U1, np.diag(np.sqrt(S)))
V = np.matmul(np.diag(np.sqrt(S)), V)

print(np.round(U1, decimals=12))
print()


V = V.reshape(16, 16)
U2, S, V = np.linalg.svd(V, full_matrices=False)
U2 = np.matmul(U2, np.diag(np.sqrt(S)))
V = np.matmul(np.diag(np.sqrt(S)), V)

print(np.round(U2, decimals=12))
print()

U3, S, V = np.linalg.svd(V, full_matrices=False)
U3 = np.matmul(U3, np.diag(np.sqrt(S)))
V = np.matmul(np.diag(np.sqrt(S)), V)

print(np.round(U3, decimals=12))
print()
print(np.round(V, decimals=12))

print(np.shape(U1))
print(np.shape(U2))
print(np.shape(U3))
print(np.shape(V))



newchi=20
wow = np.zeros((4, newchi, newchi))






