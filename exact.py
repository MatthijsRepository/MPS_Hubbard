import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

t_coupling = 1
d = 2
dt = 0.01
steps = 300




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
print(np.real(rho))

print(np.trace(np.matmul(rho, rho)))



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

for i in range(3):
    plt.plot(expvals_rho_2[i])
#plt.title("Rho_2")
plt.ylim((-1.05, 1.05))
plt.grid()
plt.show()

print(expvals_psi[:,-1])
print(expvals_rho_1[:,-1])
print(expvals_rho_2[:,-1])
















