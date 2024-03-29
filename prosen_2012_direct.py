import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigs
from scipy.linalg.lapack import dgesvd, zgesvd

import matplotlib.pyplot as plt

import pickle
import time
from datetime import datetime

from MPS_initializations import initialize_halfstate, initialize_LU_RD


########################################################################################################

class MPS:
    def __init__(self, ID, N, d, chi, is_density):
        self.ID = ID
        self.N = N
        self.d = d
        self.chi = chi
        self.is_density = is_density
        if is_density:
            self.name = "DENS"+str(ID)
        else: 
            self.name = "MPS"+str(ID)
        
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.locsize = np.zeros(N+1, dtype=int)     #locsize tells us which slice of the matrices at each site holds relevant information
        
        self.flipped_factor = np.ones(N)
        self.flipped_factor[:self.N//2*2] *= -1 # all sites start with a flip factor -1, except the last site ONLY in case of odd chain length
        
        self.spin_current_values = np.array([])
        self.normalization = np.array([])
        return
        
    def __str__(self):
        if self.is_density:
            return f"Density matrix {self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
        else:
            return f"MPS {self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
            
    def store(self):
        """ Stores the object to memory using pickle """
        time = str(datetime.now())
        timestr = time[5:7] + time[8:10] + "_" + time[11:13] + time[14:16] + "_"  #get month, day, hour, minute
        
        folder = "data\\" 
        filename = timestr+self.name+"_N"+str(self.N)+"_chi"+str(self.chi)+".pkl"
        
        file = open(folder + filename, 'wb')
        pickle.dump(self, file)
        
        print(f"Stored {filename} to memory")
        pass        
          
    def construct_vidal_supermatrices(self, newchi):
        """ Constructs the matrices for the superket MPDO in Vidal decomposition for this MPS -- see the function 'create superket' """
        sup_Gamma_mat = np.zeros((self.N, self.d**2, newchi, newchi), dtype=complex)
        sup_Lambda_mat = np.zeros((self.N+1, newchi))
        for i in range(self.N):
            sup_Gamma_mat[i,:,:,:] = np.kron(self.Gamma_mat[i], np.conj(self.Gamma_mat[i]))[:,:newchi,:newchi]
            sup_Lambda_mat[i,:] = np.kron(self.Lambda_mat[i], self.Lambda_mat[i])[:newchi]
        sup_Lambda_mat[N,:] = np.kron(self.Lambda_mat[N], self.Lambda_mat[N])[:newchi]
        sup_locsize = np.minimum(self.locsize**2, newchi)
        return sup_Gamma_mat, sup_Lambda_mat, sup_locsize
    
    def contract(self, begin, end):
        """ Contracts the gammas and lambdas between sites 'begin' and 'end' """
        theta = np.diag(self.Lambda_mat[begin,:]).copy()
        theta = theta.astype(complex)
        for i in range(end-begin+1):
            theta = np.tensordot(theta, self.Gamma_mat[begin+i,:,:,:], axes=(-1,1)) #(chi,...,d,chi)
            theta = np.tensordot(theta, np.diag(self.Lambda_mat[begin+i+1]), axes=(-1,1)) #(chi,...,d,chi)
        theta = np.rollaxis(theta, -1, 1) #(chi, chi, d, ..., d)
        return theta
    
    def decompose_contraction(self, theta, i):
        """ decomposes a given theta back into Vidal decomposition. i denotes the leftmost site contracted into theta """
        num_sites = np.ndim(theta)-2 # The number of sites contained in theta
        temp = num_sites-1           # Total number of loops required
        for j in range(temp):

            theta = theta.reshape((self.chi, self.chi, self.d, self.d**(temp-j)))
            theta = theta.transpose(2,0,3,1) #(d, chi, d**(temp-j), chi)
            theta = theta.reshape((self.d*self.chi, self.d**(temp-j)*self.chi))
            X, Y, Z = np.linalg.svd(theta); Z = Z.T
            #This can be done more efficiently by leaving out the Z=Z.T and only doing so in case of j==2
            
            self.Lambda_mat[i+j+1,:] = Y[:self.chi]
            
            X = np.reshape(X[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))
            inv_lambdas = self.Lambda_mat[i+j, :self.locsize[i+j]].copy()
            inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
            X = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i+j],:self.locsize[i+j+1]],axes=(1,1)) #(chi, d, chi)
            X = X.transpose(1,0,2)
            self.Gamma_mat[i+j, :, :self.locsize[i+j],:self.locsize[i+j+1]] = X

            theta_prime = np.reshape(Z[:self.chi*self.d**(temp-j),:self.chi], (self.d**(temp-j), self.chi, self.chi))
            if j==(temp-1):
                theta_prime = theta_prime.transpose(0,2,1)
                inv_lambdas  = self.Lambda_mat[i+j+2, :self.locsize[i+j+2]].copy()
                inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
                tmp_gamma = np.tensordot(theta_prime[:,:self.locsize[i+j+1],:self.locsize[i+j+2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
                self.Gamma_mat[i+j+1, :, :self.locsize[i+j+1],:self.locsize[i+j+2]] = tmp_gamma 
            else:
                theta_prime = theta_prime.transpose(1,2,0)
                #Here we must contract Lambda with V for the next SVD. The contraction runs over the correct index (the chi resulting from the previous SVD, not the one incorporated with d**(temp-j))
                theta_prime = np.tensordot(np.diag(Y[:self.chi]), theta_prime, axes=(1,1))
        return
    
    def apply_singlesite(self, TimeOp, i, normalize):
        """ Applies a single-site operator to site i """
        theta = self.contract(i,i)
        theta_prime = np.tensordot(theta, TimeOp, axes=(2,1)) #(chi, chi, d)
        if normalize:
            theta_prime = theta_prime / np.linalg.norm(theta_prime)
        
        inv_lambdas  = self.Lambda_mat[i].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(np.diag(inv_lambdas), theta_prime, axes=(1,0)) #(chi, chi, d) 
        
        inv_lambdas = self.Lambda_mat[i+1].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(theta_prime, np.diag(inv_lambdas), axes=(1,0)) #(chi, d, chi)
        self.Gamma_mat[i,:,:,:] = np.transpose(theta_prime, (1,0,2))
        return

    def apply_twosite(self, TimeOp, i, normalize):
        """ Applies a two-site operator to sites i and i+1 """
        theta = self.contract(i,i+1) #(chi, chi, d, d)
        theta = theta.reshape(self.chi, self.chi, self.d**2)

        theta_prime = np.tensordot(theta, TimeOp, axes=(2,1))
        theta_prime = theta_prime.reshape((self.chi, self.chi, self.d, self.d))

        self.decompose_contraction(theta_prime, i)
        return 
   
    def TEBD_3_sets(self, TimeOp_leg, TimeOp_rung, Diss_arr, normalize, Diss_bool):
        """ In this method the time evolution is split into 3 different parts that are applied consecutively """
        """ First 0-2 and 3-5, then 1-3 and 4-6, then 2-4 and 5-7 """
        for j in range(0,3,1):
            for i in range(j, self.N-2, 3):
                # Apply swap (2,3) -> (3,2)
                theta = self.contract(i+1,i+2)
                theta = theta.transpose(0,1,3,2)
                self.decompose_contraction(theta, i+1)
                
                self.apply_twosite(TimeOp_leg, i, normalize)
           
                # Apply swap (3,2) -> (2,3)
                theta = self.contract(i+1,i+2)
                theta = theta.transpose(0,1,3,2)
                self.decompose_contraction(theta, i+1)
        if Diss_bool:
            for i in range(len(Diss_arr["index"])):
                self.apply_singlesite(Diss_arr["TimeOp"][i], Diss_arr["index"][i], normalize)
        return
            
        
    def TEBD(self, TimeOp_leg, TimeOp_rung, Diss_arr, normalize, Diss_bool):
        """ TEBD algorithm """
        #"""
        for i in range(0, self.N-3, 4):
            # Apply swap (2,3) -> (3,2)
            theta = self.contract(i+1,i+2)
            theta = theta.transpose(0,1,3,2)
            self.decompose_contraction(theta, i+1)
            
            self.apply_twosite(TimeOp_leg, i, normalize)
            self.apply_twosite(TimeOp_leg, i+2, normalize)
            
            # Apply swap (3,2) -> (2,3)
            theta = self.contract(i+1,i+2)
            theta = theta.transpose(0,1,3,2)
            self.decompose_contraction(theta, i+1)
        #"""
        #"""
        for i in range(2, self.N-3, 4):
            # Apply swap (2,3) -> (3,2)
            theta = self.contract(i+1,i+2)
            theta = theta.transpose(0,1,3,2)
            self.decompose_contraction(theta, i+1)
            
            self.apply_twosite(TimeOp_leg, i, normalize)
            self.apply_twosite(TimeOp_leg, i+2, normalize)
            
            # Apply swap (3,2) -> (2,3)
            theta = self.contract(i+1,i+2)
            theta = theta.transpose(0,1,3,2)
            self.decompose_contraction(theta, i+1)
        #"""
        
        if Diss_bool:
            for i in range(len(Diss_arr["index"])):
                self.apply_singlesite(Diss_arr["TimeOp"][i], Diss_arr["index"][i], normalize)
        return
    
    def sign_flip_check(self, Sz_array):
        """ Checks which sites have experienced a sign flip <Sz> not due to a zero-crossing 
            The global variable flip_threshold determines to ingnore a site due to a likely zero-crossing """
        Sz_array[ np.where(np.abs(Sz_array[:,-2])<flip_threshold) , :] = 0
        flipped_sites = np.sign(Sz_array[:,-2]) - np.sign(Sz_array[:,-1])
        flipped_sites = np.nonzero(flipped_sites)[0]
        self.flipped_factor[flipped_sites] *= -1
        return flipped_sites
    
    def expval(self, Op, site):
        """ Calculates the expectation value of an operator Op for a single site """
        if self.is_density:     #In case of density matrices we must take the trace 
            self.apply_singlesite(Op, site, False)
            a = self.calculate_vidal_inner(NORM_state)
            self.apply_singlesite(Op, site, False)
            return a
            #return np.real(np.tensordot(theta_prime, NORM_state.singlesite_thetas, axes=([0,1,2],[2,1,0])))
        else:
            theta = self.contract(site,site) #(chi, chi, d)
            theta_prime = np.tensordot(theta, Op, axes=(2,1)) #(chi, chi, d)
            return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2],[0,1,2])))
        
    def calculate_vidal_inner(self, MPS2):
        """ Calculates the inner product of the MPS with another MPS """
        m_total = np.eye(self.chi)
        temp_gammas, temp_lambdas = MPS2.Gamma_mat, MPS2.Lambda_mat  #retrieve gammas and lambdas of MPS2
        for j in range(0, self.N):
            st1 = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[j,:,:,:],np.diag(temp_lambdas[j+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st1), st2, axes=(0,0)) #(chi, chi, chi, chi)    
            m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))    
        return np.real(m_total[0,0])
    
    def calculate_norm(self):
        """ Calculates the norm of the MPS """
        if self.is_density:
            return self.calculate_vidal_inner(NORM_state)
        else: 
            return self.calculate_vidal_inner(self)
    
    def time_evolution(self, TimeEvol_obj, normalize, steps, track_normalization):
        if TimeEvol_obj.is_density != self.is_density:
            print("Error: time evolution operator type does not match state type (MPS/DENS)")
            return
        #### Initializing operators and expectation value arrays
        TimeOp_leg = TimeEvol_obj.TimeOp_leg
        TimeOp_rung = TimeEvol_obj.TimeOp_rung
        
        Diss_arr = TimeEvol_obj.Diss_arr
        Diss_bool = TimeEvol_obj.Diss_bool
        
        Sz_expvals = np.zeros((self.N, steps),dtype=float)
        # Defining operator used for expectation value calculation
        if self.is_density:
            Sz_exp_op = np.kron(Sz, np.eye(d))
        else:
            Sz_exp_op = Sz
        
        if track_normalization:
            Normalization = np.zeros(steps)        
           
        #### Time evolution steps
        print(f"Starting time evolution of {self.name}")
        for t in range(steps):
            if (t%20==0):
                print(t)
                
            if track_normalization:
                if normalize:
                    Normalization[t] = self.calculate_norm()
                    self.Lambda_mat[1:self.N] *= (1/Normalization[t])**(1 / (self.N-1))    
                    Normalization[t] = 1
                    self.normalization = np.append(self.normalization, Normalization[t])
                else:
                    Normalization[t] = self.calculate_norm()
                    self.normalization = np.append(self.normalization, Normalization[t])
            
            for i in range(self.N):
                Sz_expvals[i,t] = self.expval(Sz_exp_op, i)
                
            if (t>=1 and self.is_density):
                Sz_expvals[:,t] *= self.flipped_factor
                sign_flips = self.sign_flip_check(Sz_expvals[:,t-1:t+1].copy())
                Sz_expvals[sign_flips,t] *= -1
                
            self.TEBD(TimeOp_leg, TimeOp_rung, Diss_arr, normalize, Diss_bool)
            #self.TEBD_3_sets(TimeOp_leg, TimeOp_rung, Diss_arr, normalize, Diss_bool)
        
            
        #### Plotting expectation values
        time_axis = np.arange(steps)*abs(TimeEvol_obj.dt)
        
        for i in range(self.N):
            plt.plot(time_axis, Sz_expvals[i,:], label="Site "+str(i))
        plt.title(f"<Sz> of {self.name} over time")
        plt.xlabel("Time")
        plt.ylabel("Sz")
        plt.ylim((-1.05,1.05))
        plt.grid()
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.show()
        
        if track_normalization:
            plt.plot(time_axis, Normalization)
            plt.title(f"Normalization of {self.name} over time")
            plt.xlabel("Time")
            plt.ylabel("Normalization")
            plt.grid()
            plt.show()
        return
       

########################################################################################  

class Time_Operator:
    def __init__(self,N, d, t_hopping, U_coulomb, s_coup, dt, Diss_bool, is_density, use_CN):
        self.N = N
        self.d = d
        self.t_hopping = t_hopping
        self.U_coulomb = U_coulomb
        self.s_coup = s_coup
        self.dt = dt
        self.is_density = is_density
        self.Diss_bool = Diss_bool
        self.use_CN = use_CN
        
        if isinstance(dt, complex):     
            self.is_imaginary = True
        else:
            self.is_imaginary = False
               
        #### Creating Hamiltonian and Time operators
        #### Note: Ham_energy is the Hamiltonian to be used for energy calculation
        if self.is_density:
            self.Ham_leg, self.Ham_rung = self.Create_Dens_Ham()
            #Ham_energy is removed due to complications with the new Hamiltonian
        else:
            self.Ham_leg, self.Ham_rung = self.Create_Ham()
        
        self.TimeOp_leg = self.Create_TimeOp(self.Ham_leg, self.dt, self.use_CN)
        self.TimeOp_rung = self.Create_TimeOp(self.Ham_rung, self.dt, self.use_CN)
        
        if (self.is_density and self.Diss_bool):
            self.Diss_arr = self.Create_Diss_Array(self.s_coup)
            self.Calculate_Diss_TimeOp(self.dt, self.use_CN)
        else:
            self.Diss_arr = None
        return
    
    def Create_Ham(self):
        """ Create Hamiltonian for purestate """
        H_arr_leg = -self.t_hopping/2 * (np.kron(Sx, Sx) + np.kron(Sy, Sy))
        H_arr_rung = self.U_coulomb/4 * np.kron(Sz+np.eye(self.d), Sz+np.eye(self.d))
        return H_arr_leg, H_arr_rung
        
    def Create_Dens_Ham(self):
        """ create effective Hamiltonian for time evolution of the density matrix """
        Sx_arr = np.array([np.kron(Sx, np.eye(self.d)) , np.kron(np.eye(self.d), Sx)])
        Sy_arr = np.array([np.kron(Sy, np.eye(self.d)) , np.kron(np.eye(self.d), Sy)])
        Sz_arr = np.array([np.kron(Sz, np.eye(self.d)) , np.kron(np.eye(self.d), Sz)])
        Identity = np.eye(self.d**2)
         
        """ Calculates (H otimes I) - (I otimes H)* """
        H_arr_leg = np.ones((2, self.d**4, self.d**4), dtype=complex)
        H_arr_rung = np.ones((2, self.d**4, self.d**4), dtype=complex)
        for i in range(2):
            #x and y hopping connections
            H_arr_leg[i] = -self.t_hopping/2 * (np.kron(Sx_arr[i], Sx_arr[i]) + np.kron(Sy_arr[i], Sy_arr[i]) )#+ np.kron(Sz_arr[i], Sz_arr[i])     )
            #Coulomb interactions
            H_arr_rung[i] = self.U_coulomb/4 * np.kron(Sz_arr[i]+Identity, Sz_arr[i]+Identity)

        #Note: H_arr[0] is the correct Hamiltonian to use for energy calculations
        #return (H_arr_leg[0] - np.conj(H_arr_leg[1])), (H_arr_rung[0] - np.conj(H_arr_rung[1]))    
        return (H_arr_leg[0] - np.transpose(H_arr_leg[1])), (H_arr_rung[0] - np.transpose(H_arr_rung[1]))    


    def Create_TimeOp(self, Ham, dt, use_CN):
        if use_CN:
            U = self.create_crank_nicolson(Ham, dt)
        else:
            U = expm(-1j*dt*Ham)
    
        U = np.around(U, decimals=15)        #Rounding out very low decimals 
        return U

    def create_crank_nicolson(self, H, dt):
        """ Creates the Crank-Nicolson operator from a given Hamiltonian """
        H_top=np.eye(H.shape[0])-1j*dt*H/2
        H_bot=np.eye(H.shape[0])+1j*dt*H/2
        return np.linalg.inv(H_bot).dot(H_top)

    def Calculate_Diss_site(self, Lind_Op):
        """ Creates the dissipative term for a single site """
        """ Lind_Op is shape (k,d,d) or (d,d) -- the k-index is in case multiple different lindblad operators act on a single site """
        Diss = np.zeros((self.d**2, self.d**2), dtype=complex)
        if Lind_Op.ndim==2:     #If only a single operator is given, this matrix is used
            Diss += 2*np.kron(Lind_Op, np.conj(Lind_Op))
            Diss -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op)), Lind_Op), np.eye(self.d))
            Diss -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op), np.conj(Lind_Op)))
        else:                   #If multiple matrices are given, the sum of Lindblad operators is used
            for i in range(np.shape(Lind_Op)[0]):
                Diss += 2*np.kron(Lind_Op[i], np.conj(Lind_Op[i]))
                Diss -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d))
                Diss -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i])))
        return Diss
    
    def Create_Diss_Array(self, s_coup):
        """ Creates the array containing dissipative term, where 'index' stores the site the corresponding Lindblad operators couple to """
        Diss_arr = np.zeros((), dtype=[
            ("index", int, 4),
            ("Operator", complex, (2, self.d**2, self.d**2)),
            ("TimeOp", complex, (2, self.d**2, self.d**2))
            ])
        
        Diss_arr["index"][0] = 0
        Diss_arr["Operator"][0,:,:] = self.Calculate_Diss_site(np.array([mu_min*Sp, mu_plus*Sm]))
        
        Diss_arr["index"][2] = N-2
        Diss_arr["Operator"][2,:,:] = self.Calculate_Diss_site(np.array([mu_plus*Sp, mu_min*Sm]))
    
    
        Diss_arr["index"][1] = 1
        Diss_arr["Operator"][1,:,:] = self.Calculate_Diss_site(np.array([mu_min*Sp, mu_plus*Sm]))
    
        Diss_arr["index"][3] = N-1
        Diss_arr["Operator"][3,:,:] = self.Calculate_Diss_site(np.array([mu_plus*Sp, mu_min*Sm]))
        return Diss_arr
    
    def Calculate_Diss_TimeOp(self, dt, use_CN):
        """ Calculates the dissipative time evolution operators """
        for i in range(len(self.Diss_arr["index"])):
            if use_CN:
                temp = self.create_crank_nicolson(self.Diss_arr["Operator"][i], dt)
            else:
                temp = expm(dt*self.Diss_arr["Operator"][i])
            temp = np.around(temp, decimals=15)    #Rounding out very low decimals 
            self.Diss_arr["TimeOp"][i,:,:] = temp
        return



###################################################################################

def load_state(folder, name, new_ID):
    """ loads a pickled state from folder 'folder' with name 'name' - note: name must include .pkl """
    filename = folder + name
    with open(filename, 'rb') as file:  
        loaded_state = pickle.load(file)
    globals()[loaded_state.name] = loaded_state
    
    loaded_state.ID = new_ID
    if loaded_state.is_density:
        loaded_state.name = "DENS"+str(new_ID)
    else: 
        loaded_state.name = "MPS"+str(new_ID)
    return loaded_state
    

def create_superket(State, newchi):
    """ create MPS of the density matrix of a given MPS """
    gammas, lambdas, locsize = State.construct_vidal_supermatrices(newchi)
    
    name = "DENS" + str(State.ID)
    newDENS = MPS(State.ID, State.N, State.d**2, newchi, True)
    newDENS.Gamma_mat = gammas
    newDENS.Lambda_mat = lambdas
    newDENS.locsize = locsize
    globals()[name] = newDENS
    return newDENS

def create_maxmixed_normstate():
    """ Creates vectorised density matrix of an unnormalized maximally mixed state, used to calculate the trace of a vectorised density matrix """
    """ since to obtain rho11 + rho22 you must take inner [1 0 0 1] [rho11 rho12 rho21 rho22]^T without a factor 1/sqrt(2) in front """
    lambdas = np.zeros((N+1,newchi))
    lambdas[:,0]= 1
    
    gammas = np.zeros((N,d**2,newchi,newchi), dtype=complex)
    diagonal = (1+d)*np.arange(d)
    gammas[:,diagonal, 0, 0] = 1        #/2  #/np.sqrt(2)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,newchi**2)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum((d**2)**arr, newchi**2)
    
    NORM_state = MPS(0, N, d**2, newchi, True)
    NORM_state.Gamma_mat = gammas
    NORM_state.Lambda_mat = lambdas
    NORM_state.locsize = locsize
    return NORM_state

def calculate_thetas_singlesite(state):
    """ contracts lambda_i gamma_i lambda_i+1 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    #Note, the lambda matrices are just a factor 1, it is possible to simply return a reshaped gamma matrix
    #temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    #return np.tensordot(temp, np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi,d,chi)
    return state.Gamma_mat[0].transpose(0,2,1)

def calculate_thetas_twosite(state):
    """ contracts lambda_i gamma_i lambda_i+1 gamma_i+1 lambda_i+2 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    temp = np.tensordot(temp,np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi, d, chi) 
    temp = np.tensordot(temp, state.Gamma_mat[1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
    return np.tensordot(temp,np.diag(state.Lambda_mat[2,:]), axes=(3,0)) #(chi, d, d, chi)

def calculate_thetas_threesite(state):
    """ contracts lambda_i gamma_i lambda_i+1 gamma_i+1 lambda_i+2 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    #temp = np.tensordot(temp,np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi, d, chi) 
    temp = np.tensordot(temp, state.Gamma_mat[1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
    temp = np.tensordot(temp, state.Gamma_mat[2,:,:,:],axes=(3,1)) #(chi, d, d, d, chi)
    return temp







####################################################################################
t0 = time.time()
#### Simulation variables
N=8
d=2
chi=20      #MPS truncation parameter
newchi=20   #DENS truncation parameter

#im_steps = 0
#im_dt = -0.03j
steps=100
dt = 0.02

normalize = False
use_CN = False #choose if you want to use Crank-Nicolson approximation
Diss_bool = False

flip_threshold = 0.02 #Threshold below which an <Sz> sign flip is not flagged as being caused by the SVD


#### Hamiltonian and Lindblad constants
t_hopping = 1
U_coulomb = 0

s_coup=1
mu=0.01
mu_plus = np.sqrt(s_coup*(1+mu))
mu_min = np.sqrt(s_coup*(1-mu))  


#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])


#### NORM_state initialization
NORM_state = create_maxmixed_normstate()
NORM_state.singlesite_thetas = calculate_thetas_singlesite(NORM_state)
NORM_state.twosite_thetas = calculate_thetas_twosite(NORM_state)
NORM_state.threesite_thetas = calculate_thetas_threesite(NORM_state)


#### Loading and saving states
loadstate_folder = "data\\"
#loadstate_filename = "0218_1613_DENS1_N8_chi25.pkl"
#loadstate_filename = "0220_2126_DENS1_N8_chi25.pkl"

loadstate_filename = "0220_2204_DENS1_N8_chi25.pkl"

save_state_bool = False
load_state_bool = False


####################################################################################

def main():
    #load state or create a new one
    if load_state_bool:
        DENS1 = load_state(loadstate_folder, loadstate_filename, 1)
    else:
        MPS1 = MPS(1, N,d,chi, False)
        #MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_halfstate(N,d,chi)
        MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_LU_RD(N,d,chi, scale_factor = 1)
        
        DENS1 = create_superket(MPS1, newchi)

    #creating time evolution object
    TimeEvol_obj_DENS = Time_Operator(N, d, t_hopping, U_coulomb, s_coup, dt, Diss_bool, True, use_CN)
    TimeEvol_obj_MPS = Time_Operator(N, d, t_hopping, U_coulomb, s_coup, dt, False, False, use_CN)
        
    #time evolution of the state
    DENS1.time_evolution(TimeEvol_obj_DENS, normalize, steps, True)
    MPS1.time_evolution(TimeEvol_obj_MPS, normalize, steps, True)
    ####################################################### ^BOOL^: whether to track normalization    
   

    if save_state_bool:
        DENS1.store()
        #MPS1.store()
    """
    final_Sz = np.zeros(N)
    for i in range(N):
        final_Sz[i] = DENS1.expval(np.kron(Sz, np.eye(d)), i) * DENS1.flipped_factor[i]
    plt.plot(final_Sz, linestyle="", marker=".")
    plt.xlabel("Site")
    plt.ylabel("<Sz>")
    plt.grid()
    plt.title(f"<Sz> for each site after {steps} steps with dt={dt}")
    plt.show()  
    """
    pass

main()

elapsed_time = time.time()-t0
print(f"Elapsed simulation time: {elapsed_time}")