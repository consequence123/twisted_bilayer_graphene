#coding:utf-8
from numpy import*
import matplotlib.pyplot as plt
import pdb
from multiprocessing import Pool
from itertools import repeat
import numexpr as ne


sigma0 = diag([1.,1])
sigma1 = array([[0.,1],[1,0]])
sigma2 = array([[0.,-1j],[1j,0]])
sigma3 = array([[1.,0],[0,-1]])

def rotate(angle):
    return array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])
def general_kron(a,b):
    ashape=[i for i in a.shape]
    bshape=b.shape
    for i in bshape:
        ashape.append(1)
    c=a.reshape(ashape)
    return c*b
def fermi_distribution(beta,energy):
    return 1-1./(exp(-energy*beta)+1.)
def Euclidean_distance(x,y):
    l = ((x-y)**2).sum()
    return sqrt(l)

class lattice():
    def get_kmesh(self,kx,ky):
        pass
    def get_distance(self,p,q):
        a = ((p-q)**2).sum()
        return sqrt(a)
    def get_kvector(self,alpha,betha,Nx,Ny):
        gamma=array([0,0,1.])
        V=dot(alpha,cross(betha,gamma))
        kalpha=cross(betha,gamma)[:-1]*2*pi/V
        kbetha=cross(gamma,alpha)[:-1]*2*pi/V
        return kalpha/Nx,kbetha/Ny
    def get_kmesh(self,alpha,betha,Nx,Ny):
        kalpha,kbetha=self.get_kvector(alpha,betha,Nx,Ny)
        X=arange(0,Nx)
        Y=arange(0,Ny)
        kmesh=X.reshape(-1,1,1)*kalpha.reshape(1,1,-1)+Y.reshape(1,-1,1)*kbetha.reshape(1,1,-1)
        return kmesh
    def get_line(self,p,q,n):
        x=linspace(p[0],q[0],n,endpoint=0)
        y=linspace(p[1],q[1],n,endpoint=0)
        z = [array([x[i],y[i]]) for i in xrange(x.shape[0])]
        return z
    def get_angle(self,vector1,vector2):
        l1 = sqrt((vector1**2).sum())
        l2 = sqrt((vector1**2).sum())
        d = (vector1*vector2).sum()
        return arccos(d/(l1*l2))
    def plot_line(self,A,B):
        plt.plot([A[0],B[0]],[A[1],B[1]],color = 'blue')
    def get_Mpoint(self,kalpha,kbetha):
        return (kalpha + kbetha)*0.5
    def get_Kpoint(self,kalpha,kbetha):
        return (2*kalpha + kbetha)/3.
        #need to be correct
    def generate_path(self,p,q,n):
        x=linspace(p[0],q[0],n,endpoint=0)
        y=linspace(p[1],q[1],n,endpoint=0)
        z=[array([x[i],y[i]]) for i in xrange(n)]
        return z
    def generate_loop(self,Kpoint,Mpoint,Kp_point):
        Gamma = array([0,0])
        distance_G2K = Euclidean_distance(Kpoint,Gamma)
        distance_K2M = Euclidean_distance(Mpoint,Kpoint)
        distance_M2G = Euclidean_distance(Gamma,Mpoint)
        distance_G2Kp = Euclidean_distance(Gamma,Kp_point)
        G2K_number = 500
        K2M_number = int(G2K_number*distance_K2M/distance_G2K)
        M2G_number = int(G2K_number*distance_M2G/distance_G2K)
        G2Kp_number = int(G2K_number*distance_G2Kp/distance_G2K)
        #print G2K_number,K2M_number,M2G_number
        G2K = self.generate_path(Gamma,Kpoint,G2K_number)
        K2M = self.generate_path(Kpoint,Mpoint,K2M_number)
        M2G = self.generate_path(Mpoint,Gamma,M2G_number)
        G2Kp = self.generate_path(Gamma,Kp_point,G2Kp_number)
        G2K.extend(K2M)
        G2K.extend(M2G)
#        G2K.extend(G2Kp)
        return array(G2K)
    def stack_hamitonian(self,hamitonian_list,sigma_list = 0):
        l = len(hamitonian_list)
        H = 0.
        for i in xrange(l):
            loc = zeros([l,l])
            loc[i,i] = 1.
            H = H + kron(hamitonian_list[i],loc)
        return H
    def block_matrix_element(self,shape,loc,matrix):
        H = zeros(shape)
        H[loc[0],loc[1]] = 1.
        return kron(matrix,H)
    def mirror_transform(self,kmesh,mirror_direction):
        aa = kmesh.dot(mirror_direction)
        bb = (mirror_direction**2).sum()
        kshape = [i for i in kmesh.shape]
        mshape = [1 for i in kmesh.shape]
        mshape[-1] = -1
        cc = mirror_direction.reshape(mshape)
        kshape[-1] = -1
        km = kmesh - aa.reshape(kshape)*cc/bb
        return kmesh - 2*km
    def get_polarmesh(self, angle, l_min,l_max,delta):
        l = arange(l_min,l_max,delta)
        polarmesh = zeros([l.shape[0],2])
        polarmesh[:,0] = l*cos(angle)
        polarmesh[:,1] = l*sin(angle)
        return polarmesh

class raman_scatter():
    def raman_xx(self,maxgap_direction):
        ni=maxgap_direction
        ns=maxgap_direction
        return ni,ns
    def raman_yy(self,mingap_direction):
        ni=mingap_direction
        ns=mingap_direction
        return ni,ns
    def raman_A1B2(self):
        ni=array([1,1])/sqrt(2)
        ns=array([1,1])/sqrt(2)
        return ni,ns
    def raman_A1B1(self):
        ni=array([0.,1.])
        ns=array([0.,1.])
        return ni,ns
    def raman_B1(self):
        ni=array([1,1])/sqrt(2)
        ns=array([1,-1])/sqrt(2)
        return ni,ns
    def raman_B2(self):
        ni=array([1.,0.])
        ns=array([0.,1.])
        return ni,ns
    def raman_linhard(self,deltafk,deltaEk,z):
        Lin=deltafk/(z+deltaEk)
        return Lin
    def raman_gamma(self,vertex,Uk):
        gammak=einsum('ijkl,ijlm->ijkm',conjugate(transpose(Uk,(0,1,3,2))),vertex)
        gammak=einsum('ijkl,ijlm->ijkm',gammak,Uk)
        return gammak
    def get_vertex1(self,normaldimension):
        return kron(eye(normaldimension),sigma3)
    def raman_gamma1(self,vertex,Uk):
        gammak=einsum('ijkl,lm->ijkm',conjugate(transpose(Uk,(0,1,3,2))),vertex)
        gammak=einsum('ijkl,ijlm->ijkm',gammak,Uk)
        return gammak
    def raman_single_z(self,z,deltaEk,deltafk,g_g,g_g1,g1_g,g1_g1,screen):
        Nx = deltaEk.shape[0]
        Ny = deltaEk.shape[1]
        Lin = ne.evaluate('deltafk/(z+deltaEk)')
        kai = -einsum("ijkl,ijlk->",Lin,g_g)/Nx/Ny
        if screen > 0 :
            kai1 = -einsum("ijkl,ijlk->",Lin,g_g1)/Nx/Ny
            kai2 = -einsum("ijkl,ijlk->",Lin,g1_g)/Nx/Ny
            kai3 = -einsum("ijkl,ijlk->",Lin,g1_g1)/Nx/Ny
            deltakai = -kai1*kai2/kai3
            kai = kai + deltakai
        return kai
    def raman_vertex(self,ni,ns,kmesh):
        eps = 1e-7
        qi = ni.reshape(1,1,-1)*eps
        qs = ns.reshape(1,1,-1)*eps
        kmesh_is = kmesh + qi + qs
        kmesh_i = kmesh + qi
        kmesh_s = kmesh + qs
        vertex = self.get_TSChk(kmesh_is) + self.get_TSChk(kmesh)
        vertex = vertex - self.get_TSChk(kmesh_i) - self.get_TSChk(kmesh_s)
        vertex = vertex/(eps**2)
        return vertex
    def raman_intensity(self,Ek,Uk,vertex,vertex1,Z,ni,ns,screen,beta=80):
        Nx,Ny = Ek.shape[0],Ek.shape[1]
        gamma1 = self.raman_gamma1(vertex1,Uk)
#        gamma = self.raman_gamma1(vertex,Uk)
        gamma = self.raman_gamma(vertex,Uk)
        g_g = gamma*transpose(gamma,(0,1,3,2))
        g_g1 = gamma*transpose(gamma1,(0,1,3,2))
        g1_g = gamma1*transpose(gamma,(0,1,3,2))
        g1_g1 = gamma1*transpose(gamma1,(0,1,3,2))
#        pdb.set_trace()
        Ekm = Ek.reshape(Nx,Ny,-1,1)
        Ekn = Ek.reshape(Nx,Ny,1,-1)
        deltaEk = Ekm - Ekn
        fk = fermi_distribution(beta,Ek)
        fkm = fk.reshape(Nx,Ny,-1,1)
        fkn = fk.reshape(Nx,Ny,1,-1)
        deltafk = fkm - fkn
        #pdb.set_trace()

#        cores=1
#        chunk=len(Z)//cores
#        if chunk==0:
#            chunk=1
#        
#        def raman_map_func(z):
#            return self.omega_single_z(z,deltaEk,deltafk,gammak)
        #ggg=partial(self.omega_single_z,deltaEk=deltaEk,deltafk=deltafk,gammak=gammak)
        #def dummy():
            #pool = ThreadPool(cores)
            #kai = pool.map(raman_map_func, Z)
            #pool.close()
            #pool.join()       
        #lambda_func=lambda z:self.omega_single_z(z,deltaEk,deltafk,gammak)
               
        #def joblib_run():
            #kai=Parallel(n_jobs=cores)(delayed(self.omega_single_z)(z,deltaEk,deltafk,gammak) for z in Z)
        #def joblib_partial_run():
            #kai=Parallel(n_jobs=cores)(delayed(ggg)(z) for z in Z)
        #def pool_partial():
            #with Pool(cores) as p:
                #kai=p.map(ggg,Z,chunk)
            #return kai
#        def pool_itertool():
#            with Pool(cores) as p:
#                kai=p.starmap(self.omega_single_z, zip(Z,repeat(deltaEk),repeat(deltafk),repeat(gammak)),chunk)
#            return kai
        #t0=time.time()
        #kai=pool_itertool()
        #t1=time.time();print(t1-t0)
        kai = [self.raman_single_z(z,deltaEk,deltafk,g_g,g_g1,g1_g,g1_g1,screen) for z in Z]
        kai = array(kai)
        return kai.imag/pi
    def raman_save(self,w,RI):
        results=zeros([w.shape[0],2])
        results[:,0]=w
        results[:,1]=RI
        savetxt("RI.dat",results)
    def raman_plot(self,w,RI,lab):
        plt.plot(w,RI,label=lab)
                            
        




