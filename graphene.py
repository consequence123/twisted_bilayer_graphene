#coding:utf-8
from numpy import*
import matplotlib.pyplot as plt 
from raman_toolkit import sigma0,sigma1,sigma2,sigma3,general_kron,rotate,raman_scatter,lattice
import pdb

class graphen(raman_scatter,lattice):
    def __init__(self,mu,Nx,Ny):
        self.mu = mu; self.t1 = -1.
        self.alpha = array([1.,0.,0.])
        self.betha = array([1./2,sqrt(3)/2.,0.])
        self.Nx = Nx; self.Ny = Ny
    def get_tba(self,k):
        alpha = self.alpha
        betha = self.betha
        mu, t1 = self.mu, self.t1
        angle = 2*pi/3
        alpha = alpha[:-1]
        betha = betha[:-1]
        delta1=(alpha+betha)/3
        delta2=rotate(angle).dot(delta1)
        delta3=-delta1-delta2
        #print delta1,delta2,delta3
        phi1=k.dot(delta1)
        phi2=k.dot(delta2)
        phi3=k.dot(delta3)
        xk=-t1*(cos(phi1)+cos(phi2)+cos(phi3))
        yk=-t1*(sin(phi1)+sin(phi2)+sin(phi3))
        #pdb.set_trace()
        hk=general_kron(xk,sigma1)
        hk=hk-general_kron(yk,sigma2)
        hk+=-mu*sigma0
        return hk
    def get_twisted_tba(self,kmesh,t_tunnel,lambda_tunnel):
        hk = self.get_tba(kmesh)
        bond = self.single_honeycomb() + array([1,0])
        xk,yk = 0,0
        for b in bond:
            xk = xk + cos(kmesh.dot(b))
            yk = yk + sin(kmesh.dot(b))
        xk = xk*t_tunnel*lambda_tunnel
        yk = yk*t_tunnel*lambda_tunnel
        Tunneling = general_kron(xk,sigma1)-general_kron(yk,sigma2) + t_tunnel*array([[1,0],[0,0]])
        Hk = kron(hk,sigma0) + kron(Tunneling,sigma1)
        return Hk
    def get_TBG_model(self,kmesh,t2,tt1,tt2):
        hk = self.get_tba(kmesh)
        bondnnn = zeros([6,2])
        bondnnn[0] = array([0,sqrt(3)])
        r_nnn = rotate(pi/3)
        for i in xrange(1,6):
            bondnnn[i] = r_nnn.dot(bondnnn[i-1])
        bondnn = zeros([3,2])
        bondnn[0] = (self.alpha + self.betha)[:-1]/3
        r_nn = rotate(2*pi/3)
        for i in xrange(1,3):
            bondnn[i] = r_nn.dot(bondnn[i-1])

        isigma2 = 1j*sigma2
        sigmap = (sigma1 + isigma2)/2.
        sigmam = sigmap.T.conj()

        warp_nnn = 0.; 
        fsign = array([1,-1,1,-1,1,-1.])
        for i in xrange(6):
            phi = kmesh.dot(bondnnn[i])
            warp_nnn = warp_nnn + general_kron(exp(1j*phi),sigma0)*fsign[i] 
        warp_nnn = warp_nnn*tt2

#        fsign = fsign*1j; hop_nnn = 0.
#        for i in xrange(6):
#            phi = kmesh.dot(bondnnn[i])
#            hop_nnn = hop_nnn + general_kron(exp(1j*phi),sigma0)*fsign[i]
#        hop_nnn *= tt1

        bondnnn = zeros([6,2])
        bondnnn[0] = array([1.,0])
        for i in xrange(1,6):
            bondnnn[i] = r_nnn.dot(bondnnn[i-1])
        hop = 0.
        for i in xrange(6):
            phi = kmesh.dot(bondnnn[i])
            hop = hop + general_kron(exp(1j*phi),sigma0)
        hop *= t2
        
        hop_nn = 0.
        for i in xrange(3):
            bond = bondnn[i]*sqrt(3)
            xx = bond[0]**2
            yy = bond[1]**2
            xy = bond[0]*bond[1]
            sigma_orbital = array([[xx-yy,2*xy],[2*xy,yy-xx]])
            phi = kmesh.dot(bondnn[i])
            hop_xy = general_kron(cos(phi),sigma1) - general_kron(sin(phi),sigma2)
            hop_nn = hop_nn + kron(sigma_orbital,hop_xy)
        hop_nn *= tt1

        Hk = 0.
        Hk = Hk + kron(sigma0,hk)
        Hk = Hk + kron(isigma2,warp_nnn)
        Hk = Hk + kron(sigma0,hop)
#        Hk = Hk + kron(sigma0,hop_nnn)
        Hk = Hk + hop_nn
#        pdb.set_trace()
        return Hk
    def get_dos(self,t2,tt1,tt2):
        Nx = 400; Ny = 400
        alpha = self.alpha; betha = self.betha
        kalpha,kbetha = self.get_kvector(alpha,betha,Nx,Ny)
        kx = arange(Nx)
        ky = arange(Ny)
        kmesh = kx.reshape(-1,1,1)*kalpha + ky.reshape(1,-1,1)*kbetha
        Hk = self.get_TBG_model(kmesh,t2,tt1,tt2)
        Ek,Uk = linalg.eigh(Hk)
        eta = 1e-2; omega = arange(-3,5,0.01)
        dos = zeros(omega.shape)
        for i,w in enumerate(omega):
            Gk_w = 1./(w - Ek + 1j*eta)
            dos_i = Gk_w.sum()
            dos[i] = -imag(dos_i)/pi/1e5
#        pdb.set_trace()
        order = argsort(dos)
        omegam = omega[order[-1]]
        n = (Ek<omegam).sum()/400.**2
        print omegam, n
        plt.plot(dos,omega,color = 'r',linewidth = 1.5)
#        plt.plot([0,12],[0,0],linestyle = '--',color = 'black')
#        for i in xrange(2):
#            ydn = -1.65 + i*0.1
#            plt.plot([0,12],[ydn,ydn],linestyle = '--',color = 'black')
            #yup = 0.4 + i*0.01
            #plt.plot([0,12],[yup,yup],linestyle = '--',color = 'b')
#        plt.plot([0,12],[1.6,1.6],linestyle = '--',color = 'black')
#        plt.plot([0,12],[1.4,1.4],linestyle = '--',color = 'black')
#        plt.plot([0,12],[-1.8,-1.8],linestyle = '--',color = 'black')
#        plt.plot([0,12],[-1.65,-1.65],linestyle = '--',color = 'black')
        plt.plot([0,Ek.shape[0]],[-1.15,-1.15],linestyle = '--',color = 'black')
        plt.minorticks_on()
        plt.xlabel('DOS(a.u.)',size = 20)
#        plt.ylabel('$\omega/t$',size = 28)
        plt.xticks(size = 16)
        plt.yticks([],size = 16)
        plt.xlim(0,4)
        plt.ylim(-2.5,4)
#        plt.show()
    def single_honeycomb(self,initbond = array([1.,0])):
        a = initbond
        r = rotate(pi/3)
        b = zeros([6,2])
        b[0] = a
        for i in xrange(1,6):
            b[i] = r.dot(b[i-1])
#        b = b - a
        return b
    def plot_single_honeycomb(self,bond,c='black'):
        for i in xrange(6):
            plt.plot([bond[i,0],bond[(i+1)%6,0]],[bond[i,1],bond[(i+1)%6,1]],color = c,lw=0.5)
    def plot_lattice(self,magic_angle,color='black'):
        scalex, scaley = 10,10
        a, b = self.alpha[:2], self.betha[:2]
        bond0 = self.single_honeycomb()
        r = rotate(magic_angle)
        for i in xrange(-scalex,scalex+1):
            opointx = i*a
            for j in xrange(-scaley,scaley+1):
                opointy = j*b
                opoint = opointx + opointy
#                bond = opoint + bond0
                bond = (opoint + bond0).T
                bond = (r.dot(bond)).T
                self.plot_single_honeycomb(bond,color)
#        plt.plot([0,a[0]],[0,a[1]],color = 'red')
#        plt.plot([0,b[0]],[0,b[1]],color = 'red')
#        self.plot_circle(7)
#        plt.axis('equal')
#        plt.axis('off')
#        plt.savefig('honeylattic')
#    def plot_circle(self,r):
#        x = linspace(-r,r,100)
#        y = sqrt(r**2 - x**2)
#        plt.plot(x,y,color = 'grey',lw = 0.4)
#        plt.plot(x,-y,color = 'grey',lw = 0.4)
    def plot_TBG(self,n,m):
        alpha = self.alpha; betha = self.betha
        alpha_t = m*alpha + n*betha
        betha_t = -n*alpha + (n + m)*betha
        plt.plot([0,alpha_t[0]],[0,alpha_t[1]],color = 'blue')
        plt.plot([0,betha_t[0]],[0,betha_t[1]],color = 'blue')
        cos_theta = (n**2 + m**2 + 4*m*n)/(n**2 + m*n + m**2)/2.
        theta = arccos(cos_theta)
        self.plot_lattice(0,'red')
        self.plot_lattice(theta,'green')
        plt.axis('equal')
        plt.axis('off')
        plt.savefig('TBG_'+str(theta)+'.png')
    def continum_model_graphene(self,k,twisted_angle):
        fermi_velocity = 1.
        t3 = cos(twisted_angle/2)*sigma0 + 1j*sigma3*sin(twisted_angle/2)
        s1 = dot(dot(t3,sigma1),t3.conj())
        s2 = dot(dot(t3,sigma2),t3.conj())
        Hk = general_kron(k[...,0],s1) + general_kron(k[...,1],s2)
#        Ek,Uk = linalg.eigh(Hk)
#        pdb.set_trace()
        return Hk
    def get_Tmatrix(self,kk,ll):
        Tab = exp(-2j*pi*(2*kk-ll)/3)
        Taa = exp(-2j*pi*(kk+ll)/3)
        Tba = 1.
        Tbb = exp(-2j*pi*(kk-2*ll)/3)
        T = array([[Taa,Tab],[Tba,Tbb]])
        return T
    def continum_model_TBG(self,k,vector,theta):
        n = vector[0]; m = vector[1]
        alpha = self.alpha; betha = self.betha
        alpha_t = m*alpha + n*betha
        betha_t = -n*alpha + (n + m)*betha
        G1,G2 = self.get_kvector(alpha_t,betha_t,1,1)
        alpha_t = alpha_t[:-1]; betha_t = betha_t[:-1]
        delta_lenth = 8*pi/3*sin(theta/2)
        delta_K = (2*G1+G2)/3
#        delta_K = (2*G1 + G2)/6
#        k10 = k + delta_K; k11 = k10 - G1; k12 = k11 - G2
#        k20 = k - delta_K; k21 = k20 + G1; k22 = k21 + G2
#        h10 = self.continum_model_graphene(k10,0)
#        h11 = self.continum_model_graphene(k11,0)
#        h12 = self.continum_model_graphene(k12,0)
#        h20 = self.continum_model_graphene(k20,theta)
#        h21 = self.continum_model_graphene(k21,theta)
#        h22 = self.continum_model_graphene(k22,theta)
#        Hk = self.stack_hamitonian([h10,h11,h12,h20,h21,h22])

        dirac_point1 = self.single_honeycomb(delta_lenth)
        self.plot_single_honeycomb(dirac_point1)
        dirac_point2 = self.single_honeycomb(2*delta_lenth)
        H_list = []
        for point in dirac_point1:
            kp = point - k
            h_kp = self.continum_model_graphene(kp,theta/2)
            H_list.append(h_kp)
            theta = -theta
        for point in dirac_point2:
            theta = -theta
            kp = point - k
            h_kp = self.continum_model_graphene(kp,theta/2)
            H_list.append(h_kp)
        Hk = self.stack_hamitonian(H_list)

        t_tunnel = 0.04
        Tm = 0.
        s = [12,12]
        phi = 2*pi/3
        T0 = ones([2,2])
        T1 = array([[exp(-1j*phi),1],[exp(1j*phi),exp(-1j*phi)]])
        T2 = T1.conj()
        T_list = [T0,T1,T2]
        dirac_shape = dirac_point1.shape[0]
#        pdb.set_trace()
        l = 0; j = 2
        for i in xrange(dirac_shape):
            if (i%2) == 0:
                T = T_list[l]
                Tm = Tm + self.block_matrix_element(s,[i+6,i],T)
                l = (l+1)%3
            else:
                T = T_list[j]
                Tm = Tm + self.block_matrix_element(s,[i,i+6],T)
                j = (j+1)%3
            pdb.set_trace()
            print i
##                pdb.set_trace()

#        j = 1; l = 0
#        for i in xrange(dirac_shape):
#            if (i%2) == 0:
#                T = T_list[j]
#                Tm = Tm + self.block_matrix_element(s,[i+1,i],T)
#                j = (j+1)%3
#            else:
#                T = T_list[l]
#                Tm = Tm + self.block_matrix_element(s,[i,(i+1)%dirac_shape],T)
#                l = (l+1)%3
        Tm = Tm + Tm.T.conj()
        return Hk + Tm

        
        
        
        
        
        
#    def effective_model_graphene(self,k,twisted_angle):
#        fermi_velocity = 1.
#        k_module = sqrt((k**2).sum(-1))
#        x = (k_module == 0)
#        k_module[x] = k_module.max()
#        cos_theta = k[...,0]/k_module
#        thetak = arccos(cos_theta)
#        a = array([[0.,1],[0,0]])
#        b = a.T
#        phase = exp(1j*(thetak - twisted_angle))*k_module*fermi_velocity
#        hk = general_kron(phase,a) + general_kron(phase.conj(),b)
##        pdb.set_trace()
#        return hk
#    def effective_model_k(self,k,vector):
#        phi = 2*pi/3.
#        T1 = array([[1.,1],[1,1]])
#        T2 = array([[exp(-1j*phi),1],[exp(1j*phi),exp(-1j*phi)]])
#        T3 = T2.conj()
#        Tone_1 = zeros([4,4]); Tone_1[0,1] = 1.
#        Tone_2 = zeros([4,4]); Tone_2[0,2] = 1.
#        Tone_3 = zeros([4,4]); Tone_3[0,3] = 1.
#        Ht = kron(T1,Tone_1) + kron(T2,Tone_2) + kron(T3,Tone_3)
#        Ht = Ht + Ht.T.conj()
##        pdb.set_trace()
#        
#        n = vector[0]; m = vector[1]
#        cos_theta = (n**2 + m**2 + 4*m*n)/(n**2 + m*n + m**2)/2.
#        theta = arccos(cos_theta)/2
#        kd = 4*pi/3*sin(theta)
#        q1 = array([0.,-1]); q2 = array([sqrt(3)/2,1./2]); q3 = array([-sqrt(3)/2,1./2])
#        q1 = 2*kd*q1; q2 = 2*kd*q2; q3 = 2*kd*q3
#        hk = self.effective_model_graphene(k, theta)
#        hk_1 = self.effective_model_graphene(k + q1, -theta)
#        hk_2 = self.effective_model_graphene(k + q2, -theta)
#        hk_3 = self.effective_model_graphene(k + q3, -theta)
#        one_0 = diag([1.,0,0,0]); one_1 = diag([0.,1,0,0])
#        one_2 = diag([0.,0,1,0]); one_3 = diag([0.,0,0,1])
#        Hk = kron(hk,one_0) + kron(hk_1,one_1) + kron(hk_2,one_2) + kron(hk_3,one_3)
##        pdb.set_trace()
#        Hk = Hk + Ht
#        return Hk

    def TBG_band(self,vector):
        alpha = self.alpha; betha = self.betha
        Nx = 200; Ny = 200
        n = vector[0]; m = vector[1]
        alpha_t = n*alpha + m*betha
        betha_t = -m*alpha + (n+m)*betha
        kalpha,kbetha = self.get_kvector(alpha_t,betha_t,Nx,Ny)
        cos_theta = (n**2 + m**2 + 4*m*n)/(n**2 + m*n + m**2)/2.
        theta = arccos(cos_theta)
        delta_lenth = 4*pi/3*sin(theta/2)
#        Kpoint=gph.get_Kpoint(kalpha,kbetha)
#        Mpoint=gph.get_Mpoint(kalpha,kbetha)
#        GKMG = gph.generate_loop(Kpoint,Mpoint)
#        PATH = GKMG
        PATH = arange(-0.2,0.2,0.0001).reshape(-1,1)*array([1,0])
        Hk = self.continum_model_TBG(PATH,vector,theta)
        Ek,Uk = linalg.eigh(Hk)
        plt.plot(arange(Ek.shape[0]),Ek[...,:],linewidth=2.)
        plt.show()
        plt.close()

    def TBG_dispersion(self,vector):
        alpha = self.alpha; betha = self.betha
        Nx = 200; Ny = 200
        n = vector[0]; m = vector[1]
        alpha_t = n*alpha + m*betha
        betha_t = -m*alpha + (n+m)*betha
        cos_theta = (n**2 + m**2 + 4*m*n)/(n**2 + m*n + m**2)/2.
        theta = arccos(cos_theta)
        delta_lenth = 8*pi/3*sin(theta/2)
        k = linspace(-2*delta_lenth,2*delta_lenth,50)
        kmesh = k.reshape(-1,1,1)*array([1,0]) + k.reshape(1,-1,1)*array([0,1])
        Hk = self.continum_model_TBG(kmesh,vector,theta)
        Ek,Uk = linalg.eigh(Hk)
        kmesh_rs = kmesh.reshape(-1,2)
        Ek_rs = Ek.reshape(-1,Hk.shape[-1])
        kmesh = meshgrid(k,k,indexing='ij')
        band = 12
        plt.pcolor(kmesh[0],kmesh[1],Ek[...,band])
#        plt.contour(kmesh[0],kmesh[1],Ek[...,band],arange(1e-2,0.1,0.02))
        plt.colorbar()
        plt.axis('equal')
        plt.show()
        data = hstack((kmesh_rs,Ek_rs))
        savetxt("dispersion.dat",data)
#        Ek_band = Ek[...,4].reshape(-1)
#        plt.scatter(kmesh[...,0].reshape(-1),kmesh[...,1].reshape(-1),vmin = Ek_band.min(),vmax = Ek_band.max(),c = Ek_band,cmap = 'Blues',s = 40)
#        plt.colorbar()
#        plt.axis('equal')
#        plt.show()
#        plt.close()
    def superlattice_angle(self,start,end):
        theta = []
        for n in xrange(start,end+1):
            m = n+1.
            cos_theta = (n**2 + m**2 + 4*m*n)/(n**2 + m*n + m**2)/2.
            theta.append(arccos(cos_theta)*180/pi)
        print theta

    def get_fs_theta(self,t2,tt1,tt2,band,l,co,si):
        polarmesh = zeros([l.shape[0],2])
        polarmesh[:,0] = l*co
        polarmesh[:,1] = l*si
        Hk = self.get_TBG_model(polarmesh,t2,tt1,tt2)
        Ek,Uk = linalg.eigh(Hk)
        Ek = abs(Ek[:,band])
        order = argsort(Ek)
        order0 = order[0]
        return Ek[order0],l[order0]

    def get_fs_contour(self,t2,tt1,tt2):
        angle = linspace(0,2*pi,600,endpoint = 0)
        s = angle.shape[0]
        fs2 = zeros([s,3])
        fs3 = zeros([s,3])
        band1 = 0; band2 = 1
        fs2[...,2] = band1 + 1
        fs3[...,2] = band2 + 1
        cos_theta = cos(angle)
        sin_theta = sin(angle)
        l0 = 2*pi/sqrt(3)
        l_angle0 = l0/cos(pi/6-angle[:100])
        l_angle = l_angle0
        for i in xrange(5):
            l_angle = hstack((l_angle,l_angle0))
        for i in xrange(s):
            l = arange(0,l_angle[i],0.01)
            co = cos_theta[i]; si = sin_theta[i]
            E,lmin = self.get_fs_theta(t2,tt1,tt2,band1,l,co,si)
#            pdb.set_trace()
            ll = arange(lmin-1e-2,lmin+1e-2,1e-4)
            E,lmin = self.get_fs_theta(t2,tt1,tt2,band1,ll,co,si)
            ll = arange(lmin-1e-4,lmin+1e-4,1e-6)
            E,lmin = self.get_fs_theta(t2,tt1,tt2,band1,ll,co,si)
#            pdb.set_trace()
            fs2[i,0] = lmin*co; fs2[i,1] = lmin*si
#            pdb.set_trace()
            E,lmin = self.get_fs_theta(t2,tt1,tt2,band2,l,co,si)
            ll = arange(lmin-1e-2,lmin+1e-2,1e-4)
            E,lmin = self.get_fs_theta(t2,tt1,tt2,band2,ll,co,si)
            ll = arange(lmin-1e-4,lmin+1e-4,1e-6)
            E,lmin = self.get_fs_theta(t2,tt1,tt2,band2,ll,co,si)
            fs3[i,0] = lmin*co; fs3[i,1] = lmin*si
#            pdb.set_trace()
        fs = vstack((fs2,fs3))
        fs[...,0:2] = fs[...,0:2]/pi
        plt.scatter(fs[:,0],fs[:,1])
        plt.axis('equal')
        plt.show()
#        pdb.set_trace()
        savetxt('fs.dat',fs)
        return fs
            


if __name__=='__main__':
    Nx = 200; Ny = 200
    mu = -0; t2 = 0.15; tt1 = 0.; tt2 = -0.02
    gph = graphen(mu,Nx,Ny)
#    plt.subplot(122)
#    plt.subplot(122)
#    gph.get_dos(t2,tt1,tt2)
    alpha = gph.alpha; betha = gph.betha
    kalpha,kbetha = gph.get_kvector(alpha,betha,Nx,Ny)
    kalpha *= Nx; kbetha *=Ny
    Kpoint=gph.get_Kpoint(kalpha,kbetha)
    Mpoint=gph.get_Mpoint(kalpha,kbetha)
    Kp_point = rotate(pi/3).dot(Kpoint)
    GKMG = gph.generate_loop(Kpoint,Mpoint,Kp_point)
    Hk = gph.get_TBG_model(GKMG,t2,tt1,tt2)
    Ek,Uk = linalg.eigh(Hk)
    plt.subplot(121)
    plt.plot([0,Ek.shape[0]],[-1.29,-1.29],linestyle = '--',color = 'black')
#    plt.plot([0,Ek.shape[0]],[-1.8,-1.8],linestyle = '--',color = 'black')
#    plt.plot([0,Ek.shape[0]],[-1.65,-1.65],linestyle = '--',color = 'black')
#    plt.plot([0,Ek.shape[0]],[1.4,1.4],linestyle = '--',color = 'black')
#    plt.plot([0,Ek.shape[0]],[1.6,1.6],linestyle = '--',color = 'black')
    plt.plot(arange(Ek.shape[0]),Ek,linewidth=1.5)
    plt.yticks(size = 16)
    plt.xticks([0,500,750,750+250*sqrt(3)],['$\Gamma$','K','M','$\Gamma$'],size = 16)
    plt.ylabel('$\epsilon_k$',size = 32)
    plt.xlim(0,750+250*sqrt(3))
    plt.ylim(-2.5,4)
    plt.minorticks_on()
    plt.show()
#    fs = gph.get_fs_contour(t2,tt1,tt2)
#    fs1 = fs[:600,:2]; fs2 = fs[600:,:2]
#    Hk1 = gph.get_TBG_model(fs1,t2,tt1,tt2)
#    Hk2 = gph.get_TBG_model(fs2,t2,tt1,tt2)
#    Ek1,Uk1 = linalg.eigh(Hk1)
#    Ek2,Uk2 = linalg.eigh(Hk2)
#    for i in xrange(4):
#        ax = plt.subplot(2,2,i+1)
#        plt.scatter(fs1[:,0],fs1[:,1],c = abs(Uk1[:,i,0]),s = abs(Uk1[:,i,0])**8*2e3,edgecolors = 'none')
#        plt.scatter(fs2[:,0],fs2[:,1],c = abs(Uk2[:,i,1]),s = abs(Uk2[:,i,0])**8*2e3,edgecolors = 'none')
#        plt.axis('equal')
##    pdb.set_trace()
#    plt.savefig('blochstate.png')
#    plt.show()

#    kx = linspace(-4*pi/3,4*pi/3,100)
#    ky = linspace(-4*pi/3,4*pi/3,100)
#    kmesh = kx.reshape(-1,1,1)*array([1,0]) + ky.reshape(1,-1,1)*array([0,1])
#    Hk = gph.get_TBG_model(kmesh,t2,tt1,tt2)
#    Ek,Uk = linalg.eigh(Hk)
#    BZ = gph.single_honeycomb(array([4*pi/3,0]))
#    gph.plot_single_honeycomb(BZ)
#    kmesh = meshgrid(kx,ky,indexing='ij')
#    band1 = 0
##    plt.pcolor(kmesh[0],kmesh[1],Ek[...,band1])
##    plt.colorbar()
##    plt.axis('equal')
##    plt.show()
#    energy_c = -1.55
#    band2 = 1
##    pdb.set_trace()
#    plt.contour(kmesh[0],kmesh[1],Ek[...,band1],levels = [energy_c,energy_c+1e-6])
#    plt.contour(kmesh[0],kmesh[1],Ek[...,band2],levels = [energy_c,energy_c+1e-6])
##    plt.colorbar()
#    plt.axis('equal')
#    plt.axis('off')
#    plt.show()

#    alpha = gph.alpha; betha = gph.betha
#    kalpha,kbetha = gph.get_kvector(alpha,betha,Nx,Ny)
#    kalpha *= Nx; kbetha *=Ny
#    Kpoint=gph.get_Kpoint(kalpha,kbetha)
#    Mpoint=gph.get_Mpoint(kalpha,kbetha)
#    pdb.set_trace()
#    GKMG = gph.generate_loop(Kpoint,Mpoint)
#    Hk = gph.get_twisted_tba(GKMG,0.2,0.)

#    Ek,Uk = linalg.eigh(Hk)
#    plt.plot(arange(Ek.shape[0]),Ek,linewidth=2.)
#    plt.savefig('band.pdf')

#    n = 4.; m=1.
#    gph.plot_TBG(n,m)
#    pdb.set_trace()

#    gph.TBG_band([30.,31.])
#    gph.TBG_dispersion([30.,31.])
#    gph.superlattice_angle(990,1000)
    









