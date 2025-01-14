import multiprocessing
import numpy as np
import bisect
c0 = 3.0e10

def curl_matrix(matrix):

    # Assuming your matrix is named 'matrix' with shape (nx, ny, nz, 3)

    # Compute the gradient along each axis
    grad_x = np.gradient(matrix[:, :, :, 0])
    grad_y = np.gradient(matrix[:, :, :, 1])
    grad_z = np.gradient(matrix[:, :, :, 2])

    # Rearrange the components to calculate the curl
    curl_x = grad_z[1] - grad_y[2]
    curl_y = grad_x[2] - grad_z[0]
    curl_z = grad_y[0] - grad_x[1]

    # Combine the components into a curl matrix
    curl_matrix = np.stack((curl_x, curl_y, curl_z), axis=-1)
    return curl_matrix

def RBSL_Anp_parallel(r_veci, Rpli):
    # initialize vector from domain to the ith part
    # r_vec = ((meshx - x_axis[i])/a)
    r_mag = np.zeros((r_veci.shape[:-1]))
    r_mag = np.sqrt(np.sum(r_veci**2, axis=-1))
    Rcr = np.cross(Rpli,r_veci)

    mask1 = np.where(r_mag<0, 1, 0)
    mask2 = 1-mask1
    # case of r_mag < 1
    r_mag1 = r_mag * mask1
    re_pi = 1/np.pi
    sqrt1r = np.sqrt(1.0-(r_mag1)**2)
    f52r = 5.0-2.0*r_mag1**2
    fsqrt6 = 1.0/np.sqrt(6.0)
    # here different usr of r_mag[1] depends on whether the expression would generate invalid value
    KIr  = 2.0*re_pi*(np.arcsin(r_mag1)/r_mag+f52r/3.0)
    KFr  = 2.0*re_pi/r_mag**2*(np.arcsin(r_mag1)/r_mag-sqrt1r) + \
        2.0*re_pi*sqrt1r + f52r*0.5*fsqrt6*(1.0- \
        2.0*re_pi*np.arcsin((1.0+2.0*r_mag1**2)/f52r))
    KIr *= mask1
    KFr *= mask1
    # case of r_mag > 1
    KIr1 = (1.0/r_mag) * mask2
    KFr1 = KIr1**3 * mask2
    # generate total A
    KIr += KIr1
    KFr += KFr1
    AIx = KIr[:,:,:,np.newaxis] * Rpli[np.newaxis,np.newaxis,np.newaxis,:]
    AFx = KFr[:,:,:,np.newaxis] * Rcr[:,:,:,:]
    return (AIx, AFx) #can be optimized

class mhdtp():
    def __init__(self, nx, ny, nz, lx, ly, lz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.lx = lx
        self.ly = ly
        self.lz = lz
    
    def meshgrids(self):
        x_coords = np.linspace(-self.lx/2, self.lx/2, self.nx+1)
        y_coords = np.linspace(-self.ly/2, self.ly/2, self.ny+1)
        z_coords = np.linspace(0         , self.lz  , self.nz+1)
        xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        # combine the coordinate grids into a single array
        return np.stack((xv, yv, zv), axis=-1)
    
    def bipolar_field(self, q_para=7.5, l_para=1.5, d_para=1.0):
        x = self.meshgrids()
        Aphi = q_para*(l_para-x[:,:,:,0])/(np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2)*
            np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2+(x[:,:,:,0]-l_para)**2))+ \
            q_para*(l_para+x[:,:,:,1])/(np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2)*
            np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2+(x[:,:,:,0]+l_para)**2))

        A = np.resize(np.zeros((self.nx+1)*(self.ny+1)*(self.nz+1)*3), ((self.nx+1),(self.ny+1),(self.nz+1),3))
        A[:,:,:,0] = 0.0
        A[:,:,:,1] = -Aphi*(x[:,:,:,2]+d_para)/np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2)
        A[:,:,:,2] = Aphi*x[:,:,:,2]/np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2)

        tmp = np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2+(x[:,:,:,0]+l_para)**2)**3
        Bbp = np.resize(np.zeros((self.nx+1)*(self.ny+1)*(self.nz+1)*3), ((self.nx+1),(self.ny+1),(self.nz+1),3))
        Bbp[:,:,:,0] = (x[:,:,:,0]+l_para)/tmp
        Bbp[:,:,:,1] = x[:,:,:,1]/tmp
        Bbp[:,:,:,2] = (x[:,:,:,2]+d_para)/tmp
        tmp = np.sqrt(x[:,:,:,1]**2+(x[:,:,:,2]+d_para)**2+(x[:,:,:,0]-l_para)**2)**3
        Bbp[:,:,:,0] = Bbp[:,:,:,0]-(x[:,:,:,0]-l_para)/tmp
        Bbp[:,:,:,1] = Bbp[:,:,:,1]-x[:,:,:,1]/tmp
        Bbp[:,:,:,2] = Bbp[:,:,:,2]-(x[:,:,:,2]+d_para)/tmp
        return Bbp

    def RBSL_flux_rope(self, kappa=1.1, inp=200, htop=1.2, Lfp=1.5, a=0.5, q_para=7.5, Bperp=1.2, positive_helicity=True):

        def RBSL_Anp(i):
            # initialize vector from domain to the ith part
            r_vec = ((meshx - x_axis[i])/a)
            r_mag = np.sqrt(np.sum(r_vec**2, axis=-1))
            Rcr = np.cross(Rpl[i],r_vec)

            mask1 = np.where(r_mag<0, 1, 0)
            mask2 = 1-mask1
            # case of r_mag < 1
            r_mag1 = r_mag * mask1
            re_pi = 1/np.pi
            sqrt1r = np.sqrt(1.0-(r_mag1)**2)
            f52r = 5.0-2.0*r_mag1**2
            fsqrt6 = 1.0/np.sqrt(6.0)
            # here different usr of r_mag[1] depends on whether the expression would generate invalid value
            KIr  = 2.0*re_pi*(np.arcsin(r_mag1)/r_mag+f52r/3.0*sqrt1r)
            KFr  = 2.0*re_pi/r_mag**2*(np.arcsin(r_mag1)/r_mag-sqrt1r) + \
                2.0*re_pi*sqrt1r + f52r*0.5*fsqrt6*(1.0- \
                2.0*re_pi*np.arcsin((1.0+2.0*r_mag1**2)/f52r))
            KIr *= mask1
            KFr *= mask1
            # case of r_mag > 1
            KIr1 = (1.0/r_mag) * mask2
            KFr1 = KIr1**3 * mask2
            # generate total A
            KIr += KIr1
            KFr += KFr1
            AIx = KIr[:,:,:,np.newaxis] * Rpl[i][np.newaxis,np.newaxis,np.newaxis,:]
            AFx = KFr[:,:,:,np.newaxis] * Rcr[:,:,:,:]
            return (AIx*I_cur/a + AFx*F_flx*0.25*re_pi/a**2) #can be optimized

        meshx = self.meshgrids()
        re_pi = 1/np.pi
        x0 = 0
        y0 = 0
        r0 = htop / (1.0-np.cos(np.pi-2.0*np.arctan(Lfp/htop)))
        z0 = htop - r0
        # 
        I_cur = -q_para*r0*Bperp/(np.log(8*r0/a)-1)*kappa
        F_flx = -4.0*np.pi*3.0*0.2/np.sqrt(2.0)*I_cur*a
        if positive_helicity:
            I_cur /= -1
        # 
        x_axis = np.zeros((inp,3))
        inparray = np.arange(inp)
        x_axis[:,0] = x0
        x_axis[:,1] = y0 - r0*np.cos(inparray*2.0*np.pi/(inp+1))
        x_axis[:,2] = z0 + r0*np.sin(inparray*2.0*np.pi/(inp+1))
        
        Rpl = np.zeros((inp,3))
        Rpl[1:-1] = 0.5*(x_axis[2:,:]-x_axis[:-2,:]) 
        Rpl[0]  = 0.5*(x_axis[1,:]-x_axis[-1,:])
        Rpl[-1] = 0.5*(x_axis[0,:]-x_axis[-2,:])

        Atotal = np.zeros(meshx.shape)
        if self.nx < 100 and self.ny < 100 and self.nz < 100:
            for i in inparray:
                Atotal += RBSL_Anp(i)
        else:
            r_vec = meshx[np.newaxis,:,:,:,:] - x_axis[:,np.newaxis,np.newaxis,np.newaxis,:] # (inp,nx,ny,nz,3)
            arguments = [(r_vec[i],Rpl[i]) for i in inparray]
            with multiprocessing.Pool(processes=10) as pool:
                results = pool.starmap(RBSL_Anp_parallel, arguments)
            for Ai in results:
                Atotal += Ai[0]*I_cur/a + Ai[1]*F_flx*0.25*re_pi/a**2

        return curl_matrix(Atotal)
        
    def generate_particle_position(self, n_particles, x0, x1, y0, y1, z0, z1):
        xp = np.zeros((n_particles,3))
        for i in range(n_particles):
            xp[i] = np.array([np.random.uniform(x0,x1), np.random.uniform(y0,y1), np.random.uniform(z0,z1)])
        return xp

    def generate_particle_velocity(self, n_particles, thermal_velocity):
        vp = np.zeros((n_particles,3))
        for i in range(n_particles):
            vp[i] = generate_maxwellian_velocity(thermal_velocity) * uniform_sphere()
        return vp
    
    def generate_particle_velocity1(self, n_particles, thermal_velocity):
        vp = np.zeros((n_particles,3))
        for i in range(n_particles):
            vp[i] = generate_maxwellian_velocity(thermal_velocity) * uniform_angle()
        return vp

    def generate_particle_velocity2(self, n_particles, thermal_velocity):
        vp = np.zeros((n_particles,3))
        for i in range(n_particles):
            vp[i] = uniform_sphere()
        return vp

    def generate_particle_velocity3(self, n_particles, thermal_velocity):
        vp = np.zeros((n_particles,3))
        for i in range(n_particles):
            vp[i] = uniform_angle()
        return vp

    def test_pitch_angle(self, N=10000, field=None):
        pitch_angle = np.zeros(N)

        if field is None:
            field = bipolar_field()+RBSL_flux_rope()

        particlex = self.generate_particle_position(N, -self.lx/2.0, self.lx/2.0, -self.ly/2.0, self.ly/2.0, 0.0, self.lz)
        particlev = self.generate_particle_velocity(N, 1)
        for i in range(N):
            b1i = self.interpolate_field(field[:,:,:,0], particlex[i])
            b2i = self.interpolate_field(field[:,:,:,1], particlex[i])
            b3i = self.interpolate_field(field[:,:,:,2], particlex[i])
            bhat = np.array([b1i,b2i,b3i])/np.linalg.norm([b1i,b2i,b3i])
            vhat = particlev[i] / np.linalg.norm(particlev[i])

            pitch_angle[i] = np.arccos(np.dot(bhat,vhat))
        return pitch_angle

    def interpolate_field(self, field, pos, meshgrid=None):
        if (meshgrid is None):
            meshgrid = self.meshgrids()
        # only for cube mesh
        xline = meshgrid[:,0,0,0]
        yline = meshgrid[0,:,0,1]
        zline = meshgrid[0,0,:,2]
        ic11, ic21 = find_neighbor_indices(xline, pos[0])
        ic12, ic22 = find_neighbor_indices(yline, pos[1])
        ic13, ic23 = find_neighbor_indices(zline, pos[2])

        xd1 = (pos[0]-meshgrid[ic11,ic12,ic13,0]) / (meshgrid[ic21,ic12,ic13,0] - meshgrid[ic11,ic12,ic13,0])
        xd2 = (pos[1]-meshgrid[ic11,ic12,ic13,1]) / (meshgrid[ic11,ic22,ic13,1] - meshgrid[ic11,ic12,ic13,1])
        xd3 = (pos[2]-meshgrid[ic11,ic12,ic13,2]) / (meshgrid[ic11,ic12,ic23,2] - meshgrid[ic11,ic12,ic13,2])

        c00 = field[ic11,ic12,ic13] * (1.0 - xd1) + field[ic21,ic12,ic13] * xd1
        c10 = field[ic11,ic22,ic13] * (1.0 - xd1) + field[ic21,ic22,ic13] * xd1
        c01 = field[ic11,ic12,ic23] * (1.0 - xd1) + field[ic21,ic12,ic23] * xd1
        c11 = field[ic11,ic22,ic23] * (1.0 - xd1) + field[ic21,ic22,ic23] * xd1

        c0  = c00 * (1.0 - xd2) + c10 * xd2
        c1  = c01 * (1.0 - xd2) + c11 * xd2

        field_loc = c0 * (1.0 - xd3) + c1 * xd3
        return field_loc


def generate_semirandom_uniform_particles(N, lx, ly, lz, r):

    # Calculate the number of particles in each dimension and the spacing between particles
    lp = (N / (lx*ly*lz))**(1/3)
    nx = int(np.ceil(lx*lp))
    ny = int(np.ceil(ly*lp))
    nz = int(np.floor(lz*lp))
    n_cubes = nx * ny * nz
    N_cubes = min(n_cubes, N)
    
    # Generate the initial particle positions on a grid
    x, y, z = np.meshgrid(np.linspace(r/2-lx/2,lx/2-r/2,nx), np.linspace(r/2-ly/2,ly/2-r/2,ny), np.linspace(r/2-lz/2,lz/2-r/2,nz))
    positions_all = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    random_indices = np.random.choice(positions_all.shape[0], size=N_cubes, replace=False)
    positions = positions_all[random_indices]
    
    return positions

def generate_random_point_on_sphere():
    # Generate random values from a standard normal distribution
    x3 = np.random.normal(0, 1, 3)
    
    # Normalize the coordinates to lie on the unit sphere
    magnitude = np.linalg.norm(x3)
    x3 /= magnitude 

    return x3

def uniform_sphere():
    costheta = np.random.uniform(-1,1)
    phi = np.random.uniform(0,2*np.pi)
    theta = np.arccos(costheta)
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),costheta])

def uniform_angle():
    phi = np.random.uniform(0,2*np.pi)
    theta = np.random.uniform(0,2*np.pi)
    return np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

def generate_maxwellian_velocity(thermal_velocity):
    v0 = np.zeros(3)
    v0[0] = thermal_velocity * np.random.uniform(0,1)
    v0[1] = thermal_velocity * np.random.uniform(0,1)
    v0[2] = thermal_velocity * np.random.uniform(0,1)
    return np.linalg.norm(v0)

def find_neighbor_indices(list0, a):
    # Find the insertion point for `a` in the sorted sequence
    index = bisect.bisect_left(list0, a)

    # If the insertion point is at index 0, there is no left neighbor
    left_idx = index - 1 if index > 0 else None

    # If the insertion point is at the last index, there is no right neighbor
    right_idx = index if index < len(list0) else None

    return left_idx, right_idx