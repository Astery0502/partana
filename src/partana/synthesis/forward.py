import numpy as np
import pickle
import os

"""
Function toolkit for HXR synthesis

ep, ek are in mass energy unit, only with etab_table keys use keV
"""

class BremsstrahlungSynthesis:
    def __init__(self):
        self.ek_range = np.arange(10, 2001)
        self.ep_range = np.arange(10, 2001)
        self.etab_table = {}
        self.load_etab_table()
        
    def correction(self, ep, ek):
        e0 = ek+1
        e = ek+1-ep
        p0 = np.sqrt(e0**2-1)
        p = np.sqrt(e**2-1)
        
        return e*p0/p/e0*(1-np.exp(-2/137*np.pi*e0/p0))/(1-np.exp(-2/137*np.pi*e/p))
        
    def etab1(self, ep, ek):
        e0 = ek+1
        e = ek+1-ep
        p0 = np.sqrt(e0**2-1)
        p = np.sqrt(e**2-1)

        return 16/3/e/p0**2*np.log((p0+p)/(p0-p))*self.correction(ep, ek)

    def etab2(self, ep, ek):
        e0 = ek+1
        e = ek+1-ep
        p0 = np.sqrt(e0**2-1)
        p = np.sqrt(e**2-1)
        k = np.log((e+p)/(e-p))
        k0 = np.log((e0+p0)/(e0-p0))

        L = 2*np.log((e*e0+p*p0-1)/ep)

        part1 = 4/3-2*e0*e*(p0**2+p**2)/p0**2/p**2+k0*e/p0**3+k*e0/p**3-k*k0/p/p0
        part2 = 8*e*e0/3/p/p0+ep**2*(e**2+e0**2+p**2+p0**2)/p**3/p0**3
        part3 = ep/2/p/p0*(k0*(e*e0+p0**2)/p0**3-k*(e*e0+p**2)/p**3+2*ep*e*e0/p**2/p0**2)

        return p/p0/ep*(part1+L*(part2+part3))*self.correction(ep,ek)

    def etab(self, ep, ek):
        if ek < 100/511:
            return self.etab1(ep, ek)
        elif ek >= 100/511 and ek < 2e5/511:
            return self.etab2(ep, ek)

    def load_etab_table(self):
        if os.path.exists('cache/etab_table.pkl'):
            with open('cache/etab_table.pkl', 'rb') as file:
                self.etab_table = pickle.load(file)
        else:
            # Pre-calculate conversion factors
            ep_scaled = self.ep_range / 511
            ek_scaled = (self.ek_range + 0.5) / 511
            
            # Initialize empty dictionary with estimated size
            self.etab_table = {}
            
            # Vectorize calculations where possible
            for ep, ep_s in zip(self.ep_range, ep_scaled):
                # Create mask for valid ek values
                valid_ek = self.ek_range >= ep
                
                # Calculate etab values for all valid ek at once
                etab_values = [self.etab(ep_s, ek_s) for ek_s in ek_scaled[valid_ek]]
                
                # Add to dictionary using zip for efficiency
                self.etab_table.update(zip(
                    ((ep, ek) for ek in self.ek_range[valid_ek]),
                    etab_values
                ))
            
            # Save to file
            os.makedirs('cache', exist_ok=True)
            with open('cache/etab_table.pkl', 'wb') as file:
                pickle.dump(self.etab_table, file, protocol=pickle.HIGHEST_PROTOCOL)

    def inte(self, ep, ek):
        if ek < ep:
            return 0
        
        ep_kev = np.floor(ep * 511).astype(int)

        ek1_range = np.arange(ep, ek)
        sum_ek1 = 0
        for ek1 in ek1_range:
            ek1_kev = np.floor(ek1 * 511).astype(int)
            sum_ek1 += ek1/511 * self.etab_table[(ep_kev, ek1_kev)] * 1/511

        return sum_ek1

    def inter(self, ep_range, ek):
        """
        Calculate the overall photon flux from the given electron
        input:
            ep_range: the range of the electron energy
            ek: the single electron energy
        output:
            the photon flux over the given ep_range
        """
        assert len(ep_range) == 2 and ep_range[1] > ep_range[0], f"Illegal ep range {ep_range}"

        ep_range_kev = np.array(ep_range)*511

        if ek < ep_range[0]:
            return 0
        
        if ek <= ep_range[1]:
            ep_right = ek
        else:
            ep_right = ep_range[1]+0.01/511

        sumr = 0
        for ep1 in np.arange(ep_range_kev[0], ep_right*511):
            sumr += self.inte(ep1/511, ek) * 1/511

        return sumr

    def eval_inter(self, eks, ep_range):
        """
        Calculate the overall photon flux from the given electron energy range
        input:
            eks: the electron energy range
            ep_range: the range of the electron energy
        output:
            the photon flux over the given ep_range
        """
        intes = np.zeros(len(eks))
        for i, ek in enumerate(eks):
            intes[i] = self.inter(ep_range, ek)
        return intes

def points_to_grid(xrange, yrange, points, grid_size=0.1):
    """
    Aggregate points into a 2D grid based on their x,y coordinates
    
    Args:
        xrange: Tuple of (x_min, x_max) defining x boundaries
        yrange: Tuple of (y_min, y_max) defining y boundaries
        points: List of (x,y,value) tuples to aggregate
        grid_size: Size of each grid cell (default 0.1)
        
    Returns:
        grid: 2D numpy array containing aggregated values
        x_centers: Array of x-coordinates of grid cell centers
        y_centers: Array of y-coordinates of grid cell centers
    """
    x_min, x_max = xrange
    y_min, y_max = yrange
    
    # Calculate the number of cells in each direction
    x_bins = int((x_max - x_min) / grid_size)
    y_bins = int((y_max - y_min) / grid_size)

    # Initialize a grid to store aggregated values
    grid = np.zeros((x_bins, y_bins))

    # Aggregate values into the grid
    for x, y, value in points:
        # Determine the grid cell for each point
        x_index = int((x - x_min) // grid_size)
        y_index = int((y - y_min) // grid_size)
        
        # Add the point's value to the respective grid cell
        grid[x_index, y_index] += value

    # Create arrays of grid cell centers for plotting
    x_centers = np.arange(x_min + grid_size/2, x_max, grid_size)
    y_centers = np.arange(y_min + grid_size/2, y_max, grid_size)
    
    return grid, x_centers, y_centers