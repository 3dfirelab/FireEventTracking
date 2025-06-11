import xarray as xr
import numpy as np
import pdb 

class projForeFire: 

    def __init__(self, WSENLBRT):
        self.W, self.S, self.E,self. N = WSENLBRT[:4]
        self.L, self.B, self.R, self.T = WSENLBRT[4:] 

        # Width and height in projected space
        self.width = self.R - self.L
        self.height = self.T - self.B

    # --- Forward: (X, Y) to (Lon, Lat) ---
    def xy_to_lonlat(self, x, y):
        lon = self.W + (x - self.L) / self.width * (self.E - self.W)
        lat = self.S + (y - self.B) / self.height * (self.N - self.S)
        return lon, lat

    # --- Inverse: (Lon, Lat) to (X, Y) ---
    def lonlat_to_xy(self, lon, lat):
        x = self.L + (lon - self.W) / (self.E - self.W) * self.width
        y = self.B + (lat - self.S) / (self.N - self.S) * self.height
        return x, y


if __name__ == "__main__":

    filename = '/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/Pickles_active_2025-05-30_1300/FOREFIRE/000000031/data_43.5_5.0.nc'
    ds = xr.open_dataset(filename)

    WSENLBRT = np.array(ds['domain'].attrs['WSENLBRT'].split(',')).astype(float)
    convert = projForeFire(WSENLBRT)

     

    #print(convert.xy_to_lonlat(2961,5730))
    #print(convert.xy_to_lonlat(2961,5740))
    
    x, y = 24993.634719669,32493.758333312
    latlon = convert.xy_to_lonlat(x,y)
    z = ds['altitude'][0,0,int(np.round(x/100)), int(np.round(y/100))].item()
    print(latlon,z)

    x, y = 24998.634719669,32493.758333312
    latlon = convert.xy_to_lonlat(x,y)
    z = ds['altitude'][0,0,int(np.round(x/100)), int(np.round(y/100))].item()
    print(latlon,z)

