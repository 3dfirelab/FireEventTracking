import numpy as np 
from past.utils import old_div
from scipy import io,ndimage,stats,signal,interpolate,spatial 
import pdb 
import cv2
import matplotlib.pyplot as plt 
from matplotlib import cm

########################################################3
def compute_ros( maps_fire, arrivalTime, arrivalTime_raw, ros_max = 10): 
    
    '''
    arrivalTime    : interpolated field 
    arrivalTime_raw: data used for the interpolation. ie only observed contour
    '''

    data_quality = np.where(arrivalTime_raw>0, 1, 0)

    #compute gradient
    #normal_y, normal_x = np.gradient(arrivalTime.T)
    #normal_x = normal_x.T
    #normal_y = normal_y.T
    normal_x, normal_y = gradient_here(arrivalTime)

    idx = np.where(maps_fire.plotMask!=2)
    normal_x[idx] = 0 # set normal nul outside the plot
    normal_y[idx] = 0


    norm = np.sqrt(normal_x**2+normal_y**2)
    idx = np.where(norm!=0)
    normal_x[idx] = old_div(normal_x[idx], norm[idx])
    normal_y[idx] = old_div(normal_y[idx], norm[idx])
    

    #compute distance and time array to calculate ros
    xi, yi = maps_fire.grid_e, maps_fire.grid_n
    dx = xi[1,1]-xi[0,0]
    dy = yi[1,1]-yi[0,0]
    xi_m = xi + .5*dx
    yi_m = yi + .5*dy

    
    dx *= 0.75
    dy *= 0.75

    distance = np.sqrt((dx*normal_x)**2+(dy*normal_y)**2)
    time_t   = np.copy(arrivalTime)
    time_pdx = np.zeros_like(arrivalTime); data_quality_pdx = np.zeros_like(arrivalTime); vertical_distance_pdx = np.zeros_like(arrivalTime)
    time_mdx = np.zeros_like(arrivalTime); data_quality_mdx = np.zeros_like(arrivalTime); vertical_distance_mdx = np.zeros_like(arrivalTime)
   
    f    = interpolate.RegularGridInterpolator((xi_m[:,0], yi_m[0,:]), arrivalTime,  method='linear')
    f_dq = interpolate.RegularGridInterpolator((xi_m[:,0], yi_m[0,:]), data_quality, method='linear') 
    fTer = interpolate.RegularGridInterpolator((xi_m[:,0], yi_m[0,:]), maps_fire.terrain,  method='linear')
   
    #remove pts on the edge of the domain
    maskdomain = np.ones_like(arrivalTime,dtype=np.uint8)
    maskdomain[0,:] = 0
    maskdomain[-1,:] = 0
    maskdomain[:,0] = 0
    maskdomain[:,-1] = 0

    # Define the kernel (structuring element)
    kernel = np.ones((3,3), np.uint8)

    # Apply the erosion operation
    maskdomain = cv2.erode(maskdomain, kernel, iterations=1)

    idx = np.where((maps_fire.plotMask==2) & (distance!=0) & (maskdomain==1) )
    
    coord_pts     = np.vstack((xi_m[idx]                   , yi_m[idx]                   )).T
    
    coord_pts_pdx = np.vstack((xi_m[idx] + dx*normal_x[idx], yi_m[idx] + dy*normal_y[idx])).T
    time_pdx[idx]         = f(coord_pts_pdx)
    data_quality_pdx[idx] = f_dq(coord_pts_pdx)
    vertical_distance_pdx[idx] = np.abs(fTer(coord_pts) - fTer(coord_pts_pdx))

    coord_pts_mdx = np.vstack((xi_m[idx] - dx*normal_x[idx], yi_m[idx] - dy*normal_y[idx])).T
    time_mdx[idx]         = f(coord_pts_mdx)
    data_quality_mdx[idx] = f_dq(coord_pts_pdx)
    vertical_distance_mdx[idx] = np.abs(fTer(coord_pts) - fTer(coord_pts_mdx))


    #define ros array
    ros         = np.zeros_like(arrivalTime) - 999
    ros_quality = np.zeros_like(arrivalTime) - 999

    distance_pdx_tot =np.sqrt( distance**2+ vertical_distance_pdx**2)
    distance_mdx_tot =np.sqrt( distance**2+ vertical_distance_mdx**2)

    print('#####', distance.max())
    print('#####', distance_pdx_tot.flatten()[distance.max().argmax()])
    print('#####', vertical_distance_pdx.max())

    #compute ros
    time_limit_min = np.sqrt(dx**2+dy**2)/ros_max
    idx = np.where( (maps_fire.plotMask==2) &   ((time_pdx-time_t)>time_limit_min ) & ((time_t-time_mdx)>time_limit_min) & (distance!=0) & (ros < 0) )
    print(len(idx[0]))
    ros[idx]         = .5 * ( old_div(distance_pdx_tot[idx],(time_pdx-time_t)[idx]) +  old_div(distance_mdx_tot[idx],(time_t-time_mdx)[idx]) )
    ros_quality[idx] = .5 * ( data_quality_pdx[idx] + data_quality_mdx[idx])

    idx = np.where( (maps_fire.plotMask==2) &   ((time_pdx-time_t)>time_limit_min ) & ((time_t-time_mdx)<time_limit_min) & (distance!=0) & (ros < 0) )
    print(len(idx[0]))
    ros[idx]         = old_div(distance_pdx_tot[idx],(time_pdx-time_t)[idx])
    ros_quality[idx] = data_quality_pdx[idx] 
    
    idx = np.where( (maps_fire.plotMask==2) &   ((time_pdx-time_t)<time_limit_min ) & ((time_t-time_mdx)>time_limit_min) & (distance!=0) & (ros < 0) )
    print(len(idx[0]))
    ros[idx]         = old_div(distance_mdx_tot[idx],(time_t-time_mdx)[idx])
    ros_quality[idx] = data_quality_mdx[idx]

    idx = np.where( (maps_fire.plotMask==2) & ( ((time_pdx-time_t)<=time_limit_min) & ((time_t-time_mdx)<=time_limit_min) & (distance!=0) ) )
    print(len(idx[0]))
    ros[idx]         = ros_max
    ros_quality[idx] = -1
   
    idx = np.where( (maps_fire.plotMask==2) & ( (distance==0) ) )
    print(len(idx[0]))
    ros[idx]         = 0.
    ros_quality[idx] = 1. 


    #print stat
    idx = np.where((maps_fire.plotMask==2) & (ros != -999))
    print('ros min, max,', ros[idx].min(),  ros[idx].max())
    print('--')
  
    return normal_x, normal_y, ros, ros_quality

#################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

#####################################################################
def gradient_here(z):

    normal_x = np.zeros_like(z)
    normal_y = np.zeros_like(z)

    #along x
    normal_x[1:-1,:   ] = .5*(z[2:,: ]-z[:-2,:  ])

    idx = np.where((z[2:,: ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_x[i+1,j] = z[i+1,j]-z[i,j]

    idx = np.where((z[:-2,:  ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_x[i+1,j] = z[i+2,j]-z[i+1,j]

    idx = np.where((z[2:,: ]==-999)&(z[:-2,:  ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_x[i+1,j] = 0

    #along y
    normal_y[:   ,1:-1] = .5*(z[: ,2:]-z[:  ,:-2])
    
    idx = np.where((z[:,2:]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_y[i,j+1] = z[i,j+1]-z[i,j]

    idx = np.where((z[:,:-2 ]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_y[i,j+1] = z[i,j+2]-z[i,j+1]

    idx = np.where((z[:,2:]==-999)&(z[:,:-2]==-999))
    for i,j in zip(idx[0],idx[1]):
        normal_y[i,j+1] = 0

    return normal_x, normal_y

