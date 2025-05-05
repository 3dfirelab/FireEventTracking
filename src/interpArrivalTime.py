import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pdb 
import pandas as pd
import geopandas as gpd 
import shapely
import datetime
import pyproj
import rasterio
import rioxarray
from rasterio.mask import mask as maskRasterio
import json 
import sys
import cv2
import socket 
from PIL import Image, ImageDraw
import multiprocessing
from scipy import io,ndimage,stats,signal,interpolate,spatial 
import psutil
import gc 
from osgeo import gdal
from pathlib import Path
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob 
import importlib 
import elevation 
from shapely.geometry import LineString, Point, MultiPoint
import xarray as xr 

#homebrewed
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir+'../src_extra/RuningAverageImage/')
sys.path.append(src_dir+'../src_extra/Factor_number/')
import runingNeighbourImage
import factor 
import tools as map_tools

#################################
def load_geotiff( out_dir,plot_name,field_name):
    
    filename = out_dir + 'geotiff/' + plot_name + field_name + '.tif'
    src_ds = gdal.Open( filename )
    srcband = src_ds.GetRasterBand(1)
    return np.array(srcband.ReadAsArray()[::-1].T,dtype=float) 


#################################
def plot_geotiff(field, field_name, maps_fire, utm, plot_name, out_dir):
   
    if utm is None: return 0

    out_dir_tiff = out_dir + 'geotiff/'
    map_tools.ensure_dir(out_dir_tiff)
    #  Initialize the Image Size
    image_size = maps_fire.shape
    
    xi       = maps_fire.grid_e
    yi       = maps_fire.grid_n
    dx = xi[1,1]-xi[0,0]
    dy = yi[1,1]-yi[0,0]
    
    # set geotransform
    nx = image_size[0]
    ny = image_size[1]
    xmin, ymin, xmax, ymax = [maps_fire.grid_e.min(), maps_fire.grid_n.min(),  maps_fire.grid_e.max()+dx,  maps_fire.grid_n.max()+dy]
    xres = dx
    yres = dy
    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    dst_ds = gdal.GetDriverByName('GTiff').Create(out_dir_tiff+plot_name+field_name+'.tif', nx, ny, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    dst_ds.SetProjection(utm.to_wkt()) # export coords to file
    band3 = dst_ds.GetRasterBand(1)
    band3.WriteArray(field.T[::-1])   # write g-band to the raster
    band3.SetNoDataValue(-999)
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
   
    return 0


###################################################################
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


######################################################
def get_perimeter(in_, plotMask, contour_nbre_pixel_threshold=10, \
                  flag_time = 'na',                               \
                  flag_debug = False,                             \
                  flag_remove_peninsula=True,                     \
                  flag_fillUp=False,                              \
                  flag_return_mask=False,                         \
                  flag_only_oneZeros=False):
    
    if in_.max() != 1 : 
        pdb.set_trace()
   
    #if flag_fillUp:
    #    idx = np.where(in_ == 1)
    #    for [i,j] in zip(idx[0], idx[1]):
    #        MERDE


    if flag_remove_peninsula:
        #remove  peninsula type point
        idx = np.where(in_ == 1)
        for i_pt,i in enumerate(idx[0]):
            j = idx[1][i_pt]
            roi = np.copy(in_[i-1:i+2,j-1:j+2]).flatten()
            if roi.sum() == 2 :
                in_[i,j] = 0

    out_ = np.zeros(in_.shape)
    out_contourTag = np.zeros(in_.shape)

    thresh = np.copy(in_) * 255
    thresh = np.array(thresh,dtype=np.uint8)
    if flag_fillUp:
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        #fill up the mask
        in_ = np.zeros(in_.shape)
        width, height = plotMask.shape
        
        for i_c, contour in enumerate(contours):
            if hierarchy[0][i_c][-1] != -1: 
                continue
            if len(contour) < contour_nbre_pixel_threshold:
                continue

            polygon =[ tuple(pt[0]) for pt in contour ]

            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            idx = np.where(np.array(img) == 1)
            in_[idx] = 1

    else:
        #plt.imshow(thresh.T,origin='lower',interpolation='nearest')
        #plt.show()
        #pdb.set_trace()
        try:
            contours, hierarchy = cv2.findContours(np.ascontiguousarray(thresh),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        except: 
            pdb.set_trace()


    i_pt = 1
    for i_c, contour in enumerate(contours):
        
        if (flag_fillUp) & (hierarchy[0][i_c][-1] != -1): 
            continue

        if len(contour) < contour_nbre_pixel_threshold:
            continue

        for i, pt in enumerate(contour):
            out_[pt[0][1],pt[0][0]] = i_pt +1
            out_contourTag[pt[0][1],pt[0][0]] = i_c + 1
            i_pt += 1
   
    if flag_debug:
        plt.figure()
        plt.imshow(np.ma.masked_where(out_==0,out_).T,origin='lower',interpolation='nearest')
        plt.figure()
        plt.imshow(np.ma.masked_where(in_==0,in_).T,origin='lower',interpolation='nearest')
        plt.show()
        pdb.set_trace()

    if flag_only_oneZeros:
        idx = np.where(out_!=0)
        out_[idx] = 1

    #ax = plt.subplot(121)
    #ax.imshow(in_.T,origin='lower',interpolation='none')

    #ax = plt.subplot(122)
    #ax.imshow(np.ma.masked_where(out_==0,out_).T,origin='lower',interpolation='none')

    #plt.show()

    if flag_time == 'tdt':
        if flag_return_mask:
            return -1 * out_, out_contourTag, in_
        else:
            return -1 * out_, out_contourTag
    
    else:
        if flag_return_mask:
            return out_, out_contourTag, in_
        else:            
            return out_, out_contourTag



################################################
def interpolate_fire_spread_between_ff_star(arg):
    return interpolate_fire_spread_between_ff(*arg)


################################################
def find_neighbors_distance(pindx, pts_in ):

    tree_neighbour_src    = spatial.cKDTree(pts_in) # all point tree
    
    dist_, inds_ = tree_neighbour_src.query(pts_in[pindx], k = pts_in.shape[0])
        
    return inds_[np.argsort(dist_)]

################################################
def interpolate_fire_spread_between_ff(idx_contour, mask_nodata, arrivalTime_fillup, arrivalTime_clean, idx_mesh, resolution, time_reso, ros_max, flag_plot=False, id_cluster=None):

    arrivalTime_FS__       = np.where(arrivalTime_clean>=0,arrivalTime_clean,0)
    arrivalTime_FS_count__ = np.where(arrivalTime_clean>=0,1,0)
   
    pts = np.array(np.dstack( [idx_contour[0],idx_contour[1]] )[0],dtype=float)
    
    if pts.shape[0] <=2: 
        idx = np.where(arrivalTime_FS_count__>0)
        arrivalTime_FS__[idx] = arrivalTime_FS__[idx]/ arrivalTime_FS_count__[idx]
        if flag_plot: pdb.set_trace()
        return  arrivalTime_FS__, arrivalTime_FS_count__

    ''' 
    plt.scatter(tri_arrival.points[:,0],tri_arrival.points[:,1],c='b',s=100)
    mm = find_neighbors(17,tri_arrival)
    plt.scatter(tri_arrival.points[mm,0],tri_arrival.points[mm,1],c='r')
    plt.scatter(tri_arrival.points[17,0],tri_arrival.points[17,1],c='k')
    plt.show()
    pdb.set_trace()
    '''

    #sort time 
    time_pts = np.zeros(pts.shape[0]) 
    for i_pt_, pt_ in enumerate(pts): 
        time_pts[i_pt_] = arrivalTime_clean[int(pt_[0]),int(pt_[1])] 
    
    idx_time_sorted_pts = np.argsort(time_pts)
      

    '''timeminplot = arrivalTime_clean[idx_contour].min()
    timemaxplot = arrivalTime_clean[idx_contour].max()
    ax = plt.subplot(111)
    im = ax.imshow(np.ma.masked_where((arrivalTime_clean<0),arrivalTime_clean).T,origin='lower',interpolation='nearest',vmin=timeminplot,vmax=timemaxplot,cmap=mpl.cm.jet)
    im.format_cursor_data = lambda x : "{0:.2f}".format(x)
    ax.scatter(pts[:,0],pts[:,1],c='w',s=10)
    ax.set_xlim(idx_contour[0].min()-10,idx_contour[0].max()+10)
    ax.set_ylim(idx_contour[1].min()-10,idx_contour[1].max()+10)
    plt.show()
    pdb.set_trace()'''
    
    
    #rect_leftbottom = idx_contour[0].min(), idx_contour[1].min()
    #rect_rightTop   = idx_contour[0].max(), idx_contour[1].max()

    (center_x, center_y), (width, height), angle  = cv2.minAreaRect(np.array(pts,dtype=np.float32))
    
    '''timeminplot = arrivalTime_clean[idx_contour].min()
    timemaxplot = arrivalTime_clean[idx_contour].max()
    ax = plt.subplot(111)
    ax.imshow(np.ma.masked_where((arrivalTime_clean<0),arrivalTime_clean).T,origin='lower',interpolation='nearest',vmin=timeminplot,vmax=timemaxplot,cmap=mpl.cm.jet)
    ax.scatter(pts[:,0],pts[:,1],c='w',s=10)
    rect = mpl.patches.Rectangle((center_x, center_y), width, height, angle=angle) # 
    pc = mpl.collections.PatchCollection([rect], facecolor='r', alpha=.5, edgecolor='r')
    # Add collection to axes
    ax.add_collection(pc)
    ax.set_xlim(idx_contour[0].min()-10,idx_contour[0].max()+10)
    ax.set_ylim(idx_contour[1].min()-10,idx_contour[1].max()+10)
    plt.show()
    pdb.set_trace()'''
    
    max_distance_between_ff = 1.5* min([width, height]) #ros_max*(timepdt-time_pts.min()) / resolution 

    #collect info for each pts to get a distribution of ROS
    ###############
    info_per_pts = []
    all_ros = []
    all_ros_direction = []
    all_distance = []
    for idx_pt_here_selected, pt_here_selected in enumerate(pts[idx_time_sorted_pts[::-1]]): 
        
        info_per_pts.append([])
        info_per_pts[-1].append( [idx_pt_here_selected, pt_here_selected] )

        while True:
            #shoot backward
            #***********
            idx_pt_here_next = idx_pt_here_selected
            pt_here_next     = pt_here_selected
            timepdt = arrivalTime_clean[int(pt_here_next[0]),int(pt_here_next[1])] 
      
            idx_pt_here_next_tri =  ((pts - pt_here_next)**2).sum(axis=1).argmin()
            idx_direct_neighbor = find_neighbors_distance(idx_pt_here_next_tri, pts) # this give all point at time t 
            
            dist_pt_here_next = np.sqrt(np.sum(np.subtract( pts[idx_direct_neighbor], pt_here_next)**2,axis=1))
            idx_dist_ok = idx_direct_neighbor[np.where( (dist_pt_here_next>=2)&(dist_pt_here_next<= max_distance_between_ff)&((timepdt-time_pts[idx_direct_neighbor])>time_reso))] #MERDE100
            idx_dist_ok_ =                    np.where( (dist_pt_here_next>=2)&(dist_pt_here_next<= max_distance_between_ff)&((timepdt-time_pts[idx_direct_neighbor])>time_reso))  #MERDE100

            if len(idx_dist_ok_[0])==0:
                break

            idx_direct_neighbor_sorted = idx_dist_ok[ np.argsort(dist_pt_here_next[idx_dist_ok_])]
            dist_pt_here_next = np.sqrt(np.sum(np.subtract( pts[idx_direct_neighbor_sorted], pt_here_next)**2,axis=1))

            #first select point that can connect to pt_here_next and tag the one with ok path
            #---
            available_close_pt     = []
            available_close_pt_tmp = pts[idx_direct_neighbor_sorted]
            for i_ind, pt_here in enumerate(available_close_pt_tmp):
          
                #real time at start
                time = arrivalTime_clean[int(pt_here[0]),int(pt_here[1])] 
                #distance traveled
                p1p2_norm = dist_pt_here_next[i_ind]
                if p1p2_norm < 2:
                    #print 'd < 2'
                    continue
                elif p1p2_norm >= max_distance_between_ff:
                    #print 'd>limi'
                    continue
                
                p2 = np.array(pt_here_next)
                p1 = np.array(pt_here)
                p1p2 = p2-p1

                p3p1 = p1-idx_mesh[:,:2]
                dd = np.abs(np.cross(np.insert(p1p2,2,0), np.insert(p3p1,2,0,axis=1),axisb=1))[:,-1]/p1p2_norm
                #dd = np.abs(np.cross(p1p2, p3p1,axisb=1))/p1p2_norm

                idx_segment = np.where((dd<=.5) & (idx_mesh[:,0]>=min([p1[0],p2[0]])) & (idx_mesh[:,0]<=max([p1[0],p2[0]])) \
                                                & (idx_mesh[:,1]>=min([p1[1],p2[1]])) & (idx_mesh[:,1]<=max([p1[1],p2[1]])) )
                idx_segment2d        = np.unravel_index(idx_segment,arrivalTime_clean.shape)
                idx_segment2d_center = np.unravel_index(idx_segment[0][1:-1],arrivalTime_clean.shape)
              
                ros_direction = p1p2[0]+1j*p1p2[1]

                if (idx_segment[0].shape[0] <=2):
                    #print ('d < 2 *')
                    #pdb.set_trace()
                    continue
                elif (0 in mask_nodata[idx_segment2d_center]) |\
                     (  (np.sum(np.where( arrivalTime_clean[idx_segment2d_center]>0, arrivalTime_clean[idx_segment2d_center], np.zeros_like(idx_segment2d_center) ) ) > 0) ):
                    #print ('arrivalTime_clean pt on the way')
                    continue
                elif -999 in arrivalTime_fillup[idx_segment2d_center]: # if an unburnt pixel is on the path, disregard this path
                    #print ('island')
                    continue
                else:
                    if                                                             (arrivalTime_fillup[idx_segment2d_center].max() > timepdt): 
                        flag_filluptime_ok_in_the_path  = 0 #not ok
                    else:
                        flag_filluptime_ok_in_the_path  = 1 #ok

                    available_close_pt.append([i_ind, 
                                               pt_here, 
                                               np.linalg.norm(p2-p1), 
                                               idx_segment2d, 
                                               resolution*np.linalg.norm(p2-p1)/ max([(timepdt-time),1.e-6]), 
                                               flag_filluptime_ok_in_the_path, 
                                               ros_direction,  
                                               max([(timepdt-time),1.e-6]) ])
            
            dist_available_close_pt    = np.array( [available_close_pt_[2] for available_close_pt_ in available_close_pt] )
            meanRos_available_close_pt = np.array( [available_close_pt_[4] for available_close_pt_ in available_close_pt] )
            pathok_available_close_pt = np.array( [available_close_pt_[5] for available_close_pt_ in available_close_pt] )
            time_available_close_pt = np.array( [available_close_pt_[7] for available_close_pt_ in available_close_pt] )
          

            #now select the ok path, and within those the one with shortest distance 
            if pathok_available_close_pt.shape[0] ==  0: 
                #print 'no path selected'
                break
           
            idx_path_ok = np.where( pathok_available_close_pt ==1 )
            if len(idx_path_ok[0])==0: 
                #print 'no path ok'
                break
            dist_available_close_pt_min = dist_available_close_pt[idx_path_ok].min()

            idx_available_close_pt_kept = np.where( (dist_available_close_pt<1.2*dist_available_close_pt_min)                                           & 
                                                    ( pathok_available_close_pt == 1)                                                                   
                                                  )[0]
            
            if idx_available_close_pt_kept.shape[0] ==  0: 
                #print 'no dist ok'
                break
            
            ##print dist_available_close_pt[idx_available_close_pt_kept].min(), time_available_close_pt.min(),  
            ##print '* '  


            # only keep point that are within 20% of the min distance of available_close_pt and build up array for statistic
            #---
            for ii, [i_ind, pt_here, p1p2_norm, idx_segment2d, meanRos, pathok, ros_direction, time_elapsed] in enumerate([ available_close_pt[i] for i in idx_available_close_pt_kept] ):
                info_per_pts[-1].append(['b',i_ind, pt_here, pt_here_next, p1p2_norm, idx_segment2d, meanRos, pathok, ros_direction ])
                all_ros.append(meanRos) 
                all_distance.append(p1p2_norm) 
                all_ros_direction.append(ros_direction) 

            break
  
        while True: 
            idx_pt_here  = idx_pt_here_selected
            pt_here      = pt_here_selected
            #shoot forward
            #***********
            time = arrivalTime_clean[int(pt_here[0]),int(pt_here[1])] 
      
            idx_pt_here_tri =  ((pts - pt_here)**2).sum(axis=1).argmin()
            idx_direct_neighbor = find_neighbors_distance(idx_pt_here_tri, pts) # this give all point at time t 
            
            dist_pt_here = np.sqrt(np.sum(np.subtract( pts[idx_direct_neighbor], pt_here)**2,axis=1))
            idx_dist_ok = idx_direct_neighbor[np.where( (dist_pt_here>=2)&(dist_pt_here<= max_distance_between_ff)&((time_pts[idx_direct_neighbor]-time)>time_reso))] #MERDE100
            idx_dist_ok_ =                    np.where( (dist_pt_here>=2)&(dist_pt_here<= max_distance_between_ff)&((time_pts[idx_direct_neighbor]-time)>time_reso))  #MERDE100

            if len(idx_dist_ok_[0])==0:
                break

            idx_direct_neighbor_sorted = idx_dist_ok[ np.argsort(dist_pt_here[idx_dist_ok_])]
            dist_pt_here = np.sqrt(np.sum(np.subtract( pts[idx_direct_neighbor_sorted], pt_here)**2,axis=1))

            #first select point that can connect to pt_here and tag the one with ok path
            #---
            available_close_pt     = []
            available_close_pt_tmp = pts[idx_direct_neighbor_sorted]
            for i_ind, pt_here_next in enumerate(available_close_pt_tmp):
         
                #print i_ind, len(available_close_pt_tmp), pt_here_next, 
                #real time at start
                timepdt = arrivalTime_clean[int(pt_here_next[0]),int(pt_here_next[1])] 
                #distance traveled
                p1p2_norm = dist_pt_here[i_ind]
                if p1p2_norm < 2:
                    #print 'd < 2'
                    continue
                elif p1p2_norm >= max_distance_between_ff:
                    #print 'd>limi'
                    continue
                
                p2 = np.array(pt_here_next)
                p1 = np.array(pt_here)
                p1p2 = p2-p1

                p3p1 = p1-idx_mesh[:,:2]
                dd = np.abs(np.cross(np.insert(p1p2,2,0), np.insert(p3p1,2,0,axis=1),axisb=1))[:,-1]/p1p2_norm
                #np.abs(np.cross(p1p2, p3p1,axisb=1))/p1p2_norm
                
                idx_segment = np.where((dd<=.5) & (idx_mesh[:,0]>=min([p1[0],p2[0]])) & (idx_mesh[:,0]<=max([p1[0],p2[0]])) \
                                                & (idx_mesh[:,1]>=min([p1[1],p2[1]])) & (idx_mesh[:,1]<=max([p1[1],p2[1]])) )
                idx_segment2d        = np.unravel_index(idx_segment,         arrivalTime_clean.shape)
                idx_segment2d_center = np.unravel_index(idx_segment[0][1:-1],arrivalTime_clean.shape)
              
                ros_direction = p1p2[0]+1j*p1p2[1]

                if (idx_segment[0].shape[0] <=2):
                    #print 'd < 2 *'
                    continue
                elif (0 in mask_nodata[idx_segment2d_center]) |\
                     (  (np.sum(np.where( arrivalTime_clean[idx_segment2d_center]>0, arrivalTime_clean[idx_segment2d_center], np.zeros_like(idx_segment2d_center) ) ) > 0) ):
                    #print 'arrivalTime_clean pt on the way'
                    continue
                elif -999 in arrivalTime_fillup[idx_segment2d_center]: # if an unburnt pixel is on the path, disregard this path
                    #print 'island'
                    continue
                else:
                    if                                                             (arrivalTime_fillup[idx_segment2d_center].max() > timepdt): 
                        flag_filluptime_ok_in_the_path  = 0 #not ok
                    else:
                        flag_filluptime_ok_in_the_path  = 1 #ok

                    available_close_pt.append([i_ind, 
                                               pt_here_next, 
                                               np.linalg.norm(p2-p1), 
                                               idx_segment2d, 
                                               resolution*np.linalg.norm(p2-p1)/ max([(timepdt-time),1.e-6]), 
                                               flag_filluptime_ok_in_the_path, 
                                               ros_direction,  
                                               max([(timepdt-time),1.e-6]) ])
            
            dist_available_close_pt    = np.array( [available_close_pt_[2] for available_close_pt_ in available_close_pt] )
            meanRos_available_close_pt = np.array( [available_close_pt_[4] for available_close_pt_ in available_close_pt] )
            pathok_available_close_pt = np.array( [available_close_pt_[5] for available_close_pt_ in available_close_pt] )
            time_available_close_pt = np.array( [available_close_pt_[7] for available_close_pt_ in available_close_pt] )
          
            #now select the ok path, and within those the one with shortest distance 
            if pathok_available_close_pt.shape[0] ==  0: 
                #print 'no path selected'
                break
           
            idx_path_ok = np.where( pathok_available_close_pt ==1 )
            if len(idx_path_ok[0])==0: 
                #print 'no path ok'
                break
            dist_available_close_pt_min = dist_available_close_pt[idx_path_ok].min()

            idx_available_close_pt_kept = np.where( (dist_available_close_pt<1.2*dist_available_close_pt_min)                                           & 
                                                    ( pathok_available_close_pt == 1)                                                                   
                                                  )[0]
            
            if idx_available_close_pt_kept.shape[0] ==  0: 
                #print 'no dist ok'
                break
            
            #print dist_available_close_pt[idx_available_close_pt_kept].min(), time_available_close_pt.min(),  
            #print '* '  


            # only keep point that are within 20% of the min distance of available_close_pt and build up array for statistic
            #---
            for ii, [i_ind, pt_here_next, p1p2_norm, idx_segment2d, meanRos, pathok, ros_direction, time_elapsed] in enumerate([ available_close_pt[i] for i in idx_available_close_pt_kept] ):
                info_per_pts[-1].append(['f',i_ind, pt_here, pt_here_next, p1p2_norm, idx_segment2d, meanRos, pathok, ros_direction ])
                all_ros.append(meanRos) 
                all_distance.append(p1p2_norm) 
                all_ros_direction.append(ros_direction) 
        
            break

    #get some stat on ROS, to filter out high value
    ###############
    ros_min = 0
    if len(all_ros)>10:
        kernel = stats.gaussian_kde(np.array(all_ros))
        ros_scale = np.linspace(0,10,1000)
        ros_pdf = kernel(ros_scale)/kernel.integrate_box_1d(ros_scale.min(),ros_scale.max())

        i=1; p=0
        while p<.99:
            ros_max_new = ros_scale[i]
            idx = np.where(ros_scale<=ros_max_new)
            p = np.trapz(ros_pdf[idx],ros_scale[idx])
            i+=1

            if i == len(ros_scale):
                break
        ros_max = min([ros_max_new,ros_max])

    if len(all_distance) > 0:
        all_distance = np.array(all_distance)
        std_all_distance  = all_distance.std()
        mean_all_distance = all_distance.mean()
    else: 
        mean_all_distance = -999
        std_all_distance  = -999
        #print 'no pts in all_distance'
        #pdb.set_trace()

    if flag_plot: print(id_cluster, ros_max) 
    if flag_plot: print(id_cluster, mean_all_distance, std_all_distance)
    
    # then loop on pts info, select the correct pair and compute ROS
    list_arrivalTime_FS__ = [[] for x in range(arrivalTime_FS__.flatten().shape[0])]
    list_direction_FS__ = [[] for x in range(arrivalTime_FS__.flatten().shape[0])]
    ###############
    for i_pts_, info_per_pts_ in enumerate(info_per_pts): 
       
        idx_pt_here_selected, pt_here_selected = info_per_pts_[0]
        
        available_close_pt_final_f = [] #np.zeros([len(info_per_pts[idx_pt_here_next]),2])
        available_close_pt_final_b = [] #np.zeros([len(info_per_pts[idx_pt_here_next]),2])
        for ii, [mode,i_ind, pt_here, pt_here_next, p1p2_norm, idx_segment2d, meanRos, pathok, ros_direction ] in enumerate(info_per_pts_[1:]):

            if (meanRos>ros_max) | (meanRos<ros_min):  continue
            if p1p2_norm-mean_all_distance > 1.5*std_all_distance : continue # only remove too long path

            if mode == 'f': available_close_pt_final_f.append(pt_here_next)
            if mode == 'b': available_close_pt_final_b.append(pt_here)

            time = arrivalTime_clean[int(pt_here[0]),int(pt_here[1])] 
            timepdt = arrivalTime_clean[int(pt_here_next[0]),int(pt_here_next[1])] 
            local_ros = (resolution*p1p2_norm)/(timepdt-time)
            for pt in np.dstack(idx_segment2d)[0]:
                arrivalTime_FS__[pt[0],pt[1]]       += resolution*np.linalg.norm(pt-pt_here)/local_ros + time
                arrivalTime_FS_count__[pt[0],pt[1]] += 1
                
                idx = np.ravel_multi_index((pt[0],pt[1]),arrivalTime_FS__.shape)
                list_arrivalTime_FS__[idx].append(resolution*np.linalg.norm(pt-pt_here)/local_ros + time)
                list_direction_FS__[idx].append(ros_direction)

        if flag_plot & ((len(available_close_pt_final_f)>0)|(len(available_close_pt_final_b)>0)):
            percent_inside_cluster_fillup = 1.*np.where( (mask_nodata==1) & (arrivalTime_FS__>0) )[0].shape[0]/ np.where(mask_nodata==1)[0].shape[0]
            mpl.rcdefaults()
            mpl.rcParams['text.usetex'] = False
            mpl.rcParams['axes.linewidth'] = 1
            mpl.rcParams['axes.labelsize'] = 18.
            mpl.rcParams['legend.fontsize'] = 'small'
            mpl.rcParams['legend.fancybox'] = True
            mpl.rcParams['font.size'] = 18.
            mpl.rcParams['xtick.labelsize'] = 18.
            mpl.rcParams['ytick.labelsize'] = 18.
            mpl.rcParams['figure.subplot.left'] = .05
            mpl.rcParams['figure.subplot.right'] = .95
            mpl.rcParams['figure.subplot.top'] = .9
            mpl.rcParams['figure.subplot.bottom'] = .1
            mpl.rcParams['figure.subplot.hspace'] = 0.1
            mpl.rcParams['figure.subplot.wspace'] = 0.2   
            fig = plt.figure(figsize=(12,12))
            timeminplot = arrivalTime_clean[idx_contour].min()
            timemaxplot = arrivalTime_clean[idx_contour].max()
            ax = plt.subplot(111)
            im1 = ax.imshow(np.ma.masked_where(arrivalTime_clean<0,arrivalTime_clean).T,origin='lower',interpolation='nearest',alpha=.5,vmin=timeminplot,vmax=timemaxplot)
            #im.format_cursor_data = lambda x : "{0:.2f}".format(x) if isinstance(x,float) else 'no data'
            im2 = ax.imshow(np.ma.masked_where((arrivalTime_clean<timeminplot)|(arrivalTime_clean>timepdt),arrivalTime_clean).T,origin='lower',interpolation='nearest',vmin=timeminplot,vmax=timemaxplot,cmap=mpl.cm.jet)
            #im.format_cursor_data = lambda x : "{0:.2f}".format(x) if isinstance(x,float) else 'no data'
            im3 = ax.imshow(np.ma.masked_where((arrivalTime_FS__==arrivalTime_clean)|(arrivalTime_FS__==0),np.where(arrivalTime_FS_count__!=0, arrivalTime_FS__/arrivalTime_FS_count__,arrivalTime_FS__ )).T,origin='lower',alpha=.5,vmin=timeminplot,vmax=timemaxplot,interpolation='nearest',cmap=mpl.cm.jet)
            #im.format_cursor_data = lambda x : "{0:.2f}".format(x) if isinstance(x,float) else 'no data'
            
            divider = make_axes_locatable(ax)
            cbaxes = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im2 ,cax = cbaxes)    
            cbar.set_label('time (s)')
            
            ax.scatter(pts[idx_direct_neighbor,0],pts[idx_direct_neighbor,1],c='w',s=10)
            ax.scatter(pt_here_selected[0],pt_here_selected[1],c='k',s=200)
            [ax.scatter(pt[0],pt[1], c='m',s=200, marker='>') for pt in available_close_pt_final_f]
            [ax.scatter(pt[0],pt[1], c='m',s=200, marker='<') for pt in available_close_pt_final_b]
            ax.set_xlim(pts[idx_direct_neighbor,0].min()-10,pts[idx_direct_neighbor,0].max()+10)
            ax.set_ylim(pts[idx_direct_neighbor,1].min()-10,pts[idx_direct_neighbor,1].max()+10)
            
            fig.savefig('/home/paugam/Desktop/batti/ff/foo_{:04d}_{:06d}'.format(id_cluster,i_pts_),dpi=400)
            plt.close(fig)   
            #pdb.set_trace()

    #ros_direction_std_lim = .34 # 20 dgre
    ros_direction_std_lim = .698 # 40 dgre
    #arrivalTime_FS_count__lim = np.percentile(arrivalTime_FS_count__,10)

    #relatice_std = np.zeros_like(arrivalTime_FS__) -999
    ros_direction_mean = np.zeros_like(arrivalTime_FS__) -999
    ros_direction_std  = np.zeros_like(arrivalTime_FS__) -999
    for idx_, list_ in enumerate(list_direction_FS__):
        if len(list_)<=2: 
            continue
        idx__ = np.unravel_index(idx_,arrivalTime_FS__.shape)
        #relatice_std[idx__[0],idx__[1]] = np.array(list_).std()
        ros_direction_mean[idx__[0],idx__[1]] = np.angle(np.array(list_).mean())
        v_mean = np.array([np.array(list_).mean().real, np.array(list_).mean().imag] )
        vec_list = []
        for v in list_ :
            if ((np.linalg.norm(v)*np.linalg.norm(v_mean))!= 0) : vec_list.append( [v.real,v.imag] ) 
        cos = np.round( [np.dot(v,v_mean)/(np.linalg.norm(v)*np.linalg.norm(v_mean)) for v in vec_list], 9)
        angles = np.arccos( cos ) 
        angles = np.array(angles)
        ros_direction_std[idx__[0],idx__[1]]  = angles[np.where(angles!=-999)].std()
    
    if False: #(timepdt-time) > 4 : #(timepdt > 637) & (timepdt < 638) :
        percent_inside_cluster_fillup = 1.*np.where( (mask_nodata==1) & (arrivalTime_FS__>0) )[0].shape[0]/ np.where(mask_nodata==1)[0].shape[0]
        fig = plt.figure(figsize=(36,12))
        timeminplot = arrivalTime_clean[idx_contour].min()
        timemaxplot = arrivalTime_clean[idx_contour].max()
        
        ax = plt.subplot(141)
        im = ax.imshow(np.ma.masked_where((arrivalTime_clean<0),arrivalTime_clean).T,origin='lower',interpolation='nearest',vmin=timeminplot,vmax=timemaxplot,cmap=mpl.cm.jet)
        im = ax.imshow(np.ma.masked_where( (arrivalTime_FS__==arrivalTime_clean)|(arrivalTime_FS__==0)|(ros_direction_std>ros_direction_std_lim),
                                    np.where(arrivalTime_FS_count__!=0, arrivalTime_FS__/arrivalTime_FS_count__,arrivalTime_FS__ )).T,
                                    origin='lower',alpha=.5,vmin=timeminplot,vmax=timemaxplot,interpolation='nearest',cmap=mpl.cm.jet)
        im.format_cursor_data = lambda x : "{0:.2f}".format(x) if isinstance(x,float) else 'no data'
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im ,cax = cbaxes)    
        ax.scatter(pts[idx_direct_neighbor,0],pts[idx_direct_neighbor,1],c='w',s=10)
        ax.set_xlim(idx_contour[0].min()-10,idx_contour[0].max()+10)
        ax.set_ylim(idx_contour[1].min()-10,idx_contour[1].max()+10)
        ax.set_title('{:2f} of cluster fill up with data'.format(percent_inside_cluster_fillup))

        ax = plt.subplot(142)
        im = ax.imshow(np.ma.masked_where(ros_direction_mean==-999,np.where( (ros_direction_mean<0)&(ros_direction_mean!=-999), ros_direction_mean + 2*np.pi, ros_direction_mean)).T,origin='lower',interpolation='nearest') 
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im ,cax = cbaxes)    
        ax.scatter(pts[idx_direct_neighbor,0],pts[idx_direct_neighbor,1],c='w',s=10)
        ax.set_xlim(idx_contour[0].min()-10,idx_contour[0].max()+10)
        ax.set_ylim(idx_contour[1].min()-10,idx_contour[1].max()+10)
        ax.set_title(r'{:2f} of cluster fill up with data'.format(percent_inside_cluster_fillup))
            
        ax = plt.subplot(143)
        im = ax.imshow(np.ma.masked_where(ros_direction_std==-999,ros_direction_std).T,origin='lower',interpolation='nearest') 
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im ,cax = cbaxes)    
        ax.scatter(pts[idx_direct_neighbor,0],pts[idx_direct_neighbor,1],c='w',s=10)
        ax.set_xlim(idx_contour[0].min()-10,idx_contour[0].max()+10)
        ax.set_ylim(idx_contour[1].min()-10,idx_contour[1].max()+10)
        ax.set_title(r'{:2f} of cluster fill up with data'.format(percent_inside_cluster_fillup))
        
        ax = plt.subplot(144)
        im = ax.imshow(arrivalTime_FS_count__.T,origin='lower',interpolation='nearest') 
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im ,cax = cbaxes)    
        ax.scatter(pts[idx_direct_neighbor,0],pts[idx_direct_neighbor,1],c='w',s=10)
        ax.set_xlim(idx_contour[0].min()-10,idx_contour[0].max()+10)
        ax.set_ylim(idx_contour[1].min()-10,idx_contour[1].max()+10)
        ax.set_title(r'{:2f} of cluster fill up with data'.format(percent_inside_cluster_fillup))

        if flag_plot : 
            plt.show()
            pdb.set_trace()
        else:
            fig.savefig('/home/paugam/Desktop/batti/ff2/foo_{:04d}'.format(id_cluster),dpi=400)
            plt.close(fig)
   
    #percent_inside_cluster_fillup = 1.*np.where( (mask_nodata==1) & (arrivalTime_FS__>0) )[0].shape[0] / np.where(mask_nodata==1)[0].shape[0]
    #if percent_inside_cluster_fillup < 0.75: # assume FS interpolation did not work, reset all to input
    #    arrivalTime_FS__       = np.where(arrivalTime_clean>=0,arrivalTime_clean,0)
    #    arrivalTime_FS_count__ = np.where(arrivalTime_clean>=0,1,0)

    idx = np.where((arrivalTime_FS_count__>0))
    arrivalTime_FS__[idx] = arrivalTime_FS__[idx]/ arrivalTime_FS_count__[idx]
   
    #remove point where std of angle of all path is above 0.34 rad = 20 degree  
    arrivalTime_FS__ = np.where( (ros_direction_std<=ros_direction_std_lim), arrivalTime_FS__, np.zeros_like(arrivalTime_FS__) )

    #if id_cluster == 49: pdb.set_trace()

    return arrivalTime_FS__ , arrivalTime_FS_count__


#####################################################################
def interpolate_arrivalTime(out_dir, maps_fire, crs,                                    \
                            arrivalTime_fillup, arrivalTime_edgePixel,                    \
                            input_info,                                                   \
                            #flag_interpolate_betwee_ff='rbf',                             \
                            flag_plot=False,                                              \
                            frame_time_period=None,                                       \
                            flag_large_scale=False,                                       \
                            large_scale_shape_raw=None,                                   \
                            large_scale_grid_raw=None,                                    \
                            set_rbf_subset=None,                                          \
                            arrivalTime_ignition_seed=None,                               ):
    
    plot_name       = input_info['plot_name']                        
    flag_parallel   = input_info['flag_parallel']                
    ros_max         = input_info['ros_max']                            

    resolution = maps_fire.grid_e[1,1]-maps_fire.grid_e[0,0]
    if input_info['flag_use_time_reso_constraint']: 
        time_reso  = np.diff(np.sort(np.unique(arrivalTime_fillup))).min()
    else: 
        time_reso = 1.e-6

    if flag_large_scale: 
        arrivalTime_fillup_in = np.copy(arrivalTime_fillup)
        
        #get unburnt island and remove point around them that show same arrival time for all their neighbour
        #-----------
        tresh = np.copy(maps_fire.plotMask)
        tresh = np.array(tresh,dtype=np.uint8)
        contours, hierarchy = cv2.findContours(tresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
       
        mask_entire_fire   = np.zeros_like(maps_fire.plotMask)
        mask_unburntIsland = np.zeros_like(maps_fire.plotMask)

        idx_dad = np.where(hierarchy[:,:,-1]==-1)[-1]
        try:
            polygon =[ tuple(pt[0]) for pt in contours[idx_dad[0]] ]
        except:
            pdb.set_trace()
        img = Image.new('L',mask_entire_fire.shape , 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        idx = np.where(np.array(img) == 1)
        
        mask_entire_fire[idx] = 1
        idx_unburnt_island = np.where( (mask_entire_fire==1) & (maps_fire.plotMask!=2) )
        mask_unburntIsland[idx_unburnt_island] = 1
        
        print('unburnt pixel = ', len(idx_unburnt_island[0]))

        #interpolate on the island
        idx = np.where(mask_unburntIsland!=1)
        coord_pts = np.vstack((maps_fire.grid_e[idx],maps_fire.grid_n[idx])).T
        data      = arrivalTime_fillup[idx]
        arrivalTime_fillup = interpolate.griddata(coord_pts , data, (maps_fire.grid_e,maps_fire.grid_n), fill_value=-999, method='nearest')
   
        #plt.imshow(np.ma.masked_where(maps_fire.plotMask!=2,arrivalTime).T,origin='lower',interpolation='nearest')
        #plt.show()
        #pdb.set_trace()

        #remove single pixel from the arrival time to add them back after the interpolation
        #---------
        idx_single_pixel = [[],[]]
        for time_ in sorted(np.unique(arrivalTime_fillup)):
            if time_ == -999: 
                continue

            idx = np.where(arrivalTime_fillup == time_)
            mask_ = np.zeros_like(arrivalTime_fillup)
            mask_[idx] = 1

            s = [[1,1,1], \
                 [1,1,1], \
                 [1,1,1]] # for diagonal
            labeled, num_cluster = ndimage.label(mask_, structure=s )
            for i_label in range(num_cluster):
                idx = np.where(labeled==i_label+1)
                if len(idx[0]) == 1:
                    idx_single_pixel[0].append(idx[0])
                    idx_single_pixel[1].append(idx[1])  

        idx_single_pixel = [np.array(idx_single_pixel[0]),np.array(idx_single_pixel[1])]
        mask_single_pixel = np.zeros_like(arrivalTime_fillup)
        mask_single_pixel[idx_single_pixel] = 1

        #plt.imshow(np.ma.masked_where(arrivalTime_fillup<0,arrivalTime_fillup).T,origin='lower',interpolation='nearest')
        #plt.imshow(np.ma.masked_where(mask_single_pixel==0,mask_single_pixel).T,origin='lower',interpolation='nearest',alpha=.3)
        #plt.scatter(idx_single_pixel[0],idx_single_pixel[1])
        #plt.show()
        #pdb.set_trace()
       
        print('single pixel = ', len(idx_single_pixel[0]))
        if len(idx_single_pixel[0]) > 1:
            #interpolate on single pixle
            idx = np.where(mask_single_pixel!=1)
            coord_pts = np.vstack((maps_fire.grid_e[idx],maps_fire.grid_n[idx])).T
            data      = arrivalTime_fillup[idx]
            arrivalTime_fillup = interpolate.griddata(coord_pts , data, (maps_fire.grid_e,maps_fire.grid_n), fill_value=-999, method='nearest')

    
    idx_time_available = np.where(np.unique(arrivalTime_fillup)>0)
    time_available=np.sort(np.unique(arrivalTime_fillup)[idx_time_available])

    idx_plot = np.where(maps_fire.plotMask ==2)

    print('  --')
    print('  create arrivalTime_clean')
    arrivalTime_clean = np.zeros_like(arrivalTime_fillup) - 999.
    if not os.path.isfile(out_dir + 'arrivalTime_clean.npy') :
        for itime, time in enumerate(time_available):
            #print time
            idx = np.where( (arrivalTime_fillup<=time) & (arrivalTime_fillup!=-999))   # needs to improve here fr situation where arrivalTime map derived from thermal cannot say if pixels were burnt
            mask = np.zeros_like(arrivalTime_fillup)
            mask[idx]=1

            roi_ff_line_t, roi_ff_contour_t_tag = get_perimeter(mask, maps_fire.plotMask, contour_nbre_pixel_threshold=4,flag_debug=False )

            idx_mask_ff = np.where(roi_ff_line_t >=1 )
            mask_ff = np.zeros_like(arrivalTime_fillup)
            mask_ff[idx_mask_ff] = 1

            fig = plt.figure(figsize=(12,5))
            ax = plt.subplot(131)
            ax.imshow(np.ma.masked_where((arrivalTime_fillup>time) | (arrivalTime_fillup<0), arrivalTime_fillup).T, origin='lower')
            ax = plt.subplot(132)
            ax.imshow(mask.T, origin='lower')
            ax = plt.subplot(133)
            ax.imshow(mask_ff.T, origin='lower')
            plt.show()

            idx = np.where((mask_ff==1)&(arrivalTime_clean<0))
            arrivalTime_clean[idx] = arrivalTime_fillup[idx]
        
        if arrivalTime_clean.max() == -999: # there is no close contour, we still give a chance to the rbf interpolation 
            arrivalTime_clean = arrivalTime_fillup
        
        np.save( out_dir+'arrivalTime_clean', arrivalTime_clean)

    else:
        arrivalTime_clean = np.load(out_dir+ 'arrivalTime_clean.npy')

    if False:
        plt.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
        plt.title('arrivalTime clean')
        plt.show()


    #remove perimeter formed by image edge. important for RxCadre data
    arrivalTime_clean_copy = np.copy(arrivalTime_clean)
    try:
        arrivalTime_tmp = np.where(arrivalTime_edgePixel!=1, np.zeros_like(arrivalTime_clean)-999, np.abs(runingNeighbourImage.runing_minmax_neighbor(arrivalTime_fillup,3,flag='max') - arrivalTime_clean) )
        arrivalTime_clean = np.where( (arrivalTime_tmp<=3) & (arrivalTime_edgePixel==1),np.zeros_like(arrivalTime_clean)-999.,arrivalTime_clean)
    except: 
        pdb.set_trace()
    #arrivalTime_clean_tmp  = np.where(arrivalTime_edgePixel==1,np.zeros_like(arrivalTime_clean)-999.,arrivalTime_clean)
    #arrivalTime_clean_tmp2 = runingNeighbourImage.runing_minmax_neighbor(arrivalTime_clean_tmp,3,flag='max') # flag point which are not connected with fire perimeterutm
    #arrivalTime_clean = np.where( (arrivalTime_clean_tmp2<0) & (arrivalTime_edgePixel==1),np.zeros_like(arrivalTime_clean)-999.,arrivalTime_clean)

    #insert ignition information. important for RxCadre data where long time (>100s) can happend between images
    tag_ignition_side = np.zeros_like(arrivalTime_clean)-999
    if arrivalTime_ignition_seed is not None:
        print('  --')
        print('  add seed from ignition path')
    
        if 'arrivalTime_ignition_seed_windDirection' in list(input_info.keys()):
            winddirection = input_info['arrivalTime_ignition_seed_windDirection']

            grdx_, grdy_ = gradient_here(arrivalTime_ignition_seed)
            norm = np.sqrt(np.add(grdx_**2,grdy_**2) )
            idx = np.where(norm>0)
            grdx = np.zeros_like(grdx_); grdy = np.zeros_like(grdx_)
            grdx[idx] = -1*grdy_[idx]/norm[idx] #pointing outside with 0 point north
            grdy[idx] =  1*grdx_[idx]/norm[idx]
            direction = np.angle(  grdx + 1.j*grdy , deg=True ) 
            idx = np.where(direction<0)
            direction[idx] = 360+direction[idx]
            idx = np.where(direction>0)
            direction[idx] = 360-direction[idx]
            
            mask_ignition = np.where(arrivalTime_ignition_seed>0, 255*np.ones(arrivalTime_ignition_seed.shape,dtype=np.uint8), np.zeros(arrivalTime_ignition_seed.shape,dtype=np.uint8) )
            contours,      hierarchy      = cv2.findContours( mask_ignition, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                mask_contour = np.zeros_like(arrivalTime_ignition_seed); mask_contour[contour[:,0,1],contour[:,0,0]] = 1
                tag_ignition_side_ = np.where( (mask_contour==1), np.abs(direction-winddirection)     , np.zeros_like(arrivalTime_ignition_seed) )
                tag_ignition_side_[np.where(tag_ignition_side_>180)] = 360 -  tag_ignition_side_[np.where(tag_ignition_side_>180)]
                
                tag_ignition_side = np.where((mask_contour==1) & (tag_ignition_side_<=90),  1*np.ones_like(direction), tag_ignition_side) #front fire
                tag_ignition_side = np.where((mask_contour==1) & (tag_ignition_side_> 90), -1*np.ones_like(direction), tag_ignition_side) #back  fire

            #idx = np.where( (runingNeighbourImage.runing_minmax_neighbor(tag_ignition_side_tmp, 3,flag='min') == -1) & (mask_ignition==255) & (np.abs(tag_ignition_side)!=1))
            #plt.imshow(np.ma.masked_where(tag_ignition_side==-999,tag_ignition_side).T,origin='lower',interpolation='nearest')
            #plt.scatter(idx[0],idx[1])
            #plt.show()
            #pdb.set_trace()

            #add an extra line on the back fire ignition
            tag_ignition_side_tmp = np.where(tag_ignition_side==-999, np.zeros_like(tag_ignition_side), tag_ignition_side)
            mask_ignition = np.where(arrivalTime_ignition_seed>0, 255*np.ones(arrivalTime_ignition_seed.shape,dtype=np.uint8), np.zeros(arrivalTime_ignition_seed.shape,dtype=np.uint8) )
            tag_ignition_side = np.where( (runingNeighbourImage.runing_minmax_neighbor(tag_ignition_side_tmp,3,flag='min') == -1) & (mask_ignition==255) & (np.abs(tag_ignition_side)!=1), -1*np.ones_like(direction), tag_ignition_side) 
     
            arrivalTime_clean = np.where((arrivalTime_ignition_seed!=0) & (np.abs(tag_ignition_side)!= 1),  arrivalTime_ignition_seed, 
                                                                                                            arrivalTime_clean) #center
            arrivalTime_clean = np.where((arrivalTime_ignition_seed!=0) & (tag_ignition_side== 1) , arrivalTime_ignition_seed, 
                                                                                                    arrivalTime_clean) #front fire
            arrivalTime_clean = np.where((arrivalTime_ignition_seed!=0) & (tag_ignition_side==-1) , np.where(arrivalTime_ignition_seed<arrivalTime_clean, arrivalTime_ignition_seed, arrivalTime_clean),
                                                                                                    arrivalTime_clean) #back fire
            #arrivalTime_clean = np.where((arrivalTime_ignition_seed!=0) & (tag_ignition_side==-1) & (arrivalTime_ignition_seed<=arrivalTime_clean),arrivalTime_clean,         arrivalTime_clean) #back fire
   
        else:
            arrivalTime_clean = np.where((arrivalTime_ignition_seed>0), np.where(arrivalTime_ignition_seed<np.where(arrivalTime_fillup!=-999,arrivalTime_fillup,1000), arrivalTime_ignition_seed, arrivalTime_fillup), arrivalTime_clean) 

        plot_geotiff(arrivalTime_ignition_seed, 'arrivalTime_ignitionSeed',      maps_fire, crs, input_info['plot_name'], out_dir)
        plot_geotiff(tag_ignition_side,         'arrivalTime_ignition_seed_tag', maps_fire, crs, input_info['plot_name'], out_dir)
    
    '''ax = plt.subplot(111)
    im = plt.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
    #im = ax.imshow(np.ma.masked_where(arrivalTime_ignition_seed<=0,arrivalTime_ignition_seed-arrivalTime_clean).T,origin='lower',interpolation='nearest')
    im.format_cursor_data = lambda x : "{0:.2f}".format(x)
    plt.show()
    pdb.set_trace()'''

    plot_geotiff(arrivalTime_clean, 'arrivalTime_clean', maps_fire, crs, plot_name, out_dir)
   
    if False:
        ax = plt.subplot(131)
        im = ax.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
        im.format_cursor_data = lambda x : "{0:.2f}".format(x)
        ax = plt.subplot(132)
        im2 = ax.imshow(np.ma.masked_where(arrivalTime_edgePixel==1,arrivalTime_clean_copy).T,origin='lower',interpolation='nearest')
        ax = plt.subplot(133)
        im2 = ax.imshow(arrivalTime_tmp.T,origin='lower',interpolation='nearest')
        plt.show()
        pdb.set_trace()
   
    if flag_plot: 
        mpl.rcdefaults()
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 16.
        mpl.rcParams['legend.fontsize'] = 'small'
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['font.size'] = 14.
        mpl.rcParams['xtick.labelsize'] = 11.
        mpl.rcParams['ytick.labelsize'] = 11.
        mpl.rcParams['figure.subplot.left'] = .05
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.top'] = .9
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.hspace'] = 0.1
        mpl.rcParams['figure.subplot.wspace'] = 0.2

        idx=np.where(arrivalTime_fillup>0)
        correc_time = 1.
        if flag_large_scale: 
            correc_time = (3600*24)
        arrivalTime_fillup_plot       = arrivalTime_fillup/correc_time
        arrivalTime_clean_plot = arrivalTime_clean/correc_time
        daymin = (arrivalTime_fillup[idx]/correc_time).min()
        daymax = (arrivalTime_fillup[idx]/correc_time).max()
        print('  ', daymin, daymax)
        fig=plt.figure(figsize=(15,7))
        ax = plt.subplot(121)
        maps_fire_extent=(maps_fire.grid_e.min(),maps_fire.grid_e.max()+resolution,              \
                          maps_fire.grid_n.min(),maps_fire.grid_n.max()+resolution)  
        ax.imshow(np.ma.masked_where( (maps_fire.plotMask==2),maps_fire.plotMask).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,cmap=cm.Greys_r)
        im = ax.imshow(np.ma.masked_where( (maps_fire.plotMask!=2),arrivalTime_fillup_plot).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,vmin=daymin,vmax=daymax)
      
        ax = plt.subplot(122)
        ax.imshow(np.ma.masked_where( (maps_fire.plotMask==2),maps_fire.plotMask).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,cmap=cm.Greys_r)
        im = ax.imshow(np.ma.masked_where( (maps_fire.plotMask!=2)|(arrivalTime_clean_plot==0)|(arrivalTime_clean_plot<daymin)|(arrivalTime_clean_plot>daymax),arrivalTime_clean_plot).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,vmin=daymin,vmax=daymax)

        cbaxes = fig.add_axes([0.3, 0.97, 0.4, 0.02])
        cbar = fig.colorbar(im ,cax = cbaxes,orientation='horizontal')
        if flag_large_scale: 
            cbar.set_label(r'time (day)',labelpad=10)
        else:
            cbar.set_label(r'time (s)',labelpad=-10)
        #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')
        cbar.ax.tick_params(rotation=45)

        if frame_time_period is not None:
            fig.savefig(out_dir+'arrivalTime_3_clean_'+plot_name+'_dx={:04.1f}_dt={:05.1f}.png'.format(resolution,frame_time_period),dpi=200)
        else:
            fig.savefig(out_dir+'arrivalTime_3_clean_'+plot_name+'_dx={:04.1f}.png'.format(resolution),dpi=200)

        plt.close(fig)

    print('  --')
    print('  create arrivalTime_fire_spread')
    flag_run_fs = True if ('fs' in input_info['interpolate_betwee_ff']) else False

    if (flag_run_fs):  
        if not ( os.path.isfile(out_dir+ 'geotiff/'+plot_name+'arrivalTime_FS.tif') ) : 
            arrivalTime_FS=np.zeros_like(arrivalTime_clean) - 999.
            arrivalTime_FS_count=np.zeros_like(arrivalTime_clean) # nbre of fire line that pass by a point
            yy,xx = np.meshgrid(np.arange(arrivalTime_clean.shape[1]),np.arange(arrivalTime_clean.shape[0]))
            idx_mesh = np.dstack((xx.flatten(),yy.flatten()))[0]
            len_loop = time_available.shape[0]-2

            tresh = np.array(np.where(arrivalTime_clean<0,np.zeros_like(arrivalTime_clean),np.ones_like(arrivalTime_clean)),dtype=np.uint8)
            tresh_copy = np.copy(tresh)
            contours,      hierarchy      = cv2.findContours(tresh,     cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contoursCCOMP, hierarchyCCOMP = cv2.findContours(tresh_copy,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
          
            #iii = 0#644
            #jjj = 0#501
            #flag_stop = False
            argsStar = []
            nbre_contours = len(contours)
            mm = np.zeros_like(arrivalTime_clean)
            size_contour = []
           
            mask_all_fsrun = np.zeros_like(arrivalTime_clean)

            for ic in range(nbre_contours):
                if ('node' not in socket.gethostname()): 
                    print('  select contours: {:05.2f}% ...\r'.format(100.*ic/nbre_contours), end=' ') 
                    sys.stdout.flush()
                if hierarchy[0,ic,3] == -1: 
                #    if (flag_stop) & (mm[iii,jjj] > 0): pdb.set_trace()
                    print(ic)
                    continue                                    # outermost contour
                
                idx_contour = [contours[ic][:,0,1],contours[ic][:,0,0]]

                plt.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
                plt.scatter(idx_contour[0],idx_contour[1],c='w')
                plt.show()
                pdb.set_trace()

                ichilds = np.where(hierarchy[0,:,3] == ic)
                idx_childs = []
                for ichild in ichilds[0]:
                    idx_childs.append([contours[ichild][:,0,1],contours[ichild][:,0,0]])

                #mask inside
                polygon =[ tuple(pt) for pt in zip(contours[ic][:,0,1],contours[ic][:,0,0]) ]
                img = Image.new('L',arrivalTime_clean.shape , 0)
                ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
                mask_inside = np.where( np.array(img).T == 1, np.ones_like(arrivalTime_clean), np.zeros_like(arrivalTime_clean) )
               
                
                for idx_child in idx_childs:
                    polygon2 =[ tuple(pt) for pt in zip(idx_child[0], idx_child[1]) ]
                    img = Image.new('L',arrivalTime_clean.shape , 0)
                    ImageDraw.Draw(img).polygon(polygon2, outline=1, fill=1)
                    mask_child = np.where( np.array(img).T == 1, np.ones_like(arrivalTime_clean), np.zeros_like(arrivalTime_clean) )
                    mask_inside = np.where(mask_child==1, np.zeros_like(arrivalTime_clean), mask_inside)

                    '''
                    ax = plt.subplot(121)
                    ax.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
                    [ax.scatter(pt[0],pt[1],c='k') for pt in polygon]
                    [ax.scatter(pt[0],pt[1],c='w') for pt in polygon2]
                    ax = plt.subplot(122)
                    ax.imshow(mask_inside.T,origin='lower',interpolation='nearest')
                    '''
                idx_inside = np.where(mask_inside==1)

                if idx_inside[0].shape[0] <= 1:  # drop contour with less than 4 pixels inside
                    continue
                
                #plt.imshow(mask_inside.T,origin='lower',interpolation='nearest')
                #plt.scatter(idx_contour[0],idx_contour[1],c='w',s=10)
                #plt.show()
                #pdb.set_trace()

                #mask contour
                img = Image.new('L',arrivalTime_clean.shape , 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
                mask_contour = np.where( np.array(img).T == 1, np.ones_like(arrivalTime_clean), np.zeros_like(arrivalTime_clean) )
                
                for idx_child in idx_childs:
                    polygon3 =[ tuple(pt) for pt in zip(idx_child[0], idx_child[1]) ]
                    img = Image.new('L',arrivalTime_clean.shape , 0)
                    ImageDraw.Draw(img).polygon(polygon3, outline=1, fill=0)
                    mask_child = np.where( np.array(img).T == 1, np.ones_like(arrivalTime_clean), np.zeros_like(arrivalTime_clean) )
                    mask_contour = np.where(mask_child==1, mask_child, mask_contour)
                        
                    #plt.figure()    
                    #ax = plt.subplot(121)
                    #ax.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
                    #[ax.scatter(pt[0],pt[1],c='k') for pt in polygon]
                    #[ax.scatter(pt[0],pt[1],c='w') for pt in polygon3]
                    #ax = plt.subplot(122)
                    #ax.imshow(mask_contour.T,origin='lower',interpolation='nearest')
                    #plt.show() 
                
                mm[idx_inside] = ic


                #if len(idx_inside[0])<=1: # fire went no where from there (=0), or only one pixel that can be handle by rbf
                #    if (flag_stop) & (mm[iii,jjj] > 0): pdb.set_trace()
                #    continue

                #if arrivalTime_clean[idx_inside].max() > 0: # continue if the contour is around a cluster with arrival time known
                #    if (flag_stop) & (mm[iii,jjj] > 0): pdb.set_trace()
                #    continue

                #if np.unique(arrivalTime_clean[idx_contour]).shape[0] == 1: # continue if all pt 
                #    if (flag_stop) & (mm[iii,jjj] > 0): pdb.set_trace()
                #    continue                                                # are at the same time. 
                                                                            # we ll use rbf
               
                flag_igni_correction = False
                if tag_ignition_side[*idx_contour].max()>0:
                    mask_contour
                    #if the contour got some ignition point, we make sure that the last ignition point is younger than the older contour point
                    maxTime_clean = arrivalTime_clean[np.where( (np.abs(tag_ignition_side)!=1) & (mask_contour==1))].max()
                    maxTime_clean_ignition = arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))].max()
                    minTime_clean_ignition = arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))].min()
                    
                    # set last point of ignition 10 second before the older point on the contour
                    if (maxTime_clean_ignition > maxTime_clean - 10 ) & (maxTime_clean_ignition>minTime_clean_ignition):
                        flag_igni_correction = True
                        maxTime_clean_ignition_new = maxTime_clean - 10
                        #rescale ignition 
                        #print arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))]
                        for i,j in zip(*np.where((tag_ignition_side==1) & (mask_contour==1))):
                            arrivalTime_clean[i,j] = (maxTime_clean_ignition_new-minTime_clean_ignition)/(maxTime_clean_ignition-minTime_clean_ignition) \
                                                     * (arrivalTime_clean[i,j] - minTime_clean_ignition )                                                \
                                                                                                        + minTime_clean_ignition
                            if np.isnan(arrivalTime_clean[i,j]): pdb.set_trace()
                        #print arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))]
                    
                    maxTime_clean = arrivalTime_clean[np.where( (np.abs(tag_ignition_side)!=1) & (mask_contour==1))].max()
                    maxTime_clean_ignition = arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))].max()
                    
                    #set first ignition in the coutour to be egal to the closter point in the contour
                    maxTime_clean_ignition = arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))].max()
                    idx_contour_ignition = np.where((tag_ignition_side==1) & (mask_contour==1))
                    minTime_clean_ignition     = arrivalTime_clean[idx_contour_ignition].min()
                    minTime_clean_ignition_loc = [idx_contour_ignition[0][minTime_clean_ignition.argmin()], idx_contour_ignition[1][minTime_clean_ignition.argmin()]]
                    idx_contour_noignition = np.where( (np.abs(tag_ignition_side)!=1) & (mask_contour==1))
                    tree_neighbour    = spatial.cKDTree(list(zip(*idx_contour_noignition)))
                    d_, inds = tree_neighbour.query(minTime_clean_ignition_loc, k = 1)
                    if d_ > 3: pdb.set_trace()
                    minTime_clean_ignition_new = arrivalTime_clean[idx_contour_noignition[0][inds],idx_contour_noignition[1][inds]]
                    if (minTime_clean_ignition > minTime_clean_ignition_new) & (maxTime_clean_ignition>minTime_clean_ignition):
                        flag_igni_correction = True
                        #rescale ignition 
                        #print arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))]
                        for i,j in zip(*np.where((tag_ignition_side==1) & (mask_contour==1))):
                            #print arrivalTime_clean[i,j], 
                            arrivalTime_clean[i,j] = (maxTime_clean_ignition-minTime_clean_ignition_new)/(maxTime_clean_ignition-minTime_clean_ignition) \
                                                     * (arrivalTime_clean[i,j] - minTime_clean_ignition )                                                \
                                                                                                        + minTime_clean_ignition_new
                            if np.isnan(arrivalTime_clean[i,j]): pdb.set_trace()
                            #print arrivalTime_clean[i,j]

                    
                        #print arrivalTime_clean[np.where((tag_ignition_side==1) & (mask_contour==1))]
                        #print arrivalTime_clean[np.where((mask_contour==1))]
                        
                
                '''
                available_arrival_time = np.sort(arrivalTime_clean[contours[ic][:,0,1],contours[ic][:,0,0]])
                bin_time_size = input_info['nbre_img_overpass'] * scipy.stats.mode(np.diff(np.sort(np.unique(arrivalTime_clean))))[0][0]
                range_hist = (available_arrival_time.min(),available_arrival_time.max())
                nbrebins   = max([2,np.int(np.round((range_hist[1]-range_hist[0])/bin_time_size,0))])
                hist, time_scale = np.histogram(available_arrival_time,range=range_hist,bins=nbrebins )

                idx_extrema_hist = scipy.signal.argrelextrema(np.insert(hist,[0,len(hist)],[0,0]), np.greater)[0] - 1


                ratio_pts_before_after = 0.
                i_hist = -1
                while ratio_pts_before_after < .3:
                    i_hist += 1
                    time_mark = time_scale[i_hist+1]
                    idx_t   = np.where( (mask_contour == 1) & (arrivalTime_clean <= time_mark) )
                    idx_pdt = np.where( (mask_contour == 1) & (arrivalTime_clean >  time_mark) )
                    
                    try: 
                        ratio_pts_before_after = 1.*len(idx_t[0])/len(idx_pdt[0])
                    except: 
                        pdb.set_trace()


                    if ratio_pts_before_after > 1.3:
                        if i_hist == 0: break
                        while ratio_pts_before_after > 1.3:
                            i_hist -= 1
                            try: 
                                time_mark = time_scale[i_hist+1]
                            except: 
                                pdb.set_trace()
                            idx_t   = np.where( (mask_contour == 1) & (arrivalTime_clean <= time_mark) )
                            idx_pdt = np.where( (mask_contour == 1) & (arrivalTime_clean >  time_mark) )
                            ratio_pts_before_after = 1.*len(idx_t[0])/len(idx_pdt[0])
                            if i_hist == 0 : 
                                break
                        break
                    

                    if i_hist == len(time_scale)-3: break
                '''
                ####
                ###
                '''
                if scipy.signal.argrelextrema(pdf, np.greater)[0].shape[0] <= 1:
                    idx_t   = np.where( (mask_contour == 1) & (arrivalTime_clean == available_arrival_time.min()) )
                    idx_pdt = np.where( (mask_contour == 1) & (arrivalTime_clean >  available_arrival_time.min()) )

                else:
                    ii = scipy.signal.argrelextrema(pdf, np.greater)[0][0]
                    while pdf[ii+1] ==0: 
                        ii+=1
                        if ii == pdf.shape[0]: break

                    if ii == pdf.shape[0]: 
                        pdb.set_trace()
                        continue

                    idx_t   = np.where( (mask_contour == 1) & (arrivalTime_clean <= time_scale[ii]   ) )
                    idx_pdt = np.where( (mask_contour == 1) & (arrivalTime_clean >  time_scale[ii]   ) )

                    if idx_pdt[0].shape[0] ==0: continue
                    
                    #plt.imshow(arrivalTime_clean.T,origin='lower',interpolation='nearest')
                    #plt.scatter(contours[ic][:,0,1],contours[ic][:,0,0],c='k')
                    #plt.scatter(idx_inside[1],idx_inside[0],c='w')
                    #plt.show()
                '''

                #if False: #ic == 114: #mm[260,280] > 0:  # L2G
                #if (flag_stop) & (mm[iii,jjj] > 0):  # L2G
                if ic == -999 :#49 #674 : #167: #3136: #4701 : #3136 : #1388:#  3607 :#1860: #3607: 

                    #plt.imshow(mm.T,origin='lower')
                    #iidx = np.where(mask_contour==1)
                    #plt.scatter(iidx[0],iidx[1],c='k')
                    #plt.scatter(iii,jjj,c='w')
                    #plt.show()
                    #pdb.set_trace()
                    #if flag_igni_correction:
                    #    print 'apply ignition correction'
                    arg = [idx_contour, mask_inside, arrivalTime_fillup, arrivalTime_clean, idx_mesh, resolution, time_reso, ros_max, True, ic]

                    interpolate_fire_spread_between_ff_star(arg)
                    
                argsStar.append( [idx_contour, mask_inside, arrivalTime_fillup, arrivalTime_clean, idx_mesh, resolution, time_reso, ros_max, False, ic])
                size_contour.append(np.where(mask_inside==1)[0].shape[0]) 
                
                mask_all_fsrun[np.where(mask_inside==1)] = ic

            plt.imshow(np.ma.masked_where(mask_all_fsrun==0,mask_all_fsrun).T, origin='lower')
            plt.show()
            pdb.set_trace()
            nx,ny =  arrivalTime_fillup.shape
            #draw contour
            '''
            ax = plt.subplot(111)
            for argsStar_ in argsStar:
                idx_contour_ = argsStar_[0]
                if min(idx_contour_[0]).min()>nx/2: 
                    print(argsStar_[-1])
                    for x,y in zip(idx_contour_[0],idx_contour_[1]):
                        ax.scatter(x,y, s=2, c='k')
            plt.show()
            '''
            print('  select contours: done; {:d} contours selected'.format(len(argsStar))) 

            #plt.imshow(np.ma.masked_where(mm<=0,mm).T,origin='lower')
            #plt.show()
            #pdb.set_trace()
           
            idx_sort_size_contour = np.argsort(np.array(size_contour))
            '''
            #old selection only base on consecutive time
            for itime, (time,timepdt) in enumerate(zip(time_available[:-1],time_available[1:])):
                
                #select empty cluster, keep the one with only 2 times on the contour and run the FS interpolation 
                
                idx_pdt = np.where( (arrivalTime_clean==timepdt) ) 
                idx_t   = np.where( (arrivalTime_clean==time   ) )
                args.append( [idx_t, idx_pdt, arrivalTime_fillup, arrivalTime_clean, idx_mesh, resolution, ros_max])
            '''

            len_loop = len(argsStar)
            
            
            if len_loop == 0: 
                print('  no interpolation on linear FS') 
                arrivalTime_FS       = np.where(arrivalTime_clean>=0,arrivalTime_clean,-999)
                arrivalTime_FS_count = np.where(arrivalTime_clean>=0,1,0)
           
            else:   
                if flag_parallel:
                    print('  interpolate linear FS: ...', end=' ') 
                    # set up a pool to run the parallel processing
                    cpus= map_tools.cpu_count() 
                    pool = multiprocessing.Pool(processes=cpus)

                    # then the map method of pool actually does the parallelisation  
                    results = pool.map(interpolate_fire_spread_between_ff_star, argsStar) 
                    pool.close()
                    pool.join()
                    
                    for itime, (arrivalTime_FS_new, arrivalTime_FS_new_count) in enumerate(results):
                        arrivalTime_FS = np.where(arrivalTime_FS_new!=0, arrivalTime_FS_new ,arrivalTime_FS)
                        arrivalTime_FS_count += np.where(argsStar[itime][1]==1, arrivalTime_FS_new_count, np.zeros_like(arrivalTime_FS_new_count))
                        
                else:
                    for itime, arg in enumerate( [argsStar[ii] for ii in idx_sort_size_contour[::-1]] ):

                        if ('node' not in socket.gethostname()): 
                            print('  interpolate linear FS: {:05.2f}% ...\r'.format(100.*itime/len_loop), end=' ') 
                            sys.stdout.flush()
                        arrivalTime_FS_new, arrivalTime_FS_new_count = interpolate_fire_spread_between_ff_star(arg)
                        arrivalTime_FS = np.where(arrivalTime_FS_new!=0, arrivalTime_FS_new ,arrivalTime_FS)
                        arrivalTime_FS_count += np.where(arg[2]==1, arrivalTime_FS_new_count, np.zeros_like(arrivalTime_FS_new_count))
                        
                        
                        '''
                        if itime == 19:
                            correc_time = 1.
                            if flag_large_scale: 
                                correc_time = (3600*24)  
                            fig = plt.figure(figsize=(10,8))
                            ax = plt.subplot(121)
                            ax.imshow(np.ma.masked_where( (arrivalTime_clean<arg[0]) | (arrivalTime_clean>arg[1]),arrivalTime_clean/correc_time).T,origin='lower',interpolation='nearest',vmin=arg[0]/correc_time,vmax=arg[1]/correc_time)
                            ax = plt.subplot(122)
                            ax.imshow(np.ma.masked_where(arrivalTime_FS<=0,arrivalTime_FS/correc_time).T,origin='lower',interpolation='nearest',vmin=arg[0]/correc_time,vmax=arg[1]/correc_time)
                            plt.show()
                            pdb.set_trace() 
                        '''
                if not(flag_parallel):
                    print('  interpolate linear FS: {:05.2f}% done'.format(100.*itime/len_loop))
                else: 
                    print('  interpolate linear FS: done') 

            #arrivalTime_FS_count = np.where(arrivalTime_clean>0, np.zeros_like(arrivalTime_FS_count)-999, arrivalTime_FS_count)
            #plt.imshow(np.ma.masked_where(arrivalTime_FS_count<0,arrivalTime_FS_count).T , origin='lower', interpolation='nearest' )
            #plt.colorbar()
            #plt.show()
            #pdb.set_trace()

            #remove point from FS interpolation where we only have two points
            #arrivalTime_FS = np.where((arrivalTime_FS_count<=2)&(arrivalTime_FS_count>0), np.zeros_like(arrivalTime_FS)-999, arrivalTime_FS)


            if flag_plot: 
                mpl.rcdefaults()
                mpl.rcParams['text.usetex'] = False
                mpl.rcParams['font.family'] = 'DejaVu Sans'
                mpl.rcParams['axes.linewidth'] = 1
                mpl.rcParams['axes.labelsize'] = 16.
                mpl.rcParams['legend.fontsize'] = 'small'
                mpl.rcParams['legend.fancybox'] = True
                mpl.rcParams['font.size'] = 14.
                mpl.rcParams['xtick.labelsize'] = 11.
                mpl.rcParams['ytick.labelsize'] = 11.
                mpl.rcParams['figure.subplot.left'] = .05
                mpl.rcParams['figure.subplot.right'] = .95
                mpl.rcParams['figure.subplot.top'] = .9
                mpl.rcParams['figure.subplot.bottom'] = .1
                mpl.rcParams['figure.subplot.hspace'] = 0.1
                mpl.rcParams['figure.subplot.wspace'] = 0.2

                correc_time = 1.
                if flag_large_scale: 
                    correc_time = (3600*24)
                arrivalTime_FS_plot = arrivalTime_FS/correc_time
                arrivalTime_clean_plot = arrivalTime_clean/correc_time
                idx=np.where(arrivalTime_fillup>0)
                daymin = (arrivalTime_fillup[idx]/correc_time).min()
                daymax = (arrivalTime_fillup[idx]/correc_time).max()
                print('  ', daymin, daymax)
                maps_fire_extent=(maps_fire.grid_e.min(),maps_fire.grid_e.max()+resolution,              \
                                  maps_fire.grid_n.min(),maps_fire.grid_n.max()+resolution)  
                
                fig=plt.figure(figsize=(15,7))
                
                ax = plt.subplot(121)
                ax.imshow(np.ma.masked_where( (maps_fire.plotMask==2),maps_fire.plotMask).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,cmap=cm.Greys_r)
                im = ax.imshow(np.ma.masked_where( (maps_fire.plotMask!=2)|(arrivalTime_clean_plot<=0)|(arrivalTime_clean_plot<daymin)|(arrivalTime_clean_plot>daymax),arrivalTime_clean_plot).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,vmin=daymin,vmax=daymax)
                
                ax = plt.subplot(122)
                ax.imshow(np.ma.masked_where( (maps_fire.plotMask==2),maps_fire.plotMask).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,cmap=cm.Greys_r)
                im = ax.imshow(np.ma.masked_where( (maps_fire.plotMask!=2)|(arrivalTime_FS_plot<=0)|(arrivalTime_FS_plot<daymin)|(arrivalTime_FS_plot>daymax),arrivalTime_FS_plot).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,vmin=daymin,vmax=daymax)
              

                cbaxes = fig.add_axes([0.3, 0.97, 0.4, 0.02])
                cbar = fig.colorbar(im ,cax = cbaxes,orientation='horizontal')
                if flag_large_scale: 
                    cbar.set_label(r'time (day)',labelpad=10)
                else:
                    cbar.set_label(r'time (s)',labelpad=10)  
                #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')
                cbar.ax.tick_params(rotation=45)

                if frame_time_period is not None:
                    fig.savefig(out_dir+'arrivalTime_4_FS_interpolation_'+plot_name+'_dx={:04.1f}_dt={:05.1f}.png'.format(resolution,frame_time_period),dpi=200)
                else:
                    fig.savefig(out_dir+'arrivalTime_4_FS_interpolation_'+plot_name+'_dx={:04.1f}.png'.format(resolution),dpi=200)

                plt.close(fig)

            
            plot_geotiff(arrivalTime_FS,       'arrivalTime_FS',       maps_fire, crs, plot_name, out_dir)
            plot_geotiff(arrivalTime_FS_count, 'arrivalTime_FS_count', maps_fire, crs, plot_name, out_dir)

        else: 
            arrivalTime_FS       = load_geotiff( out_dir,plot_name,'arrivalTime_FS')
            arrivalTime_FS_count = load_geotiff( out_dir,plot_name,'arrivalTime_FS_count')

    else: # no FS
        arrivalTime_FS = arrivalTime_clean
        arrivalTime_FS_count = np.zeros_like(arrivalTime_FS)

    print('  --')
    print('  final interpolation of arrival Time')
    if not ( os.path.isfile(out_dir+ 'geotiff/'+plot_name+'arrivalTime_interp.tif') ) : 
        plotMask = maps_fire.plotMask
        xi, yi = maps_fire.grid_e, maps_fire.grid_n
        xi = xi + .5*resolution 
        yi = yi + .5*resolution

        zi = np.copy(arrivalTime_FS) # np.zeros_like(xi)

        #plt.imshow(zi.T,origin='lower',interpolation='nearest')
        #plt.show()
        #pdb.set_trace()

        mask_nodata = np.where(arrivalTime_FS<0, np.ones_like(arrivalTime_FS), np.zeros_like(arrivalTime_FS))
        s = [[1,1,1], \
             [1,1,1], \
             [1,1,1]]
        mask_nodata_labled, num_cluster = ndimage.label(mask_nodata, structure=s )
        dimension_cluster_nodata = np.zeros(num_cluster)
        for i_cluster in range(num_cluster):
            idx_ = np.where(mask_nodata_labled==i_cluster+1) 
            pts = np.array(np.dstack( [idx_[0],idx_[1]] )[0],dtype=float)
            (center_x, center_y), (width, height), angle  = cv2.minAreaRect(np.array(pts,dtype=np.float32))
            dimension_cluster_nodata[i_cluster] = min([width,height])

        if 'rbf' in  input_info['interpolate_betwee_ff']:
            flag_interpolate_betwee_ff = 'rbf'
        elif 'griddata' in input_info['interpolate_betwee_ff']: 
            flag_interpolate_betwee_ff = 'griddata'
        else:
            print('interpolation between ff not understood in input parameter :', input_info['interpolate_betwee_ff']) 
            pdb.set_trace()

        #interpolate the data with the matlab griddata function
        # try first griddata if fit the request, if point are missing (most probably value on the bndf), then we fall back to rbf 
        if flag_interpolate_betwee_ff == 'griddata':
            idx = np.where(arrivalTime_FS>=0)
            x = xi[idx]
            y = yi[idx]
            z = arrivalTime_FS[idx]
            coord_pts = np.vstack((x, y)).T
            data      = z.flatten()
            fill_val  = -999
            #method    = 'cubic'
            #zi = interpolate.griddata(coord_pts , data, (xi,yi), fill_value=fill_val, method=method)
            
            interp = interpolate.LinearNDInterpolator(coord_pts, data, fill_value=fill_val, rescale=True)
            zi = interp(xi, yi)
            
            #plt.imshow(zi.T,origin='lower',interpolation='nearest',vmin=0,vmax=1000)
            #plt.show()

            if np.where( (zi<=0) &  (plotMask==2) )[0].shape[0] != 0:  
                flag_interpolate_betwee_ff = 'rbf' 

        print('interpolation is :',  flag_interpolate_betwee_ff)
        
        if (flag_interpolate_betwee_ff == 'rbf') : 
            time_now = datetime.datetime.now()
            
            if set_rbf_subset is not None:
                subset = set_rbf_subset

            else:
                
                distance_interaction = input_info['distance_interaction_rbf']

                possible_subset = []
                list1 = factor.get_factor(plotMask.shape[0])
                list2 = factor.get_factor(plotMask.shape[1])
                for ii in list1 :
                    for jj in list2 :
                    
                        nrows = plotMask.shape[0]/ii
                        ncols = plotMask.shape[1]/jj

                        mem= psutil.virtual_memory()
                        memavailable = mem.available
                        if socket.gethostname() == 'ibo': memavailable *= .5
                        if (2*((3*nrows)*(3*ncols))**2*8 < memavailable)     &\
                           (nrows>distance_interaction/resolution)            &\
                           (ncols>distance_interaction/resolution)              : 
                               possible_subset.append([ii,jj])
               
                if len(possible_subset) == 0:
                    print('cannot call rbf interpolation')
                    print('no set up found for available mem and distance =', distance_interaction)
                    print('mem   =',  mem)
                    print('list1 =', list1)
                    print('list2 =', list2)
                    pdb.set_trace()
                    sys.exit()

                print('  possible split of the domain for rbf call')
                print(possible_subset)
                print('  select')
                if len(possible_subset)==1:
                    id_select = 0
                else:
                    diff = []; mean = []
                    for id_ in range(len(possible_subset)):
                        diff.append(abs(possible_subset[id_][0]-possible_subset[id_][1]))
                        mean.append(.5*(possible_subset[id_][0]+possible_subset[id_][1]))
                    diff = np.array(diff);mean = np.array(mean)
                    id_select_1 = np.where(diff <= 1.1*diff.min())
                    id_select_2  = mean[id_select_1].argmin()
                    id_select = id_select_1[0][id_select_2]

                #possible_subset[id_select] = [50,50]
                subset = possible_subset[id_select]
                print(subset)

            nrows = plotMask.shape[0]//subset[0]
            ncols = plotMask.shape[1]//subset[1]
            print('  sub domain = {:3.1f} x {:3.1f}'.format(nrows*resolution,ncols*resolution)) 
            nx,ny = xi.shape
            ii_sub = blockshaped(np.arange(xi.size).reshape(xi.shape), nrows, ncols) 

            #count number of point in each zone to sort the order to run interpolation
            density_point = np.zeros(ii_sub.shape[0])
            for i_sub in range(ii_sub.shape[0]):
                i_min_zz, j_min_zz = np.unravel_index(ii_sub[i_sub].min(),xi.shape) # zone we want
                i_max_zz, j_max_zz = np.unravel_index(ii_sub[i_sub].max(),xi.shape)
                
                i_min = i_min_zz-nrows # zone we want + extra point around
                i_max = i_max_zz+nrows
                j_min = j_min_zz-ncols
                j_max = j_max_zz+ncols
                
                #print max([i_min,0]),min([i_max,nx]),max([j_min,0]),min([j_max,ny])
                xi_tmp = xi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
                yi_tmp = yi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
                

                flag_zz = np.zeros_like(xi)
                flag_zz[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = 1
                flag_zz_tmp = flag_zz[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]

                arrivalTime_FS_tmp = arrivalTime_FS[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
                plotMask_tmp = plotMask[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            
                idx = np.where(arrivalTime_FS_tmp>0)
                
                if (len(idx[0]) == 0) :
                    density_point[i_sub] = -999
                else: 
                    density_point[i_sub] = 1.*len(idx[0])/(arrivalTime_FS_tmp.shape[0]*arrivalTime_FS_tmp.shape[1])
               

            for i_loop, i_sub in enumerate(np.argsort(density_point)[::-1]):

                if i_sub < 0:
                    continue

                i_min_zz, j_min_zz = np.unravel_index(ii_sub[i_sub].min(),xi.shape) # zone we want
                i_max_zz, j_max_zz = np.unravel_index(ii_sub[i_sub].max(),xi.shape)
                
                i_min = i_min_zz-nrows # zone we want + extra point around
                i_max = i_max_zz+nrows
                j_min = j_min_zz-ncols
                j_max = j_max_zz+ncols
                
                #print max([i_min,0]),min([i_max,nx]),max([j_min,0]),min([j_max,ny])
                xi_tmp = xi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
                yi_tmp = yi[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
                

                flag_zz = np.zeros_like(xi)
                flag_zz[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = 1
                flag_zz_tmp = flag_zz[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]

                arrivalTime_FS_tmp = arrivalTime_FS[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
                plotMask_tmp = plotMask[max([i_min,0]):min([i_max,nx-1])+1,max([j_min,0]):min([j_max,ny-1])+1]
            
                idx = np.where(arrivalTime_FS_tmp>0)
                if (len(idx[0]) == 0) :
                    zi[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = -999.
                    continue
                
                idx_pt_to_interp = np.where((plotMask_tmp==2) & (arrivalTime_FS_tmp<0 ))
                if len(idx_pt_to_interp[0]) == 0 : # if we only have 1 point with data we continue
                    continue

                x = xi_tmp[idx]
                y = yi_tmp[idx]
                z = arrivalTime_FS_tmp[idx]
                z_init = np.copy(arrivalTime_FS_tmp)

                try: 
                    interp = interpolate.Rbf(x, y, z, function='multiquadric',epsilon=resolution)
                except : 
                    pdb.set_trace()
                #print 'rbf interp 1 elapse time (h):', (datetime.datetime.now()-time_now).total_seconds() / 3600
                sys.stdout.flush()
                zi_tmp = np.array([interp(xi_tmp.flatten()[i], yi_tmp.flatten()[i]) for i in range(xi_tmp.flatten().shape[0])]).reshape(xi_tmp.shape)
                print('  {:3d}% | elapse time (min): {:4.1f} |  point density: {:3.2} | mem availaible = {:3.1f}'.format(int(100.*i_loop/len(np.where(density_point>=0)[0])), (datetime.datetime.now()-time_now).total_seconds()/ 60, density_point[i_sub], 100.*psutil.virtual_memory().available/psutil.virtual_memory().total ))

                zi_tmp2 = np.copy(zi_tmp)

                idx = np.where(plotMask_tmp!=2)
                zi_tmp[idx] = -999
      
                idx = np.where(flag_zz_tmp==1)
                zi[max([i_min_zz,0]):min([i_max_zz,nx-1])+1,max([j_min_zz,0]):min([j_max_zz,ny-1])+1] = zi_tmp[idx].reshape((nrows,ncols))
               
                '''
                fig = plt.figure(figsize=(15,8))
                ax=plt.subplot(141)
                plt.imshow(np.ma.masked_where(zi<=0,zi).T,origin='lower',interpolation='nearest')
                ax=plt.subplot(142)
                plt.imshow(np.ma.masked_where(z_init<=0,z_init).T,origin='lower',interpolation='nearest')
                ax=plt.subplot(143)
                plt.imshow(np.ma.masked_where(zi_tmp<=0,zi_tmp).T,origin='lower',interpolation='nearest')
                ax=plt.subplot(144)
                plt.imshow(np.ma.masked_where(zi_tmp2<=0,zi_tmp2).T,origin='lower',interpolation='nearest')
                plt.show()
                '''
                zi_tmp = None
                interp = None
                arrivalTime_FS_tmp = None
                plotMask_tmp = None
                gc.collect() # collect memory carbage

                #plt.imshow(np.ma.masked_where(zi<0,zi).T,origin='lower',interpolation='nearest')
                #plt.show()
                #pdb.set_trace()
                

        idx = np.where(plotMask!=2)
        zi[idx]=-999
        
        if flag_plot: 
            mpl.rcdefaults()
            mpl.rcParams['text.usetex'] = False
            mpl.rcParams['font.family'] = 'DejaVu Sans'
            mpl.rcParams['axes.linewidth'] = 1
            mpl.rcParams['axes.labelsize'] = 16.
            mpl.rcParams['legend.fontsize'] = 'small'
            mpl.rcParams['legend.fancybox'] = True
            mpl.rcParams['font.size'] = 14.
            mpl.rcParams['xtick.labelsize'] = 11.
            mpl.rcParams['ytick.labelsize'] = 11.
            mpl.rcParams['figure.subplot.left'] = .05
            mpl.rcParams['figure.subplot.right'] = .95
            mpl.rcParams['figure.subplot.top'] = .9
            mpl.rcParams['figure.subplot.bottom'] = .1
            mpl.rcParams['figure.subplot.hspace'] = 0.1
            mpl.rcParams['figure.subplot.wspace'] = 0.2

            correc_time = 1.
            if flag_large_scale: 
                correc_time = (3600*24)
            arrivalTime_FS_plot = arrivalTime_FS/correc_time
            zi_plot       = zi/correc_time
            idx=np.where(zi_plot>0)
            daymin = zi_plot[idx].min()
            daymax = zi_plot[idx].max()
            print('  ' ,daymin, daymax)
            fig=plt.figure(figsize=(15,7))
            ax = plt.subplot(121)
            maps_fire_extent=(maps_fire.grid_e.min(),maps_fire.grid_e.max()+resolution,              \
                              maps_fire.grid_n.min(),maps_fire.grid_n.max()+resolution)  
            ax.imshow(np.ma.masked_where( (maps_fire.plotMask==2),maps_fire.plotMask).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,cmap=cm.Greys_r)
            im = ax.imshow(np.ma.masked_where( (maps_fire.plotMask!=2)|(arrivalTime_FS_plot==0)|(arrivalTime_FS_plot<daymin)|(arrivalTime_FS_plot>daymax),arrivalTime_FS_plot).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,vmin=daymin,vmax=daymax)
          
            ax = plt.subplot(122)
            ax.imshow(np.ma.masked_where( (maps_fire.plotMask==2),maps_fire.plotMask).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,cmap=cm.Greys_r)
            im = ax.imshow(np.ma.masked_where( (maps_fire.plotMask!=2)|(zi_plot<daymin)|(zi_plot>daymax),zi_plot).T,origin='lower',interpolation='nearest',extent=maps_fire_extent,vmin=daymin,vmax=daymax)

            cbaxes = fig.add_axes([0.3, 0.97, 0.4, 0.02])
            cbar = fig.colorbar(im ,cax = cbaxes,orientation='horizontal')
            if flag_large_scale: 
                cbar.set_label(r'time (day)',labelpad=10)
            else:
                cbar.set_label(r'time (s)',labelpad=10)  
            #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='45')
            cbar.ax.tick_params(rotation=45)

            if frame_time_period is not None:
                fig.savefig(out_dir+'arrivalTime_5_rbf_interpolation_'+plot_name+'_dx={:04.1f}_dt={:05.1f}.png'.format(resolution,frame_time_period),dpi=200)
            else:
                fig.savefig(out_dir+'arrivalTime_5_rbf_interpolation_'+plot_name+'_dx={:04.1f}.png'.format(resolution),dpi=200)

            plt.close(fig)

        if flag_large_scale: 
            #add single pixel
            idx = np.where(mask_single_pixel==1)
            zi[idx] = arrivalTime_fillup_in[idx]

            #large_scale_zoomFactor
            large_scale_zoomFactor = 1.*zi.shape[0]/large_scale_shape_raw[0]
            plotMask = large_scale_grid_raw.plotMask
            xi       = large_scale_grid_raw.grid_e
            yi       = large_scale_grid_raw.grid_n
            if large_scale_zoomFactor > 1: 
                zi       = map_tools.downgrade_resolution(zi,       large_scale_shape_raw , flag_interpolation='average')
            elif large_scale_zoomFactor < 1:
                zi = ndimage.zoom(zi, (int(1./large_scale_zoomFactor),int(1./large_scale_zoomFactor)) , order=0)
            else:
                zi = zi
    
        plot_geotiff(zi,       'arrivalTime_interp',       maps_fire, crs, plot_name, out_dir)
    
    else: 
        zi       = load_geotiff( out_dir,plot_name,'arrivalTime_interp')
    
    return zi, arrivalTime_FS, arrivalTime_clean


