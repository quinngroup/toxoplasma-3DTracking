import numpy as np
import math
import cv2
import glob
from scipy import ndimage
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import scipy.misc
from skimage import exposure
import os
import time 
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.measurements import label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def read_images(folders_address, file_extension):

	'''
	This Function reads all the images and put them into a 4D numpy array , first dimension indicates the frame numbers
	, the second one shows the slices(z axis) and the 3rd and 4th are the x and y axises. 
	To use this function , you need to put the images of each frame inside a seperate folder. 
	
	Parameters
    ----------
    folders_address: string
		root address of image folders    
	file_extension : string
        it shows the type of the images that should be imported ( e.g.: '*.tiff')

    Returns
    -------
    all_images_nparray : array, shape (frames, z, x, y)
        all the images are arranged in a 4D numpy array

	'''

	directory_list = list()
	for root, dirs, files in os.walk(folders_address , topdown=False):
	    for name in dirs:
	        if not (name.startswith('.')):
	            directory_list.append(os.path.join(root, name))

	sorted_path = sorted(directory_list)

	all_images = list ()
	extension = file_extension
	for i in range(len(sorted_path)):
	    all_images.append([cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(os.path.join(sorted_path[i],extension)))]) 
	all_images_nparray = np.asarray(all_images)
	return all_images_nparray


def thresholding(all_images):

	BW = np.array([i if i > 42 else 0 for i in range(0,256)]).astype(np.uint8)
	total = np.asarray(all_images)

	# Thresholding the total cells :
	all_img_thr = [[]]
	for i in range(total.shape[0]):
	    for j in range(total.shape[1]):
	        all_img_thr[i].append(cv2.LUT(total[i][j],BW))
	    all_img_thr.append([])

	all_img_thr = [pick for pick in all_img_thr if len(pick) > 0 ]

	return(all_img_thr)


def ccl_3d (all_image_arr):

	'''
	First we extract the labels of the componets for all the cells
	across all the frames. Thus the number of components and their labels
	are discovered here:
	'''

	all_image_arr = np.asarray(all_image_arr)

	structure = np.ones((3,3,3), dtype = np.int)
	all_labeled = np.zeros(shape=(63, 41, 500, 502))
	all_ncomponents = np.zeros(63)

	for frames in range (all_image_arr.shape[0]): 
	    all_labeled[frames], all_ncomponents[frames] = label(all_image_arr[frames], structure)

	return all_labeled, all_ncomponents

def noise_removal(all_img_arr, all_labeled):
	'''
	For Removing the noise,, i've considered only the components with 
	the volume greater tha 1 pixel. Thus first I computed the volume of 
	each component and then extracted the centers only for the components 
	with the volume greater than 1 pixels:
	'''
	all_img_arr = np.asarray(all_img_arr)

	unique = list()
	counts = list()
	for frames in range(all_img_arr.shape[0]):
	    unique.append(np.unique(all_labeled[frames], return_counts = True)[0])
	    counts.append(np.unique(all_labeled[frames], return_counts = True)[1])

	# Here I'm selecting only the center of the components with the volume 
	# less than 1 pixel and put them in thr_idxs list  :

	thr_idxs = [[]]
	for i in range (len(counts)):
	    for j in range (1, len (counts[i])):
	        if counts[i][j] > 1 : 
	            thr_idxs[i].append(unique[i][j])
	    thr_idxs.append([])
	thr_idxs = [pick for pick in thr_idxs if len(pick) > 0 ]
	
	return thr_idxs		

def center_detection(all_img_arr, all_labeled, thr_idxs):
	
	all_img_arr = np.asarray(all_img_arr)

	all_centers_noisefree = []
	for frames in range (all_img_arr.shape[0]):
		# print(frames)
		# print(thr_idxs[frames])
		all_centers_noisefree.append(center_of_mass(all_img_arr[frames], labels=all_labeled[frames], index= thr_idxs[frames]))
	
	return all_centers_noisefree



def tracker(all_centers):

	''' Now let's initialize the first elements of each object with the
	first centers in our frame 0, then we should append the points to these elements
	as a result, at the begining our number of objects will be equal to 
	number of centers in first frame.
	'''
	all_cen = all_centers #[ [x[0] for x in frame ] for frame in all_centers]
	new_objects = [ [(0,x)] for x in all_centers[0] ]

	# we need to set a threshold for adding only the closest points
	# within a threshold to our object list

	t_limit = 20

	# Now, we need to iterate on the frames and running our points matching module
	for i in range (1, len(all_cen)-1):
	    
	    '''in every step we need to check the points in current frame with 
	    last selected points in our object list
	    '''
	    current_frame = all_cen[i]
	    last_known_centers = [ obj[-1][1] for obj in new_objects if len(obj)>0 ] 
	    
	    # We are going to use Hungarian algorithm which is built in scipy
	    # As linear_sum_assignment. we need to pass a cost to that function
	    # the function will assign the points based on minimum cost. Here we 
	    # define the distance between the above mentioned points as our cost 
	    # function 
	    cost = distance.cdist(last_known_centers, current_frame,'euclidean')
	    # in this function row_ind will act as object_ids and the col_ind
	    # will play the role of new_centers_ind for us so we have : 
	    obj_ids, new_centers_ind = linear_sum_assignment(cost)
	    
	    all_center_inds = set(range(len(current_frame)))
	    # now we should iterate on obj_id and new_center_ind 
	    # checking the min acceptable distance , appending the points to 
	    # our frames and finally removing those points from our set.
	    
	    for  obj_id, new_center_ind  in zip(obj_ids,new_centers_ind):
	        if( distance.euclidean(np.array(current_frame[new_center_ind]),np.array(new_objects[obj_id][-1][1]) ) <= t_limit):
	            all_center_inds.remove(new_center_ind)
	            new_objects[obj_id].append((i,current_frame[new_center_ind]))
	    # at the end if the points are not matched with the previous objects 
	    # we will consider them as new objects and appending them to the end 
	    # of our object list.

	    for new_center_ind in all_center_inds:
	        new_objects.append([ (i,current_frame[new_center_ind])])
	xx = [[]]
	yy = [[]]
	zz = [[]]
	for i in range (len(new_objects)):
	    for j in range (len(new_objects[i])):
	        zz[i].append(new_objects[i][j][1][0])
	        xx[i].append(new_objects[i][j][1][1])
	        yy[i].append(new_objects[i][j][1][2])
	    xx.append([])
	    zz.append([])
	    yy.append([])

	zz = [pick for pick in zz if len(pick) > 0 ]
	xx = [pick for pick in xx if len(pick) > 0 ]
	yy = [pick for pick in yy if len(pick) > 0 ]

	xx = np.asarray(xx)
	yy = np.asarray(yy)  
	zz = np.asarray(zz)


	return (xx, yy, zz)


def visualization_3d_detection(all_cnf, image_width, image_height):

	znf3d = [[]]
	xnf3d = [[]]
	ynf3d = [[]]

	for frames in range(all_cnf.shape[0]):
	    for i in range (np.shape(all_cnf[frames])[0]):
	        znf3d[frames].append(all_cnf[frames][i][0])
	        xnf3d[frames].append(all_cnf[frames][i][1])
	        ynf3d[frames].append(all_cnf[frames][i][2])
	    znf3d.append([])
	    xnf3d.append([])
	    ynf3d.append([])

	# ok! Now let us visualize the final detection results:
	fig = plt.figure(figsize=(image_width, image_height))
	ax = fig.add_subplot(111, projection='3d')
	for i in range(all_cnf.shape[0]):
	    ax.scatter(ynf3d[i], xnf3d[i], znf3d[i], 
	               zdir='znf3d[i]', marker = "o",  c= znf3d[i], cmap='gist_heat')

	ax.w_xaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
	ax.w_yaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
	ax.w_zaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
	ax.view_init(35, 45)
	plt.grid(False)
	plt.savefig("demo2center-allnf.png", dpi=200)
	plt.show()

def Trajectory_3D_TimeVarying(frame_num, single_flag, point_num, s, x, y, z, number_of_points, video_file):
    '''*********************************************************** 
    This Function  will  plot a 3D  representation of the motility
    x, y and z are the  axis  values  which are defined inside the 
    main function.  single_flag is a  flag which indicates we want
    1 trajectory plotting or all of  them ?  if True means we need
    just one trajectory. The point_num indicates the number of the
    point which we are going to plot. written by MS.Fazli
    **************************************************************
    '''
    n = frame_num
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(35, 45)



    if single_flag :
        traj_points = 1
        T = np.linspace(0,1,np.size(x[point_num]))

    else : 
        traj_points = number_of_points
        T = np.linspace(0,1,np.size(x[0]))

    for i in range(traj_points):
        for j in range(0, n-s, s):
            if single_flag :             
                ax.plot(yy[point_num][j:j+s+1],  xx[point_num][j:j+s+1] ,zz[point_num][j:j+s+1], zdir='zz[i]', linewidth =5, color = ( 0.0, 0.9*T[j], 0.0))
#                 plt.pause(0.06)
            else : 
                ax.plot(yy[i][j:j+s+1],  xx[i][j:j+s+1] ,zz[i][j:j+s+1], zdir='zz[i]', linewidth =3, color = (T[j], 0.0, 0.0))
#                 plt.pause(0.06)

    #for angle in range(0, 360):
        # ax.view_init(5, i)
        # plt.pause(0.01)
        # plt.draw()
    ax.w_xaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
    ax.view_init(35, 45)

    plt.grid(False)
    
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    plt.savefig(str(video_file) + '4d_timeVariying_.jpg', dpi=600)

    plt.show()

def simple_visualization_tracked_points(xx, yy, zz, traj_length, image_width, image_height, savefig_quality, savefig_name):


	fig = plt.figure(figsize=(image_width,image_height))
	ax = fig.add_subplot(111, projection='3d')
	for i in range(traj_length):
	    ax.plot(yy[i], xx[i], zz[i], 
	               zdir='zz[i]', linewidth = 3)

	ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.view_init(35, 45)
	plt.grid(False)
	plt.savefig(savefig_name, dpi = savefig_quality)
	plt.show()


def main():
	
	folders = '/Users/mojtaba/Desktop/Toxoplasma/3d_video/'
	extension =  '*.tif'
	denoising_thresh = 1

	all_image_arr = read_images(folders, extension)
	print(all_image_arr.shape)



	#================ Thresholding: =============
	all_img_thresh = thresholding(all_image_arr) 
	print(np.shape(all_img_thresh))
	#================ 3d CCL and labeling: =============
	all_labeled, all_ncomponents = ccl_3d(all_img_thresh)
	print(np.shape(all_ncomponents),np.shape(all_labeled))

	#================ Computing the volume of each compnent and denoising : ===============
	thr_idxs = noise_removal(all_img_thresh, all_labeled)
	thr_idxs = np.asarray(thr_idxs)
	print(thr_idxs.shape)
	#print(thr_idxs[0])
	#================ Computing the centers: ==================================
	all_centers_noisefree = center_detection(all_img_thresh, all_labeled, thr_idxs)
	all_centers_noisefree = np.asarray(all_centers_noisefree)
	#================ Computing the centers: ===============

	visualization_3d_detection(all_centers_noisefree, 15, 10)

	#================= Tracking Part : ======================
	
	xx, yy, zz = tracker(all_centers_noisefree)
	simple_visualization_tracked_points(xx, yy, zz, xx.shape[0], 15, 10, 150, 'test_track_serial')


if __name__ == "__main__":
    main()