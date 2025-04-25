Runing Average using the 8 nearest neighbor in a 2D array.

The function is based on `scipy.ndimage`

If you want to change the average pattern to the 4 nearest neighbor, 
`footprint` and `nbre_neighbour` need to be modified accordingly. 

**Example:**

	python runingAverageImage.py 
	input
	----
	[[ 1.  2.  3.]
	 [ 4.  5.  6.]
	 [ 7.  8.  9.]]

	runing average
	----
	[[ 3.66666667  3.8         4.33333333]
	 [ 4.6         5.          5.4       ]
	 [ 3.4         6.2         6.33333333]]

