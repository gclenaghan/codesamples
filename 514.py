from random import randint, sample
from scipy.stats import binom
import itertools
import numpy as np
import argparse

def polygon_area(npoints):
	# computes twice the area_list of a polygon with points given
	# using a vectorized shoelace formula
	points = npoints[1]
	if points.shape[0] < 3:
		return (npoints[0], 0)
	x = points[:, 0]
	y = points[:, 1]
	area_list = np.abs(np.dot(x, np.roll(y, 1)) -
		np.dot(y, np.roll(x, 1)))
	return (npoints[0], area_list)

def cross(o, a, b):
	# Computes cross product (a-o) x (b-o)
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    

def generate_hull(n, m=None):
	# Sample random points in grid, then take the convex hull
	if m is None:
		m = binom.rvs((n+1)**2, 1/(n+1)) # compute number of points to be included
	# sample m points uniformly across grid
	points = [divmod(p, n+1) for p in sorted(sample(range((n+1)**2), m))]
	if m <= 3:
		return (m, np.array(points))
	
	# Compute convex hull of the points chosen,
	# uses the monotone chain algorithm
	lower = points.copy() # preallocate array
	i = 0
	for p in points:
		while i >= 2 and cross(lower[i-2], lower[i-1], p) <= 0:
			i -= 1
		lower[i] = p
		i += 1
 
	upper = points.copy() 
	j = 0
	for p in reversed(points):
		while j >= 2 and cross(upper[j-2], upper[j-1], p) <= 0:
			j -= 1
		upper[j] = p
		j += 1
 
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
	return (len(points), np.array(lower[:i-1] + upper[:j-1]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-N','--grid_size', type=int, default=100)
	parser.add_argument('-M','--samples', type=int, default=10000)
	parser.add_argument('--method', type=str, choices=['randomsize','uniformsize'],
						default='randomsize')
	args = parser.parse_args()
	N = args.grid_size
	M = args.samples
	
	print("Computing areas...")
	if args.method == 'randomsize':
		# Generate M random polygons, keeping track of the original number of points
		area_list = [polygon_area(generate_hull(N, x)) for x in np.random.binomial((N+1)**2, 1/(N+1), M)]
		print("Grouping...")
		list.sort(area_list)
	else:
		# Generate M polygons for each size from 3 to (N+1)**2
		area_list = [polygon_area(generate_hull(N, x//M + 3)) for x in range(((N+1)**2 - 2)*M)]
	
	# freeze a binomial distribution corresponding to the expected number of polygons
	# for a particular point count. The average will be computed separately for each
	# point count, and then the averages can be weighted by the number of occurances
	# expected of that count. This method reduces the variance.
	b = binom((N+1)**2, 1/(N+1))
	avg = 0
	print("Averaging...")
	for n, ptlist in itertools.groupby(area_list, lambda x: x[0]):
		if n < 3:
			continue
		n_arealist = [x[1] for x in ptlist]
		l = len(n_arealist)
		if l > 0:
			avg += b.pmf(n)*np.mean(n_arealist)/2 # /2 as previously computed twice the area
		print(n, l, b.pmf(n)*M, np.mean(n_arealist), np.var(n_arealist)/4)
	print("Done!")
	print(avg, np.var([x[1] for x in area_list])/4)
