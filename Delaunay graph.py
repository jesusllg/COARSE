import numpy as np
import scipy

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

tested = np.random.rand(20,3)
cloud  = np.random.rand(50,3)

print(in_hull(tested,cloud))

vertex = np.array(vertex)
        #print('vertex: ', vertex)
        #print('p: ', p)
        from scipy.spatial import Delaunay
        if not isinstance(vertex, Delaunay):
            hull = Delaunay(vertex)

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")
# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
for s in hull1.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(vertex[s, 0], vertex[s, 1], "r-")

# Make axis label
for i in ["x", "y", "z"]:
    eval("ax.set_{:s}label('{:s}')".format(i, i))

    ax.plot(p.T[0], p.T[1], "ko")

    # Plot defining corner points
ax.plot(vertex.T[0], vertex.T[1], "ko")
plt.show()
