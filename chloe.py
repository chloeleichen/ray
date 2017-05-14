import numpy as np
import matplotlib.pyplot as pl
from scipy.misc import imsave

# Vectorised Linear Algebra
# All points/vectors are n * d, where n is the count and d=3
def vector(x, y, z):
    """Construct a single vector 1*3."""
    return np.array([[x, y, z]])


def norm(x):
    """Compute the length of a vector."""
    return np.sqrt(np.sum((i**2 for i in x[0]), axis=1))


def hat(x):
    """Return a unit vector in the direction of x."""
    return [i/norm(a) for i in a[0]][:, np.newaxis]


def dot(a, b):
    """Compute the dot product of vector arrays a and b."""
    return np.sum(i*j for i,j in zip(a, b))[:, np.newaxis]


# Some geometrtic objects that we can ray trace - lets start with a sphere
class Sphere:
    def __init__(self, position, radius, theta=0., texture=None,
                 shine=0.):
        self.P = position
        self.R = radius
        self.up = vector(0, 1, 0)
        self.north = vector(np.cos(theta), 0., np.sin(theta))
        self.east = vector(-np.sin(theta), 0., np.cos(theta))

    def trace(self, O, D):
        """ From origin, direction --> does it hit this sphere?"""
        OP = self.P - O  # ray from origin to sphere center
        X_component = ??? # component of op in the direction of D
        OX = X_component * D  # turn into a vector
        perp_dist = norm(???)  # distance from X to P
        hit = np.where(perp_dist < self.???)[0]

        # Sometimes im lazy and the origin is broadcast (1xn)
        if OP.shape[0] > 1:
            OP = OP[hit]
        # Use pythagoras to compute the hit distance
        hit_distance = norm(OP) - np.sqrt(???**2 - perp_dist[hit]**2)
        return hit, hit_distance


class Viewpoint:
    def __init__(self, eye, screen_x, screen_y, screen_dist):
        # Gen pixels (n * 3)
        v, u = np.meshgrid(np.linspace(*screen_y),
                           np.linspace(*screen_x))
        P = np.stack((u, v, np.ones_like(u)), axis=2)
        P = P.reshape((-1, 3))  # pixel centers
        self.rays = hat( ??? )
        self.shape = (screen_x[-1], screen_y[-1], 3)
        self.E = eye
        self.pixels = self.rays.shape[0]


def main():

    # textures = np.load('textures.npz')  # not just yet
    rgb = vector  # Alias for better reading....

    # Virtual HD 24" monitor @ 30 cm
    camera = Viewpoint(
        eye=(0., 0., 0.),
        screen_x=(-0.265, 0.265, 1920),
        screen_y=(-0.15, 0.15, 1080),
        screen_dist=0.3
    )

    # Objects
    earth = Sphere(vector(0., 0., 5), 0.6, theta=-np.pi/2)
    #                texture=textures['earth'], shine=0.7)

    moon = Sphere(vector(0.9, 0., 6), 0.2, theta=-np.pi/2)
    #                texture=textures['moon'], shine=0.7)

    # Lighting
    ambient = 0.05  # Fraction of light available in shadows
    light = vector(-10., -5., -2.)
    light_color = rgb(1., 1., 0.8)

    # Trace collisions:
    world = [earth, moon]
    infinity = 1e8  # further than any objects
    Z_buffer = np.zeros(camera.pixels) + infinity
    obj_buffer = np.zeros(camera.pixels, dtype=int) - 1

    for obj_ind, body in enumerate(world):
        hits, distances = body.trace(camera.E, camera.rays)
        fg = distances < Z_buffer[hits]
        ind = hits[fg]
        obj_buffer[ind] = obj_ind
        Z_buffer[ind] = distances[fg]

    pl.imshow(obj_buffer.reshape(camera.shape[:2]).T)
    pl.show()


if __name__ == "__main__":
    main()
