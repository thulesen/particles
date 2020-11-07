# This project simulates wall and interparticular collisions for an arbitrary number of particles.
# The simulation is event-driven and implemented with a priority queue.
# As the simulation proceeds, it gets progressively slower because the queue fills up with invalid
# events. Better performance could be achieved if we knew in advance the simulation duration so we could
# discard some of these events that occur too far in the future.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from numpy import inner
from numpy.linalg import norm
from math import pi
import itertools
from heapq import *
from heapdict import heapdict

# modified from https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit/42972469#42972469
class LineDataMarkerSize(Line2D):
    def __init__(self, *args, **kwargs):
        _ms_data = kwargs.pop("markersize", 1)
        super().__init__(*args, **kwargs)
        self._ms_data = _ms_data

    def _get_ms(self):
        if self.axes is not None:
            ppd = 72. / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((self._ms_data, 0)) - trans((0, 0))) * ppd)[0]
        else:
            return 1

    def _set_ms(self, ms):
        pass
        #self._ms_data = ms

    _markersize = property(_get_ms, _set_ms)

class WallCollisionHandler:
    # for the _i'th particle, this class handles wall collisions
    
    def __init__(self, particle, i):
        self._particle = particle
        self._i = i
        # self._j is the velocity coordinate that we will invert on collision
        self._j = -1 # invalid assignment, will be set in predict()
        self.idxs = [i]

    def unfold(self):
        self._particle.vel[self._i, self._j] *= -1

    def predict(self):
        particle = self._particle
        pos = particle.pos
        vel = particle.vel
        rad = particle.rad
        i = self._i
        
        ts = np.array([[np.inf, 0],
                       [np.inf, 1]]) # invalid times
        for j in range(2):
            if vel[i][j] > 0:
                ts[j][0] = (1 - pos[i][j] - rad[i]) / vel[i][j]
            elif vel[i][j] < 0:
                ts[j][0] = (-pos[i][j] + rad[i]) / vel[i][j]
        t, j = ts[np.argmin(ts, axis=0)[0]]

        self._j = int(j)
        return t

class ParticleCollisionHandler:
    # for the _i1, _i2'th particles, this class handles wall collisions

    def __init__(self, particle, i1, i2):
        self._particle = particle
        self._i1 = i1
        self._i2 = i2
        self.idxs = [i1, i2]

    def unfold(self):
        dif = self._particle.pos[self._i1] - self._particle.pos[self._i2]
        proj = inner(self._particle.vel[self._i1] - self._particle.vel[self._i2], dif) / norm(dif)**2 * dif
        self._particle.vel[self._i1] -= proj
        self._particle.vel[self._i2] += proj

    def predict(self):
        dif = self._particle.pos[self._i1] - self._particle.pos[self._i2]
        veldif = self._particle.vel[self._i1] - self._particle.vel[self._i2]
        b = inner(dif, veldif)
        if b >= 0:
            return np.inf
        r = (self._particle.rad[self._i1] + self._particle.rad[self._i2])[0]
        a = norm(veldif)**2
        c = norm(dif)**2 - r**2
        D = b**2 - a * c
        if D < 0:
            return np.inf
        else:
            return -(b + D**(1/2)) / a

class Particle:
    # particle is a generalised particle, i.e., the position and velocity is a (n, 2) matrix
    
    def __init__(self, pos, vel, rad):
        # shapes: pos & vel: (n, 2), rad: (n, 1)
        self.pos = pos
        self.vel = vel
        self.rad = rad
        counter = itertools.count()
        self._counter = counter # count for breaking ties
        
        # event-driven simulation
        self._t = 0.0 # time
        handlers = {i: [] for i in range(pos.shape[0])}
        hd = heapdict()
        for i in range(pos.shape[0]):
            handler = WallCollisionHandler(self, i)
            handlers[i].append(handler)
            t = handler.predict()
            hd[handler] = (t, next(counter))
        for i in range(pos.shape[0]):
            for j in range(i):
                handler = ParticleCollisionHandler(self, i, j)
                handlers[i].append(handler)
                handlers[j].append(handler)
                t = handler.predict()
                hd[handler] = (t, next(counter))
        self._handlers = handlers
        self._hd = hd
        self._next_event = hd.popitem()

    def advance_to(self, t):
        while(self._next_event[1][0] <= t):
            # move particles into place
            self.pos += (self._next_event[1][0] - self._t) * self.vel
            self._t = self._next_event[1][0]

            # let the next event unfold
            handler = self._next_event[0]
            handler.unfold()

            # change priority for the relevant events
            for i in handler.idxs:
                for handler in self._handlers[i]:
                    dt = handler.predict()
                    self._hd[handler] = (self._t + dt, next(self._counter))

            self._next_event = self._hd.popitem()
        # after this: self._next_event[0] > t
        
        # move particles the rest of the way
        self.pos += (t - self._t) * self.vel
        self._t = t

# overall parameters
n_particles = 20
radius = 0.01

# Create simulation objects
shape = (n_particles, 2)
rng = np.random.default_rng()
pos = (1 - 2 * radius) * rng.random(shape) + radius
vel = rng.random(shape)
vel = vel / np.linalg.norm(vel)
rad = np.array(n_particles * [radius]).reshape((n_particles, 1))
particle_sim = Particle(pos, vel, rad)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1), aspect='equal')
particle_plot = LineDataMarkerSize([], [], marker='o', ls=' ', markersize=2 * radius)
ax.add_line(particle_plot)

# initialization function: plot the background of each frame
def init():
    #particle_plot.set_data([], [])
    return particle_plot,

# animation function.  This is called sequentially
dt = 1 / 60.
def animate(i):
    global dt
    particle_sim.advance_to(i * dt)
    pos = particle_sim.pos
    
    particle_plot.set_data(pos[:, 0], pos[:, 1])
    return particle_plot,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, interval=20, blit=True)

plt.show()
