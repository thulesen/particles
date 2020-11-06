"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from numpy import inner
from numpy.linalg import norm
from math import pi
from heapq import *

dt = 1 / 60.

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

class WallCollisionEvent:
    def __init__(self, particle, data):
        # data should be a tuple (i, s) where i is the index of the colliding particle
        # and s is 'h' resp. 'v' corresponding to a collision with the horizontal
        # resp. vertical wall
        
        self._particle = particle
        i, s = data
        # self._refl_idx is the index of the velocity to be inverted
        if s == 'h':
            self._refl_idx = (i, 0)
        elif s == 'v':
            self._refl_idx = (i, 1)

    def unfold():
        self._particle.vel[self._refl_idx] *= -1

class Particle:
    # particle is a generalised particle, i.e., the position and velocity is a (n, 2) matrix
    
    def __init__(self, pos, vel, rad):
        # shapes: pos & vel: (n, 2), rad: (n, 1)
        self.pos = pos
        self.vel = vel
        self.rad = rad

        # event-driven simulation
        self._t = 0.0 # time
        pq = [] # priority queue
        for i in range(pos.shape[0]):
            t, j = self.predict_wall_collision(i)
            event = (t, i, j)
            heappush(pq, event)
        self._pq = pq
        self._next_event = heappop(pq)

    def advance_to(self, t):
        while(self._next_event[0] <= t):
            # move particles into place
            self.pos += (self._next_event[0] - self._t) * self.vel
            self._t = self._next_event[0]
            
            # let the next event unfold
            i, j = self._next_event[1:]
            self.vel[i][j] *= -1
            #self.vel[i][(self.pos[i] + self.rad[i] > 1) | (self.pos[i] - self.rad[i] < 0)] *= -1
            
            # create new event and push it to queue
            s, j = self.predict_wall_collision(i)
            event = (s + self._next_event[0], i, j)
            self._next_event = heappushpop(self._pq, event)

        # after this: self._next_event[0] > t
        
        # move particles the rest of the way
        self.pos += (t - self._t) * self.vel
        self._t = t

    def predict_wall_collision(self, i):
        ts = np.array([[np.inf, 0],
                       [np.inf, 1]]) # invalid times
        for j in range(2):
            if self.vel[i][j] > 0:
                ts[j][0] = (1 - self.pos[i][j] - self.rad[i]) / self.vel[i][j]
            elif self.vel[i][j] < 0:
                ts[j][0] = (-self.pos[i][j] + self.rad[i]) / self.vel[i][j]
        t, j = ts[np.argmin(ts, axis=0)[0]]
        return t, int(j)
        
    def tick(self):
        global dt

        self._t += dt
        # if self._next_event is None:
        #     self._next_event = heappop(pq)
        if self._t > self._next_event[0]:
            # let the next event unfold
            i = self._next_event[1]
            self.vel[i][(self.pos[i] + self.rad[i] > 1) | (self.pos[i] - self.rad[i] < 0)] *= -1

            # create new event and push it to queue
            event = (self.predict_wall_collision(i) + self._t, i)
            self._next_event = heappushpop(self._pq, event)
            
        # reverse direction if we are hitting the walls
        #vel[(pos + rad > 1) | (pos - rad < 0)] *= -1

        # collision detection between particles
        # for i in range(n_particles):
        #     for j in range(i+1, n_particles):
        #         dif = pos[i] - pos[j]
        #         if norm(dif) < rad[i] + rad[j]:
        #             proj = inner(vel[i] - vel[j], dif) / norm(dif)**2 * dif
        #             vel[i] = vel[i] - proj
        #             vel[j] = vel[j] + proj
        
        self.pos += dt * self.vel


# overall parameters
n_particles = 5
radius = 0.01

# Create simulation objects
shape = (n_particles, 2)
rng = np.random.default_rng()
#pos = (1 - 2 * radius) * rng.random(shape) + radius
pos = np.array(n_particles * [[0.5, 0.5]])
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
time = 0
dt = 1 / 60.
def animate(i):
    global time, dt
    #particle_sim.advance_to(time)
    particle_sim.advance_to(i * dt)
    #particle_sim.tick()
    pos = particle_sim.pos
    particle_plot.set_data(pos[:, 0], pos[:, 1])
    time += dt
    return particle_plot,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
