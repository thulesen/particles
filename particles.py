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

class Particle:
    # particle is a generalised particle, i.e., the position and velocity is a (n, 2) matrix
    
    def __init__(self, pos, vel, rad):
        # shapes: pos & vel: (n, 2), rad: (n, 1)
        self.pos = pos
        self.vel = vel
        self.rad = rad

    def tick(self):
        global dt

        # reverse direction if we are hitting the walls
        vel[(pos + rad > 1) | (pos - rad < 0)] *= -1

        # collision detection between particles
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                dif = pos[i] - pos[j]
                if norm(dif) < rad[i] + rad[j]:
                    proj = inner(vel[i] - vel[j], dif) / norm(dif)**2 * dif
                    vel[i] = vel[i] - proj
                    vel[j] = vel[j] + proj
        
        self.pos += dt * vel

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
def animate(i):
    particle_sim.tick()
    pos = particle_sim.pos
    particle_plot.set_data(pos[:, 0], pos[:, 1])
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
