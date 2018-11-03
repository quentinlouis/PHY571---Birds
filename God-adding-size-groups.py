import numpy as np
import itertools
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import cmocean
import scipy.optimize
import time
import datetime


class Bird:
    def __init__(self, pos: np.ndarray, vel: float, angle: float):
        self.vel = vel
        self.angle = angle
        self.pos = pos
        self.lookV = None
        self.speedV = None
        self.update_calculated_props()

    def __repr__(self):
        return "Bird(%.2f, %.2f)" % (self.pos[0], self.pos[1])

    def update_calculated_props(self):
        self.lookV = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.speedV = self.lookV * self.vel


class Sky:
    def __init__(self, L: float, gridstep: float):
        self.L = L
        self.gridstep = gridstep

        # Generate grid
        self.gridL = int(np.ceil(L/gridstep))
        self.grid = None
        self.init_grid()

        self.birds = []

    def get_avg_speed(self):
        angles = np.array([bird.angle for bird in self.birds])
        velocities = np.array([bird.vel for bird in self.birds])
        vel_x_mean, vel_y_mean = np.mean(np.cos(angles)* velocities), np.mean(np.sin(angles)* velocities)
        return np.sqrt(np.dot(vel_x_mean, vel_x_mean) ** 2 + np.dot(vel_y_mean, vel_y_mean) ** 2)

    def add_bird(self, bird: Bird):
        self.birds.append(bird)

    def init_grid(self):
        self.grid = np.zeros((self.gridL, self.gridL), dtype=object)
        for i, j in itertools.product(range(self.gridL), range(self.gridL)):
            self.grid[i, j] = []

    def update_grid(self):
        self.init_grid()
        for bird in self.birds:
            gridpos = bird.pos // self.gridstep
            self.grid[int(gridpos[0]), int(gridpos[1])].append(bird)

    def add_n_random_birds(self, n: int, vel: float):
        for _ in range(n):
            theta = np.random.rand() * np.pi * 2
            bird = Bird(np.random.rand(2) * self.L, vel, theta)
            self.add_bird(bird)

    def plot(self):
        pos = np.array([bird.pos for bird in self.birds])
        angles = np.array([bird.angle for bird in self.birds])
        plt.quiver(pos[:, 0], pos[:, 1], np.cos(angles), np.sin(angles))
        plt.show()


class Physics:
    def __init__(self, sky: Sky, interaction_radius: float, eta: float):
        self.sky = sky; sky.update_grid()
        self.interaction_radius = interaction_radius
        self.eta = eta

    def advance(self, dt: float):
        for bird in self.sky.birds:
            
            #--> à mettre dans une fonction à part
            # Collect all bird in interaction range
            interact_with = []
            gridpos = bird.pos // self.sky.gridstep
            grid_interaction_radius = self.interaction_radius / self.sky.gridstep

            grid_xmin = int(gridpos[0]-grid_interaction_radius)
            grid_xmax = int(np.ceil(gridpos[0]+grid_interaction_radius))
            grid_ymin = int(gridpos[1] - grid_interaction_radius)
            grid_ymax = int(np.ceil(gridpos[1] + grid_interaction_radius))

            for i, j in itertools.product(range(grid_xmin, grid_xmax + 1), range(grid_ymin, grid_ymax + 1)):
                interact_with += self.sky.grid[i % self.sky.gridL, j % self.sky.gridL]


            # Apply interaction
            # Enforce radius
            interact_with_close = []
            for other_bird in interact_with:
                distance = np.linalg.norm(other_bird.pos - bird.pos)
                if distance <= self.interaction_radius:
                    interact_with_close.append(other_bird)
            
            #Enforce bird movement
            median_cos = 0
            median_sin = 0
            for other_bird in interact_with_close:
                median_cos += np.cos(other_bird.angle)
                median_sin += np.sin(other_bird.angle)
            median_angle = np.arctan2(median_sin, median_cos)
            delta_angle = self.eta * (np.random.rand()-.5)
            bird.angle = (median_angle + delta_angle) % (2 * np.pi)

        # Verlet movement *after* updating directions
        for bird in self.sky.birds:
            bird.update_calculated_props()
            bird.pos = (bird.pos + bird.speedV * dt) % L

        sky.update_grid()
    
    #TO DO - Create an idependent for computing the interactions. To be used in the computation of the movement and of the groups.
    """def get_interactions(self):
        all_final_interactions = []
        for bird in self.sky.birds:
            # Collect all bird in interaction range
            interact_with = []
            gridpos = bird.pos // self.sky.gridstep
            grid_interaction_radius = self.interaction_radius / self.sky.gridstep

            grid_xmin = int(gridpos[0]-grid_interaction_radius)
            grid_xmax = int(np.ceil(gridpos[0]+grid_interaction_radius))
            grid_ymin = int(gridpos[1] - grid_interaction_radius)
            grid_ymax = int(np.ceil(gridpos[1] + grid_interaction_radius))

            for i, j in itertools.product(range(grid_xmin, grid_xmax + 1), range(grid_ymin, grid_ymax + 1)):
                interact_with += self.sky.grid[i % self.sky.gridL, j % self.sky.gridL]

            # Apply interaction
            # Enforce radius
            interact_with_close = []
            for other_bird in interact_with:
                distance = np.linalg.norm(other_bird.pos - bird.pos)
                if distance <= self.interaction_radius:
                    interact_with_close.append(other_bird)

            all_final_interactions.append([bird, interact_with_close])
        return all_final_interactions"""

    def get_angles_correlations(self, pos, angles):
        n_birds = angles.shape[0]
        projected = np.cos(angles)  # fuck circular variables

        correlations = np.full((n_birds, n_birds), 1.)  # upper triangular
        for i in range(n_birds):
            for j in range(i+1, n_birds):
                correlations[i, j] = projected[i] * projected[j]

        distances = np.zeros((n_birds, n_birds))  # upper triangular
        for i in range(n_birds):
            for j in range(i + 1, n_birds):
                distances[i, j] = np.linalg.norm(pos[i] - pos[j])

        distances_flat = distances[np.triu_indices(n_birds, k=1)]
        correlations_flat = correlations[np.triu_indices(n_birds, k=1)]

        return np.array(sorted(distances_flat)), np.array([x for _, x in sorted(zip(distances_flat, correlations_flat))])
    
    """Deux oiseaux sont dans un même groupe si ils sont dans un cercle de rayon r=1 et si ils ont des angles qui ne divergent par pas plus de eta. Il faut plotter un histogramme avec en abscisse la taille des groupes et en hauteur le nombre de groupes de cette taille donnée"""
    def get_size_groups(self):
        #Je recalcule toutes les interactions, ce qui ne parait pas judicieux d'un point de vue computationnel
        #TO DO : Il faut isoler une méthode pour simplement calculer les intéractions.
        
        all_final_interactions = []
        for bird in self.sky.birds:
            # Collect all bird in interaction range
            interact_with = []
            gridpos = bird.pos // self.sky.gridstep
            grid_interaction_radius = self.interaction_radius / self.sky.gridstep

            grid_xmin = int(gridpos[0]-grid_interaction_radius)
            grid_xmax = int(np.ceil(gridpos[0]+grid_interaction_radius))
            grid_ymin = int(gridpos[1] - grid_interaction_radius)
            grid_ymax = int(np.ceil(gridpos[1] + grid_interaction_radius))

            for i, j in itertools.product(range(grid_xmin, grid_xmax + 1), range(grid_ymin, grid_ymax + 1)):
                interact_with += self.sky.grid[i % self.sky.gridL, j % self.sky.gridL]

            # Apply interaction
            # Enforce radius
            interact_with_close = []
            for other_bird in interact_with:
                distance = np.linalg.norm(other_bird.pos - bird.pos)
                angle_difference = np.abs(other_bird.angle - bird.angle)%(2*np.pi)
                if (distance <= self.interaction_radius and angle_difference <= self.eta):
                    interact_with_close.append(other_bird)

            all_final_interactions.append([bird, interact_with_close])
    
        interactions = all_final_interactions
        #print(interactions)
        #--> I now have the table of interaction of all the birds. The question is now how do I define a group ?
        
        groups = []
        for interaction in interactions:
            bird, other_birds = interaction[0], interaction[1]
            bird_was_added = False
            for group in groups:
                if (bird in group):
                    #print(1)
                    bird_was_added = True
                    idx = groups.index(group)
                    for other_bird in other_birds:
                        if (other_bird not in groups[idx]):
                            groups[idx].append(other_bird)
            if (not bird_was_added):
                new_group = [bird]
                for other_bird in other_birds:
                    new_group.append(other_bird)
                groups.append(new_group)
        #print(groups)
        #grouping in group sizes with simple code
        ##Code valid for a scatter plot
        group_sizes = np.zeros(len(self.sky.birds))
        for group in groups:
            nb_birds = len(group)
            group_sizes[nb_birds] += 1
        ##Similar code, but for a histogram
        """group_sizes = np.zeros(len(groups))
        for i in range (len(groups)):
            nb_birds = len(groups[i])
            group_sizes[i] = nb_birds"""
        
        return group_sizes
    
    def animate(self, total_time: float, dt: float, verbose_prop: float=.01):
        Writer = animation.writers['ffmpeg']  # to save video
        writer = Writer(fps=15, bitrate=-1)  # to save video

        gs = GridSpec(2, 2)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        fig.set_size_inches(15, 15, True)

        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
        timestamps = np.arange(0, total_time, dt)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cm = cmocean.cm.phase

        pos = np.array([bird.pos for bird in self.sky.birds])
        angles = np.array([bird.angle for bird in self.sky.birds])
        Q = ax1.quiver(pos[:, 0], pos[:, 1], np.cos(angles), np.sin(angles), cmap=cm, angles='xy', scale_units='xy', scale=1)
        ax1.set_xlim(0, self.sky.L)
        ax1.set_ylim(0, self.sky.L)

        avg_speed, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, total_time)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("avg. speed (m/s)")

        all_corr, = ax3.plot([], [], lw=2)
        fitted_corr, = ax3.plot([], [], lw=2)
        ax3.set_xlim(0, self.sky.L)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("distance (m)")
        ax3.set_ylabel("correlation (normed)")
        
        size_groups, = ax4.plot([], [], "-o", lw=2)
        #ax4.set_xlim(0, total_time)
        ax4.set_ylim(0, 10) #len(self.sky.birds))
        ax4.set_xlim(0, len(self.sky.birds))
        #ax4.set_ylim(0, len(self.sky.birds))
        ax4.set_xlabel("group size")
        ax4.set_ylabel("number of groups")

        avg_speeds = []

        total_frames = int(total_time / dt)
        
        def update_quiver(num, Q, time_text, avg_speed, all_corr, fitted_corr, size_groups):
            """updates the horizontal and vertical vector components by a
            fixed increment on each frame
            """
            if num % (1+int(total_frames*verbose_prop)) == 0:
                print("Doing frame %d/%d" % (num, total_frames))

            self.advance(dt)
            
            #General graph
            pos = np.array([bird.pos for bird in self.sky.birds])
            angles = np.array([bird.angle for bird in self.sky.birds])
            
            X, Y = pos[:, 0], pos[:, 1]
            N = len(self.sky.birds)
            new_offsets = np.zeros((N, 2))
            for i in range(N):
                new_offsets[i][0] = X[i]
                new_offsets[i][1] = Y[i]
            U = np.cos(angles)
            V = np.sin(angles)
            Q.set_offsets(new_offsets)
            Q.set_UVC(U*self.interaction_radius, V*self.interaction_radius, norm(angles))
            
            #average_speed
            avg_speeds.append(self.sky.get_avg_speed())
            avg_speed.set_data(timestamps[:num + 1], avg_speeds[:num + 1])
            
            #correlations
            dists, corrs = physics.get_angles_correlations(pos, angles)
            regular_dists = np.linspace(0, self.sky.L, 250)
            regular_corrs = []

            for i in range(len(regular_dists) - 1):
                mask = np.logical_and(dists > regular_dists[i], dists < regular_dists[i + 1])
                mean = np.mean(corrs[mask])
                regular_corrs.append(mean)
            regular_dists = regular_dists[:-1]
            all_corr.set_data(regular_dists, regular_corrs)

            def func_fit(x, a, zeta):
                return a * np.exp(- x / zeta)
            try:
                popt, _ = scipy.optimize.curve_fit(func_fit, dists, corrs)
                fitted_corr.set_data(dists, func_fit(dists, *popt))
            except Exception:
                pass
            
            #size groups
            get_size_groups = self.get_size_groups()
            #print(get_size_groups)
            N_abscisse = np.linspace(1, len(self.sky.birds), len(self.sky.birds))
            size_groups.set_data(N_abscisse, get_size_groups)
            #ax4.hist(get_size_groups, normed=True, bins=np.linspace(1, len(self.sky.birds), len(self.sky.birds)))
                
            #affichage du temps
            time_text.set_text('time = %.1f s' % round(dt * num, 3))

            return Q, time_text, avg_speed, all_corr, fitted_corr, size_groups

        start_t = time.time()
        print("Simulation start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))

        anim = animation.FuncAnimation(fig, update_quiver, frames=total_frames, fargs=(Q, time_text, avg_speed, all_corr, fitted_corr, size_groups),
                                       interval=200, blit=True, repeat=True)
        anim.save('lines.mp4', writer=writer)
        elapsed = time.time()-start_t
        print("Simulation ended at t=%s, elapsed: {%d}h{%d}m{%d}s" % (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                                                         elapsed// 3600 % 24, elapsed // 60 % 60, elapsed % 60))
        # plt.show()



L = 70
gridstep = .5
sky = Sky(L, gridstep)
sky.add_n_random_birds(400, 1)

physics = Physics(sky, 1, .2)

physics.animate(200, 1, verbose_prop=.1)



























































