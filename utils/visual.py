import glob 
import imageio
import numpy as np
import os
import torch 
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import List
from natsort import natsorted

class Visual:
    def __init__(self, outdir, full_screen: bool = False, background: str = 'white') -> None:
        """ Initializes the visualization class. 
        
        Inputs
        ------
            outdir[str]: output directory path.
            full_screen[bool]: toggles full-screen if True
            background[str]: sets the color of the background. 
        """
        self.full_screen = full_screen
        self.background = background
        self.hh = []
        self.fig_count = 0
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.reset()
        
    def reset(self) -> None:
        """ Resets the visualization radar. """
        self.fig = plt.figure(figsize=(10, 10))

        self.sp = self.fig.add_subplot(111)
        self.sp.set_facecolor(f'xkcd:{self.background}')
        self.sp.axis('equal')
        self.radar()
        # self.runway()
        plt.grid(True)
        plt.xlabel("X (in m)")
        plt.ylabel("Y (in m)")
        plt.xlim([-350,350])
        plt.ylim([-350,350])

        if self.full_screen:
            self.fig.canvas.manager.full_screen_toggle()
    
    def radar(self) -> None:
        self.sp.plot(
            100*np.sin(np.linspace(0, 2*np.pi, 100)), 100*np.cos(np.linspace(0, 2*np.pi, 100)), 
            color='grey', linestyle='--')
        self.sp.text(70,70,'100m', color = 'grey')
        self.sp.text(-70,-70,'100m', color = 'grey')

        self.sp.plot(
            200*np.sin(np.linspace(0, 2*np.pi, 100)), 200*np.cos(np.linspace(0, 2*np.pi, 100)), 
            color='grey', linestyle='--')
        self.sp.text(141,141,'200m', color = 'grey')
        self.sp.text(-141,-141,'200m', color = 'grey')

        self.sp.plot(
            300*np.sin(np.linspace(0, 2*np.pi, 100)), 300*np.cos(np.linspace(0, 2*np.pi, 100)), 
            color='grey', linestyle='--')
        self.sp.text(212,212,'300m', color = 'grey')
        self.sp.text(-212,-212,'300m', color = 'grey')

    def runway(self) -> None:
        self.sp.plot(
            np.linspace(-200, 200, 100), np.linspace(0, 0, 100), color="green", lw=70, alpha=0.2, 
            zorder=10, label='Runway 2')
        

    
    def plot(self, agents, show: bool = False, show_tree: bool = False, agent_id = None) -> None:
        for agent in agents:
            state = agent.state
            color = agent.color

            first_state = agent.trajectory[0].numpy()
            last_state = agent.trajectory[-1].numpy()
            # print(last_state[-1,:],color)
            cur_traj = torch.cat(agent.trajectory).numpy()
            ref_traj = agent.reference_trajectory
            id_ = agent.id
            
            alpha, alpha_ref = 0.2, 0.1
            # text = f'A{id_}\n Done!'
            # if not agent.done:
            #     alpha, alpha_ref = 1.0, 0.2

            #     speed = str(int(np.linalg.norm(last_state[-1,:2]-last_state[-2,:2]) * 1943)) + "Knots \n"
            #     alt = str(int(last_state[-1, 2] * 3280.84)) + "MSL"
            #     text = f'A{id_}\n {speed} {alt}'
            # self.sp.text(first_state[0,0]+0.5, first_state[0,1]+0.5, text, color=color, fontsize=8)

            # reference trajectory 
            # self.sp.plot(ref_traj[:, 0], ref_traj[:, 1] , color=color, linewidth=10, alpha=alpha_ref, zorder=0)
            
            # executed trajectory so far:
            self.sp.plot(last_state[:10, 0], last_state[:10, 1], color=color, linestyle='-', linewidth=4, alpha=1)
            
            # markers for the start and end of last agent's state:
            self.sp.plot(last_state[0, 0], last_state[0, 1], color=color, marker='o', markersize=6, alpha=1)
            self.sp.plot(last_state[9, 0], last_state[9, 1], color=color, marker='o', markersize=10, alpha=1)
            plt.axes
            # plot tree expansions
            lz1 = plt.Circle(( 300 , 0.0 ), 20 ,alpha = 0.5, facecolor = 'darkorange', linestyle = '--', edgecolor = 'darkorange')
            self.sp.add_artist(lz1)
            lz2 = plt.Circle(( -300 , 0.0 ), 20 ,alpha = 0.5, facecolor = 'dodgerblue', linestyle = '--', edgecolor = 'dodgerblue')
            self.sp.add_artist(lz2)

            if show_tree:
                for tree in agent.tree:
                    # if len(tree)>0:
                        # tree = torch.cat(agent.tree).numpy()
                        self.sp.plot(tree[:, 0], tree[:, 1], color='limegreen', linestyle='-', linewidth=2, markersize=4,alpha=alpha)
                
                # self.sp.plot(tree[:, 0], tree[:, 1], color='magenta', linestyle='-', linewidth=2, markersize=4)

                # self.sp.plot(state[:, 0], state[:, 1], color='magenta', linestyle='-', linewidth=6)

        if show:
            plt.show()
            plt.waitforbuttonpress()
            # plt.pause(pause)
        else:
            plt.savefig(f"{self.outdir}/{self.fig_count}.png", bbox_inches='tight', dpi=100)
            self.fig_count += 1
        
        plt.close()

    def save(self, num_episode: int) -> None:
        self.fig_count = 0
        imgs = natsorted(glob.glob(f"{self.outdir}/*.png"))
        with imageio.get_writer(f"{self.outdir}/ep-{num_episode}.gif", mode='I', duration=0.2) as writer:
            for img in imgs:
                writer.append_data(imageio.imread(img))
                os.remove(img)

        