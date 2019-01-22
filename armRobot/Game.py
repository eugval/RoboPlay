from framework.Game import  BaseGame
import pygame


class Game(BaseGame):
    def __init__(self, world,  size = 1000, background_color =(0,0,0),target_color = (255,0,0),
                 effector_color =(0,255,0),  effector_diameter = 7,arm_color = (255,255,255),
                 arm_thickness = 5, target_diameter = 21, randomise_dynamics = True):
        super(Game,self).__init__(world,size,background_color,target_color,target_diameter,randomise_dynamics)

        self.effector_color= effector_color
        self.effector_diameter  = effector_diameter
        self.arm_color = arm_color
        self.arm_thickness = arm_thickness


    def snap(self):
        target_pose = self.world.target_pos

        self.screen.fill(self.background_color)
        pygame.draw.circle(self.screen, self.target_color, self.discretize_position(*target_pose), self.target_diameter)

        joints = []
        for eff_pos in self.world.robot_arm.effector_positions:
            new_coords = self.discretize_position(*eff_pos)
            pygame.draw.circle(self.screen, self.effector_color, new_coords, self.effector_diameter)
            if len(joints) > 0: pygame.draw.line(self.screen, self.arm_color, joints[-1], new_coords,
                                                 self.arm_thickness)
            joints.append(new_coords)

        pygame.display.flip()