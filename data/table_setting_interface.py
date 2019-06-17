from copy import deepcopy
import os

import matplotlib.pyplot as plt
import numpy as np
import pygame
import pygame.freetype

import pydrake
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
import pydrake.geometry as pygeom
from pydrake.math import RigidTransform


# Maps XY -> pixel coordinates

SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1000

scaling = min(SCREEN_WIDTH, SCREEN_HEIGHT)
default_view_matrix = np.array([[scaling, 0., SCREEN_WIDTH/2. - scaling/2.],
                                [0., scaling, SCREEN_HEIGHT/2. - scaling/2.],
                                [0., 0., 1.]])
def apply_homogenous_tf(view, xy):
    return view[:2, :2].dot(xy) + view[:2, 2]
def invert_view_matrix(view):
    return np.linalg.inv(view)
    # Inverse for rigid transform:
    #inverted_view = np.eye(3)
    #inverted_view[:2, :2] = view[:2, :2].T
    #inverted_view[:2, 2] = -view[:2, :2].T.dot(view[:2, 2])
    #return inverted_view
def rotmat(r):
    return np.array([[np.cos(r), -np.sin(r)],
                     [np.sin(r), np.cos(r)]])

class PygameShape(object):
    def __init__(self, selectable=False, view_matrix=default_view_matrix):
        self.view_matrix = view_matrix.copy()
        self.surface = None
        self.pose = None
        self.selectable = selectable

    def draw(self, screen):
        if self.surface is None:
            raise NotImplementedError("Subclasses must populate their surface attribute.")
        if self.pose is None:
            raise NotImplementedError("Subclasses must populate their pose attribute.")
        rotated_surf = pygame.transform.rotate(self.surface, 180*self.pose[2]/np.pi)
        desired_rect = rotated_surf.get_rect()
        desired_rect.center = apply_homogenous_tf(self.view_matrix, self.pose[:2])
        screen.blit(rotated_surf, desired_rect)

    def get_center(self):
        return apply_homogenous_tf(self.view_matrix, self.pose[:2])

    def set_center(self, xy):
        self.pose[:2] = apply_homogenous_tf(invert_view_matrix(self.view_matrix), xy)
        
    def query_click(self, mouse_xy):
        # True if this landed in a visible pixel of our surface.
        if not self.selectable:
            return False

        # Transform mouse_xy backwards through rotation
        curr_center = apply_homogenous_tf(self.view_matrix, self.pose[:2])
        mouse_xy_local = mouse_xy - curr_center
        mouse_xy_local = rotmat(self.pose[2]).dot(mouse_xy_local) + self.surface.get_rect().center
        mouse_xy_local = mouse_xy_local.astype(np.int)
        if (mouse_xy_local[0] >= 0 and mouse_xy_local[1] >= 0 and 
            mouse_xy_local[0] < self.surface.get_width() and 
            mouse_xy_local[1] < self.surface.get_height()):
            at_pt = self.surface.get_at(mouse_xy_local)
            if at_pt[3] > 0:
                return True
        return False


class FixedTable(PygameShape):
    def __init__(self, pose, radius, color=None, img_path=None, view_matrix=default_view_matrix):
        PygameShape.__init__(self, selectable=False, view_matrix=view_matrix)
        assert(len(pose) == 3)
        assert(color is None or (len(color) == 3 or len(color) == 4))
        assert(color is not None or img_path is not None)
        self.radius = radius
        self.pose = np.array(pose).copy()
        self.pixel_extent = (np.diag(self.view_matrix[:2, :2])*radius).astype(int)
        self.surface = pygame.Surface(self.pixel_extent, pygame.SRCALPHA)

        if img_path is None:
            pygame.draw.ellipse(self.surface, color, pygame.Rect(0, 0, self.pixel_extent[0], self.pixel_extent[1]))
            pygame.draw.ellipse(self.surface, (0, 0, 0), pygame.Rect(0, 0, self.pixel_extent[0], self.pixel_extent[1]), 2)
        else:
            self.image = pygame.image.load(os.path.abspath(img_path)).convert_alpha()
            self.image = pygame.transform.scale(self.image, self.pixel_extent)
            self.surface.blit(self.image, self.surface.get_rect())

class InteractableDish(PygameShape):
    def __init__(self, pose, radius, color=None, img_path=None, view_matrix=default_view_matrix):
        PygameShape.__init__(self, selectable=True, view_matrix=view_matrix)
        assert(len(pose) == 3)
        assert(color is None or (len(color) == 3 or len(color) == 4))
        assert(color is not None or img_path is not None)
        self.radius = radius
        self.pose = np.array(pose).copy()
        self.pixel_extent = (np.diag(self.view_matrix[:2, :2])*radius).astype(int)
        self.surface = pygame.Surface(self.pixel_extent, pygame.SRCALPHA)

        if img_path is None:
            pygame.draw.ellipse(self.surface, color, pygame.Rect(0, 0, self.pixel_extent[0], self.pixel_extent[1]))
            pygame.draw.ellipse(self.surface, (0, 0, 0), pygame.Rect(0, 0, self.pixel_extent[0], self.pixel_extent[1]), 2)
        else:
            self.image = pygame.image.load(os.path.abspath(img_path)).convert_alpha()
            self.image = pygame.transform.scale(self.image, self.pixel_extent)
            self.surface.blit(self.image, self.surface.get_rect())


class InteractableUtensil(PygameShape):
    def __init__(self, pose, size, color=None, img_path=None, view_matrix=default_view_matrix):
        PygameShape.__init__(self, selectable=True, view_matrix=view_matrix)
        assert(len(pose) == 3)
        assert(color is None or (len(color) == 3 or len(color) == 4))
        assert(color is not None or img_path is not None)
        self.size = np.array(size)
        self.pose = np.array(pose).copy()
        self.pixel_extent = (self.view_matrix[:2, :2].dot(self.size)).astype(int)
        self.surface = pygame.Surface(self.pixel_extent, pygame.SRCALPHA)

        if img_path is None:
            pygame.draw.rect(self.surface, color,
                             pygame.Rect(0, 0, self.pixel_extent[0], self.pixel_extent[1]))
            pygame.draw.rect(self.surface, (0, 0, 0),
                             pygame.Rect(0, 0, self.pixel_extent[0], self.pixel_extent[1]),
                             5)
        else:
            self.image = pygame.image.load(os.path.abspath(img_path)).convert_alpha()
            self.image = pygame.transform.scale(self.image, self.pixel_extent)
            self.surface.blit(self.image, self.surface.get_rect())


instructions_string = '''
Left click and drag: translate object
Right click and drag: rotate object about origin
Middle click: Delete object
Keys [1, 2, 3, 4]: Add [plate, fork, knife, spoon] at cursor.
'''

WHITE = (255, 255, 255)
RED   = (255,   0,   0)

FPS = 30

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Set Some Tables")

    font = pygame.freetype.SysFont("", 12)

    instructions = []
    for instructions_substr in instructions_string.split("\n"):
        ins, _ = font.render(instructions_substr, (255, 255, 255))
        instructions.append(ins)

    bg_image = pygame.image.load(os.path.abspath("table_setting_assets/wood_floor.png")).convert_alpha()
    bg_image = pygame.transform.scale(bg_image, screen.get_rect().size)

    all_objects = [
        FixedTable(pose=[0.5, 0.5, 0.], radius=0.9, img_path="table_setting_assets/tabletop_wood.png"),
    ]

    # Interface state:

    object_being_dragged = None
    drag_start_pos = None
    drag_start_offset = None
    
    object_being_rotated = None
    rotate_start_offset = None
    rotate_start_angle = None

    clock = pygame.time.Clock()
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Add object at current mouse position with
                # random orientation.
                # key 1: Add plate
                # key 2: Add fork
                # key 3: Add knife
                # key 4: Add spoon
                new_object = None
                if event.key == ord('1'):
                    new_object = InteractableDish(pose=np.zeros(3), radius=0.2, img_path="table_setting_assets/plate_red.png")
                elif event.key == ord('2'):
                    new_object = InteractableUtensil(pose=np.zeros(3), size=[0.02, 0.14], img_path="table_setting_assets/fork.png")
                elif event.key == ord('3'):
                    new_object = InteractableUtensil(pose=np.zeros(3), size=[0.015, 0.15], img_path="table_setting_assets/knife.png")
                elif event.key == ord('4'):
                    new_object = InteractableUtensil(pose=np.zeros(3), size=[0.02, 0.12], img_path="table_setting_assets/spoon.png")
                if new_object is not None:
                    new_object.set_center(pygame.mouse.get_pos())
                    new_object.pose[2] = np.random.uniform(0., 2.*np.pi)
                    all_objects.append(new_object)


            elif event.type == pygame.MOUSEBUTTONDOWN:
                # event.button 1: Left -- click and drag to translate
                # event.button 3: Right -- click and drag to rotate
                # event.button 2: Middle -- click to delete object
                if (not object_being_dragged) and (not object_being_rotated):
                    # See if we clicked an object. Objects are prioritized
                    # in the reverse order they appear in the all_objects list
                    # (i.e. in their z-order).
                    selected_object = None
                    for pygame_shape in reversed(all_objects):
                            grab = pygame_shape.query_click(event.pos)
                            if (grab):
                                selected_object = pygame_shape
                                break
                    if selected_object is not None:
                        if event.button == 1:            
                            # Pick up objects.
                            object_being_dragged = selected_object
                            drag_start_pos = np.array(event.pos)
                            drag_start_offset = (selected_object.get_center() - np.array(event.pos))
                        elif event.button == 3:
                            # Rotate object.
                            object_being_rotated = selected_object
                            rotate_start_angle = selected_object.pose[2]
                            offset = selected_object.get_center() - np.array(event.pos)
                            rotate_start_offset = np.arctan2(offset[1], offset[0])
                        elif event.button == 2:
                            # Remove object
                            all_objects.remove(selected_object)


            elif event.type == pygame.MOUSEBUTTONUP:
                # Stop whatever dragging is happening.
                if event.button == 1:
                    object_being_dragged = None
                    drag_start_pos = None
                    drag_start_offset = None
                if event.button == 3:
                    object_being_rotated = None
                    rotate_start_offset = None
                    rotate_start_angle = None

            elif event.type == pygame.MOUSEMOTION:
                # Process dragging motion.
                if object_being_dragged is not None:
                    object_being_dragged.set_center(
                        np.array(event.pos) + drag_start_offset)
                elif object_being_rotated is not None:
                    offset = selected_object.get_center() - np.array(event.pos)
                    rotate_current_offset = np.arctan2(offset[1], offset[0])
                    object_being_rotated.pose[2] = rotate_start_angle - (rotate_current_offset - rotate_start_offset)


        # Background
        #screen.fill(WHITE)
        screen.blit(bg_image, screen.get_rect())
        for i, ins in enumerate(instructions):
            screen.blit(ins, (10, 10 + i * 15))
        for pygame_shape in all_objects:
            pygame_shape.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()