import torch
import pyro

device = torch.device('cpu')

# Use Spatial Transformer Network strategy from the 
# Pyro AIR tutorial to construct a differentiable
# planar box-drawing system.

expansion_indices = torch.LongTensor([0, 1, 2, 3, 4, 5])
def convert_pose_to_matrix(pose):
    # Converts a [x, y, theta] pose to a tf matrix:
    # [[cos(theta) -sin(theta) x]]
    #  [sin(theta) cos(theta)  y]]
    # (except in vectorized form -- so input is n x 3,
    #  output is n x 2 x 3)
    n = pose.size(0)
    out = torch.cat((torch.cos(pose[:, 0]).view(n, -1),
    				 -torch.sin(pose[:, 0]).view(n, -1),
    				 pose[:, 1].view(n, -1),
    				 torch.sin(pose[:, 0]).view(n, -1),
    				 torch.cos(pose[:, 0]).view(n, -1),
    				 pose[:, 2].view(n, -1)), 1)
    out = out.view(n, 2, 3)
    return out

poses = torch.FloatTensor([[0., 1., 2.],
						   [1.57, 4., 5.]]).view(2, -1)
print poses
print convert_pose_to_matrix(poses)

def draw_sprites_at_poses(pose, sprite_size_x, sprite_size_y, image_size_x, image_size_y, sprites):
    n = sprites.size(0)
    assert sprites.size(1) == sprite_size_x * sprite_size_y, 'Size mismatch.'
    theta = convert_pose_to_matrix(pose)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size_x, image_size_y)))
    out = F.grid_sample(sprites.view(n, 1, sprite_size_x, sprite_size_y), grid)
    return out.view(n, image_size_x, image_size_y)

print("
	LOAD A SPRITE FROM AN IMAGE
	TRY OUT DRAWING IT AT A POSE!"
