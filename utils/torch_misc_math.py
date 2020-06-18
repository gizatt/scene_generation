import torch

def skew_symmetric_batch(p):
    out = torch.zeros(p.shape[0], 3, 3).to(p.device)
    out[:, 0, 1] = -p[:, 2]
    out[:, 0, 2] = p[:, 1]
    out[:, 1, 0] = p[:, 2]
    out[:, 1, 2] = -p[:, 0]
    out[:, 2, 0] = -p[:, 1]
    out[:, 2, 1] = p[:, 0]
    return out

def rotation_matrix_from_two_vectors(p, q):
    batch_size = p.shape[0]
    assert(p.shape[1] == 3)
    assert(p.shape == q.shape)
    assert(p.device == q.device)

    r = torch.cross(p, q)
    rx = skew_symmetric_batch(r)
    R = torch.eye(3).reshape((1, 3, 3)).repeat(batch_size, 1, 1).to(p.device)
    R = R + rx
    R = R + torch.matmul(rx, rx) / (1. + torch.sum(p*q, dim=-1)).view(batch_size, 1, 1)
    return R

if __name__ == "__main__":
    p = torch.tensor([0., 0., 1.]).view(1, -1)
    q = torch.tensor([1., 0., 0.]).view(1, -1)
    print("Output: ", rotation_matrix_from_two_vectors(p, q))

    p = p.repeat([5, 1])
    q = q.repeat([5, 1])
    rotation_matrix_from_two_vectors(p, q)

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)

https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def conv_t_2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = (h_w[0] - 1)*stride - 2*pad + dilation*(kernel_size[0]-1) + pad + 1
    w = (h_w[1] - 1)*stride - 2*pad + dilation*(kernel_size[1]-1) + pad + 1
    return h, w