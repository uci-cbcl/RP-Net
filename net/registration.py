import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.nn import functional as F
import numpy as np
import cv2
from torch import nn
from utils.util import normalize
from torchviz import make_dot


"""
Create a one dimensional gaussian kernel matrix
"""
def gaussian_kernel_1d(sigma, asTensor=False, dtype=torch.float32, device='cpu'):

    kernel_size = int(2*np.ceil(sigma*2) + 1)

    x = np.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, num=kernel_size)

    kernel = 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)

    if asTensor:
        kernel = torch.tensor(kernel, dtype=dtype)

        if device != 'cpu':
            kernel = kernel.cuda()

    return kernel


"""
Create a two dimensional gaussian kernel matrix
"""
def gaussian_kernel_2d(sigma, asTensor=False, dtype=torch.float32, device='cpu'):

    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])

    kernel = np.tensordot(y_1, y_2, 0)
    kernel = kernel / np.sum(kernel)

    if asTensor:
        kernel = torch.tensor(kernel, dtype=dtype)

        if device != 'cpu':
            kernel = kernel.cuda()

    return kernel

"""
Create a three dimensional gaussian kernel matrix
"""
def gaussian_kernel_3d(sigma, asTensor=False, dtype=torch.float32, device='cpu'):

    kernel_2d = gaussian_kernel_2d(sigma[:2])
    kernel_1d = gaussian_kernel_1d(sigma[-1])

    kernel = np.tensordot(kernel_2d, kernel_1d, 0)
    kernel = kernel / np.sum(kernel)

    if asTensor:
        kernel = torch.tensor(kernel, dtype=dtype)

        if device != 'cpu':
            kernel = kernel.cuda()

    return kernel


"""
    Create a Gaussian kernel matrix
"""
def gaussian_kernel(sigma, dim=1, asTensor=False, dtype=torch.float32, device='cpu'):

    assert dim > 0 and dim <=3

    if dim == 1:
        return gaussian_kernel_1d(sigma, asTensor=asTensor, dtype=dtype, device=device)
    elif dim == 2:
        return gaussian_kernel_2d(sigma, asTensor=asTensor, dtype=dtype, device=device)
    else:
        return gaussian_kernel_3d(sigma, asTensor=asTensor, dtype=dtype, device=device)

    
class _DemonsRegulariser():
    def __init__(self, pixel_spacing, dtype=torch.float32, device='cpu'):
        super(_DemonsRegulariser, self).__init__()

        self._dtype = dtype
        self._device = device
        self._weight = 1
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self.name = "parent"


class GaussianRegulariser(_DemonsRegulariser):
    def __init__(self, pixel_spacing, sigma, dtype=torch.float32, device='cpu'):
        super(GaussianRegulariser, self).__init__(pixel_spacing, dtype=dtype, device=device)

        sigma = np.array(sigma)

        if sigma.size != self._dim:
            sigma_app = sigma[-1]
            while sigma.size != self._dim:
                sigma = np.append(sigma, sigma_app)


        self._kernel = gaussian_kernel(sigma, self._dim, asTensor=True, dtype=dtype, device=device)

        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1, dtype=int) * -1).tolist()))
        if device != 'cpu':
            self._kernel = self._kernel.to(dtype=dtype).cuda()

        if self._dim == 2:
            self._regulariser = self._regularise_2d
        elif self._dim == 3:
            self._regulariser = self._regularise_3d


    def _regularise_2d(self, data):

#         data.data = data.data.unsqueeze(0)
        data.data = F.conv2d(data.data, self._kernel.contiguous(), padding=self._padding, groups=2)
#         data.data = data.data.squeeze()
        
    def _regularise_3d(self, data):

#         data.data = data.data.unsqueeze(0)
        data.data = F.conv3d(data.data, self._kernel, padding=self._padding, groups=3)
#         data.data = data.data.squeeze()

    def regularise(self, data):
        for parameter in data:
            # no gradient calculation for the demons regularisation
            with torch.no_grad():
                self._regulariser(parameter)


def MSE(y_pred, y_true, mask=None):
    if mask is not None:
        value = torch.mean((y_true - y_pred) ** 2)
        value = torch.masked_select(value, mask)
        
        return value.mean()
    else:
        return torch.mean((y_true - y_pred) ** 2)


def NCC(moving_image_valid, fixed_image_valid, mask=None):
    value = -1.*torch.sum((fixed_image_valid - torch.mean(fixed_image_valid))*(moving_image_valid - torch.mean(moving_image_valid)))\
        /torch.sqrt(torch.sum((fixed_image_valid - torch.mean(fixed_image_valid))**2)*torch.sum((moving_image_valid - torch.mean(moving_image_valid))**2) + 1e-10)
    return value


def _l2_regulariser_2d(displacement, pixel_spacing=[1, 1]):
    displacement = displacement.squeeze(0)
    dx = (displacement[1:, 1:, :] - displacement[:-1, 1:, :]).pow(2) * pixel_spacing[0]
    dy = (displacement[1:, 1:, :] - displacement[1:, :-1, :]).pow(2) * pixel_spacing[1]

    return torch.mean(F.pad(dx + dy, (0, 1, 0, 1)))


def compute_grid(img_size):
    vectors = [torch.arange(0, s) for s in img_size]
    grids = torch.meshgrid(vectors)
    
    # pytorch meshgrid is y, x, so should inverse
    grid = torch.stack(grids[::-1])
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)
    grid.requires_grad = False

    # should be x, y so should inverse the shape
    shape = grid.shape[2:][::-1]
    
    for i in range(len(shape)):
        grid[:, i, ...] = 2 * (grid[:, i, ...] / (shape[i] - 1) - 0.5)
        
    return grid


class Diffeomorphic(nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        
    def forward(self, displacement, grid):
        if len(displacement.shape[2:]) == 2:
            return Diffeomorphic.diffeomorphic_2D(displacement, grid, self.scaling)
        elif len(displacement.shape[2:]) == 3:
            return Diffeomorphic.diffeomorphic_3D(displacement, grid, self.scaling)

    @staticmethod
    def diffeomorphic_2D(displacement, grid, scaling):
        # B*2*H*W
        grid = grid.permute(0, 2, 3, 1).contiguous()
        displacement = displacement / (2 ** scaling)
        
        for i in range(scaling):
            displacement_trans = displacement.transpose(1, 2).transpose(2, 3)
            displacement = displacement + F.grid_sample(displacement, displacement_trans + grid)

        return displacement

    @staticmethod
    def diffeomorphic_3D(displacement, grid, scaling=-1):
        grid = grid.permute(0, 2, 3, 4, 1).contiguous()
        displacement = displacement / (2 ** scaling)

        for i in range(scaling):
            displacement_trans = displacement.transpose(1, 2).transpose(2, 3).transpose(3, 4)
            displacement = displacement + F.grid_sample(displacement, displacement_trans + grid)

        return displacement


class DemonsRegistration(nn.Module):
    def __init__(self, img_size, use_diffeomorphic=False, use_GPU=False):
        super().__init__()
        self.ndim = len(img_size)
        if self.ndim == 2:
            flow = nn.Parameter(torch.zeros([1, 2] + list(img_size)), requires_grad=True)
        elif self.ndim == 3:
            flow = nn.Parameter(torch.zeros([1, 3] + list(img_size)), requires_grad=True)
        else:
            raise NotImplementedError

        self.use_GPU = use_GPU
        flow.data.fill_(0)
        self.flow = flow

        if use_diffeomorphic:
            self.diffeomorphic = Diffeomorphic(10)
        else:
            self.diffeomorphic = None
        
    
    def forward(self, x, grid):
        flow = self.flow

        if self.diffeomorphic is not None:
            flow = self.diffeomorphic(flow, grid)

        new_locs = grid + flow
        if self.ndim == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif self.ndim == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        warped = F.grid_sample(x, new_locs)
                
        return warped

    
    def plot_flow_field(self):
        x = self.grid[0][0].detach().numpy()
        y = self.grid[0][1].detach().numpy()
        u = self.flow[0][0].detach().numpy()
        v = self.flow[0][1].detach().numpy()

        # u, v = np.zeros_like(u) + 0.5, np.zeros_like(v) + 0.5

        plt.figure(figsize=(10,10), dpi=72)
        plt.quiver(y, x, v, u, scale_units='xy', scale=1, width=0.001)
        plt.show()
        
        
    def plot_flow_magnitude(self):
        flow = self.flow
        if self.use_GPU:
            flow = flow.cpu()
            
        u = flow[0][0].detach().numpy()
        v = flow[0][1].detach().numpy()

        # u, v = np.zeros_like(u) + 0.5, np.zeros_like(v) + 0.5

        plt.figure(figsize=(10,10), dpi=72)
        plt.imshow(np.sqrt(u**2 + v**2), cmap='jet')
        plt.show()


    def train_registraion(self, moving, fixed, optimizer, loss_fn=MSE, regulariser=None, iters=200, regularise_displacement=False, verbose=False):
        img_size = moving.shape[2:]
        for i in range(iters):
            grid = compute_grid(img_size)
            if self.use_GPU:
                grid = grid.cuda()

            optimizer.zero_grad()
            warped = self.forward(moving, grid)
            
            loss = loss_fn(warped, fixed, mask=None)
            if regularise_displacement:
                loss += _l2_regulariser_2d(self.flow)

            loss.backward()
            optimizer.step()

            if regulariser is not None:        
                regulariser.regularise(self.parameters())

            if verbose:
                print(f'Demons registration train {i}, loss {loss}')


        
class AffineRegistration(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        if len(img_size) == 2:
            self.theta = nn.Parameter(torch.zeros((1, 2, 3)))
            self.theta.data[0][0][0] = 1
            self.theta.data[0][1][1] = 1
            
            self._stop_shear = torch.Tensor([[[1,0,1],[0,1,1]]])
        elif len(img_size) == 3:
            self.theta = nn.Parameter(torch.zeros((1, 3, 4)))
            self.theta.data[0][0][0] = 1
            self.theta.data[0][1][1] = 1
            self.theta.data[0][2][2] = 1
            
            self._stop_shear = torch.Tensor([[[1,0,0,1],[0,1,0,1],[0,0,1,1]]])
        else:
            raise NotImplementedError

        self.register_buffer('stop_shear', self._stop_shear)
    
    def forward(self, x, stop_shear=False):
        theta = self.theta 

        if stop_shear:
            theta = theta * self.stop_shear
        grid = F.affine_grid(theta, x.size())
        
        return F.grid_sample(x, grid)


    def train_registraion(self, moving, fixed, optimizer, loss_fn=MSE, iters=20, verbose=False):
        for i in range(iters):
            optimizer.zero_grad()
            warped = self.forward(moving)
            
            loss = loss_fn(warped, fixed, mask=None)
            loss.backward()
            optimizer.step()

            if verbose:
                print(f'Affine registration train {i}, loss {loss}')


class DEEDSRegistration(nn.Module):
    def __init__(self, grid_size=128, disp_range=0.1, displacement_width=15, use_GPU=False, mode='nearest'):
        super().__init__()

        self.grid_size = grid_size
        self.disp_range = disp_range#0.25
        self.displacement_width = displacement_width#11#17
        self.sample_grid = None

        self.alpha = torch.nn.Parameter(torch.Tensor([1,.1,1,0,.1,10]))
        self.pad1 = torch.nn.ReplicationPad2d(3)
        self.avg1 = torch.nn.AvgPool2d(3,stride=1)
        self.max1 = torch.nn.MaxPool2d(3,stride=1)
        self.pad2 = torch.nn.ReplicationPad2d(2)
        self.mode = mode
        
    
    def forward(self, img):
        assert self.sample_grid is not None
        warped = F.grid_sample(img, self.sample_grid)

        return warped

    
    def plot_flow_field(self):
        x = self.grid[0][0].detach().numpy()
        y = self.grid[0][1].detach().numpy()
        u = self.flow[0][0].detach().numpy()
        v = self.flow[0][1].detach().numpy()

        # u, v = np.zeros_like(u) + 0.5, np.zeros_like(v) + 0.5

        plt.figure(figsize=(10,10), dpi=72)
        plt.quiver(y, x, v, u, scale_units='xy', scale=1, width=0.001)
        plt.show()
        
        
    def plot_flow_magnitude(self):
        flow = self.flow
        if self.use_GPU:
            flow = flow.cpu()
            
        u = flow[0][0].detach().numpy()
        v = flow[0][1].detach().numpy()

        # u, v = np.zeros_like(u) + 0.5, np.zeros_like(v) + 0.5

        plt.figure(figsize=(10,10), dpi=72)
        plt.imshow(np.sqrt(u**2 + v**2), cmap='jet')
        plt.show()


    def train_registraion(self, moving, fixed, verbose=False):
        img_size = moving.shape[2:]

        alpha0 = self.alpha[0]
        alpha1 = self.alpha[1]
        alpha2 = self.alpha[2]
        alpha3 = self.alpha[3]
        alpha4 = self.alpha[4]
        alpha5 = self.alpha[5]
        _, _, H, W = moving.shape

        grid_size = self.grid_size
        disp_range = self.disp_range
        displacement_width = self.displacement_width
        grid_xyz = F.affine_grid(torch.eye(2,3).unsqueeze(0),(1,1,grid_size,grid_size)).to(moving.device)
        shift_xyz = F.affine_grid(
            disp_range*torch.eye(2,3).unsqueeze(0),
            (1,1,displacement_width,displacement_width)
        ).to(moving.device)

        new_grid = grid_xyz.view(1,-1,1,2) + shift_xyz.view(1,1,-1,2)

        moving_grid = F.grid_sample(moving, new_grid)

        fixed_grid = F.grid_sample(fixed, grid_xyz.view(1,-1,1,2))

        deeds_cost = alpha1 + alpha0 * torch.pow(fixed_grid-moving_grid,2)
        deeds_cost = deeds_cost.view(1,-1,displacement_width,displacement_width)

        pad1 = self.pad1
        avg1 = self.avg1
        max1 = self.max1
        pad2 = self.pad2
        
        # approximate min convolution / displacement compatibility
        cost = avg1(avg1(-max1(-pad1(deeds_cost))))
        
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,0,1).view(1,displacement_width**2,grid_size,grid_size)
        cost_avg = avg1(avg1(pad2(cost_permute))).permute(0,2,3,1).view(1,-1,displacement_width,displacement_width)

        # second path
        cost = alpha4 + alpha2 * deeds_cost + alpha3 * cost_avg
        cost = avg1(avg1(-max1(-pad1(cost))))

        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,0,1).view(1,displacement_width**2,grid_size,grid_size)
        cost_avg = avg1(avg1(pad2(cost_permute))).permute(0,2,3,1).view(grid_size**2,displacement_width**2)

        cost_soft = F.softmax(-alpha5*cost_avg,1)
        pred_xyz = torch.sum(cost_soft.unsqueeze(2)*shift_xyz.view(1,-1,2),1)
        pred_xyz = pred_xyz.view(1,grid_size,grid_size,2)


        shift = pred_xyz.view(1, grid_size, grid_size, 2)

        new_grid = grid_xyz + shift
        new_grid = F.upsample(new_grid.permute(0, 3, 1, 2).contiguous(), size=(H, W), mode=self.mode).permute(0, 2, 3, 1)

        self.sample_grid = new_grid
                    

class AffineDemonsRegistration(nn.Module):
    def __init__(self, img_size, use_diffeomorphic=False, use_GPU=False, stop_shear=False):
        super().__init__()
        self.affine_reg = AffineRegistration(img_size)
        self.demons = DemonsRegistration(img_size, use_GPU=use_GPU, use_diffeomorphic=use_diffeomorphic)
        self.use_GPU = use_GPU
        self.stop_shear = stop_shear
        
    
    def forward(self, x, grid):
        x = self.affine_reg(x, stop_shear=self.stop_shear)
        x = self.demons(x, grid)

        return x


    def train_registraion(self, moving, fixed, optimizers, loss_fn=[MSE, NCC], regulariser=None, iters=[100, 100], regularise_displacement=False, verbose=False):
        self.affine_reg.train_registraion(moving, fixed, optimizers[0], loss_fn=loss_fn[0], iters=iters[0], verbose=verbose)
        affined_moving = self.affine_reg(moving).detach()
        self.demons.train_registraion(
            affined_moving, 
            fixed, 
            optimizers[1], 
            loss_fn=loss_fn[1], 
            regulariser=regulariser,
            iters=iters[1],
            regularise_displacement=regularise_displacement,
            verbose=verbose
        )


class AffineDEEDSRegistration(nn.Module):
    def __init__(self, img_size, use_diffeomorphic=False, use_GPU=False, stop_shear=False):
        super().__init__()
        self.affine_reg = AffineRegistration(img_size)
        self.deeds = DEEDSRegistration()
        self.use_GPU = use_GPU
        self.stop_shear = stop_shear
        
    
    def forward(self, x, grid):
        x = self.affine_reg(x, stop_shear=self.stop_shear)
        x = self.deeds(x)

        return x


    def train_registraion(self, moving, fixed, optimizers, loss_fn=[MSE, NCC], regulariser=None, iters=[100, 100], regularise_displacement=False, verbose=False):
        self.affine_reg.train_registraion(moving, fixed, optimizers[0], loss_fn=loss_fn[0], iters=iters[0], verbose=verbose)
        affined_moving = self.affine_reg(moving).detach()
        self.deeds.train_registraion(affined_moving, fixed)