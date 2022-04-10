import numpy as np
import torch
nn = torch.nn
F = nn.functional

from models.common import positional_encoding

class INR2d(nn.Module):
    def __init__(self, C=256, num_freqs=6):
        super().__init__()
        self.encode = positional_encoding
        self.num_freqs = num_freqs
        self.layer1 = nn.Linear(4 * num_freqs, C)
        self.layer2 = nn.Linear(C, C)
        self.layer3 = nn.Linear(C, C)
        self.layer4 = nn.Linear(C, 3)

    def forward(self, coords):
        x = self.encode(coords, num_freqs=self.num_freqs)
        x = F.relu(self.layer1(x), inplace=True)
        x = F.relu(self.layer2(x), inplace=True)
        x = F.relu(self.layer3(x), inplace=True)
        return self.layer4(x), coords

    def produce_image(self):
        mg = torch.meshgrid(torch.arange(H).cuda(), torch.arange(W).cuda(), indexing='xy')
        coords = torch.stack(mg, dim=-1).reshape(-1, 2)
        coords = common.positional_encoding(coords, include_input=False)
        rgb = self.forward(coords)
        return rgb.reshape(1,3,H,W)



class Siren(nn.Module):
    def __init__(self, C=256, in_channels=2, out_channels=3, layers=3, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = [SineLayer(in_channels, C, 
                      is_first=True, omega_0=first_omega_0)]

        for i in range(layers):
            self.net.append(SineLayer(C, C, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(C, out_channels)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / C) / hidden_omega_0, 
                                              np.sqrt(6 / C) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(C, out_channels, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def produce_image(self):
        tensors = [torch.linspace(-1, 1, steps=H), torch.linspace(-1, 1, steps=W)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        coords = mgrid.reshape(-1, 2)
        rgb,_ = self.forward(coords)
        return rgb.reshape(1,3,H,W)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))



from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image
import skimage
class ImageFitting(Dataset):
    def __init__(self, img):
        super().__init__()
        #img = Image.fromarray(img)
        transform = Compose([
            #ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        img = transform(img)
        H,W = img.shape[-2:]
        self.pixels = img.permute(2,3, 0,1).view(-1, 3)
        tensors = [torch.linspace(-1, 1, steps=H), torch.linspace(-1, 1, steps=W)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        self.coords = mgrid.reshape(-1, 2)
        self.H,self.W = H,W

    def __len__(self):
        return 1
    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels


