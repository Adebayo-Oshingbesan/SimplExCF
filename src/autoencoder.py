import torch
    
class AutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim, out, mask):
        super(AutoEncoder, self).__init__()
        self.mask = mask
        self.lin1 = torch.nn.Linear(latent_dim, latent_dim//2)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(latent_dim//2, out)
        
    def forward(self, inp, inp_latent_approx, eps):
        outp = inp_latent_approx + eps #
        outp = self.lin1(outp)
        outp = self.relu(outp)
        outp = self.lin2(outp)
        outp = self.mask * outp + (1 - self.mask) * inp.reshape(*outp.shape)
        return outp