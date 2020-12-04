import torch
from dcgan import Generator
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class LatentOptim:
    def __init__(self, generator, z_size, lr, device, target_image, logProbWeight=0):
        self.generator = generator
        self.z_size = z_size
        self.z_pdf = torch.distributions.Normal(0, 1)

        self.z_vec = torch.randn(1, z_size, 1, 1, device=device).clone().detach()
        self.z_vec.requires_grad = True

        self.params = [self.z_vec]
        self.optim = torch.optim.Adam(self.params, lr=lr)
        #self.optim = torch.optim.SGD(self.params, lr=lr, momentum=0.9)
        #self.scheduler = CosineAnnealingWarmRestarts(self.optim, 100, 1)
        self.target_image = target_image

        self.logProbWeight = logProbWeight

    def step(self):
        self.optim.zero_grad()

        # Compute Loss
        image = self.generator(self.z_vec)

        # Step optimizer
        logProb = self.z_pdf.log_prob(self.z_vec).mean()  # From https://github.com/ToniCreswell/InvertingGAN/blob/master/scripts/invert.py
        loss = torch.mean((self.target_image - image)**2) - self.logProbWeight * logProb
        print(f"Loss: {loss}")

        loss.backward()
        self.optim.step()
        #self.scheduler.step()


if __name__ == '__main__':
    # Get GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Set up Generator
    generator = Generator().to(device)
    save = torch.load('iter45300.save')
    generator.load_state_dict(save['gen_params'])
    generator.eval()

    # Load target image
    target_image = Image.open('target.png')
    target_image = target_image.resize((64, 64))
    target_image = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).double() / 255.0  # Rescale to [0, 1]
    target_image = (target_image - 0.5) / 0.5  # Rescale [0, 1] to [-1, 1]
    print(target_image.shape)

    latent_optim = LatentOptim(generator=generator, z_size=100, lr=0.1, device=device, target_image=target_image.to(device))

    for i in range(1, 10000):
        print(f"{i}: ", end="")
        latent_optim.step()

    image = generator(latent_optim.z_vec).cpu().detach()
    plt.imshow(image.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()
    print(torch.min(image), torch.max(image))