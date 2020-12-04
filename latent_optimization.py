import torch
from dcgan import Generator
from PIL import Image

class LatentOptim:
    def __init__(self, generator, z_size, lr, device, target_image):
        self.generator = generator
        self.z_size = z_size
        self.z_vec = torch.randn(1, z_size, 1, 1, device=device)
        self.params = [self.z_vec]
        self.optim = torch.optim.Adam(self.params, lr=lr)
        self.target_image = target_image

    def step(self):
        print("-------------")
        print(f"z_vec: {self.z_vec}")
        # Compute Loss
        image = self.generator(self.z_vec)

        # Step optimizer
        loss = torch.mean((self.target_image - image)**2)
        print(f"Loss: {loss}")

        loss.backward()
        self.optim.step()

        print(f"z_vec: {self.z_vec}")
        print("---------------")

if __name__ == '__main__':
    # Get GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Set up Generator
    generator = Generator().to(device)
    save = torch.load('iter44600.save')
    generator.load_state_dict(save['gen_params'])

    # Load target image
    image = Image.open('dimakis-alex.jpg')
    image = image.convert('L')
    image = image.resize((64, 64))

    latent_optim = LatentOptim(generator=generator, z_size=100, lr=0.1, device=device, target_image=)