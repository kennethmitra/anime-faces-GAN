import torch
from dcgan import Generator
from latent_optimization import tensor2numpy_image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from pathlib import Path
import numpy as np
from super_resolution import SuperResModel

if __name__ == '__main__':
    torch.manual_seed(420)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Set up Generator
    generator = Generator().to(device)
    save = torch.load('iter57300.save')
    generator.load_state_dict(save['gen_params'])
    generator.eval()

    superres_model = SuperResModel().to(device)
    superres_model.load_state_dict(torch.load('saves/superres_trial1/iter27177.save')['model_params'])

    z_vec = torch.randn(1, 100, 1, 1, device=device).clone().detach()

    vec1 = torch.load('results/vec1.save')
    vec2 = torch.load('results/vec2.save')
    save_dir = "results/interpolation2/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    i = 1
    for t in np.linspace(0, 1, 300):
        vec = t * vec1 + (1-t) * vec2

        with torch.no_grad():
            image_tens = generator(vec)
            image_upsample = superres_model(image_tens)

        assert torch.min(image_upsample) >= -1 and torch.max(image_upsample) <= 1, f"{torch.min(image_upsample)}, {torch,max(image_upsample)}"
        save_image(image_upsample.cpu(), f"{save_dir}/frame_{i}.png", range=(-1, 1))
        print(f"Frame {i}, t={t}")
        i += 1