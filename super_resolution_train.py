import torch
from torch.utils.data import DataLoader
from super_resolution import SuperResDataset, SuperResModel
from torch.utils.tensorboard import SummaryWriter
import lpips
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def tensor2numpy_image(image_tensor):
    """
    Convert a tensor (N, C, H, W) to a numpy array (H, W, C) [for plotting with plt.imshow()]
    :param image_tensor: input image in tensor form
    :return: numpy array
    """
    assert torch.min(image_tensor) >= -1 and torch.max(image_tensor) <= 1, f"min: {torch.min(image_tensor)}, max: {torch.max(image_tensor)}"
    return np.array(image_tensor.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5)

def img2tensor(image_path, resize_dims=(64, 64)):
    """
    Read image from disk, return pytorch tensor (N, C, H, W)
    :param image_path: path to image on disk
    :return: numpy array
    """
    target_image = Image.open(image_path)
    target_image = target_image.resize(resize_dims)
    target_image = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Rescale to [0, 1]
    target_image = (target_image - 0.5) / 0.5  # Rescale [0, 1] to [-1, 1]
    return target_image

def make_progress_pic(low_res, bicubic, enhanced, high_res, save_path):
    assert len(low_res) == len(high_res)
    num_imgs = len(low_res)
    fig, ax = plt.subplots(num_imgs, 4, figsize=(12, 8))
    for i, axis in enumerate(ax):
        axis[0].imshow(tensor2numpy_image(low_res[i].unsqueeze(0).cpu()))
        axis[0].set_title("Low Resolution Input")
        axis[1].imshow(tensor2numpy_image(bicubic[i].unsqueeze(0).cpu()))
        axis[1].set_title("Bicubic Upsample")
        axis[2].imshow(tensor2numpy_image(enhanced[i].unsqueeze(0).cpu()))
        axis[2].set_title("Model Output")
        axis[3].imshow(tensor2numpy_image(high_res[i].unsqueeze(0).cpu()))
        axis[3].set_title("Ground Truth High Res")
    fig.savefig(save_path)

if __name__ == '__main__':
    logger = SummaryWriter()

    batch_size = 8
    num_workers = 4
    num_epochs = 100
    lr = 1e-3
    mse_loss_factor = 0.5
    lpips_loss_factor = (1-mse_loss_factor)
    run_name = "superres_trial1"

    logger.add_text("hparams/batch_size", str(batch_size))
    logger.add_text("hparams/num_workers", str(num_epochs))
    logger.add_text("hparams/lr", str(lr))
    logger.add_text("hparams/run_name", run_name)
    logger.add_text("hparams/mse_loss_factor", str(mse_loss_factor))
    logger.add_text("hparams/lpips_loss_factor", str(lpips_loss_factor))

    Path(f'./saves/{run_name}').mkdir(parents=True, exist_ok=True)
    save_path = f'./saves/{run_name}'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = SuperResDataset("./data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = SuperResModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    batch_iters = 1

    lpips_loss = lpips.LPIPS(net="vgg").to(device)

    mse_loss_hist = []
    lpips_loss_hist = []
    loss_hist = []

    # Get images for progress pics
    display_batch = next(iter(dataloader))

    for epoch in range(0, num_epochs):
        for batch_num, batch_data in enumerate(dataloader):
            model.train()

            low_res, high_res = batch_data
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            optim.zero_grad()

            preds = model(low_res)

            # Loss Functions (https://medium.com/beyondminds/an-introduction-to-super-resolution-using-deep-learning-f60aff9a499d)
            mse_loss = mse_loss_factor * torch.pow(high_res - preds, 2).mean()
            mse_loss_hist.append(mse_loss.item())

            lpips_loss_val = lpips_loss_factor * lpips_loss(high_res, preds).mean()
            lpips_loss_hist.append(lpips_loss_val.item())

            loss = mse_loss + lpips_loss_val
            loss_hist.append(loss.item())

            loss.backward()

            optim.step()

            if batch_num % 100 == 0:
                torch.save({
                            'epoch': epoch,
                            'batch_iters': batch_iters,
                            'optim_params': optim.state_dict(),
                            'model_params': model.state_dict(),
                            'seed': torch.seed(),
                        }, f'{save_path}/iter{batch_iters}.save')


                model.eval()
                with torch.no_grad():
                    output = model(display_batch[0][:2].to(device))
                    bicubic = model.upsample(display_batch[0][:2].to(device)).clamp(-1, 1)
                make_progress_pic(display_batch[0][:2], bicubic, output, display_batch[1][:2], f'{save_path}/iter{batch_iters}.png')
                logger.add_image("progress_pic", np.array(Image.open(f'{save_path}/iter{batch_iters}.png')), batch_iters, dataformats='HWC')

                logger.add_scalar("metrics/loss", np.array(loss_hist).mean(), batch_iters)
                logger.add_scalar("metrics/loss_mse", np.array(mse_loss_hist).mean(), batch_iters)
                logger.add_scalar("metrics/loss_lpips", np.array(lpips_loss_hist).mean(), batch_iters)
                print(f"Iter: {batch_iters}, Loss: {np.array(lpips_loss_hist).mean()}, loss_mse: {np.array(mse_loss_hist).mean()}, loss_lpips: {np.array(loss_hist).mean()}")
                loss_hist.clear()
                mse_loss_hist.clear()
                lpips_loss_hist.clear()

            batch_iters += 1
        print(f"FINISHED EPOCH {epoch}")