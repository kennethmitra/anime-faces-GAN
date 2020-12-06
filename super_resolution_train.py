import torch
from torch.utils.data import DataLoader
from super_resolution import SuperResDataset, SuperResModel

batch_size = 64
num_workers = 4
num_epochs = 100
lr = 1e-3

save_path = "superRes_save"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

dataset = SuperResDataset("./data")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = SuperResModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

batch_iters = 0

for epoch in range(num_epochs):
    for batch_num, batch_data in enumerate(dataloader):
        
        optim.zero_grad()

        preds = model(batch_data[0])
        loss = torch.pow(batch_data[1] - preds, 2) 
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
        
        batch_iters += 1