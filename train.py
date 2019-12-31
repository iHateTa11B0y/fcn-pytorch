from coco import COCODataset
import torch
from generalized_fcn import GeneralizedFCN
from torch import optim
from transforms import build_transforms
from torch.utils.data import DataLoader, random_split
from optimizer import OptimizerScheduler
from logger import Logger


def train_net(model, 
              data_loader,
              optimizer,
              device,
              epoch,
              logg,
              ):
    model.to(device)
    model.train()
    for i in range(epoch):
        epoch_iter = len(data_loader)
        for iteration, (images, targets, _) in enumerate(data_loader):
            
            images = images.to(device)
            targets = targets.to(device)
            loss = model(images, targets) 
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(iteration)
            
            logg.log(i * epoch_iter + iteration, loss=loss.item())

            if iteration % 10 == 0:
                logg.print(flush=True)
        torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(i))

def train():
    json_file = "/core1/data/home/niuwenhao/data/tiny_bg.json"
    lr = 0.1
    epoch = 10
    val_ratio = 0.1
    batch_size = 2
    steps = [100,200,300]

    logg = Logger()
    logg.add('loss')

    model = GeneralizedFCN()

    data_set = COCODataset(json_file, '', True, transforms=build_transforms(), clamp=False)

    val_num = int(val_ratio * len(data_set))
    train_num = len(data_set) - val_num
    train, val = random_split(data_set, [train_num, val_num])
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    optimizer = OptimizerScheduler(model, base_lr=lr, steps=steps)     
    train_net(model, train_dataloader, optimizer, 'cuda', epoch, logg)

if __name__=='__main__':
    train()
        
