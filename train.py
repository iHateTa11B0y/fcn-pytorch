from coco import COCODataset
import torch
from generalized_fcn import GeneralizedFCN
from torch import optim
from transforms import build_transforms

def train_net(model, 
              data_loader,
              optimizer,
              device,
              epoch,
              ):
    model.to(device)
    model.train()
    for i in range(epoch):
        for iteration, (images, targets, _) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.to(device)
            loss = model(images, targets) 
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % 20 == 0:
                print('[epoch/iter]: [{:<4d}/{:<4d}], loss: {}'.format(i, iteration, loss.item()))
        torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(i))

def train():
    json_file = "/home/core/tiny_bg.json"
    lr = 0.1
    epoch = 10
    model = GeneralizedFCN()
    data_loader = COCODataset(json_file, '', True, transforms=build_transforms(), clamp=False)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8)     
    train_net(model, data_loader, optimizer, 'cuda', epoch)

if __name__=='__main__':
    train()
        
