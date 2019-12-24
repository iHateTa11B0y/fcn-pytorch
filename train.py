from coco import COCODataset
from generalized_fcn import GeneralizedFCN
from torch import optim
from transforms import build_transforms

def train_net(model, 
              data_loader,
              optimizer,
              device,
              epoch,
              ):
    model.train()
    for i in range(epoch):
        for iteration, (images, targets, _) in enumerate(data_loader):
            print(images.shape) 
            images = images.to(device)
            targets = targets.to(device)
            loss = model(images, targets) 
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % 20:
                print('iter: {}, loss: {}'.format(iteration, loss.item()))

def train():
    json_file = "/core1/data/home/liuhuawei/data-manager/data/background/coco_background_all_train_new.json"
    lr = 0.1
    epoch = 10
    model = GeneralizedFCN()
    data_loader = COCODataset(json_file, '', True, transforms=build_transforms(), clamp=False)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8)     
    train_net(model, data_loader, optimizer, 'cpu', epoch)

if __name__=='__main__':
    train()
        
