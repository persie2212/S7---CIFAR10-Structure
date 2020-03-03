def train(model, device, train_loader, optimizer, epoch, loss_type, plot_flag, plot_after_epochs):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    count = 0
    loss = 0
    
    train_losses = []
    train_acc = []
    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if epoch >= plot_after_epochs and plot_flag:
        figure = plt.figure(figsize=(15,20))
        figure.suptitle('Training Data : Misclassification', fontsize=16)
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
    
        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
    
        # Predict
        y_pred = model(data)
        # print(y_pred)
    
        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        # print(loss)
    
        # if loss_type == 'Normal' or loss_type == 'L2':
        #   train_losses.append(np.sum(loss.item()))
            
        if loss_type == 'L1':
          # l1_crit =  nn.L1Loss(size_average=False,reduction='sum')
          reg_loss = 0
          for param in model.parameters():
            reg_loss += torch.sum(abs(param))
          factor = 0.0005
          loss += factor * reg_loss 
        #   train_losses.append(np.sum(loss.item()))
        # elif loss_type == 'L2':
        #   train_losses.append(loss) 
        elif loss_type == 'ElastiNet':
          # l1_crit = nn.L1Loss(size_average=False,reduction='sum')
          reg_loss = 0
          for param in model.parameters():
            reg_loss += torch.sum(abs(param)) #l1_crit(param).item()
          factor = 0.0005
          loss += factor * reg_loss 
        #   train_losses.append(np.sum(loss.item()))
    
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
        x = torch.tensor((~pred.eq(target.view_as(pred))), dtype=torch.float, device = device).clone().detach().requires_grad_(True)
        
        for index,image in enumerate(data):      
            if x[index] == 1 and count < 25 and epoch>=plot_after_epochs and plot_flag:
              plt.subplot(5, 5, count+1)
              plt.axis('off')
              plt.title("Classified as : "+ str(classes[pred[index].item()])+"\n Label : "+ str(classes[target.view_as(pred)[index].item()]))
              data[index][0] = data[index][0]*0.24703223 + 0.49139968
              data[index][1] = data[index][1]*0.24348513 + 0.48215841
              data[index][2] = data[index][2]*0.26158784 + 0.44653091
              npimg = data[index].cpu().numpy()
              plt.imshow(np.transpose(npimg, (1, 2, 0)))
              count = count+1        
    
    # loss /= len(train_loader.dataset)
    train_losses.append(loss)
    train_acc.append(100*correct/processed)
        
    return (train_acc,train_losses)
          