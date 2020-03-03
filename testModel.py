def test(model, device, test_loader,epoch,loss_type,plot_flag, plot_after_epochs):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    
    test_losses = []
    test_acc = []
    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if epoch >=plot_after_epochs and plot_flag:
      figure = plt.figure(figsize=(15,20))
      figure.suptitle('Test Data : Misclassification', fontsize=16)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            x = torch.tensor((~pred.eq(target.view_as(pred))), dtype=torch.float, device = device).clone().detach().requires_grad_(True)
    
            if epoch>=plot_after_epochs and plot_flag :
                for index,image in enumerate(data):      
                    if x[index] == 1 and count < 25:
                      plt.subplot(5, 5, count+1)
                      plt.axis('off')
                      plt.title("Classified as : "+ str(classes[pred[index].item()])+"\n Label : "+ str(classes[target.view_as(pred)[index].item()]))
                      data[index][0] = data[index][0]*0.24703223 + 0.49139968
                      data[index][1] = data[index][1]*0.24348513 + 0.48215841
                      data[index][2] = data[index][2]*0.26158784 + 0.44653091
                      npimg = data[index].cpu().numpy()
                      plt.imshow(np.transpose(npimg, (1, 2, 0)))
                      count = count+1   

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return (test_acc,test_losses)
    
      
      