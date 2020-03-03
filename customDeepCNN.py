import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,no_input_channels):
        dropout_value = 0.1
        super(Net, self).__init__()
         # Input Block
        self.inputblock = nn.Sequential(
            nn.Conv2d(in_channels=no_input_channels, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),   # output_size = 30, RF = 3
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        ) 

        # CONVOLUTION BLOCK 1 - DepthwiseSeparable
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=4, bias=False, groups = 32),     # output_size = 36, RF = 5
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=4, bias=False),                  # output_size = 44, RF = 5
            nn.BatchNorm2d(32), 
            nn.Dropout(dropout_value),
            nn.ReLU(),

        ) 

        # TRANSITION BLOCK 1
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2) ,                                                                                    # output_size = 22, RF = 6
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),                  # output_size = 22, RF = 6
        ) 
        # CONVOLUTION BLOCK 2 - Dilated Conv block
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=2, bias=False, dilation=2),      # output_size = 22, RF = 14
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2),      # output_size = 22, RF = 22
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, bias=False, dilation=2),     # output_size = 22, RF = 30
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )
        
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2) ,                                                                                    # output_size = 11, RF = 32
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),                 # output_size = 11, RF = 32
        ) 
            
        # CONVOLUTION BLOCK 3 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2 ),     # output_size = 11, RF = 48
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),            
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, bias=False, dilation=2 ),     # output_size = 11, RF = 64
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            nn.ReLU(),            
        ) 

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)                                                                             # output_size = 1, RF = 96
        ) 
        # CONVOLUTION BLOCK 4 
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),                  # output_size = 1, RF = 96
            # nn.ReLU() NEVER!
        ) 


    def forward(self, x):
        x = self.inputblock(x)
        x = self.convblock1(x)
        x = self.transition1(x)
        x = self.convblock2(x)
        x = self.transition2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.convblock4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
        
        
        
        