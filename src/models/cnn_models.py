import torch
import torch.nn as nn
import torch.nn.functional as F


class OneD_ResCNN(nn.Module):
    def __init__(self, input_shape):
        super(OneD_ResCNN, self).__init__()
        self.conv_initial = nn.Conv1d(input_shape, 32, 5, padding='same')
        self.bn_initial = nn.BatchNorm1d(32)
        self.resblock1 = ResidualBlock_OneD_ResCNN(32, 32, 3)
        self.resblock2 = ResidualBlock_OneD_ResCNN(32, 32, 5)
        self.resblock3 = ResidualBlock_OneD_ResCNN(32, 32, 7)
        self.conv_final = nn.Conv1d(96, 32, 1, padding='same')
        self.bn_final = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 19)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = F.relu(self.bn_initial(self.conv_initial(x)))  
        x1 = self.resblock1(x)
        x2 = self.resblock2(x)
        x3 = self.resblock3(x)


        x = torch.cat((x1, x2, x3), dim=1)
    

        x = F.relu(self.bn_final(self.conv_final(x)))

        x = x.permute(0, 2, 1)  # Change (batch, 32, 1280) → (batch, 1280, 32)

        x = self.fc(x)  # Should output (batch, 1280, 19)
        #print(f"Shape after fc: {x.shape}")  # Expect (batch, 1280, 19)

        return x

        
#Modelo do Fábio 
class OneD_DCNN(nn.Module): 
    def __init__(self, input_channels):
        super(OneD_DCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, input_channels, kernel_size=3, stride=1, padding="same"),
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x=self.model(x)
        x = x.permute(0, 2, 1)
        return x





class ReslBlock_RestNet_1D_DCNN(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride=1):
        super(ReslBlock_RestNet_1D_DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.1)  

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.1)  

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels)



        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride,padding="same"),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
	    
        return out

class RestNet_1D_DCNN(nn.Module):
    def __init__(self, input_channels):
        super(RestNet_1D_DCNN, self).__init__()
        # self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=9, stride=1, padding="same")
        # self.bn1 = nn.BatchNorm1d(32)
        # self.relu = nn.LeakyReLU(0.2)
        # self.dropout = nn.Dropout(0.1)

        self.res_block1 = ReslBlock_RestNet_1D_DCNN(input_channels, 32,7)   
        self.res_block2 = ReslBlock_RestNet_1D_DCNN(32, 64,5)   
        self.res_block3 = ReslBlock_RestNet_1D_DCNN(64, 128,3)

        self.final = nn.Conv1d(128, input_channels, kernel_size=3, stride=1, padding="same")
       
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        # x = self.relu(self.bn1(self.conv1(x)))
        # x=self.dropout(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.final(x)
        x = x.permute(0, 2, 1)
        return x





