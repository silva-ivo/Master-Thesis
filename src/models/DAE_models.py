import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_channels=19):
        super(Conv1DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=3, padding="same"),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, input_channels, kernel_size=3, padding="same"),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1) 
        return x
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, N_o=25, N_i=10, N_LS=32, num_layers=2):
        """
        LSTM Autoencoder for EEG signals.
        
        Parameters:
        - input_size: Number of EEG input features (channels)
        - N_o: Hidden size of the first LSTM layer
        - N_i: Hidden size of the second LSTM layer
        - N_LS: Latent space size (user-defined)
        - num_layers: Number of LSTM layers
        """
        super( LSTMAutoencoder, self).__init__()
        
        # Encoder: Two LSTM layers with No = 50 and Ni = 25
        self.encoder_lstm1 = nn.LSTM(input_size, N_o, num_layers=1, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(N_o, N_i, num_layers=1, batch_first=True)
        
        # Fully connected layer to project to latent space
        self.encoder_fc = nn.Linear(N_i, N_LS)
        
        # Decoder: Fully connected layer to expand back to Ni
        self.decoder_fc = nn.Linear(N_LS, N_i)
        
        # Decoder LSTM layers (reverse of encoder)
        self.decoder_lstm1 = nn.LSTM(N_i, N_o, num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(N_o, input_size, num_layers=1, batch_first=True)

    def forward(self, x):
        # Encoder
        window_size = x.shape[1]
        x, _ = self.encoder_lstm1(x)  # LSTM Layer 1 (No = 50)
        x, (hidden, _) = self.encoder_lstm2(x)  # LSTM Layer 2 (Ni = 25)
        
        # Flatten last hidden state and project to latent space
        latent = self.encoder_fc(hidden[-1])  # [batch_size, N_LS]
        
        # Decoder
        hidden_dec = self.decoder_fc(latent).unsqueeze(1)  # Expand to match LSTM input
        x, _ = self.decoder_lstm1(hidden_dec.expand(-1, window_size, -1)) 
        x, _ = self.decoder_lstm2(x) 
        
        return x

class CNN_LSTM_Autoencoder(nn.Module):
    def __init__(self, input_channels=19, seq_length=7680, N_o=50, N_i=25, N_LS=32):
        super(CNN_LSTM_Autoencoder, self).__init__()

        # 1️⃣ CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  

            nn.Conv1d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  
        )
        
        # Compute CNN output shape
        cnn_output_size = seq_length // 4  # Because of 2 MaxPools (each halves the size)

        # 2️⃣ LSTM Encoder
        self.encoder_lstm1 = nn.LSTM(64, N_o, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(N_o, N_i, batch_first=True)

        # 3️⃣ Fully Connected Latent Space
        self.encoder_fc = nn.Linear(N_i, N_LS)
        self.decoder_fc = nn.Linear(N_LS, N_i)

        # 4️⃣ LSTM Decoder (Reverse of Encoder)
        self.decoder_lstm1 = nn.LSTM(N_i, N_o, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(N_o, 64, batch_first=True)

        # 5️⃣ CNN Decoder
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output between 0-1
        )

    def forward(self, x):
        window_size = x.shape[1]
        # CNN Feature Extraction (permute to match Conv1d: batch, channels, time)
        x = x.permute(0, 2, 1) 
        
        x = self.cnn(x)

        # LSTM expects (batch, seq_len, features) → Permute back
        x = x.permute(0, 2, 1)  

        # LSTM Encoder
       
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)

        # Latent space projection
        latent = self.encoder_fc(hidden[-1])

        # Decoder: Expand latent space and LSTM decode
        x = self.decoder_fc(latent).unsqueeze(1).expand(-1, window_size, -1)
        x, _ = self.decoder_lstm1(x)
        x, _ = self.decoder_lstm2(x)

        # CNN Decoder
        x = x.permute(0, 2, 1)  # Back to (batch, channels, time)
        
        x = self.deconv(x)
        x = x.permute(0, 2, 1) 
    
        return x


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, ks):
        super().__init__()
        padding = int((ks - 1) / 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=ks, padding=padding)
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=ks, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class UNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128,256 ]

        self.pool = nn.MaxPool1d(2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        # input_channel => 32; 32 => 64; 64=>128; 128=>256
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], ks=7)
        
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], ks=5)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], ks=5)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], ks=3)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], ks=3)

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3], ks=3)
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], ks=5)
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], ks=5)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], ks=7)

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        x0_0 = self.conv0_0(input)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))  # Fixed skip connection
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        output = self.final(x0_4)
        output = output.permute(0, 2, 1)
        return output