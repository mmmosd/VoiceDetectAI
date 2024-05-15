import torch
import torch.nn as nn
import numpy as np

EPOCH = 100
BATCH_SIZE = 32
LR = 0.0001

class GRU(nn.Module) :
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) :
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                         num_layers=num_layers,batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    
def train():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print('GPU count: {}'.format(torch.cuda.device_count()))

    model = GRU()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    target = torch.ones(BATCH_SIZE, 1, device=device)

    for epoch in range(1, EPOCH+1):
        model.zero_grad()

        out = model.forward(data)
        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        print("Epoch {} | loss : {} |".format(epoch, loss))
    