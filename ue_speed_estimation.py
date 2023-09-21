
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Tuple, Any, Union, Dict, List
from functools import partial
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange, repeat
#---------------
# hyper parameters
#---------------
learning_rate = 1e-3
batch_size = 64
epochs = 5

input_vocab_size = 120
output_vocab_size = 300

#---------------
# prepare data
#---------------

def produce_rand_sample(size: int):
    x = np.random.uniform(0, 120, size=size)
    y = np.random.uniform(0, 300, size=size)
    return (x, y)

def discretize(input:Tuple, x_range:int=1, y_range:int=1):
    x, y = input
    x_bins = np.arange(x_range) - 0.5
    y_bins = np.arange(y_range) - 0.5
    x_d = np.digitize(x, x_bins) - 1
    y_d = np.digitize(y, y_bins) - 1
    return (x_d, y_d)

z = produce_rand_sample(10)
print(z)
z_d = discretize(z, 120, 300)
print(z_d)


class FTDataset(Dataset):
    def __init__(self, num_sample:int, transform=None, target_transform=None, input_target_transform = None):
        self.data = self.create_fake_dataset(num_samples=num_sample)
        self.transform = transform 
        self.target_transform = target_transform
        self.input_target_transform = input_target_transform
    def __gen_fake_sample(self) -> Tuple:
        """
        return a fake sample
        """
        seq_len = np.random.randint(low=10, high = 200)
        return produce_rand_sample(seq_len)
    
    def create_fake_dataset(self, num_samples) -> List:
        return [self.__gen_fake_sample() for i in range(num_samples)]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature, label = self.data[idx] # z is a tuple contains (x,y)
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        x_d, y_d = self.input_target_transform((feature, label))
        return (x_d, y_d)

dataset = FTDataset(100, input_target_transform= partial(discretize, x_range = input_vocab_size, y_range=output_vocab_size))
print(dataset[0])



#---------------
# prepare model
#---------------

class rnn(torch.nn.Module):
    def __init__(self, hidden_size=200, proj_size = 100, output_vocab=300, num_layers=3):
        super().__init__()
        # LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, proj_size)
        self.rnn_backbone = torch.nn.LSTM(1, hidden_size, num_layers=num_layers, bias = True, batch_first= True, dropout = 0.1, bidirectional= False,
                        proj_size = proj_size)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(proj_size * 1, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, output_vocab),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        output, (h_n, c_n) = self.rnn_backbone(x)
        logits = self.head(output)
        return logits


model = rnn(hidden_size=200, proj_size = 100, output_vocab=300, num_layers=3).to("cuda:0")
input, label = dataset[0]
# test forward
x = torch.from_numpy(input)[:,None].float().cuda()
print(x.shape) # (seq,1)
output = model(x)
print(output)
print(output.shape) # (seq,)

train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)


train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
test_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False)

#---------------
# prepare loss function
#---------------
# Initialize the loss function
loss_fct = torch.nn.CrossEntropyLoss() # ignore_index=- 100

#---------------
# prepare optimizer
#---------------
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X = X.float().to("cuda:0")
        y = y.to("cuda:0")
        #print("y", y)
        X = rearrange(X, "1 (n d) -> 1 n d", d=1)
        # Compute prediction and loss
        #print(X.shape)
        #print(y.shape)
        pred = model(X)
        #print("pred shape: ", pred.shape)
        loss = loss_fn(pred.view(-1, output_vocab_size), y.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_loop(train_dataloader, model, loss_fct, optimizer)

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.float().cuda(), y.cuda()
            X = rearrange(X, "1 (n d) -> 1 n d", d=1)
            pred = model(X) # (1, n, out_vocab)
            test_loss += loss_fn(pred.view(-1, output_vocab_size), y.view(-1)).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


test_loop(test_dataloader, model, loss_fct)
