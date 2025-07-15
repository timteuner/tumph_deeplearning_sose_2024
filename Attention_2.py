import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import grad
import numpy as np

#HYPERPARAMETERS
BATCH_SIZE = 500
LR = 5e-5
EPOCHS = 1000

def generate_data(N):
    X = np.random.randint(0, 10, size=(N, 10))
    counts_2 = np.sum(X == 2, axis=1)
    counts_4 = np.sum(X == 4, axis=1)
    y = (counts_2 > counts_4).astype(int).reshape(-1, 1)
    return X, y

class AttentionModel(torch.nn.Module):
    
    def __init__(self):
        super(AttentionModel,self).__init__()
        self.embed_dim = 5
        self.att_dim = 5
        self.embed = torch.nn.Embedding(10,self.embed_dim)
        
        #one query
        self.query  = torch.nn.Parameter(torch.randn(1,self.att_dim))
        
        #used to compute keys
        self.WK = torch.nn.Linear(self.embed_dim,self.att_dim)
        
        #used to compute values
        self.WV = torch.nn.Linear(self.embed_dim,1)
        
        #final decision based on attention-weighted value
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1,200),
            torch.nn.ReLU(),
            torch.nn.Linear(200,1),
            torch.nn.Sigmoid(),
        )

    def attention(self,x):
        # compute keys
        keys = self.WK(x)
        # compute attention scores
        scores = torch.matmul(self.query, keys.transpose(-2, -1)) / np.sqrt(self.att_dim)
        # compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights
    
    def values(self,x):
        weighted_values = torch.matmul(self.attention(x), self.WV(x))
        return weighted_values
                
    def forward(self,x):
        x_embedded = self.embed(x)
        weighted_values = self.values(x_embedded)

        return self.nn(weighted_values)
    # for understanding
    def Vals(self, x):
        return self.WV(x)

def main():

    X_np, y_np = generate_data(10000)
    X = torch.tensor(X_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.float32)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    criterion = torch.nn.BCELoss()
    model = AttentionModel()
    optimizer = Adam(model.parameters(), lr=LR)

    N = X_train.shape[0]
    loss_trajectory = []

    for epoch in range(EPOCHS):
    # Shuffle indices
        indices = torch.randperm(N)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
        running_loss = 0.0

        for i in range(0, N, BATCH_SIZE):
            xb = X_train[i:i+BATCH_SIZE]
            yb = y_train[i:i+BATCH_SIZE]

            optimizer.zero_grad()
            outputs = model(xb).squeeze(-1)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        with torch.no_grad():
            model.eval()
            val_preds = model(X_val).squeeze(-1)
            val_loss = criterion(val_preds, y_val)
        
        epoch_loss = running_loss / N
        
        loss_trajectory.append(epoch_loss)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss.item():.4f}')

            sample_idx = np.random.randint(0, len(X_val))
            sample_input = X_val[sample_idx].unsqueeze(0)
            sample_output = model(sample_input).item()
            print(f"Sample input: {X_val[sample_idx].tolist()}")
            print(f"Attention weights: {model.attention(model.embed(sample_input)).squeeze().detach().numpy()}")
            print(f"Values: {model.Vals(model.embed(sample_input)).squeeze().detach().numpy()}")

    # Plotting attention weights
    
    def plot(model,N,traj):
        x,y = generate_data(N)
        f,axarr = plt.subplots(1,3)
        f.set_size_inches(13,2)
        ax = axarr[0]
        at = model.attention(model.embed(torch.tensor(x, dtype=torch.long)))[:,0,:].detach().numpy()
        ax.imshow(at)
        ax = axarr[1]
    
    
        vals = model.forward(torch.tensor(x, dtype=torch.long)).squeeze(-1).detach().numpy()

        nan = np.ones_like(vals)*np.nan
        nan = np.where(at > 0.1, vals, nan)
        ax.imshow(nan,vmin = -1, vmax = 1)
        for i,xx in enumerate(x):
            for j,xxx in enumerate(xx):
                ax = axarr[0]
                ax.text(j,i,int(xxx), c = 'r' if (xxx in [2,4]) else 'w')    
                ax = axarr[1]
                ax.text(j,i,int(xxx), c = 'r' if (xxx in [2,4]) else 'w')    
        ax = axarr[2]
        ax.plot(traj)
        ax.set_yscale("log")
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        
        axarr[0].set_title("Attention Weights")
        axarr[1].set_title("Value Outputs (Masked)")
        axarr[0].set_xlabel("Input Position")
        axarr[1].set_xlabel("Input Position")
        axarr[0].set_ylabel("Batch Index")
        axarr[1].set_ylabel("Batch Index")

        plt.tight_layout()
        plt.show()

    plot(model=model, N=20, traj=loss_trajectory)

if __name__ == "__main__":
    main()

# We can see that the attention weights focus on the positions of 2 and 4 and that the values are strongly positive for 2s and negative for 4s.