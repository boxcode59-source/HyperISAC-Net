# ==========================================================
# HyperISAC-Net : Step 1
# Data Loading + Preprocessing + STFT
# ==========================================================

# Install (Google Colab)
# !pip install scipy librosa timm mamba-ssm torch_geometric -q

import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, stft
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

##############################################################
# Load Dataset
##############################################################

DATASET_PATH = "/content/DeepSense6G"

csv_files = glob.glob(DATASET_PATH + "/**/*.csv", recursive=True)

dfs = []

for f in csv_files:
    try:
        dfs.append(pd.read_csv(f))
    except:
        pass

dataset = pd.concat(dfs, ignore_index=True)

print(dataset.shape)

##############################################################
# Label
##############################################################

label_col = dataset.columns[-1]

X = dataset.drop(label_col, axis=1)
y = dataset[label_col]

##############################################################
# Missing Values
##############################################################

X = X.fillna(X.mean())

##############################################################
# Savitzky-Golay Filtering
##############################################################

filtered = []

for col in X.columns:

    sig = X[col].values

    try:
        sig = savgol_filter(
            sig,
            window_length=11,
            polyorder=3
        )
    except:
        pass

    filtered.append(sig)

filtered = np.array(filtered).T

##############################################################
# Z-score Normalization
##############################################################

filtered = zscore(filtered, axis=0)

filtered = np.nan_to_num(filtered)

##############################################################
# STFT
##############################################################

spectrograms = []

for sample in filtered:

    f, t, Z = stft(
        sample,
        nperseg=256,
        noverlap=128
    )

    spec = np.abs(Z)

    spectrograms.append(spec)

spectrograms = np.array(spectrograms)

print("Spectrogram Shape :", spectrograms.shape)

##############################################################
# Train Test Split
##############################################################

X_train,X_test,y_train,y_test=train_test_split(
    spectrograms,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train,X_val,y_train,y_val=train_test_split(
    X_train,
    y_train,
    test_size=0.176,
    random_state=42,
    stratify=y_train
)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

##############################################################
# Torch Dataset
##############################################################

class ISACDataset(Dataset):

    def __init__(self,X,y):

        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y.values,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        return self.X[idx],self.y[idx]

train_dataset=ISACDataset(X_train,y_train)
val_dataset=ISACDataset(X_val,y_val)
test_dataset=ISACDataset(X_test,y_test)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32)
test_loader=DataLoader(test_dataset,batch_size=32)

print("Data Ready")

# ==========================================================
# HyperISAC-Net : Step 2
# ConvNeXt-V2 + TKAN + GRE
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ----------------------------------------------------------
# ConvNeXt-V2 Backbone
# ----------------------------------------------------------

class ConvNeXtEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "convnextv2_tiny",
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

    def forward(self,x):

        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        return self.backbone(x)


# ----------------------------------------------------------
# Temporal KAN
# ----------------------------------------------------------

class TKAN(nn.Module):

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,x):

        x = self.fc1(x)

        x = torch.tanh(x)

        x = x.unsqueeze(1)

        out,_ = self.gru(x)

        out = out.squeeze(1)

        out = self.fc2(out)

        out = self.norm(out)

        return out


# ----------------------------------------------------------
# Gated Residual Encoding
# ----------------------------------------------------------

class GRE(nn.Module):

    def __init__(self,
                 feature_dim=512):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(feature_dim,feature_dim),

            nn.Sigmoid()

        )

        self.proj = nn.Linear(
            feature_dim,
            feature_dim
        )

    def forward(self,x):

        gate = self.gate(x)

        residual = self.proj(x)

        out = gate * residual + (1-gate) * x

        return out


# ----------------------------------------------------------
# Hybrid Feature Extractor
# ----------------------------------------------------------

class ConvNeXt_TKAN_GRE(nn.Module):

    def __init__(self):

        super().__init__()

        self.convnext = ConvNeXtEncoder()

        self.tkan = TKAN()

        self.gre = GRE()

    def forward(self,x):

        x = self.convnext(x)

        x = self.tkan(x)

        x = self.gre(x)

        return x


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

feature_extractor = ConvNeXt_TKAN_GRE().to(device)

dummy = torch.randn(2,1,224,224).to(device)

features = feature_extractor(dummy)

print("Feature Shape :",features.shape)


# ==========================================================
# HyperISAC-Net : Step 3
# SGAE-MIMP Feature Selection
# Graph Attention + Sparse AE + Mamba + MI Pruning
# ==========================================================

# !pip install torch-geometric mamba-ssm -q

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from mamba_ssm import Mamba

# ----------------------------------------------------------
# Graph Attention Network (GATv2)
# ----------------------------------------------------------

class GATFeatureExtractor(nn.Module):

    def __init__(self,
                 in_dim=512,
                 hidden_dim=64,
                 heads=8):

        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim*heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False
        )

    def forward(self,x,edge_index):

        x = self.gat1(x,edge_index)
        x = F.elu(x)

        x = self.gat2(x,edge_index)

        return x


# ----------------------------------------------------------
# Sparse AutoEncoder
# ----------------------------------------------------------

class SparseAutoEncoder(nn.Module):

    def __init__(self,
                 input_dim=64,
                 latent_dim=32):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.ReLU(),

            nn.Linear(128,latent_dim),
            nn.ReLU()

        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim,128),
            nn.ReLU(),

            nn.Linear(128,input_dim)

        )

    def forward(self,x):

        latent = self.encoder(x)

        recon = self.decoder(latent)

        return latent,recon


# ----------------------------------------------------------
# Mamba Feature Selector
# ----------------------------------------------------------

class MambaSelector(nn.Module):

    def __init__(self,
                 d_model=32):

        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

    def forward(self,x):

        x = x.unsqueeze(1)

        x = self.mamba(x)

        return x.squeeze(1)


# ----------------------------------------------------------
# Mutual Information Pruning
# ----------------------------------------------------------

class MIPruning(nn.Module):

    def __init__(self,
                 feature_dim=32):

        super().__init__()

        self.score = nn.Sequential(

            nn.Linear(feature_dim,64),
            nn.ReLU(),

            nn.Linear(64,feature_dim),
            nn.Sigmoid()

        )

    def forward(self,x):

        importance = self.score(x)

        mask = importance > 0.5

        selected = x * mask.float()

        return selected,importance


# ----------------------------------------------------------
# Complete SGAE-MIMP
# ----------------------------------------------------------

class SGAE_MIMP(nn.Module):

    def __init__(self):

        super().__init__()

        self.gat = GATFeatureExtractor()

        self.autoencoder = SparseAutoEncoder()

        self.mamba = MambaSelector()

        self.pruning = MIPruning()

    def forward(self,x,edge_index):

        # Graph Attention
        x = self.gat(x,edge_index)

        # Sparse AE
        latent,reconstruction = self.autoencoder(x)

        # Mamba
        latent = self.mamba(latent)

        # MI Pruning
        selected,importance = self.pruning(latent)

        return {
            "selected_features":selected,
            "importance":importance,
            "latent":latent,
            "reconstruction":reconstruction
        }


# ----------------------------------------------------------
# Example
# ----------------------------------------------------------

model = SGAE_MIMP().to(device)

num_nodes = 512

node_features = torch.randn(num_nodes,512).to(device)

edge_index = torch.randint(
    0,
    num_nodes,
    (2,4000)
).to(device)

output = model(node_features,edge_index)

print("Selected :",output["selected_features"].shape)
print("Latent :",output["latent"].shape)
print("Reconstruction :",output["reconstruction"].shape)

# ==========================================================
# HyperISAC-Net : Step 4
# Sparse Attention PPO Decision Layer
# ==========================================================

# !pip install entmax stable-baselines3 gymnasium -q

import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import entmax15

# ----------------------------------------------------------
# Sparse Attention
# ----------------------------------------------------------

class SparseAttention(nn.Module):

    def __init__(self, feature_dim=32):

        super().__init__()

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key   = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self,x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        score = torch.matmul(Q, K.transpose(-2,-1))
        score = score / (Q.size(-1) ** 0.5)

        attention = entmax15(score)

        output = torch.matmul(attention,V)

        return output


# ----------------------------------------------------------
# PPO Actor
# ----------------------------------------------------------

class Actor(nn.Module):

    def __init__(self,
                 state_dim=32,
                 beam_classes=64,
                 power_levels=8):

        super().__init__()

        self.attention = SparseAttention(state_dim)

        self.fc = nn.Sequential(

            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU()

        )

        self.beam = nn.Linear(64,beam_classes)
        self.power = nn.Linear(64,power_levels)

    def forward(self,state):

        state = state.unsqueeze(1)

        state = self.attention(state)

        state = state.squeeze(1)

        x = self.fc(state)

        beam_logits = self.beam(x)
        power_logits = self.power(x)

        return beam_logits,power_logits


# ----------------------------------------------------------
# PPO Critic
# ----------------------------------------------------------

class Critic(nn.Module):

    def __init__(self,state_dim=32):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1)

        )

    def forward(self,state):

        return self.net(state)


# ----------------------------------------------------------
# PPO Agent
# ----------------------------------------------------------

class PPOAgent(nn.Module):

    def __init__(self):

        super().__init__()

        self.actor = Actor()
        self.critic = Critic()

    def forward(self,state):

        beam,power = self.actor(state)

        value = self.critic(state)

        return beam,power,value


# ----------------------------------------------------------
# Reward Function
# ----------------------------------------------------------

def reward_function(rate,
                    sensing,
                    power,
                    a=0.5,
                    b=0.4,
                    c=0.1):

    reward = a*rate + b*sensing - c*power

    return reward


# ----------------------------------------------------------
# PPO Loss
# ----------------------------------------------------------

def ppo_loss(old_log_prob,
             new_log_prob,
             advantage,
             clip=0.2):

    ratio = torch.exp(new_log_prob-old_log_prob)

    s1 = ratio*advantage

    s2 = torch.clamp(ratio,
                     1-clip,
                     1+clip)*advantage

    return -torch.min(s1,s2).mean()


# ----------------------------------------------------------
# Build Model
# ----------------------------------------------------------

ppo = PPOAgent().to(device)

optimizer = torch.optim.Adam(
    ppo.parameters(),
    lr=1e-4
)

# ----------------------------------------------------------
# Example Training
# ----------------------------------------------------------

state = torch.randn(16,32).to(device)

beam_logits,power_logits,value = ppo(state)

beam_action = beam_logits.argmax(1)
power_action = power_logits.argmax(1)

reward = reward_function(
            rate=torch.rand(16).to(device),
            sensing=torch.rand(16).to(device),
            power=torch.rand(16).to(device)
         )

advantage = reward.unsqueeze(1)-value.detach()

beam_dist = torch.distributions.Categorical(
            logits=beam_logits)

old_log = beam_dist.log_prob(beam_action)

new_log = beam_dist.log_prob(beam_action)

loss = ppo_loss(
        old_log,
        new_log,
        advantage.squeeze()
      )

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("Beam :",beam_action.shape)
print("Power :",power_action.shape)
print("Reward :",reward.mean().item())
print("Loss :",loss.item())

# ==========================================================
# HyperISAC-Net : Step 5
# Chaotic Lévy-flight Osprey Optimization Algorithm (CLOOA)
# ==========================================================

import numpy as np
from scipy.special import gamma

# ----------------------------------------------------------
# Lévy Flight
# ----------------------------------------------------------

def levy_flight(beta=1.5):

    sigma = (
        gamma(1+beta) *
        np.sin(np.pi*beta/2) /
        (
            gamma((1+beta)/2) *
            beta *
            2**((beta-1)/2)
        )
    )**(1/beta)

    u = np.random.randn() * sigma
    v = np.random.randn()

    step = u / (abs(v)**(1/beta))

    return step


# ----------------------------------------------------------
# Logistic Chaotic Initialization
# ----------------------------------------------------------

def chaotic_population(pop_size,
                       dim,
                       lb,
                       ub):

    population = np.zeros((pop_size,dim))

    x = np.random.rand()

    for i in range(pop_size):

        for j in range(dim):

            x = 4*x*(1-x)

            population[i,j] = lb[j] + x*(ub[j]-lb[j])

    return population


# ----------------------------------------------------------
# Fitness Function
# Replace with PPO validation reward
# ----------------------------------------------------------

def fitness(position):

    learning_rate = position[0]
    gamma_rl      = position[1]
    clip_ratio    = position[2]
    entropy       = position[3]
    gae_lambda    = position[4]

    # Dummy objective
    score = (
        learning_rate
        + gamma_rl
        + clip_ratio
        + entropy
        + gae_lambda
    )

    return -score


# ----------------------------------------------------------
# CLOOA Optimizer
# ----------------------------------------------------------

class CLOOA:

    def __init__(self,
                 population=20,
                 iterations=50):

        self.population = population
        self.iterations = iterations

        self.lb = np.array([
            1e-5,
            0.80,
            0.10,
            0.001,
            0.80
        ])

        self.ub = np.array([
            1e-3,
            0.999,
            0.30,
            0.050,
            0.99
        ])

        self.dimension = len(self.lb)

    def optimize(self):

        pop = chaotic_population(
            self.population,
            self.dimension,
            self.lb,
            self.ub
        )

        fit = np.array([fitness(i) for i in pop])

        best = pop[np.argmin(fit)].copy()

        best_score = fit.min()

        history = []

        for epoch in range(self.iterations):

            for i in range(self.population):

                step = levy_flight()

                new = pop[i] + \
                      np.random.rand(self.dimension) * \
                      step * \
                      (best-pop[i])

                new = np.clip(
                    new,
                    self.lb,
                    self.ub
                )

                new_fit = fitness(new)

                if new_fit < fit[i]:

                    pop[i] = new
                    fit[i] = new_fit

                if new_fit < best_score:

                    best_score = new_fit
                    best = new.copy()

            history.append(best_score)

            print(
                f"Iteration {epoch+1:02d} | "
                f"Fitness : {best_score:.5f}"
            )

        return best,history


# ----------------------------------------------------------
# Run CLOOA
# ----------------------------------------------------------

optimizer = CLOOA(
    population=20,
    iterations=30
)

best_parameter,history = optimizer.optimize()

print("\nBest Hyperparameters\n")

print("Learning Rate :",best_parameter[0])
print("Gamma         :",best_parameter[1])
print("Clip Ratio    :",best_parameter[2])
print("Entropy       :",best_parameter[3])
print("GAE Lambda    :",best_parameter[4])
# ==========================================================
# HyperISAC-Net : Step 6
# Complete Training Pipeline
# ==========================================================

import torch
import torch.nn as nn
from tqdm import tqdm

# ----------------------------------------------------------
# Complete HyperISAC-Net
# ----------------------------------------------------------

class HyperISACNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.extractor = ConvNeXt_TKAN_GRE()

        self.selector = SGAE_MIMP()

        self.classifier = nn.Sequential(

            nn.Linear(32,128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,64)     # 64 Beam Classes

        )

    def forward(self,image,edge_index):

        features = self.extractor(image)

        out = self.selector(features,edge_index)

        selected = out["selected_features"]

        prediction = self.classifier(selected)

        return prediction,out


# ----------------------------------------------------------
# Build Model
# ----------------------------------------------------------

model = HyperISACNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=best_parameter[0],

    weight_decay=1e-5

)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)

# ----------------------------------------------------------
# Edge Index Generator
# ----------------------------------------------------------

def create_edge_index(nodes):

    row = torch.arange(nodes)

    col = torch.roll(row,1)

    edge = torch.stack([row,col],0)

    return edge.long()


# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

epochs = 30

train_loss_history = []
val_acc_history = []

for epoch in range(epochs):

    model.train()

    running_loss = 0

    total = 0
    correct = 0

    loop = tqdm(train_loader)

    for images,labels in loop:

        images = images.to(device)

        labels = labels.to(device)

        edge_index = create_edge_index(
            images.size(0)
        ).to(device)

        optimizer.zero_grad()

        outputs,extra = model(

            images,

            edge_index

        )

        loss = criterion(

            outputs,

            labels

        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pred = outputs.argmax(1)

        correct += (pred==labels).sum().item()

        total += labels.size(0)

        loop.set_description(

            f"Epoch {epoch+1}/{epochs}"

        )

    train_acc = 100*correct/total

    train_loss_history.append(

        running_loss/len(train_loader)

    )

    # ---------------- Validation ----------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            edge_index = create_edge_index(

                images.size(0)

            ).to(device)

            outputs,_ = model(

                images,

                edge_index

            )

            pred = outputs.argmax(1)

            correct += (pred==labels).sum().item()

            total += labels.size(0)

    val_acc = 100*correct/total

    scheduler.step(val_acc)

    val_acc_history.append(val_acc)

    print("-----------------------------------")

    print("Epoch :",epoch+1)

    print("Train Loss :",train_loss_history[-1])

    print("Train Accuracy :",train_acc)

    print("Validation Accuracy :",val_acc)

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

torch.save(

    model.state_dict(),

    "HyperISACNet.pth"

)

print("\nModel Saved Successfully")

# ----------------------------------------------------------
# Save Optimized Hyperparameters
# ----------------------------------------------------------

torch.save(

    {

        "learning_rate":best_parameter[0],

        "gamma":best_parameter[1],

        "clip":best_parameter[2],

        "entropy":best_parameter[3],

        "gae_lambda":best_parameter[4]

    },

    "CLOOA_Optimized_Parameters.pth"

)

print("Hyperparameters Saved")

print("\nTraining Completed Successfully.")