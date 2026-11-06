### NNTRAINCASE4.py  

import os, tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from Rockets.Kang.saburnbackfinal import solve_burnback, pressure_profile, solve_SA, energy_sf
from time import perf_counter
start = perf_counter()



LEARNING_RATE = 1e-4
EPOCHS = 5000
BATCH_SIZE = 512
TRAIN_SPLIT = 0.8



PCO = 101325.0
MPA = 1e6



class SA_Dataset(Dataset):
    def __init__(self, pc_file, coeff_file):
        PC_DATA = []
        with open(pc_file, "r") as f:
            for line in f:
                row = np.array(list(map(float, line.strip().split())))
                Pc = row[:-1]
                t_burn = row[-1]
                t_vals = np.linspace(0, t_burn, len(Pc)
                )
                t_interp = np.linspace(0, t_burn, 50)
                Pc_interp = np.interp(t_interp, t_vals,
                    Pc)
                Pc_interp = np.append(Pc_interp, t_burn
                )
                PC_DATA.append(Pc_interp)

        PC_DATA = np.array(PC_DATA)
        coeff_data = np.loadtxt(coeff_file)
        COEFF_DATA = coeff_data[:, [0, 2, 3, 4]] # n1 constant and a = b and n2 = n3

        self.X = torch.tensor(PC_DATA, dtype=torch.float32)
        self.y = torch.tensor(COEFF_DATA, dtype=torch.float32)

    def __len__(self):
        return len(self.X)



def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

model = nn.Sequential(
    nn.Linear(51, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 4)

)

loss_fn = nn.MSELoss()
# loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.L1Loss()
# loss_fn = nn.HuberLoss()

# optimizer = optim.SGD(model.parameters(), lr=    LEARNING_RATE)
# optimizer = optim.RMSprop(model.parameters(), lr=    LEARNING_RATE)
# optimizer = optim.AdamW(model.parameters(), lr=    LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = optim.Adam(model.parameters(), lr=    LEARNING_RATE)

dataset = SA_Dataset("PC_DATA_CASE_4.txt", "COEFF_DATA_CASE_4.txt")

train_size = round(len(dataset) * TRAIN_SPLIT)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_losses, val_losses = [], []
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    for Xbatch, ybatch in train_loader:
        pred = model(Xbatch)
        loss = loss_fn(pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for Xval, yval in test_loader:
            val_pred = model(Xval)
            loss = loss_fn(val_pred, yval)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

torch.save(model.state_dict(), "NN_MODEL_CASE_4.pth")

end = perf_counter()
code_run_time = end-start
print(f"Code Run Time: {code_run_time:.4f}")

plt.figure()
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="red", linestyle="--")
plt.title("Training and Validation MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

t1 = 0.01
t2 = 0.05
mse_list = []
c1 = 0
c2 = 0
for i in range(100):
    model.eval()
    sample_X, sample_y = test_dataset[i]
    sample_X = sample_X.unsqueeze(0)
    coeff_pred = model(sample_X).detach().numpy().squeeze()
    coeff_true = sample_y.numpy()
    # Pc_true = sample_X.numpy()

    a_in, m_in, n1_in, n2_in = coeff_true
    b_in = a_in
    m_in = int(m_in)
    # n1_in = 5.0
    # n2_in = 10.0
    n3_in = n2_in

    a_pred, m_pred, n1_pred, n2_pred = coeff_pred
    b_pred = a_pred
    m_pred = int(m_pred)
    # n1_pred = n1_in
    # n2_pred = n2_in
    n3_pred = n2_pred
    x_pred = np.array([a_pred, b_pred, m_pred, n1_pred, n2_pred, n3_pred])

    mesh_in, S_in = solve_burnback(a_in, b_in, m_in, n1_in, n2_in, n3_in, plot=False)
    web_in, Ps_in, time_in, perimeters, npoints = pressure_profile(mesh_in, S_in, plot=False)

    mse_val = energy_sf(x_pred, web_in, Ps_in, n_div=100)
    mse_list.append(mse_val)

    if mse_val <= t1:
        c1 += 1

    if mse_val <= t2:
        c2 += 1



mse_arr = np.array(mse_list)
mean_mse = np.mean(mse_arr)
median_mse = np.median(mse_arr)
p90_mse = np.percentile(mse_arr, 90)
acc_frac1 = (c1 / len(mse_arr))
acc_frac2 = (c2 / len(mse_arr))

print(f"Samples evaluated: {len(mse_arr)}")
print(f"Mean MSE: {mean_mse:.4f}")
print(f"Median MSE: {median_mse:.4f}")
print(f"P90 MSE: {p90_mse:.4f}")
print(f"Accuracy Percent (0.01): {acc_frac1*100:.2f}%\n")
print(f"Accuracy Percent (0.05): {acc_frac2*100:.2f}%\n")


