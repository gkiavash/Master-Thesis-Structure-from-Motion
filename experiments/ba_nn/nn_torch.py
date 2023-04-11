import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from utils import get_initial_matches_and_matrices

all_vals = []


class MyDataset(Dataset):
    def __init__(self):
        K, R, t, F, E, ptsLeft, ptsRight = get_initial_matches_and_matrices()
        x_ = np.concatenate((ptsLeft, ptsRight), axis=1)
        y_ = np.zeros((x_.shape[0], 1))
        print("MyDataset", x_.shape, y_.shape)

        scaler_x = MinMaxScaler()
        scaler_x.fit(x_)
        x_ = scaler_x.transform(x_)

        x_data, y_data = x_, y_

        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


class MyLayer(nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)

        self.weight = nn.Parameter(torch.empty(3, 3), requires_grad=True)
        nn.init.uniform_(self.weight, -0.1, 0.1)

    def rotation_matrix(self, angles):
        # Calculate the sine and cosine of each rotation angle
        c_x, s_x = torch.cos(angles[0]), torch.sin(angles[0])
        c_y, s_y = torch.cos(angles[1]), torch.sin(angles[1])
        c_z, s_z = torch.cos(angles[2]), torch.sin(angles[2])

        # Construct the rotation matrix using the Euler angles
        Rx = torch.tensor([[1, 0, 0], [0, c_x, -s_x], [0, s_x, c_x]], requires_grad=True)
        Ry = torch.tensor([[c_y, 0, s_y], [0, 1, 0], [-s_y, 0, c_y]], requires_grad=True)
        Rz = torch.tensor([[c_z, -s_z, 0], [s_z, c_z, 0], [0, 0, 1]], requires_grad=True)
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)

        return R

    def matrices_(self, initial_input, output_fc1):
        R = self.rotation_matrix(output_fc1[:3])
        t = output_fc1[3:6]
        t_x = torch.tensor([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], requires_grad=True)
        fx = output_fc1[6]
        fy = output_fc1[7]
        cx = output_fc1[8]
        cy = output_fc1[9]
        K_camera = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], requires_grad=True)
        E = torch.matmul(t_x, R)
        # print("E", E)

        # calc F:
        F_1 = torch.matmul(
            torch.transpose(torch.inverse(K_camera), 0, 1),
            E
        )
        F = torch.matmul(
            F_1,
            torch.inverse(K_camera)
        )
        p_left = torch.tensor([initial_input[0], initial_input[1], 1], requires_grad=True)
        p_right = torch.tensor([initial_input[2], initial_input[3], 1], requires_grad=True)

        output_1 = torch.matmul(p_left, F)
        output = torch.matmul(output_1, p_right)
        return output

    def matrices_F(self, initial_input, output_fc1):
        F = torch.reshape(output_fc1[:9], (3, 3))
        # print(F)

        p_left, p_right = torch.split(initial_input, 2)
        p_left = torch.cat((p_left, torch.tensor([1])), 0)
        p_right = torch.cat((p_right, torch.tensor([1])), 0)

        output_1 = torch.matmul(p_left, F)
        output = torch.matmul(output_1, p_right)
        return output

    def matrices_F_2(self, initial_input, output_fc1):
        p_left, p_right = torch.split(initial_input, 2)
        p_left = torch.cat((p_left, torch.tensor([1])), 0)
        p_right = torch.cat((p_right, torch.tensor([1])), 0)

        output_1 = torch.matmul(p_left, self.weight)
        output = torch.matmul(output_1, p_right)
        return output

    def forward(self, input):
        output_fc1 = self.fc1(input)
        output_fc2 = self.fc2(output_fc1)

        output_list = []
        for each_input, each_output_fc1, each_output_fc2 in zip(input, output_fc1, output_fc2):
            # output_list.append(self.matrices_(each_input, each_output_fc1))
            # output_list.append(self.matrices_F(each_input, each_output_fc1))
            output_list.append(self.matrices_F_2(each_input, each_output_fc1))

        output_ = torch.tensor(output_list, requires_grad=True)
        # print("output_", output_)
        # print("weight", self.weight)

        # return output_fc2
        return output_


def run():
    model = MyLayer()
    num_epochs = 50
    loss_model = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(list(model.parameters()))

    train_dataset = MyDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    for epoch in range(num_epochs):
        print("epoch:", epoch)
        print("weight", model.weight)

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_model(output, target)

            print("loss", loss)
            loss_ = loss.detach().numpy()
            all_vals.append(loss_)

            loss.backward()
            optimizer.step()

    plt.plot(all_vals)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    # val_dataset = MyDataset(X_test, y_test)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
    #
    # with torch.no_grad():
    #     model.eval()
    #     total_correct = 0
    #     total_samples = 0
    #     for data, target in val_loader:
    #         output = model(data)
    #         loss = loss_model(output, target)
    #         print("eval", loss)
    #     val_acc = total_correct / total_samples
