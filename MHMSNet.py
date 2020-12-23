import torch
import Config
import torch.nn as nn
from torch.nn import utils as nn_utils
from torch.utils.data import DataLoader,TensorDataset
import pickle
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

RANDOM_SEED = Config.RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
batch_size = Config.batch_size
input_size = Config.input_size
embed_dim = Config.embed_dim
num_layers = Config.num_layers
hidden_size = Config.hidden_size
workers = Config.workers
learning_rate = Config.learning_rate
epochs = Config.epochs
device = Config.device
datatype = Config.datatype
save_model_dir = '/home/lb/test1/mimic_model_mine_4.2_'+datatype+'_test.pth'
if datatype=="mimic3":
    batch_size = 256
    learning_rate = 1e-2
else:
    batch_size = 512
    learning_rate = 1e-3

showindex = 66

def get_data():
    x_torch = pickle.load(open('lb_'+datatype+'_x.p', 'rb'))
    y_torch = pickle.load(open('lb_'+datatype+'_y.p', 'rb'))
    x_lens = pickle.load(open('lb_'+datatype+'_len.p', 'rb'))

    train_size = int(len(x_torch) * 0.7)
    train_X = x_torch[:train_size]
    train_X_lens = x_lens[:train_size]
    train_Y = y_torch[:train_size]
    test_X = x_torch[train_size:]
    test_X_lens = x_lens[train_size:]
    test_Y = y_torch[train_size:]

    train_deal_dataset = TensorDataset(train_X, train_Y,train_X_lens)

    train_loader = DataLoader(dataset=train_deal_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=workers)

    test_deal_dataset = TensorDataset(test_X, test_Y,test_X_lens)

    test_loader = DataLoader(dataset=test_deal_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True,
                             num_workers=workers)

    return train_loader,test_loader


class MHMSNet(nn.Module):

    def __init__(self, input_size):
        super(MHMSNet, self).__init__()
        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2*hidden_size, nhead=2), num_layers=2)
        self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                          padding=1, dilation=1)
        self.chomp1 = Chomp1d(1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=input_size, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                        padding=2, dilation=2)
        self.chomp2 = Chomp1d(2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                               padding=4, dilation=4)
        self.chomp3 = Chomp1d(4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(in_channels=input_size, out_channels=2 * hidden_size, kernel_size=2, stride=1,
                               padding=8, dilation=8)
        self.chomp4 = Chomp1d(8)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.convseq1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.convseq2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.convseq3 = nn.Sequential(self.conv3, self.chomp3, self.relu3, self.dropout3)
        self.convseq4 = nn.Sequential(self.conv4, self.chomp4, self.relu4, self.dropout4)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embed_dim),
            nn.ReLU()
        )

        self.att = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.Softmax()
        )
        self.Wc = nn.Sequential(
            nn.Linear(2*hidden_size,hidden_size),
            nn.Tanh()
        )
        self.predict = nn.Sequential(
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x,lens):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        v = self.embedding(x)
        pack = nn_utils.rnn.pack_padded_sequence(v, lens, batch_first=True, enforce_sorted=False)
        out_packed,h = self.rnn(pack)
        out_unpacked = nn_utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        outs,outlens = out_unpacked
        att = self.transformerEncoder(outs)
        out2 = torch.mul(att, outs)
        pack2 = nn_utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        unpack2 = nn_utils.rnn.pad_packed_sequence(pack2, batch_first=True)
        v_conv = unpack2[0].permute(0, 2, 1)
        v_conv1 = self.convseq1(v_conv).permute(0,2,1)
        v_conv2 = self.convseq2(v_conv).permute(0,2,1)
        v_conv3 = self.convseq3(v_conv).permute(0,2,1)
        v_conv4 = self.convseq4(v_conv).permute(0, 2, 1)
        totalvconv = v_conv1+v_conv2+v_conv3+v_conv4
        out3 = torch.mul(totalvconv,out2)
        out3 = self.Wc(out3)
        out3 = self.predict(out3)
        return out3

model = MHMSNet(input_size)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model,train_loader):
    model.train()
    train_loss_array = []
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            lens = lens.to(device)
            labels = labels.float()
            out = model(inputs,lens)
            out = out.to(device)
            batch_loss = torch.tensor(0,dtype=float).to(device)
            for j in range(len(lens)):
                intlenj = int(lens[j])
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0))
                batch_loss += oneloss

            batch_loss /= batch_size
            optimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            optimizer.step()

        if epoch % 4 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        if (epoch + 1) % 1 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch + 1, batch_loss.data))
            train_loss_array.append(batch_loss.data)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, save_model_dir)

    print(epochs)
    print(train_loss_array)
    plt.xlim((0, epochs))
    plt.ylim((0, 1))
    my_x_ticks = np.arange(0, epochs, 1.0)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.plot(np.arange(epochs), train_loss_array)
    plt.title('Train loss')
    plt.show()


def test_model(model,test_loader):
    device = torch.device("cpu")
    model.eval()
    test_loss_array = []
    outs = list()
    labelss = list()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, lens = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            lens = lens.to(device)
            labels = labels.float()
            out = model(inputs, lens)
            out = out.to(device)
            batch_loss = torch.tensor(0, dtype=float).to(device)
            for j in range(len(lens)):
                intlenj = int(lens[j])
                lossF = torch.nn.BCELoss(size_average=True).to(device)
                oneloss = lossF(out[j, intlenj - 1, :], labels[j].unsqueeze(dim=0))
                outs.extend(list(out[j, intlenj - 1, :].cpu().numpy()))
                templabel = [int(labels[j])]
                labelss.extend(templabel)
                batch_loss += oneloss

            batch_loss /= batch_size
            test_loss_array.append(float(batch_loss.data))
            print('Test loss:{}'.format(float(batch_loss.data)))

    print(len(test_loss_array))
    print(test_loss_array)
    outs = np.array(outs)
    labelss = np.array(labelss)
    print(labelss)
    print(outs)
    print('AUROC:')
    print(metrics.roc_auc_score(labelss, outs))
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(labelss, outs)
    auprc = metrics.auc(recalls, precisions)
    print('AUPRC:')
    print(auprc)

def main():
    train_loader,test_loader = get_data()
    checkpoint = torch.load(save_model_dir)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_model(model, test_loader)
    return

if __name__ == '__main__':
    main()

