import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
from torch.utils.data import Dataset, DataLoader

#-----------------------Data preprocessing----------------------------
class CoupledDataset(Dataset):
    def __init__(self, hsi_data, lidar_data, labels):
        self.hsi_data = hsi_data
        self.lidar_data = lidar_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.hsi_data[idx], self.lidar_data[idx]), self.labels[idx]

    def __labels__(self):
        return self.labels

def data_load(hsi_data, lidar_data, labels,patchsize, val_percent, train_num, tr_bsize=64, te_bsize=1000, use_val=True):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    h, w = labels.shape[0], labels.shape[1]
    labels = labels.reshape(h * w)
    num_class = np.max(labels)

    train_label = np.zeros_like(labels)
    test_label = np.zeros_like(labels)

    for i in range(num_class):
        r = random.random
        random.seed(0)
        index = np.where(labels == i + 1)[0]
        random.shuffle(index, r)
        train_index = index[:train_num[i]]
        test_index = index[train_num[i]:]
        train_label[train_index] = labels[train_index]
        test_label[test_index] = labels[test_index]

    labels = labels.reshape(h, w)
    train_label = train_label.reshape(h, w)
    test_label = test_label.reshape(h, w)

    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_test_sample(
        train_label, test_label, labels, num_class)

    margin = int(patchsize/ 2)
    zeroPaddedX_hsi = padWithZeros(hsi_data, margin=margin)
    zeroPaddedX_lidar = padWithZeros(lidar_data, margin=margin)

    x_train_hsi, x_test_hsi, x_true_hsi = train_and_test_data(zeroPaddedX_hsi, band=zeroPaddedX_hsi.shape[-1],
                                                              train_point=total_pos_train, test_point=total_pos_test,
                                                              true_point=total_pos_true, patch=patchsize)
    x_train_lidar, x_test_lidar, x_true_lidar = train_and_test_data(zeroPaddedX_lidar, band=zeroPaddedX_lidar.shape[-1],
                                                                    train_point=total_pos_train, test_point=total_pos_test,
                                                                    true_point=total_pos_true, patch=patchsize)

    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_class)

    del total_pos_train, total_pos_test, number_train, number_test, number_true, zeroPaddedX_hsi, zeroPaddedX_lidar

    if use_val:
        hsi_x_val, hsi_x_test, lidar_x_val, lidar_x_test, y_val, y_test = split_data(
            x_test_hsi, x_test_lidar, y_test, val_percent, rand_state=0
        )
    else:
        hsi_x_val, lidar_x_val, y_val = None, None, None
        hsi_x_test, lidar_x_test, y_test = x_test_hsi, x_test_lidar, y_test

    train_dataset = CoupledDataset(
        np.transpose(x_train_hsi, (0, 3, 1, 2)).astype("float32"),
        np.transpose(x_train_lidar, (0, 3, 1, 2)).astype("float32"),
        y_train
    )
    test_dataset = CoupledDataset(
        np.transpose(hsi_x_test, (0, 3, 1, 2)).astype("float32"),
        np.transpose(lidar_x_test, (0, 3, 1, 2)).astype("float32"),
        y_test
    )

    if use_val:
        val_dataset = CoupledDataset(
            np.transpose(hsi_x_val, (0, 3, 1, 2)).astype("float32"),
            np.transpose(lidar_x_val, (0, 3, 1, 2)).astype("float32"),
            y_val
        )
    else:
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=tr_bsize, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=te_bsize, shuffle=False, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=te_bsize, shuffle=False, **kwargs) if use_val else None

    return train_loader, test_loader, val_loader

class HyperData(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels

def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float32)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float32)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float16)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    return x_train, x_test, x_true

def choose_train_test_sample(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}

    # for train data
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # for test data
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # for true data
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]

    if x + patch <= mirror_image.shape[0] and y + patch <= mirror_image.shape[1]:
        temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    else:
        temp_image = np.zeros((patch, patch, mirror_image.shape[2]), dtype=mirror_image.dtype)

    return temp_image

def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes + 1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)

    return y_train, y_test, y_true

def split_data(hsi_pixels, lidar_pixels, labels, percent, rand_state=345):

    hsi_train, hsi_test, lidar_train, lidar_test, y_train, y_test = train_test_split(
        hsi_pixels, lidar_pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state
    )

    return hsi_train, hsi_test, lidar_train, lidar_test, y_train, y_test

def standardize_data(data):
    m1, n1, l1 = np.shape(data)
    data_reshaped = data.reshape(m1 * n1, l1)
    scaler = StandardScaler()
    data_reshaped = scaler.fit_transform(data_reshaped)
    data_standardized = data_reshaped.reshape(m1, n1, l1)
    return data_standardized

#-----------------------Training and validation----------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class PolyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0):
        super(PolyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)

        pt = F.softmax(outputs, dim=1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        poly_loss = self.alpha * (1 - pt) ** self.beta

        total_loss = ce_loss + poly_loss.mean()
        return total_loss
def train_epoch(model, train_loader, criterion, optimizer, use_cuda=True):
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    model.train()

    accs = np.ones((len(train_loader))) * -1000.0
    losses = np.ones((len(train_loader))) * -1000.0

    for batch_idx, ((hsi_inputs, lidar_inputs), targets) in enumerate(train_loader):
        hsi_inputs, lidar_inputs, targets = hsi_inputs.to(device), lidar_inputs.to(device), targets.to(device)

        targets = targets.long()

        optimizer.zero_grad()
        outputs, con_loss = model(hsi_inputs, lidar_inputs)

        cls_loss = criterion(outputs, targets)
        loss = torch.exp(-model.log_vars[0]) * cls_loss + torch.exp(-model.log_vars[1]) * con_loss + model.log_vars.sum()

        losses[batch_idx] = loss.item()
        accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()

        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)

    return avg_loss, avg_acc

def valid_epoch(model, valid_loader, criterion, use_cuda=True):
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    model.eval()

    accs = []
    losses = []

    with torch.no_grad():
        for batch_idx, ((hsi_inputs, lidar_inputs), targets) in enumerate(valid_loader):
            hsi_inputs, lidar_inputs, targets = hsi_inputs.to(device), lidar_inputs.to(device), targets.to(device)

            targets = targets.long()
            outputs, con_loss = model(hsi_inputs, lidar_inputs)
            cls_loss = criterion(outputs, targets)

            loss = torch.exp(-model.log_vars[0]) * cls_loss + torch.exp(-model.log_vars[1]) * con_loss + model.log_vars.sum()

            losses.append(loss.item())
            accs.append(accuracy(outputs.data, targets.data, topk=(1,))[0].item())

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)

    return avg_loss, avg_acc

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#-----------------------Loss----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        target = target.long()
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

#-----------------------Prediction----------------------------
def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for (hsi_inputs, lidar_inputs), _ in testloader:
            if use_cuda:
                hsi_inputs, lidar_inputs = hsi_inputs.to(device), lidar_inputs.to(device)

            outputs = model(hsi_inputs, lidar_inputs)[0]
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            predicted.append(probs)
    return np.vstack(predicted)

#-----------------------Classification report----------------------------
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
