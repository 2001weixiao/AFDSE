from utils import train_epoch,valid_epoch,PolyLoss,predict,reports, data_load,standardize_data
import numpy as np
import torch
import time
from model import AFDSE
from sklearn.decomposition import PCA
from scipy.io import loadmat
import os

def main(config):
    num_classes = len(np.unique(gt)) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFDSE(num_classes=num_classes, encoder_dim=config['encoder_dim'], d1=dimension, d2=band2,
                                  wavelet=config['wavelet_type'] ).to(device)
    criterion = PolyLoss(alpha=1.0, beta=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    print("start training")
    tic1 = time.time()

    best_acc = -1
    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(config['epochs']):
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_acces.append(train_acc)

        eval_loss, eval_acc = valid_epoch(model, val_loader, criterion)
        eval_losses.append(eval_loss)
        eval_acces.append(eval_acc)

        scheduler.step()
        print('epoch: {} | train loss: {} train acc: {} | eval loss: {} eval acc: {}'
              .format(epoch, train_loss, train_acc, eval_loss, eval_acc))

        if eval_acc > best_acc:
            best_acc = eval_acc
            state = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(state, "best_model.pth.tar")

    toc1 = time.time()

    checkpoint = torch.load("best_model.pth.tar", weights_only=False)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])

    print("FINAL:      ACCURACY", best_acc)
    tr_time = toc1 - tic1
    tic2 = time.time()
    prediction = predict(test_loader, model, criterion, config['use_best_model'])
    prediction = np.argmax(prediction, axis=1)


    classification, confusion, results = reports(prediction, np.array(test_loader.dataset.__labels__()))

    toc2 = time.time()
    te_time = toc2 - tic2

    print("Final records:")
    print("Running Traing Time: {:.2f}".format(tr_time))
    print("Running Test Time: {:.2f}".format(te_time))
    print("**************************************************")

    return results[0], results[1], results[2], results[3:len(results)], tr_time, te_time

if __name__ == '__main__':
    config = {
        # 'num_iterations': 10
        'num_iterations': 1,
        'wavelet_type': 'bior2.2',
        'patchsize': 7,
        'pca_dimension': 20,
        'epochs': 80,
        'learning_rate': 1e-3,
        'val_percent': 0.15,
        'use_val': True,
        'encoder_dim': 64,
        'scheduler_patience': 5,
        'scheduler_factor': 0.9,
        'use_best_model': True,
        'train_num': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    }

    data_path = 'C:/Users/new/Desktop/AFDSE-main/datasets/houston2013/'
    hsi = loadmat(os.path.join(data_path, 'HoustonU_hsi.mat'))['hsi']
    lidar = loadmat(os.path.join(data_path, 'HoustonU_lidar.mat'))['lidar'][:, :, None]
    gt = loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['img_gt']

    hsi = hsi.astype(np.float32)
    lidar = lidar.astype(np.float32)
    gt = gt.astype(np.int8)
    height2, width2, band2 = lidar.shape

    print('Hyperspectral data size: ', hsi.shape)
    print('Lidar data size: ', lidar.shape)

    dimension = config['pca_dimension']
    newX = np.reshape(hsi, (-1, hsi.shape[2]))
    pca = PCA(n_components=dimension, whiten=True)
    newX = pca.fit_transform(newX)
    hsi = np.reshape(newX, (hsi.shape[0], hsi.shape[1], dimension))

    hsi = standardize_data(hsi)
    lidar = standardize_data(lidar)

    train_loader, test_loader, val_loader = data_load(hsi, lidar, gt, config['patchsize'],
                                                      val_percent=config['val_percent'],
                                                      train_num=config['train_num'], use_val=config['use_val'])

    OA = []
    AA = []
    KAPPA = []
    CA = []
    Train_t = []
    Test_t = []

    for i in range(config['num_iterations']):
        print('iteration {}'.format(i + 1))
        oa, aa, kappa, ca, tr_time, te_time = main(config)
        OA.append(oa)
        AA.append(aa)
        KAPPA.append(kappa)
        CA.append(ca)
        Train_t.append(tr_time)
        Test_t.append(te_time)

    OA_average = np.mean(OA)
    OA_std = np.std(OA)
    AA_average = np.mean(AA)
    AA_std = np.std(AA)
    KAPPA_average = np.mean(KAPPA)
    KAPPA_std = np.std(KAPPA)
    CA_average = np.mean(CA, axis=0)
    CA_std = np.std(CA, axis=0)
    Train_t_avg = np.mean(Train_t)
    Test_t_avg = np.mean(Test_t)

    print("OA avg: {:.4f}, AA avg: {:.4f}, Kappa avg: {:.4f}".format(OA_average, AA_average, KAPPA_average))
    print("OA std: {:.4f}, AA std: {:.4f}, Kappa std: {:.4f}".format(OA_std, AA_std, KAPPA_std))
    print("CA avg: {}".format(CA_average))
    print("CA std: {}".format(CA_std))
    print("Train_time avg: {:.2f}".format(Train_t_avg))
    print("Test_time avg: {:.2f}".format(Test_t_avg))


