# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import datetime
import tqdm
from sklearn import metrics
import pickle
import csv

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import MusicSelfAttModel

class Solver():
    def __init__(self, data_loader1, data_loader2, valid_loader, tag_list, config):
        # Data loader
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        self.valid_loader = valid_loader

        # Training settings
        self.n_epochs = 120
        self.lr = 1e-4
        self.log_step = 100
        self.is_cuda = torch.cuda.is_available()
        self.model_save_path = config['log_dir']
        self.batch_size = config['batch_size']
        self.tag_list = tag_list
        self.num_class = 56
        self.writer = SummaryWriter(config['log_dir'])
        self.model_fn = os.path.join(self.model_save_path, 'best_model.pth')

        # Build model
        self.build_model()

    def build_model(self):
        # model and optimizer
        model = MusicSelfAttModel()

        if self.is_cuda:
            self.model = model
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def train(self):
        start_t = time.time()
        current_optimizer = 'adam'
        best_roc_auc = 0
        drop_counter = 0
        reconst_loss = nn.BCELoss()

        for epoch in range(self.n_epochs):
            print('Training')
            drop_counter += 1
            # train
            self.model.train()
            ctr = 0
            step_loss = 0
            epoch_loss = 0
            for i1, i2 in zip(self.data_loader1, self.data_loader2):
                ctr += 1

                # mixup---------
                alpha = 1
                mixup_vals = np.random.beta(alpha, alpha, i1[0].shape[0])
                
                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
                inputs = (lam * i1[0]) + ((1 - lam) * i2[0])
                
                lam = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1))
                labels = (lam * i1[1]) + ((1 - lam) * i2[1])

                # variables to cuda
                x = self.to_var(inputs)
                y = self.to_var(labels)

                # predict
                att,clf = self.model(x)
                loss1 = reconst_loss(att, y)
                loss2 = reconst_loss(clf,y)
                loss = (loss1+loss2)/2

                step_loss += loss.item()
                epoch_loss += loss.item()

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            epoch+1, self.n_epochs, ctr, len(self.data_loader1), (step_loss/self.log_step),
                            datetime.timedelta(seconds=time.time()-start_t)))
                    step_loss = 0

            self.writer.add_scalar('Loss/train', epoch_loss/len(self.data_loader1), epoch)

            # validation
            roc_auc, _ = self._validation(start_t, epoch)

            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))

            if epoch%10 ==0:
                print(f'Saving model at epoch {epoch}')
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'model_epoch_{epoch}.pth'))

            # schedule optimizer
            current_optimizer, drop_counter = self._schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def _validation(self, start_t, epoch):
        prd1_array = []  # prediction
        prd2_array = []
        gt_array = []   # ground truth
        ctr = 0
        self.model.eval()
        reconst_loss = nn.BCELoss()
        for x, y in self.valid_loader:
            ctr += 1

            # variables to cuda
            x = self.to_var(x)
            y = self.to_var(y)

            # predict
            att,clf = self.model(x)
            loss1 = reconst_loss(att, y)
            loss2 = reconst_loss(clf,y)
            loss = (loss1+loss2)/2

            # print log
            if (ctr) % self.log_step == 0:
                print("[%s] Epoch [%d/%d], Iter [%d/%d] valid loss: %.4f Elapsed: %s" %
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.valid_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

            # append prediction
            att = att.detach().cpu()
            clf = clf.detach().cpu()
            y = y.detach().cpu()
            for prd1 in att:
                prd1_array.append(list(np.array(prd1)))
            for prd2 in clf:
                prd2_array.append(list(np.array(prd2)))
            for gt in y:
                gt_array.append(list(np.array(gt)))

        val_loss1 = reconst_loss(torch.Tensor(prd1_array), torch.Tensor(gt_array))
        val_loss2 = reconst_loss(torch.Tensor(prd2_array), torch.Tensor(gt_array))
        print(f'Val Loss: {val_loss1}, {val_loss2}')
        self.writer.add_scalar('Loss/val1', val_loss1, epoch)
        self.writer.add_scalar('Loss/val2', val_loss2, epoch)

        # get auc
        list_all = True if epoch==self.n_epochs else False

        roc_auc1, pr_auc1, _, _ = self.get_auc(prd1_array, gt_array, list_all)
        roc_auc2, pr_auc2, _, _ = self.get_auc(prd2_array, gt_array, list_all)
        self.writer.add_scalar('AUC/ROC2', roc_auc1, epoch)
        self.writer.add_scalar('AUC/PR2', pr_auc1, epoch)
        self.writer.add_scalar('AUC/ROC2', roc_auc2, epoch)
        self.writer.add_scalar('AUC/PR2', pr_auc2, epoch)
        return roc_auc1, pr_auc1

    def get_auc(self, prd_array, gt_array, list_all=False):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)

        roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
        pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

        if list_all==True:            
            for i in range(self.num_class):
                print('%s \t\t %.4f , %.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
        
        return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all

    def _schedule(self, current_optimizer, drop_counter):
        if current_optimizer == 'adam' and drop_counter == 60:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def test(self):
        start_t = time.time()
        reconst_loss = nn.BCELoss()
        epoch = 0

        self.load(self.model_fn)
        self.model.eval()
        ctr = 0
        prd_array = []  # prediction
        gt_array = []   # ground truth
        for x, y in self.data_loader1:
            ctr += 1

            # variables to cuda
            x = self.to_var(x)
            y = self.to_var(y)

            # predict
            out1, out2 = self.model(x)
            out = (out1+out2)/2
            loss = reconst_loss(out, y)

            # print log
            if (ctr) % self.log_step == 0:
                print("[%s] Iter [%d/%d] test loss: %.4f Elapsed: %s" %
                        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        ctr, len(self.data_loader1), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

            # append prediction
            out = out.detach().cpu()
            y = y.detach().cpu()
            for prd in out:
                prd_array.append(list(np.array(prd)))
            for gt in y:
                gt_array.append(list(np.array(gt)))

        #np.save('./pred_array.npy', np.array(prd_array))
        #np.save('./gt_array.npy', np.array(gt_array))

        # get auc
        roc_auc, pr_auc, roc_auc_all, pr_auc_all = self.get_auc(prd_array, gt_array)

        return (np.array(prd_array), np.array(gt_array), roc_auc, pr_auc)

        # save aucs
        #np.save(open(self.roc_auc_fn, 'wb'), roc_auc_all)
        #np.save(open(self.pr_auc_fn, 'wb'), pr_auc_all)

