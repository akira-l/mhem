#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(model_recieve, data_loader, val_version, epoch_num, config):

    model = model_recieve['base']

    model.train(False)
    #as_model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0

    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    relu_layer = nn.ReLU()

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version)

    confidence_dict = {}
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            # print data
            #inputs,  labels, labels_swap, law_swap, img_name = data_val
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            img_names = data_val[-1]
            # forward
            outputs, cls_feat, avg_feat, top_avg_feat = model(inputs)
            main_score = F.softmax(outputs, 1)

            pred_val, pred_ind = outputs.max(1)
            sm_pred_val, sm_pred_ind = main_score.max(1)
            for sub_name, sub_val, sub_pos, sub_label in zip(img_names, sm_pred_val.tolist(), sm_pred_ind.tolist(), labels.tolist() ):
                confidence_dict[sub_name] = {'val': sub_val,
                                             'pos': sub_pos,
                                             'gt': sub_label}

            loss = 0
            ce_loss = get_ce_loss(outputs, labels).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss))

            top3_val, top3_pos = torch.topk(outputs, 3)
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

        val_acc1 = val_corrects1 / item_count
        val_acc2 = val_corrects2 / item_count
        val_acc3 = val_corrects3 / item_count


        t1 = time.time()
        since = t1-t0
        print('--'*30)
        print('noraml eval: % 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=False), val_version, val_acc1, val_version, val_acc2, val_version, val_acc3, since))
        print('--' * 30)
        print('\n')

    if epoch_num > 100:
        confidence_save_name = 'conf_save_acc' + str(round(val_acc1, 5)) + '.pt'
        torch.save(confidence_dict, confidence_save_name)

    return val_acc1, val_acc2, val_acc3


