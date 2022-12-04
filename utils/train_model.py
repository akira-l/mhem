#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime
import random
import gc

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from utils.eval_model import eval_turn
from utils.adjust_lr import adjust_learning_rate
#from utils.logger import Logger

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def train(Config,
          model_recieve,
          epoch_num,
          start_epoch,
          optimizer_recieve,
          scheduler_recieve,
          data_loader,
          save_dir,
          data_ver='all',
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):


    if isinstance(model_recieve, dict):
        model = model_recieve['base']

    if isinstance(optimizer_recieve, dict):
        optimizer = optimizer_recieve['common']

    if isinstance(scheduler_recieve, dict):
        exp_lr_scheduler = scheduler_recieve['common']

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []
    max_record = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    #logger = Logger('./tb_logs')

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    get_ce_loss = nn.CrossEntropyLoss()

    for epoch in range(start_epoch,epoch_num-1):

        #optimizer = adjust_learning_rate(optimizer, epoch)
        model.train(True)

        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)

            inputs, labels, img_names = data
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            optimizer.zero_grad()

            outputs, cls_feat, avg_feat, top_avg_feat = model(inputs, labels, img_names)

            ce_loss = get_ce_loss(outputs, labels)
            main_score = F.softmax(outputs, 1)

            # alpha 0~1
            alpha = 0.9
            # beta 1.2 1 1/2 1/3 1/4 1/5
            beta = 1.0 / 5.0
            gamma = 2.0
            tmp_score = beta * (alpha + main_score)
            sel_mask = torch.FloatTensor(len(tmp_score), Config.numcls).zero_().cuda()
            sel_mask.scatter_(1, labels.unsqueeze(1), 1.0)
            sel_prob = (tmp_score * sel_mask).sum(1).view(-1, 1)
            sel_prob = torch.clamp(sel_prob, 1e-8, 1 - 1e-8)
            modulate_factor = torch.pow((1 - sel_prob), gamma)

            mem_focal = - modulate_factor * main_score.log()
            mem_focal = mem_focal.mean()

            loss = ce_loss + mem_focal
            loss.backward()
            #clip_gradient(model)
            optimizer.step()

            if step % 10 == 0:
                print('step: {:-8d} / {:d} loss=ce+focal: {:6.4f} = {:6.4f} + {:6.4f}'.format(step, train_epoch_step,
                                                                                                                   loss.detach().item(),
                                                                                                                   ce_loss.detach().item(),
                                                                                                                   mem_focal,#.detach().item(),
                                                                                                                   ))
            train_loss_recorder.update(loss.detach().item())

            exp_lr_scheduler.step(epoch)
            #torch.cuda.empty_cache()

            # evaluation & save
            if step % checkpoint == 0:
                model_dict = {}
                model_dict['base'] = model
                rec_loss = []
                print(32*'-')
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()))
                print('current lr:%s' % exp_lr_scheduler.get_lr())
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(model_dict, data_loader['trainval'], 'trainval', epoch, Config)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(model_dict, data_loader['val'], 'val', epoch, Config)


                save_path = os.path.join(save_dir, 'weights__base-pick50__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))

                torch.cuda.synchronize()
                if epoch %10 == 0:# and val_acc1 > max(max_record):
                    pass
                    #torch.save(model.state_dict(), save_path)

                print('saved model to %s' % (save_path))
                max_record.append(val_acc1)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint-main__weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                #torch.save(model.state_dict(), save_path)

                torch.cuda.empty_cache()

        if epoch % 100 == 0 and epoch != 0:
            pass
            #torch.save(gather_score_all, 'gather_score_end-' + str(epoch) +  '.pt')

        gc.collect()




