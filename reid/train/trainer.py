from __future__ import print_function, absolute_import
import time
import torch
from torch import nn
from reid.evaluator import accuracy
from utils.meters import AverageMeter
import torch.nn.functional as F
from utils import to_numpy

def pp(name, var):
    print('[%s] %r' % (name, var.shape if hasattr(var, 'shape') else None))




class BaseTrainer(object):

    def __init__(self, model_flow, criterion):
        super(BaseTrainer, self).__init__()
        self.numflow = len(model_flow)
        self.model_flow1 = model_flow[0]
        if self.numflow > 1:
            self.model_flow2 = model_flow[1]
        # self.model_flow1 = model_flow1
        # self.model_flow2 = model_flow2
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, data_loader, optimizer1, optimizer2,tensorboardWrite):
        # self.model.train()
        self.model_flow1.train()
        if self.numflow > 1:
            self.model_flow2.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)

            loss, prec_oim, prec_score = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))

            precisions.update(prec_oim, targets.size(0))
            precisions1.update(prec_score, targets.size(0))


            for o in range(len(optimizer1)):
                optimizer1[o].zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            for o in range(len(optimizer1)):
                optimizer1[o].step()
            optimizer2.step()

            tensorboardWrite.add_scalar("data/loss",loss.data,epoch)
            tensorboardWrite.add_scalar("data/prec_omi",prec_oim.data,epoch)
            tensorboardWrite.add_scalar("data/prec_score",prec_score.data,epoch)

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 50
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.5f} ({:.5f})\t'
                      'prec_oim {:.3%} ({:.3%})\t'
                      'prec_score {:.3%} ({:.3%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              precisions1.val, precisions1.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class SEQTrainer(BaseTrainer):

    def __init__(self, cnn_model, att_model, classifier_model, criterion_veri, criterion_oim, rate, flow1_rate):
        # super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
        super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.regular_criterion = criterion_oim
        self.rate = rate
        self.flow1_rate = flow1_rate

    def _parse_data(self, inputs):
        imgs, flows, pose, pids, _ = inputs
        imgs = imgs.cuda()
        flows = flows.cuda()
        pose = pose.cuda()

        inputs = [imgs, flows, pose]
        

        targets = pids.cuda()
        targets = targets.long()
        return inputs, targets

    def _forward(self, inputs, targets):
        # log
        # pp('imgs', inputs[0])
        # pp('flows', inputs[1])
        # pp('pose', inputs[2])
         
        # flow1: feat, feat_raw 
        # flow2: feat_2, feat_raw_2
        feat, feat_raw = self.model_flow1(inputs[0], inputs[1], inputs[2])
        # pp('feat', feat)
        # pp('feat_raw', feat_raw)
        feat_ = feat
        if self.numflow > 1:
            feat_2, feat_raw_2 = self.model_flow2(inputs[0], inputs[1], inputs[2])
            # feat_ = self.flow1_rate* feat + (1-self.flow1_rate)* feat_2
            # pp('feat_2', feat_2)
            # pp('feat_raw_2', feat_raw_2)


        featsize = feat.size()
        featbatch = featsize[0]
        seqlen = featsize[1]

        # expand the target label ID loss
        featX = feat_.view(featbatch * seqlen, -1)
        targetX = targets.unsqueeze(1)
        targetX = targetX.expand(featbatch, seqlen)
        targetX = targetX.contiguous()
        targetX = targetX.view(featbatch * seqlen, -1)
        targetX = targetX.squeeze(1)
        # pp('featX', featX); pp('targetX', targetX)

        loss_id, outputs_id = self.regular_criterion(featX, targetX)
        prec_id, = accuracy(outputs_id.data, targetX.data)
        # prec_id = prec_id[0]

        # verification label

        featsize = feat.size()
        sample_num = featsize[0]
        targets = targets.data
        targets = targets.view(int(sample_num / 2), -1)
        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1]

        # pooled_probe, pooled_gallery = self.att_model(feat, feat_raw,flow_idx=0)
        # pooled_probe_2, pooled_gallery_2 = self.att_model(feat_2, feat_raw_2,flow_idx=1)

        # pooled_probe, pooled_gallery = (pooled_probe + pooled_probe_2) / 2., (pooled_gallery + pooled_gallery_2) / 2.

        if self.numflow > 1:
            # -----flow2: HS, flow1: key value-----
            probe_x, gallery_x, HS = self.att_model(feat_2, feat_raw_2, flow_idx=1)
            # pp('probe_x', probe_x)
            # pp('gallery_x', gallery_x)
            probe_x = probe_x.sum(1)
            probe_x = probe_x.squeeze(1)
            gallery_x = gallery_x.sum(1)
            gallery_x = gallery_x.squeeze(1)
            pooled_probe, pooled_gallery = self.att_model(feat, feat_raw, flow_idx=0, HS=HS)
            # pp('pooled_probe', pooled_probe)
            # pp('pooled_gallery', pooled_gallery)

            # pooled_probe, pooled_gallery = (probe_x + pooled_probe) / 2., (galley_x + pooled_gallery) / 2.
            pooled_probe, pooled_gallery = (1-self.flow1_rate) * probe_x + self.flow1_rate * pooled_probe, (1-self.flow1_rate) * gallery_x + self.flow1_rate* pooled_gallery

            # ----------
        else:
            # only one flow
            pooled_probe, pooled_gallery = self.d(feat, feat_raw,flow_idx=-1)

        encode_scores = self.classifier_model(pooled_probe, pooled_gallery)

        encode_size = encode_scores.size()
        encodemat = encode_scores.view(-1, 2)
        encodemat = F.softmax(encodemat)
        encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
        encodemat = encodemat[:, :, 1]

        loss_ver, prec_ver = self.criterion(encodemat, tar_probe, tar_gallery)
        # PX: 
        if loss_ver > 0:
            loss = loss_id * self.rate + 100 * loss_ver
            # loss = loss_id + 50 * loss_ver
        else:
            loss = loss_id
        # PX

        return loss, prec_id, prec_ver

    def train(self, epoch, data_loader, optimizer1, optimizer2, rate, tensorboardWrite):
        self.att_model.train()
        self.classifier_model.train()
        self.rate = rate
        super(SEQTrainer, self).train(epoch, data_loader, optimizer1, optimizer2, tensorboardWrite)
