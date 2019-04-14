from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable
from utils.meters import AverageMeter
from utils import to_torch
from .eva_functions import cmc, mean_ap
import numpy as np
import torch.nn.functional as F


def evaluate_seq(distmat, query_pids, query_camids, gallery_pids, gallery_camids, cmc_topk=(1, 5, 10, 20)):

    query_ids = np.array(query_pids)
    gallery_ids = np.array(gallery_pids)
    query_cams = np.array(query_camids)
    gallery_cams = np.array(gallery_camids)

    ##
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return mAP


class ATTEvaluator(object):

    def __init__(self, cnn_model_flow, att_model, classifier_model, rate):
        super(ATTEvaluator, self).__init__()
        self.numflow = len(cnn_model_flow)
        self.cnn_model_flow1 = cnn_model_flow[0]
        if self.numflow > 1:
            self.cnn_model_flow2 = cnn_model_flow[1]
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.rate = rate

    def extract_feature(self, data_loader):
        print_freq = 50
        self.cnn_model_flow1.eval()
        if self.numflow > 1:
            self.cnn_model_flow2.eval()
        self.att_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        allfeatures = 0
        
        for i, (imgs, flows, poses, _, _) in enumerate(data_loader):
            imgs = to_torch(imgs)
            flows = to_torch(flows)
            poses = to_torch(poses)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            imgs = imgs.cuda()
            flows = flows.cuda()
            poses = poses.cuda()
            with torch.no_grad():
                if i == 0:
                    # out_feat, out_raw = self.cnn_model(imgs, flows, poses)
                    # out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)
                    # allfeatures = out_feat
                    # allfeatures_raw = out_raw

                    out_feat,out_raw = self.cnn_model_flow1(imgs, flows, poses)
                    if self.numflow > 1:
                        out_feat_2,out_raw_2 = self.cnn_model_flow2(imgs, flows, poses)
                        # -----flow2: HS, flow1: key value-----
                        value_probe, HS_probe = self.att_model.selfpooling_model(out_feat_2, out_raw_2,singleflow=False)
                        value_probe = value_probe.sum(1)
                        value_probe = value_probe.squeeze(1)

                        out_feat,out_raw = self.att_model.selfpooling_model(out_feat, out_raw, singleflow=False, Hs=HS_probe)
                        # ----------
                        allfeatures = self.rate * out_feat + (1-self.rate) * value_probe
                    else:
                        out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)
                        allfeatures = out_feat
                    # out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)
                    # out_feat_2, out_raw_2 = self.att_model.selfpooling_model(out_feat_2, out_raw_2)

                    # allfeatures = out_feat
                    preimgs = imgs
                    preflows = flows
                    preposes = poses
                elif imgs.size(0) < data_loader.batch_size:
                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = data_loader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)
                    poses = torch.cat((poses, preposes[0:cat_batchsize]), 0)


                    # out_feat, out_raw = self.cnn_model(imgs, flows, poses)
                    # out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)

                    out_feat,out_raw = self.cnn_model_flow1(imgs, flows, poses)
                    if self.numflow > 1:
                        out_feat_2,out_raw_2 = self.cnn_model_flow2(imgs, flows, poses)
                        # -----flow2: HS, flow1: key value-----
                        value_probe, HS_probe = self.att_model.selfpooling_model(out_feat_2, out_raw_2,singleflow=False)
                        value_probe = value_probe.sum(1)
                        value_probe = value_probe.squeeze(1)
                        out_feat,out_raw = self.att_model.selfpooling_model(out_feat, out_raw, singleflow=False, Hs=HS_probe)
                        # out_feat = (out_feat +  value_probe) / 2.
                        out_feat = self.rate * out_feat + (1-self.rate) * value_probe
                        # ----------
                    else:
                        out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)
                    # out_feat_2, out_raw_2 = self.att_model.selfpooling_model(out_feat_2, out_raw_2)
                    # out_feat, out_raw = (out_feat + out_feat_2) / 2. , (out_raw + out_raw_2) / 2.
                    

                    out_feat = out_feat[0:flaw_batchsize]

                    allfeatures = torch.cat((allfeatures, out_feat), 0)
                else:
                    # out_feat, out_raw = self.cnn_model(imgs, flows, poses)
                    # out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)

                    out_feat,out_raw = self.cnn_model_flow1(imgs, flows, poses)
                    if self.numflow > 1:
                        out_feat_2,out_raw_2 = self.cnn_model_flow2(imgs, flows, poses)
                        # -----flow2: HS, flow1: key value-----
                        value_probe, HS_probe = self.att_model.selfpooling_model(out_feat_2, out_raw_2,singleflow=False)
                        value_probe = value_probe.sum(1)
                        value_probe = value_probe.squeeze(1)
                        # out_raw_2 = out_raw_2.sum(1)
                        # out_raw_2 = out_raw_2.squeeze(1)
                        out_feat,out_raw = self.att_model.selfpooling_model(out_feat, out_raw, singleflow=False, Hs=HS_probe)
                        # out_feat = (out_feat +  value_probe) / 2.
                        out_feat = self.rate * out_feat + (1-self.rate) * value_probe

                        # out_raw = (out_raw + out_raw_2) / 2.
                        # ----------
                    else:
                        out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)
                    # out_feat_2, out_raw_2 = self.att_model.selfpooling_model(out_feat_2, out_raw_2)
                    # out_feat, out_raw = (out_feat + out_feat_2) / 2. , (out_raw + out_raw_2) / 2.
                    

                    allfeatures = torch.cat((allfeatures, out_feat), 0)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        return allfeatures

    def evaluate(self, query_loader, gallery_loader, queryinfo, galleryinfo):

        
        # self.cnn_model_flow1.eval()
        # self.cnn_model_flow2.eval()
        self.cnn_model_flow1.eval()
        if self.numflow > 1:
            self.cnn_model_flow2.eval()
        self.att_model.eval()
        self.att_model.eval()
        self.classifier_model.eval()

        querypid = queryinfo.pid
        querycamid = queryinfo.camid
        querytranum = queryinfo.tranum

        gallerypid = galleryinfo.pid
        gallerycamid = galleryinfo.camid
        gallerytranum = galleryinfo.tranum

        pooled_probe = self.extract_feature(query_loader)


        querylen = len(querypid)
        gallerylen = len(gallerypid)
        # print('querylen: %2d, gallerylen: %2d' % (querylen, gallerylen))

        # online gallery extraction
        single_distmat = np.zeros((querylen, gallerylen))
        gallery_resize = 0
        gallery_popindex = 0
        gallery_popsize = gallerytranum[gallery_popindex]

        gallery_resfeatures = 0
        gallery_resraw = 0
        gallery_resfeatures_2 = 0
        gallery_resraw_2 = 0

        gallery_empty = True
        preimgs = 0
        preflows = 0

        # time
        gallery_time = AverageMeter()
        end = time.time()

        for i, (imgs, flows,poses,  _, _) in enumerate(gallery_loader):
            # print('----- gallery_loader: ', i)
            imgs = to_torch(imgs)
            flows = to_torch(flows)
            poses = to_torch(poses)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            imgs = imgs.cuda()
            flows = flows.cuda()
            poses = poses.cuda()


            with torch.no_grad():
                seqnum = imgs.size(0)
                if i == 0:
                    preimgs = imgs
                    preflows = flows
                    preposes = poses

                if gallery_empty:
                    # out_feat, out_raw, out_feat_2, out_raw_2  = self.cnn_model(imgs, flows, poses)
                    out_feat,out_raw = self.cnn_model_flow1(imgs, flows, poses)
                    if self.numflow > 1:
                        out_feat_2,out_raw_2 = self.cnn_model_flow2(imgs, flows, poses)
                        gallery_resfeatures_2 = out_feat_2
                        gallery_resraw_2 = out_raw_2

                    gallery_resfeatures = out_feat
                    gallery_resraw = out_raw
                    

                    gallery_empty = False

                elif imgs.size(0) < gallery_loader.batch_size:
                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = gallery_loader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)
                    poses = torch.cat((poses, preposes[0:cat_batchsize]), 0)
                    # out_feat, out_raw = self.cnn_model(imgs, flows,poses)

                    out_feat,out_raw = self.cnn_model_flow1(imgs, flows, poses)
                    if self.numflow > 1:
                        out_feat_2,out_raw_2 = self.cnn_model_flow2(imgs, flows, poses)
                        out_feat_2 = out_feat_2[0:flaw_batchsize]
                        out_raw_2  = out_raw_2[0:flaw_batchsize]
                        gallery_resfeatures_2 = torch.cat((gallery_resfeatures_2, out_feat_2), 0)
                        gallery_resraw_2 = torch.cat((gallery_resraw_2, out_raw_2), 0)


                    out_feat = out_feat[0:flaw_batchsize]
                    out_raw  = out_raw[0:flaw_batchsize]

                    gallery_resfeatures = torch.cat((gallery_resfeatures, out_feat), 0)
                    gallery_resraw = torch.cat((gallery_resraw, out_raw), 0)

                else:
                    # out_feat, out_raw, out_feat_2, out_raw_2  = self.cnn_model(imgs, flows, poses)
                    out_feat,out_raw = self.cnn_model_flow1(imgs, flows, poses)
                    if self.numflow > 1:
                        out_feat_2,out_raw_2 = self.cnn_model_flow2(imgs, flows, poses)
                        gallery_resfeatures_2 = torch.cat((gallery_resfeatures_2, out_feat_2), 0)
                        gallery_resraw_2 = torch.cat((gallery_resraw_2, out_raw_2), 0)

                    # out_feat, out_raw = (out_feat + out_feat_2) / 2. , (out_raw + out_raw_2) / 2.

                    gallery_resfeatures = torch.cat((gallery_resfeatures, out_feat), 0)
                    gallery_resraw = torch.cat((gallery_resraw, out_raw), 0)
                    

            gallery_resize = gallery_resize + seqnum


            while gallery_popsize <= gallery_resize:

                # print('gallery_popsize(%02d) gallery_resize(%02d) gallery_popindex(%02d)' %(gallery_popsize, gallery_resize, gallery_popindex))

                if (gallery_popindex + 1) % 50 == 0:
                    print('gallery--{:04d}'.format(gallery_popindex))
                gallery_popfeatures = gallery_resfeatures[0:gallery_popsize, :]
                gallery_popraw = gallery_resraw[0:gallery_popsize, :]
                if self.numflow > 1:
                    gallery_popfeatures_2 = gallery_resfeatures_2[0:gallery_popsize, :]
                    gallery_popraw_2 = gallery_resraw_2[0:gallery_popsize, :]

                if gallery_popsize < gallery_resize:
                    gallery_resfeatures = gallery_resfeatures[gallery_popsize:gallery_resize, :]
                    gallery_resraw = gallery_resraw[gallery_popsize:gallery_resize, :]
                    if self.numflow > 1:
                        gallery_resfeatures_2 = gallery_resfeatures_2[gallery_popsize:gallery_resize, :]
                        gallery_resraw_2 = gallery_resraw_2[gallery_popsize:gallery_resize, :]
                else:
                    gallery_resfeatures = 0
                    gallery_resraw = 0

                    gallery_resfeatures_2 = 0
                    gallery_resraw_2 = 0

                    gallery_empty = True

                gallery_resize = gallery_resize - gallery_popsize


                if self.numflow > 1:
                    # -----flow2: HS, flow1: key value-----
                    value_gallery, HS_gallery = self.att_model.selfpooling_model(gallery_popfeatures_2, gallery_popraw_2,singleflow=False)
                    value_gallery = value_gallery.sum(1)
                    value_gallery = value_gallery.squeeze(1)
                    pooled_gallery,pooled_raw = self.att_model.selfpooling_model(gallery_popfeatures, gallery_popraw, singleflow=False, Hs=HS_gallery)
                    # pooled_gallery = (pooled_gallery + value_gallery) / 2.
                    pooled_gallery = self.rate * pooled_gallery + (1-self.rate) * value_gallery
                    # ----------
                else:
                    pooled_gallery, pooled_raw = self.att_model.selfpooling_model(gallery_popfeatures, gallery_popraw)

                    
                probesize = pooled_probe.size()
                gallerysize = pooled_gallery.size()
                probe_batch = probesize[0]
                gallery_batch = gallerysize[0]
                gallery_num = gallerysize[1]
                pooled_gallery.unsqueeze(0)
                pooled_gallery = pooled_gallery.expand(probe_batch, gallery_batch, gallery_num)

                encode_scores = self.classifier_model(pooled_probe, pooled_gallery)

                encode_size = encode_scores.size()
                encodemat = encode_scores.view(-1, 2)
                encodemat = F.softmax(encodemat)
                encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
                distmat_qall_g = encodemat[:, :, 0]

                q_start = 0
                # print('access distmat[0-%d, %d]' % (len(querytranum), gallery_popindex))
                for qind, qnum in enumerate(querytranum):
                    distmat_qg = distmat_qall_g[q_start:q_start + qnum, :]
                    distmat_qg = distmat_qg.data.cpu().numpy()
                    percile = np.percentile(distmat_qg, 20)

                    if distmat_qg[distmat_qg <= percile] is not None:
                        distmean = np.mean(distmat_qg[distmat_qg <= percile])
                    else:
                        distmean = np.mean(distmat_qg)

                    single_distmat[qind, gallery_popindex] = distmean
                    q_start = q_start + qnum

                gallery_popindex = gallery_popindex + 1

                if gallery_popindex < gallerylen:
                    gallery_popsize = gallerytranum[gallery_popindex]
                gallery_time.update(time.time() - end)
                end = time.time()

        return evaluate_seq(single_distmat, querypid, querycamid, gallerypid, gallerycamid)
