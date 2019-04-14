from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np

from .reid_utils import mkdir_if_missing, write_json, read_json


"""Dataset classes"""

class DukeMTMCVidReID(object):
    """
    DukeMTMCVidReID

    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.

    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID
    
    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train) + 2636 (test)
    """
    dataset_dir = 'dukemtmc-vidreid'

    def __init__(self, root='data', min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/gallery')
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self.min_seq_len = min_seq_len
        self._download_data()
        self._check_before_run()
        print("Note: if root path is changed, the previously generated json files need to be re-generated (so delete them first)")

        train = self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query = self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, _, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, _, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, _, self.num_gallery_cams = self.get_videodata_info(self.gallery)

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(train, return_tracklet_stats=True)
        
        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(query, return_tracklet_stats=True)
        
        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")
    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-VideoReID dataset")
        urllib.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, cams, tracklet_stats = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing '{}' with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
        }
        write_json(split_dict, json_path)

        return tracklets

class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = './data/mars'
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, seq_len=10, seq_srd=4, min_seq_len=0):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train, num_train_tracklets, num_train_pids, num_train_imgs,_,_,_ = \
          self._process_data(train_names, track_train, seq_len, seq_srd, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)
        query, num_query_tracklets, num_query_pids, num_query_imgs,query_pid,query_camid,query_num = \
          self._process_data(test_names, track_query, seq_len, seq_srd, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs,gallery_pid,gallery_camid,gallery_num = \
          self._process_data(test_names, track_gallery, seq_len, seq_srd, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_tracklets = num_train_tracklets

        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = query_num

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = gallery_num

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, seq_len, seq_srd, home_dir=None, relabel=False, min_seq_len=0):
        # assert home_dir in ['bbox_train', 'bbox_test']
        # num_tracklets = meta_data.shape[0]
        # pid_list = list(set(meta_data[:,2].tolist()))
        # num_pids = len(pid_list)

        # if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        # tracklets = []
        # pids = []
        # camids = []
        # tra_num = []
        # num_imgs_per_tracklet = []

        # for tracklet_idx in range(num_tracklets):
        #     data = meta_data[tracklet_idx,...]
        #     start_index, end_index, pid, camid = data
        #     if pid == -1: continue # junk images are just ignored
        #     assert 1 <= camid <= 6
        #     if relabel: 
        #         pid = pid2label[pid]
        #     camid -= 1 # index starts from 0
        #     img_names = names[start_index-1:end_index]

        #     # make sure image names correspond to the same person
        #     pnames = [img_name[:4] for img_name in img_names]
        #     assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

        #     # make sure all images are captured under the same camera
        #     camnames = [img_name[5] for img_name in img_names]
        #     assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

        #     # append image names with directory information
        #     img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
        #     if len(img_paths) >= min_seq_len:
        #         img_paths = tuple(img_paths)
        #         tracklets.append((img_paths, [], pid, camid))
        #         num_imgs_per_tracklet.append(len(img_paths))

        # num_tracklets = len(tracklets)

        # return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,pids,camids,tra_num
        #############################PX#################################
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        pids = []
        camids = []
        tra_num = []
        num_imgs_per_tracklet = []
        person_dir_rgb = "mars_rgb"
        person_dir_flow = "mars_flow"
        person_dir_pose = "mars_pose"
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = sorted(names[start_index-1:end_index])

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"


            seqall = len(img_names)
            snippet_start_ind = 0
            seq_inds = [(snippet_start_ind, snippet_start_ind + seq_len) for snippet_start_ind in range(0, seqall - seq_len, seq_srd)]
            if not seq_inds:
                seq_inds = [(0, seqall)]
            for seq_ind in seq_inds:
                imgseq = img_names[seq_ind[0]:seq_ind[1]]
                while len(imgseq) < seq_len:
                    imgseq.append(imgseq[-1])
            # append image names with directory information
                img_paths = [osp.join(self.root, person_dir_rgb, home_dir, img_name[:4], img_name) for img_name in imgseq]
            # if len(img_paths) >= min_seq_len:
                # img_paths = tuple(img_paths)
                flow_paths = osp.join(self.root,person_dir_flow,home_dir,imgseq[0][:4])
                pose_paths = osp.join(self.root,person_dir_pose,home_dir,imgseq[0][:4])
                tracklets.append((tuple(img_paths), [flow_paths, pose_paths], pid, camid))
            num_imgs_per_tracklet.append(len(img_names))
            pids.append(pid)
            camids.append(camid)
            tra_num.append(len(seq_inds))

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,pids,camids,tra_num
        ################################################################
class infostruct(object):
    pass

class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    root = './data/ilids-vid'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'

    # PX 6CHANNEL
    data_dir = osp.join(root, 'i-LIDS-VID')
    flow_dir = osp.join(root, 'i-LIDS-VID-FLOW')
    pose_dir = osp.join(root, 'i-LIDS-VID-POSE')
    split_dir = osp.join(root, 'train-test people splits')
    split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
    split_path = osp.join(root, 'splits.json')
    cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
    cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')
    cam_1_path_flow = osp.join(flow_dir,'sequences/cam1')
    cam_2_path_flow = osp.join(flow_dir,'sequences/cam2')
    cam_1_path_pose = osp.join(pose_dir,'sequences/cam1')
    cam_2_path_pose = osp.join(pose_dir,'sequences/cam2')
    # PX

    def __init__(self, seq_len=10, seq_srd=4, split_id=0):
        print(seq_len, seq_srd, '-'*20)
        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        print ("train")
        train, num_train_tracklets, num_train_pids, num_imgs_train,_,_,_ = \
          self._process_data(train_dirs, seq_len, seq_srd, cam1=True, cam2=True)
        print ("query")
        query, num_query_tracklets, num_query_pids, num_imgs_query,query_pid,query_camid,query_num = \
          self._process_data(test_dirs, seq_len, seq_srd, cam1=True, cam2=False)
        print ("gallary")
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery,gallery_pid,gallery_camid,gallery_num = \
          self._process_data(test_dirs, seq_len, seq_srd, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")
        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_tracklets = num_train_tracklets

        # self.query, query_pid, query_camid, query_num = self._pluckseq_cam(self.identities, self.split['query'],
        #                                                                    seq_len, seq_srd, 0)
        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = query_num

        # self.gallery, gallery_pid, gallery_camid, gallery_num = self._pluckseq_cam(self.identities,
        #                                                                            self.split['gallery'],
        #                                                                            seq_len, seq_srd, 1)
        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = gallery_num

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, seq_len, seq_srd, cam1=True, cam2=True):
        tracklets = []
        pids = []
        camids = []
        tra_num = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        # dirname2pid = [re.sub("\D","",dirname) for i, dirname in enumerate(dirnames)]

        # ##################################################################
        # tracklets = []
        # pids = []
        # camids = []
        # tra_num = []
        # num_imgs_per_tracklet = []
        # # dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}

        # self.cam_rgb_path  = [self.cam_1_path, self.cam_2_path]
        # self.cam_flow_path = [self.cam_1_path_flow, self.cam_2_path_flow]
        # self.cam_pose_path = [self.cam_1_path_pose, self.cam_2_path_pose]
        # for pid, dirname in enumerate(dirnames):
        #     #PX:6CHENNEL
        #     for cami in range(len(camnum)):
        #         if camnum[cami]:
        #             person_dir = osp.join(self.cam_rgb_path[cami], dirname) 
        #             person_dir_flow = osp.join(self.cam_flow_path[cami], dirname)
        #             person_dir_pose = osp.join(self.cam_pose_path[cami], dirname)
        #             img_names = glob.glob(osp.join(person_dir, '*.png'))
        #             assert len(img_names) > 0
        #             img_names = sorted(img_names)
        #             seqall = len(img_names)
        #             start_ind = 0
        #             seq_inds = [(start_ind, start_ind + seq_len) for start_ind in range(0, seqall - seq_len, seq_srd)]
        #             if not seq_inds:
        #                 seq_inds = [(0, seqall)]
        #             for seq_ind in seq_inds:
        #                 imgseq = img_names[seq_ind[0]:seq_ind[1]]
        #                 while len(imgseq) < seq_len:
        #                     imgseq.append(imgseq[-1])
        #                 tracklets.append((tuple(imgseq), [person_dir_flow,person_dir_pose],pid, cami))
        #             num_imgs_per_tracklet.append(len(img_names))
        #             pids.append(pid)  
        #             camids.append(0)
        #             tra_num.append(len(seq_inds))

        # ##################################################################
        
        for dirname in dirnames:
            #PX:6CHENNEL
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname) 
                person_dir_flow = osp.join(self.cam_1_path_flow, dirname)
                person_dir_pose = osp.join(self.cam_1_path_pose, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = sorted(img_names)
                pid = dirname2pid[dirname]
                seqall = len(img_names)
                start_ind = 0
                seq_inds = [(start_ind, start_ind + seq_len) for start_ind in range(0, seqall - seq_len, seq_srd)]
                if not seq_inds:
                    seq_inds = [(0, seqall)]
                for seq_ind in seq_inds:
                    imgseq = img_names[seq_ind[0]:seq_ind[1]]
                    while len(imgseq) < seq_len:
                        imgseq.append(imgseq[-1])
                    tracklets.append((tuple(imgseq), [person_dir_flow,person_dir_pose],pid, 0))
                num_imgs_per_tracklet.append(len(img_names))
                pids.append(pid)  
                camids.append(0)
                tra_num.append(len(seq_inds))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                person_dir_flow = osp.join(self.cam_2_path_flow, dirname)
                person_dir_pose = osp.join(self.cam_2_path_pose, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))    
                assert len(img_names) > 0
                img_names = sorted(img_names) 
                pid = dirname2pid[dirname]
                seqall = len(img_names)
                start_ind = 0
                seq_inds = [(start_ind, start_ind + seq_len) for start_ind in range(0, seqall - seq_len, seq_srd)]
                if not seq_inds:
                    seq_inds = [(0, seqall)]
                for seq_ind in seq_inds:
                    imgseq = img_names[seq_ind[0]:seq_ind[1]]
                    while len(imgseq) < seq_len:
                        imgseq.append(imgseq[-1])
                    tracklets.append((tuple(imgseq), [person_dir_flow,person_dir_pose],pid, 1))
                num_imgs_per_tracklet.append(len(img_names))
                pids.append(pid)  
                camids.append(1)
                tra_num.append(len(seq_inds))
            #PX

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,pids,camids,tra_num

class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = './data/prid2011'
    # dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    # split_path = osp.join(root, 'splits_prid2011.json')
    # cam_a_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_a')
    # cam_b_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_b')

    # PX 6CHANNEL
    rgb_dir = osp.join(root, 'prid_2011')
    flow_dir = osp.join(root, 'prid_2011_flow')
    pose_dir = osp.join(root, 'prid_2011_pose')
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(rgb_dir, 'multi_shot', 'cam_a')
    cam_b_path = osp.join(rgb_dir, 'multi_shot', 'cam_b')
    cam_a_path_flow = osp.join(flow_dir, 'multi_shot', 'cam_a_f')
    cam_b_path_flow = osp.join(flow_dir, 'multi_shot', 'cam_b_f')
    cam_a_path_pose = osp.join(pose_dir, 'multi_shot', 'cam_a')
    cam_b_path_pose = osp.join(pose_dir, 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, seq_len=10, seq_srd=4, min_seq_len=0):
        self._check_before_run()
        # self._prepare_split()

        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train, _,_,_ = \
          self._process_data(train_dirs, seq_len, seq_srd, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query, query_pid, query_camid, query_num = \
          self._process_data(test_dirs, seq_len, seq_srd, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery, gallery_pid, gallery_camid, gallery_num = \
          self._process_data(test_dirs, seq_len, seq_srd, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_tracklets = num_train_tracklets

        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = query_num

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = gallery_num

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.rgb_dir))
    def _process_data(self, dirnames, seq_len, seq_srd, cam1=True, cam2=True):
        tracklets = []
        pids = []
        camids = []
        tra_num = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        for dirname in dirnames:
            #PX:6CHENNEL
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                person_dir_flow = osp.join(self.cam_a_path_flow, dirname)
                person_dir_pose = osp.join(self.cam_a_path_pose, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = sorted(img_names)
                pid = dirname2pid[dirname]
                seqall = len(img_names)
                start_ind = 0
                seq_inds = [(start_ind, start_ind + seq_len) for start_ind in range(0, seqall - seq_len, seq_srd)]
                if not seq_inds:
                    seq_inds = [(0, seqall)]
                for seq_ind in seq_inds:
                    imgseq = img_names[seq_ind[0]:seq_ind[1]]
                    while len(imgseq) < seq_len:
                        imgseq.append(imgseq[-1])
                    tracklets.append((tuple(imgseq), [person_dir_flow,person_dir_pose],pid, 0))
                num_imgs_per_tracklet.append(len(img_names))
                pids.append(pid)  
                camids.append(0)
                tra_num.append(len(seq_inds))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                person_dir_flow = osp.join(self.cam_b_path_flow, dirname)
                person_dir_pose = osp.join(self.cam_b_path_pose, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))    
                assert len(img_names) > 0
                img_names = sorted(img_names) 
                pid = dirname2pid[dirname]
                seqall = len(img_names)
                start_ind = 0
                seq_inds = [(start_ind, start_ind + seq_len) for start_ind in range(0, seqall - seq_len, seq_srd)]
                if not seq_inds:
                    seq_inds = [(0, seqall)]
                for seq_ind in seq_inds:
                    imgseq = img_names[seq_ind[0]:seq_ind[1]]
                    while len(imgseq) < seq_len:
                        imgseq.append(imgseq[-1])
                    tracklets.append((tuple(imgseq), [person_dir_flow,person_dir_pose],pid, 1))
                num_imgs_per_tracklet.append(len(img_names))
                pids.append(pid)  
                camids.append(1)
                tra_num.append(len(seq_inds))
            #PX

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        # return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,pids,camids,tra_num

"""Create dataset"""

__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

# if __name__ == '__main__':
#     # test
#     #dataset = Market1501()
#     #dataset = Mars()
#     dataset = iLIDSVID()
#     dataset = PRID()







