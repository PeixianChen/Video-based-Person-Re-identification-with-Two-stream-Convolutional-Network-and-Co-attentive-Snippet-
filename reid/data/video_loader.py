from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import os.path as osp

# def read_image(img_path):
#     """Keep reading image until succeed.
#     This can avoid IOError incurred by heavy IO process."""
#     got_img = False
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#             pass
#     return img

def read_image(img_path,flow_pose_dir):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            flow_img = osp.join(flow_pose_dir[0], osp.basename(img_path))
            pose_img = osp.join(flow_pose_dir[1], osp.basename(img_path))
            img = Image.open(img_path).convert('RGB')
            flow = Image.open(flow_img).convert('RGB')
            pose = Image.open(pose_img).convert('RGB')
            got_img = True
        except IOError as err:
            print (err)
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img,flow,pose


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, seq_srd=4, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.seq_srd = seq_srd
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # img_paths, pid, camid = self.dataset[index]

        #PX:6CHENNEL
        # img_paths,flow_pose_dir, pid, camid = self.dataset[index]
        img_paths,flow_pose_dir, pid, camid = self.dataset[index]
        #PX

        num = len(img_paths)
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)

            # PX:6CHANNEL
            imgs,flows, poses = [],[],[]
            for index in indices:
                index=int(index)
                img_path = img_paths[int(index)]
                img,flow,pose = read_image(img_path,flow_pose_dir)
                if self.transform is not None:
                    img = self.transform(img)
                    flow = self.transform(flow)
                    pose = self.transform(pose)
                
                imgs.append(img)
                flows.append(flow)
                poses.append(pose)
            img_tensor = torch.stack(imgs, 0)
            flow_tensor = torch.stack(flows, 0)
            pose_tensor = torch.stack(poses, 0)


            return img_tensor, flow_tensor, pose_tensor, pid, camid
        # PX

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            
            # PX:6CHENNEL

            imgs,flows,poses = [],[],[]
            for index in range(len(img_paths)):
                img_path = img_paths[int(index)]
                img,flow,pose = read_image(img_path,flow_pose_dir)
                if self.transform is not None:
                    img = self.transform(img)
                    flow = self.transform(flow)
                    pose = self.transform(pose)
                # imagePixelData = torch.zeros((6, img.shape[1],img.shape[2]))
                # for c in range(3):
                #     imagePixelData[c,:,:] = img[c,:,:]
                # for c in range(2):
                #     imagePixelData[c+3,:,:] = flow[c,:,:]
                # imagePixelData[5,:,:] = pose[0,:,:]
                # imagePixelData = imagePixelData.unsqueeze(0)
                imgs.append(img)
                flows.append(flow)
                poses.append(pose)
            img_tensor = torch.stack(imgs, 0)
            flow_tensor = torch.stack(flows, 0)
            pose_tensor = torch.stack(poses, 0)
            return img_tensor, flow_tensor, pose_tensor, pid, camid
            #PX

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))







