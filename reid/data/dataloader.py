from __future__ import print_function
import os.path as osp
from torch.utils.data import DataLoader
from reid.dataset import get_sequence
# from reid.data import seqtransforms as T
# from reid.data import SeqTrainPreprocessor
# from reid.data import SeqTestPreprocessor
from reid.data import RandomPairSampler, RandomIdentitySampler

from . import data_manager
from . import transforms as T
from torch.utils.data import DataLoader
from .video_loader import VideoDataset


def get_data(args, dataset_name, split_id, data_dir, batch_size, seq_len, seq_srd, workers):

    # PX:
    dataset = data_manager.init_dataset(name=args.dataset, seq_len=seq_len, seq_srd=seq_srd) # NOTE: seq_len and seq_srd should be passed to dataset's init here

    transform_train = T.Compose([
        T.Random2DTranslation(256, 128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if args.use_gpu else False


    train_processor = VideoDataset(dataset.train, seq_len=args.seq_len, seq_srd = args.seq_srd, sample='dense',transform=transform_train)

    query_processor = VideoDataset(dataset.query, seq_len=args.seq_len, seq_srd = args.seq_srd, sample='dense', transform=transform_test)

    gallery_processor = VideoDataset(dataset.gallery, seq_len=args.seq_len, seq_srd = args.seq_srd, sample='dense', transform=transform_test)
    
    num_classes=dataset.num_train_pids
    


    # trainloader = DataLoader(
    #     VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
    #     sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),

    #     batch_size=args.train_batch, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=True,
    # )

    # queryloader = DataLoader(
    #     VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )

    # galleryloader = DataLoader(
    #     VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
    #     batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    #     pin_memory=pin_memory, drop_last=False,
    # )
    # :PX

    # print ("xx:",dataset.train[63494])
    train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers,
                                sampler=RandomPairSampler(dataset.train), pin_memory=True, drop_last=True)

    query_loader = DataLoader(query_processor, batch_size=8, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    gallery_loader = DataLoader(gallery_processor, batch_size=8, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_loader, query_loader, gallery_loader
