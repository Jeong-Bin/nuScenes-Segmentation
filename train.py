"""
Fine-tune a Mask-RCNN model, pretrained on the COCO dataset,
to predict instance masks on the NuImages dataset.

Reference:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import os
import time
import argparse
import logging
import wandb

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from nuimages import NuImages
from nuimages_dataset import NuImagesDataset

from utils.functions import *
from utils.loss import apply_focal_loss
from utils.engine import Solver
#rom utils.model_utils import collate_fn
from utils.functions import collate_fn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation
from utils.transforms import *
from utils.models import Mask2FormerFocal
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser()

parser.add_argument('--wandb', type=int, default=0)
parser.add_argument('--wandb_project', type=str, default="nuimage_seg")
parser.add_argument('--training_name', type=str, default="debug") #icvt2cvt_pre_cold_veh_noise04  icvt2cvt_main_cold_veh
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--dataroot', type=str, default='/home/etri/DATASET/nuimages')

parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--gpu_idx_ddp', type=str, default='0,1,2,3')
parser.add_argument('--ddp', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--drop_last', type=int, default=0)

parser.add_argument('--model_name', type=str, default="facebook/mask2former-swin-small-cityscapes-semantic")



parser.add_argument('--img_h', type=int, default=224) # 224 or 448
parser.add_argument('--img_w', type=int, default=480) # 480 or 960 
parser.add_argument('--img_top_crop', type=int, default=46)
parser.add_argument('--remove_empty', type=int, default=1)
parser.add_argument('--custom_aug', type=int, default=1)
parser.add_argument('--val_keep', type=float, default=1.0)

parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--save_every', type=int, default=3)
parser.add_argument('--remain_num_chkpts', type=int, default=3)

parser.add_argument('--optimizer_type', type=str, default='AdamW')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--lr_schd_type', type=str, default='OnecycleLR', choices=['StepLR', 'OnecycleLR'])
parser.add_argument('--div_factor', type=float, default=10.0)
parser.add_argument('--pct_start', type=float, default=0.1)
parser.add_argument('--final_div_factor', type=float, default=100.0)

parser.add_argument('--target_tag', type=str, default='DVP')
parser.add_argument('--targets', type=str, nargs='*', default=['driveable', 'vehicle', 'pedestrian'], choices=['driveable', 'vehicle', 'pedestrian'])
parser.add_argument('--bg_plus', type=int, default=1)

parser.add_argument('--custom_loss', type=int, default=0)
parser.add_argument('--w_bg', type=float, default=1.0)
parser.add_argument('--w_dri', type=float, default=1.0)
parser.add_argument('--w_veh', type=float, default=1.0)
parser.add_argument('--w_ped', type=float, default=1.0)
parser.add_argument('--iou_threshold', type=float, default=0.5)
parser.add_argument('--random_seed', type=int, default=2025)

args = parser.parse_args()

def main(args):
    seed_fixer(args.random_seed)

    args.targets = []
    if 'D' in args.target_tag :
        args.targets.append('driveable')
    if 'V' in args.target_tag :
        args.targets.append('vehicle')
    if 'P' in args.target_tag :
        args.targets.append('pedestrian')

    logging.basicConfig(
        filename=args.save_dir + '/training.log',
        filemode="w",
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.DEBUG,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logger = logging.getLogger(__name__)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(level=logging.DEBUG)
    logger.addHandler(consoleHandler)

    # model_name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    # processor = SegformerImageProcessor.from_pretrained(model_name)
    # model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    #if len(args.targets) == 1:
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=len(args.targets)+1, 
        ignore_mismatched_sizes=True
    )
    # else:
    #     class_weights = torch.tensor([
    #         1.0,             # background (0) - 가중치를 0으로 설정
    #         args.w_veh,      # vehicle (1)
    #         args.w_dri,      # driveable (2)
    #         args.w_ped,      # pedestrian (3)
    #     ])
        
    #     model = Mask2FormerFocal.from_pretrained(
    #         args.model_name,
    #         num_labels=len(args.targets)+1, 
    #         ignore_mismatched_sizes=True,
    #         class_weights=class_weights
    #     )
    weight_dict = {'driveable': args.w_dri,
                   'vehicle': args.w_veh,
                   'pedestrian': args.w_ped}
    class_weights = [args.w_bg]
    for target in args.targets:
        class_weights.append(weight_dict[target])
    class_weights.append(0.1) # non-object
    
    #class_weights = [0.01, 0.01, 0.01, 10.0, 0.01]
    model.criterion.empty_weight = torch.tensor(class_weights)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    processor.num_labels = len(args.targets)+1
    processor.size['height'] = args.img_h
    processor.size['width'] = args.img_w
    
    # processor.do_normalize = True 
    # processor.image_mean = [0.363, 0.367, 0.355]
    # processor.image_std = [0.173, 0.169, 0.172]
    
    if args.custom_aug:
        train_transforms = Compose([
            Resize((args.img_h + args.img_top_crop, args.img_w)),  
            TopCrop(args.img_top_crop),
            RandomHorizontalFlip(p=0.5), 
            #RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.9),  
            # GaussianBlur(kernel_size=3, sigma=(0.5, 1.5), p=0.5),
            ToTensor(), 
            #MultiRandomErasing(scale=(0.01, 0.03), ratio=(0.3, 3.3), values=(0.1, 1), max_nums=3, p1=0.5, p2=0.7), # aug1
            #MultiRandomErasing(scale=(0.005, 0.01), ratio=(0.3, 3.3), values=(0.1, 1), max_nums=5, p1=0.5, p2=0.6), # aug2
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            Normalize(mean=[0.363, 0.367, 0.355], std=[0.173, 0.169, 0.172])
            
        ])
        val_transforms = Compose([
            Resize((args.img_h + args.img_top_crop, args.img_w)),  
            TopCrop(args.img_top_crop),
            ToTensor(),  
            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            Normalize(mean=[0.363, 0.367, 0.355], std=[0.173, 0.169, 0.172])
        ])

        # mean = [255*x for x in [0.363, 0.367, 0.355]] # [0.485, 0.456, 0.406]
        # std = [255*x for x in [0.173, 0.169, 0.172]] # [0.229, 0.224, 0.225]
        # train_transforms = A.Compose([
        #     A.Resize(args.img_h + args.img_top_crop, args.img_w),
        #     TopCrop(args.img_top_crop),
        #     A.HorizontalFlip(p=0.5),  
        #     #A.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.0, hue=0.05, p=0.8),  
        #     #A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=1),  
        #     A.RandomFog(alpha_coef=0.3, fog_coef_range=[0.2, 0.9], p=0.3),
        #     A.Normalize(mean=mean, std=std),  
            
        #     ToTensorV2()
        # ])

        # val_transforms = A.Compose([
        #     A.Resize(args.img_h + args.img_top_crop, args.img_w),  
        #     TopCrop(args.img_top_crop),
        #     A.Normalize(mean=mean, std=std),
        #     ToTensorV2()
        # ])

        processor.do_resize = False       
        processor.do_rescale = False      
        processor.do_normalize = False  
    else: 
        train_transforms = None
        val_transforms = None
      

    

    # if len(args.targets) != 1:
    #     model.config.ignore_value = len(args.targets)+1 # background의 label(=class 개수-1+1)은 무시.
    # else:
    #     model.config.ignore_value = 0



    train_nuimages = NuImages(dataroot=args.dataroot, version="v1.0-train", verbose=True, lazy=False)
    val_nuimages = NuImages(dataroot=args.dataroot, version="v1.0-val", verbose=True, lazy=False)

    #collate_fn = None
    if args.ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx_ddp
        backend = 'nccl'
        dist_url = 'env://'
        rank = int(os.environ['RANK'])
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) #int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend=backend, init_method=dist_url, rank=rank, world_size=world_size)
        dist.barrier()
        if rank==0: print(f'DDP 사용')

        # Model
        torch.cuda.set_device(rank)
        model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=False)
        #class_weights = class_weights.to(rank)

        # Dataset
        train_dataset = NuImagesDataset(args, train_nuimages, processor=processor, transforms=train_transforms, remove_empty=args.remove_empty, mode='train', rank=rank, logger=logger)
        val_dataset = NuImagesDataset(args, val_nuimages, processor=processor, transforms=val_transforms, remove_empty=args.remove_empty, mode='val', rank=rank, logger=logger)

        if args.val_keep < 1.0:
            train_dataset, val_dataset = augment_train_with_val(train_dataset, val_dataset, val_keep=args.val_keep)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=args.drop_last)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, drop_last=args.drop_last)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        rank, world_size, local_rank = 0, 1, 0
        print('DDP 아님')

        # Model
        model = model.cuda()
        #class_weights = class_weights.cuda()

        # Dataset
        train_dataset = NuImagesDataset(args, train_nuimages, processor=processor, transforms=train_transforms, remove_empty=args.remove_empty, mode='train', rank=rank, logger=logger)
        val_dataset = NuImagesDataset(args, val_nuimages, processor=processor, transforms=val_transforms, remove_empty=args.remove_empty, mode='val', rank=rank, logger=logger)

        if args.val_keep < 1.0:
            train_dataset, val_dataset = augment_train_with_val(train_dataset, val_dataset, val_keep=args.val_keep)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )


    # model = apply_focal_loss(
    #     model,
    #     class_weights,
    #     alpha=0.5,  # 클래스 불균형 조정
    #     gamma=2.0   # 잘못 분류된 샘플에 대한 가중치
    # )


    len_train_dataset = len(train_dataset)
    len_val_dataset =len(val_dataset)
    logger.info(f"{len_train_dataset} training samples and {len_val_dataset} val samples.")
    
    solver = Solver(args, model, len_train_dataset, len_val_dataset, rank, world_size, logger)

    if args.wandb and rank==0:
        CFG = solver.wandb_tracker()
        run_epoch = run_wandb(args, wandb)
        run_epoch.config.update({"args":vars(args)})
    
    if rank==0: print_training_info(args, logger)

    for epoch in range(1, args.num_epochs+1):
        start = time.time()
        solver.train_one_epoch(args, train_dataloader, epoch)
        end = time.time()

        time_left = (end - start) * (args.num_epochs - epoch - 1) / 3600.0
        solver.normalize_loss_tracker()
        solver.print_epoch_status(epoch, time_left)
        
        if args.val_keep > 0:
            if epoch<=1 or (epoch % args.save_every == 0) or (epoch >= args.num_epochs-5):
                solver.evaluate(args, val_dataloader, processor, epoch)
        else:
            if epoch >= args.num_epochs-3:
                solver.evaluate(args, val_dataloader, processor, epoch)

        if args.wandb and rank==0:
            CFG = solver.wandb_tracker()
            run_epoch.log({ 
                "Train Loss": CFG['Loss'],
                "Valid mIoU": CFG['mIoU'],
                }, step=epoch)
        solver.reset_loss_tracker()

    logger.info("The training has been completed! 🚩")

if __name__ == "__main__":
    args.save_dir = os.path.join(args.save_dir, args.training_name)
    if args.save_dir != '' and not os.path.exists(args.save_dir):
        try: os.makedirs(args.save_dir)
        except: print(f'>> [{args.save_dir}] seems to already exist!!')

    main(args)

