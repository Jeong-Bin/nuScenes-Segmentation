"""
Source: https://github.com/pytorch/vision
"""
import time
import math
import sys, os
import torch
import torch.functional as F

# from utils.coco_utils import get_coco_api_from_dataset
# from utils.coco_eval import CocoEvaluator
import utils
from utils.loss import *
from utils.functions import *
import torchvision.transforms as T
from utils.visualizer import *
from PIL import Image
import numpy as np


class Solver:
    def __init__(self, args, model, len_train_dataset, len_val_dataset, rank, world_size, logger):
        self.model = model
        self.rank = rank
        self.logger = logger
        self.len_val = len_val_dataset / (args.batch_size * world_size)

        #params = [p for p in model.parameters() if p.requires_grad]
        init_lr = args.learning_rate / args.div_factor
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=init_lr, weight_decay=args.weight_decay)

        if args.lr_schd_type == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        elif args.lr_schd_type == 'OnecycleLR':
            if args.drop_last:
                self.num_batches = int(len_train_dataset / (args.batch_size * world_size))
            else:
                self.num_batches = math.ceil(len_train_dataset / (args.batch_size * world_size))

            total_steps = args.num_epochs * self.num_batches #-1 # 가장 첫 번째 step 제외
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                                max_lr=args.learning_rate,
                                                                div_factor=args.div_factor, 
                                                                pct_start=args.pct_start, 
                                                                final_div_factor=args.final_div_factor,
                                                                total_steps=total_steps,
                                                                cycle_momentum=False)

        # if lr_scheduler = None
        #     if epoch == 1:
        #         warmup_factor = 1. / 1000
        #         warmup_iters = min(1000, len(data_loader) - 1)

        #         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        self.monitor = {'iter': 0,
                        'loss':0,
                        'total_loss': 0,
                        'prev_IoU': 0,
                        'current_mIoU': 0,
                        'cur_lr': args.learning_rate}

        self.iou_scores = {}

        if len(args.targets) == 1:
            self.class_id = {args.targets[0]: 1}
            self.iou_scores[args.targets[0]] = 0

        elif len(args.targets) == 2:
            self.class_id = {'driveable': 1,
                             'vehicle': 2}
            for target in args.targets:
                self.iou_scores[target] = 0

        elif len(args.targets) == 3:
            self.class_id = {'driveable': 1,
                             'vehicle': 2,
                             'pedestrian': 3}
            for target in args.targets:
                self.iou_scores[target] = 0
            
        self.class_id2name = {
            0: 'background',
            1: 'driveable',
            2: 'vehicle',
            3: 'pedestrian'
        }

    def train_one_epoch(self, args, data_loader, epoch):
        
        seed_fixer(args.random_seed)
        self.model.train()

        #for b, (pixel_values, pixel_masks, mask_labels, class_labels) in enumerate(data_loader):
        for b, batch in enumerate(data_loader):
            start_batch = time.time()
            # pixel_values = torch.stack(list(pixel_value for pixel_value in pixel_values), dim=0).to(self.rank)
            # pixel_masks = torch.stack(list(pixel_mask for pixel_mask in pixel_masks), dim=0).to(self.rank)
            
            # # mask_labels = list(mask_label[0].to(self.rank) for mask_label in mask_labels)
            # # class_labels = list(class_label[0].to(self.rank) for class_label in class_labels)
            # mask_labels = [mask_label.to(self.rank) for mask_label in mask_labels]
            # class_labels = [class_label.to(self.rank) for class_label in class_labels]

            # if type(pixel_values) in {list, tuple}:
            #     pixel_values = torch.stack(pixel_values, dim=0)
            # if type(pixel_masks) in {list, tuple}:
            #     pixel_masks = torch.stack(pixel_masks, dim=0)
            # if type(mask_labels) in {list, tuple}:
            #     mask_labels = torch.stack(mask_labels, dim=0)
            # if type(class_labels) in {list, tuple}:
            #     class_labels = torch.stack(class_labels, dim=0)


            # # 이제 모든 텐서가 동일한 크기를 가지므로 직접 스택 가능
            # pixel_values = pixel_values.to(self.rank) # [b, 3, h, w]
            # pixel_masks = pixel_masks.to(self.rank)   # [b, h, w]
            # mask_labels = mask_labels.to(self.rank)   # [c+1, h, w]
            # # class_labels = class_labels.to(self.rank) # [b, c+1]

            # #mask_labels = [mask_label.to(self.rank) for mask_label in mask_labels]
            # class_labels = [class_label.to(self.rank) for class_label in class_labels]

            # inputs = {
            #     "pixel_values": pixel_values,
            #     "pixel_mask": pixel_masks,
            #     "mask_labels": mask_labels,
            #     "class_labels": class_labels
            # }

            inputs = {
                "pixel_values": batch["pixel_values"].to(self.rank),
                "pixel_mask": batch["pixel_mask"].to(self.rank),
                "mask_labels": [mask_label.to(self.rank) for mask_label in batch["mask_labels"]],
                "class_labels": [class_label.to(self.rank) for class_label in batch["class_labels"]]
            }

            outputs = self.model(**inputs) # [b, c+1, h, w]
            if not args.custom_loss:
                loss = outputs.loss
            else:
                # 클래스 예측에 대한 focal loss
                class_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
                
                # 각 쿼리의 예측 중 클래스별 최대값을 선택하여 최종 클래스 예측을 얻음
                class_predictions = class_logits.max(dim=1)[0]  # [batch_size, num_classes+1]
                
                # focal loss 계산
                class_loss = self.class_focal_loss(class_predictions, class_labels.float())

                # 마스크 예측에 대한 focal loss
                mask_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height/4, width/4]
                mask_preds = torch.nn.functional.interpolate(mask_logits, size=(args.img_h, args.img_w), mode="bilinear", align_corners=False)  # [batch_size, num_queries, height, width]
                
                # 클래스별 마스크 생성
                final_masks = torch.zeros(args.batch_size, args.num_classes+1, args.img_h, args.img_w).to(mask_preds.device)
                
                # 각 쿼리의 마스크를 해당하는 클래스에 할당
                class_labels = class_logits.argmax(dim=-1)  # [batch_size, num_queries]
                batch, query, h, w = mask_logits.shape
                
                # inplace 연산을 피하기 위해 새로운 방식으로 구현
                for c in range(args.num_classes+1):
                    # 현재 클래스에 해당하는 쿼리들의 마스크만 선택
                    class_mask = (class_labels == c).unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_queries, 1, 1]
                    class_mask = class_mask.expand(-1, -1, args.img_h, args.img_w)  # [batch_size, num_queries, height, width]
                    
                    # 해당 클래스의 모든 쿼리 마스크에 대해 최대값 계산
                    masked_preds = mask_preds * class_mask.float()
                    final_masks[:, c] = masked_preds.max(dim=1)[0]
                
                # loss 계산을 위한 reshape
                mask_preds = final_masks.reshape(-1, args.img_h * args.img_w)  # [batch_size * (num_classes+1), height * width]
                target_masks = inputs["mask_labels"].reshape(-1, args.img_h * args.img_w)  # [batch_size * (num_classes+1), height * width]
                mask_loss = self.mask_focal_loss(mask_preds, target_masks)

                # 전체 loss 계산
                loss = class_loss + mask_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            self.monitor['loss'] = loss.item()
            self.monitor['total_loss'] += loss.item()

            end_batch = time.time()
            self.print_training_progress(epoch, b, time=end_batch-start_batch)


            #if b == 50: break


    #@torch.no_grad()
    def evaluate(self, args, data_loader, processor, epoch):
        seed_fixer(args.random_seed)

        self.model.eval()
        if args.val_keep > 0:
            valid_batchs = {
                'background' : len(data_loader),
                'driveable' : len(data_loader),
                'vehicle' : len(data_loader),
                'pedestrian' : len(data_loader)
            }
            for b, batch in enumerate(data_loader):

                # pixel_values = torch.stack(list(pixel_value for pixel_value in pixel_values), dim=0).to(self.rank)
                # pixel_masks = torch.stack(list(pixel_mask for pixel_mask in pixel_masks), dim=0).to(self.rank)
                # mask_labels = list(mask_label[0].to(self.rank) for mask_label in mask_labels)
                # class_labels = list(class_label[0].to(self.rank) for class_label in class_labels)

                # pixel_values = pixel_values.to(self.rank)
                # pixel_masks = pixel_masks.to(self.rank)
                # mask_labels = mask_labels.to(self.rank) # [b, c, h, w]
                # class_labels = [label.to(self.rank) for label in class_labels]


                pixel_values = batch["pixel_values"].to(self.rank)
                pixel_mask = batch["pixel_mask"].to(self.rank)
                mask_labels = [mask_label.to(self.rank) for mask_label in batch["mask_labels"]]
                class_labels = [class_label.to(self.rank) for class_label in batch["class_labels"]]

                inputs = {
                    "pixel_values": batch["pixel_values"].to(self.rank),
                    "pixel_mask": batch["pixel_mask"].to(self.rank),
                    "mask_labels": [mask_label.to(self.rank) for mask_label in batch["mask_labels"]],
                    "class_labels": [class_label.to(self.rank) for class_label in batch["class_labels"]]
                }
                with torch.no_grad():
                    outputs = self.model(**inputs)
                target_sizes = [(args.img_h, args.img_w)] * pixel_values.size(0)
                predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                # if len(args.targets)==1:
                #     predicted_semantic_map = torch.stack(predicted_semantic_map, dim=0) # [b, h, w]
                #     mask_labels = mask_labels.argmax(dim=1) # [b, c, h, w] -> [b, h, w]


                # # 클래스 예측
                # class_preds = outputs.class_queries_logits.softmax(dim=-1)  # [batch_size, num_queries, num_classes]
                # # 마스크 예측
                # mask_preds = outputs.masks_queries_logits.sigmoid()  # [batch_size, num_queries, height/4, width/4]
                # mask_preds = torch.nn.functional.interpolate(mask_preds, size=(args.img_h, args.img_w), mode="bilinear", align_corners=False)
                # mask_labels = torch.stack(mask_labels, dim=0)
                # #iou_per_query = filter_valid_queries(class_preds, mask_preds, mask_labels, threshold=args.iou_threshold, class_threshold=args.iou_threshold)
                # binary_mask_preds = (mask_preds > 0.5).float()
                # matching_indices = hungarian_matching(class_preds, mask_preds, mask_labels, class_labels)
                # iou_per_image = compute_iou(binary_mask_preds, mask_labels, matching_indices)

                # 각 배치에 대해 최종 세그멘테이션 마스크 생성
                # final_masks = []
                # for i in range(len(mask_labels)):  # 각 이미지에 대해
                #     if len(args.targets) != 1:
                #         current_mask = torch.full((args.img_h, args.img_w), 
                #                                len(args.targets),  # background
                #                                dtype=torch.long, 
                #                                device=pixel_values.device)
                        
                #         ## confidence를 고려하지 않은 방식
                #         # for q in range(class_preds.shape[1]):
                #         #     class_id = class_preds[i, q].argmax()
                #         #     if class_id < len(args.targets):  # background(3)가 아닌 경우만
                #         #         mask = mask_preds[i, q] > args.iou_threshold
                #         #         current_mask[mask] = class_id

                #         ## confidence를 고려한 방식
                #         confidence_map = torch.zeros_like(current_mask, dtype=torch.float)
                #         for q in range(class_preds.shape[1]):
                #             class_probs = class_preds[i, q]  # 클래스별 확률
                #             class_id = class_probs.argmax()
                #             confidence = class_probs[class_id]
                            
                #             if class_id < len(args.targets):  # background가 아닌 경우
                #                 mask = mask_preds[i, q] > args.iou_threshold
                #                 # confidence가 더 높은 경우에만 업데이트
                #                 update_mask = mask & (confidence > confidence_map)
                #                 current_mask[update_mask] = class_id
                #                 confidence_map[update_mask] = confidence

                #     else:
                #         current_mask = torch.zeros((args.img_h, args.img_w), 
                #                                 dtype=torch.long, 
                #                                 device=pixel_values.device)
                #         confidence_map = torch.zeros_like(current_mask, dtype=torch.float)
                        
                #         for q in range(class_preds.shape[1]):
                #             class_prob = class_preds[i, q, 1]  # 객체에 대한 확률 (인덱스 1)
                #             confidence = class_prob

                #             mask = mask_preds[i, q] > args.iou_threshold
                #             update_mask = mask & (confidence > confidence_map)
                #             current_mask[update_mask] = 1  # 객체는 1로 설정
                #             confidence_map[update_mask] = confidence
                    
                #     final_masks.append(current_mask)
                
                # final_masks = torch.stack(final_masks, dim=0)  # [B, H, W]
                
                # # ground truth 마스크 생성
                # true_masks = []
                # for img_idx, img_masks in enumerate(mask_labels):  # 각 이미지의 객체 마스크들에 대해
                #     img_true_mask = torch.zeros((args.img_h, args.img_w), dtype=torch.long, device=pixel_values.device)
                #     for mask_idx, mask in enumerate(img_masks):
                #         img_true_mask[mask > 0.5] = class_labels[img_idx][mask_idx]  # 해당 객체의 클래스로 설정
                #     true_masks.append(img_true_mask)
                
                # true_masks = torch.stack(true_masks, dim=0)  # [B, H, W]
                
                # 전체 배치에 대해 한 번에 IoU 계산
                # if len(args.targets) != 1:
                #     if args.bg_plus:
                #         for class_id in range(len(args.targets)+1): 
                #             pred_binary = (predicted_semantic_map == class_id)  # [B, H, W]
                #             true_binary = (mask_labels == class_id)   # [B, H, W]
                            
                #             intersection = (pred_binary & true_binary).sum(dim=(1,2))  # [B]
                #             union = (pred_binary | true_binary).sum(dim=(1,2))        # [B]
                            
                #             # 유효한 union > 0인 경우만 계산
                #             valid_mask = (union > 0)
                #             if valid_mask.any():
                #                 iou = (intersection[valid_mask] / union[valid_mask]).mean()
                #                 class_name = self.class_id2name[class_id]
                #                 self.iou_scores[class_name] += iou.item()
                # else:
                if args.bg_plus:
                    # for class_id in range(1, len(args.targets)+1):  # background(0)는 제외
                    #     class_name = self.class_id2name[class_id]

                    # for class_name in args.targets:
                    #     class_id = self.class_id[class_name]
                    #     valid_minibatchs = len(mask_labels)
                    #     batch_iou = 0
                    #     for i in range(len(mask_labels)): 
                    #         # mask_labels의 2번째 차원(C)의 크기가 항상 일정하지는 않기 때문에 stack을 통해 [B, C, H, W] 크기로 만들 수가 없다.
                    #         # 따라서 [B, H, W] 크기의 텐서 끼리 벡터화 계산도 할 수 없다.
                    #         # 그래서 배치 차원에 대한 반복문으로 하나씩 비교해야만 한다.
                    #         if class_id not in class_labels[i]:
                    #             valid_minibatchs -= 1 # 해당 클래스가 타겟 레이블에 없는 경우엔 그만큼 계산에서 제외한다.

                    #         if len(mask_labels[i].shape)==3:
                    #             mask_labels[i] = mask_labels[i].argmax(dim=0) # [C, H, W] -> [H, W]

                    #         pred_binary = (predicted_semantic_map[i] == class_id)  # [H, W]
                    #         true_binary = (mask_labels[i] == class_id)             # [H, W]
                            
                    #         intersection = (pred_binary & true_binary).sum()  
                    #         union = (pred_binary | true_binary).sum()         
                            
                    #         iou = intersection / (union + 1e-6)
                    #         batch_iou += iou.item()
                    #         # 배치 평균을 위해 배치 크기로 미리 나눈다.
                    #         # 단, drop_last=False일 경우 마지막 배치의 크기가 args.batch_size와 다를 수 있기 때문에 len(mask_labels)를 사용한다.
                    #     if valid_minibatchs > 0 :
                    #         self.iou_scores[class_name] += batch_iou / valid_minibatchs
                    #     else:
                    #         valid_batchs[class_name] -= 1



                    fixed_mask_labels = [torch.zeros(4, args.img_h, args.img_w, device=mask_labels[i].device) for i in range(len(mask_labels))]
                    for i in range(len(mask_labels)): 
                        for j, cls in enumerate(class_labels[i]):
                            fixed_mask_labels[i][cls.item()] = mask_labels[i][j]
                        fixed_mask_labels[i] = fixed_mask_labels[i].argmax(dim=0) # [C, H, W] -> [H, W]

                    predicted_mask = torch.stack(predicted_semantic_map, dim=0) # [B, H, W]
                    fixed_mask_labels = torch.stack(fixed_mask_labels, dim=0) # [B, H, W]

                    for class_name in args.targets:
                        class_id = self.class_id[class_name]

                        if class_id in fixed_mask_labels.unique():
                            pred_binary = (predicted_mask == class_id)   # [B, H, W]
                            true_binary = (fixed_mask_labels == class_id) # [B, H, W]

                            intersection = (pred_binary & true_binary).sum()  
                            union = (pred_binary | true_binary).sum()         

                            batch_iou = intersection / (union + 1e-6)
                            self.iou_scores[class_name] += batch_iou 

                        else:
                            valid_batchs[class_name] -= 1 # 해당 클래스가 타겟 레이블 배치에 하나도 없는 경우엔 그만큼 계산에서 제외한다.


                else:
                    for i in range(len(mask_labels)): 
                        # mask_labels의 2번째 차원(C)의 크기가 항상 일정하지는 않기 때문에 stack을 통해 [B, C, H, W] 크기로 만들 수가 없다.
                        # 따라서 [B, H, W] 크기의 텐서 끼리 벡터화 계산도 할 수 없다.
                        # 그래서 배치 차원에 대한 반복문으로 하나씩 비교해야만 한다.

                        if len(mask_labels[i].shape)==3:
                            mask_labels[i] = mask_labels[i].argmax(dim=0) # [C, H, W] -> [H, W]

                        pred_binary = (predicted_semantic_map[i] == 1)  # [H, W]
                        true_binary = (mask_labels[i] == 1)   # [C, H, W] -> [H, W]
                        
                        intersection = (pred_binary & true_binary).sum()  
                        union = (pred_binary | true_binary).sum()         
                        
                        # 유효한 union > 0인 경우만 계산
                        # valid_mask = (union > 0)
                        if union > 0:
                            iou = (intersection / union)
                            self.iou_scores[args.targets[0]] += iou.item() / len(mask_labels)
                            # 배치 평균을 위해 배치 크기로 미리 나눈다.
                            # 단, drop_last=False일 경우 마지막 배치의 크기가 args.batch_size와 다를 수 있기 때문에 len(mask_labels)를 사용한다.
                        else:
                            if torch.equal(pred_binary, true_binary): # 전부 배경인 타겟을 배경으로 잘 예측한 경우
                                self.iou_scores[args.targets[0]] += 1.0 / len(mask_labels)

                self.print_validation_progress(b, self.len_val-1)


                # 시각화
                if b==0 : # 
                    output_dir = os.path.join(args.save_dir, 'output_images')
                    #save_images(args, output_dir, pixel_values, predicted_semantic_map, mask_labels, self.rank)
                    save_images(args, output_dir, pixel_values, predicted_mask, fixed_mask_labels, self.rank)

                if b == 50: break
                

            for k, v in self.iou_scores.items():
                self.iou_scores[k] = v / valid_batchs[k] # 전체 validation의 평균 iou

            miou = sum(self.iou_scores.values()) / len(self.iou_scores)
            self.monitor['current_mIoU'] = miou
            if self.rank == 0:
                for k, v in self.iou_scores.items():
                    self.logger.info(f">> [Valid] | IoU@{args.iou_threshold}: {k}: {v:.4f}")
                self.logger.info(f">> [Valid] Epoch {epoch} | mIoU@{args.iou_threshold}: {miou:.4f} ✅")
                if self.monitor['prev_IoU'] < miou:
                    self.monitor['prev_IoU'] = miou
                    if hasattr(self.model, 'module'): # DDP
                        #self.save_trained_network_params(self.model.module, save_dir=args.save_dir, name='seg', e=epoch, remain_num_chkpts=args.remain_num_chkpts)
                        self.model.module.save_pretrained(args.save_dir + f'/epoch{epoch}')
                        processor.save_pretrained(args.save_dir + f'/epoch{epoch}')
                    else:
                        #self.save_trained_network_params(self.model, save_dir=args.save_dir, name='seg', e=epoch, remain_num_chkpts=args.remain_num_chkpts)
                        self.model.save_pretrained(args.save_dir + f'/epoch{epoch}')
                        processor.save_pretrained(args.save_dir + f'/epoch{epoch}')
        else:
            if hasattr(self.model, 'module'): # DDP
                #self.save_trained_network_params(self.model.module, save_dir=args.save_dir, name='seg', e=epoch, remain_num_chkpts=args.remain_num_chkpts)
                self.model.module.save_pretrained(args.save_dir + f'/epoch{epoch}')
                processor.save_pretrained(args.save_dir + f'/epoch{epoch}')
            else:
                #self.save_trained_network_params(self.model, save_dir=args.save_dir, name='seg', e=epoch, remain_num_chkpts=args.remain_num_chkpts)
                self.model.save_pretrained(args.save_dir + f'/epoch{epoch}')
                processor.save_pretrained(args.save_dir + f'/epoch{epoch}')


    def save_trained_network_params(self, model, save_dir, name, e, remain_num_chkpts):
        # save trained model
        _ = save_read_latest_checkpoint_num(os.path.join(save_dir), e, isSave=True)
        if name != None:
            file_name = save_dir + f'/saved_chk_point_{name}_{e}.pt'
        else:
            file_name = save_dir + f'/saved_chk_point_{e}.pt'

        check_point = {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'opt': self.optimizer.state_dict(),
            'prev_IoU': self.monitor['prev_IoU'],
            }
    
        torch.save(check_point, file_name)
        self.logger.info(f">> ⭐ current network is saved ...")
        remove_past_checkpoint(os.path.join('./', save_dir), remain_num_chkpts, name)

    def normalize_loss_tracker(self):
        self.monitor['total_loss'] /= self.num_batches

    def reset_loss_tracker(self):
        self.monitor['total_loss'] = 0

    def print_epoch_status(self, e, tl):
        if self.rank==0:
            total_loss = self.monitor['total_loss']
            cur_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'[Epoch {e:d}, {tl:.2f} hrs left] loss: {total_loss:.4f}, (cur lr: {cur_lr:.7f})')

    def print_training_progress(self, e, b, time):
        if self.rank==0:
            if (b >= self.num_batches - 2): sys.stdout.write('\r')
            else: sys.stdout.write(f"\r [Epoch {e}] {b} / {self.num_batches} ({time:.4f} sec/sample), loss: {self.monitor['loss']:.4f}")
            sys.stdout.flush()

    def print_validation_progress(self, b, num_batchs):
        if self.rank==0:
            if (b >= num_batchs - 2): sys.stdout.write('\r')
            else: sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),
            sys.stdout.flush()

    def wandb_tracker(self,):
        CFG = {
            'Loss' : self.monitor['total_loss'],
            'mIoU' : self.monitor['current_mIoU'],
        }
        return CFG