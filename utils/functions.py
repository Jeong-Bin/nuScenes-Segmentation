import os
import sys
import numpy as np
import random
import torch
from torch.utils.data import Subset, ConcatDataset

def seed_fixer(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_read_latest_checkpoint_num(path, val, isSave):
    file_name = path + '/checkpoint.txt'
    index = 0
    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def load_pretrained_models(model, ckp_path, name='m2f', logger=None, rank=0, strict=True):

    ckp_idx = save_read_latest_checkpoint_num(path=ckp_path, val=0, isSave=False)
    file_name = ckp_path + f'/saved_chk_point_{name}_{ckp_idx}.pt'
    checkpoint = torch.load(file_name, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if rank==0:
        logger.info('>> trained parameters are loaded from {%s}' % file_name)
        if 'prev_IoU' in checkpoint:
            logger.info(">> Pretrained Model status : %.4f IoU" % checkpoint['prev_IoU']) 

    return model

def remove_past_checkpoint(path, num_remain=5, name=None):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if name == None:
            if "saved" in fname:
                chk_index = get_file_number(fname)
                fname_list.append(fname)
                fnum_list.append(chk_index)
        else:
            if f"saved_chk_point_{name}" in fname:
                chk_index = get_file_number(fname)
                fname_list.append(fname)
                fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)


def run_wandb(args, wandb):
    run = wandb.init(project=f"{args.wandb_project}", name=args.training_name, save_code=False)
    return run

def print_training_info(args, logger):
    logger.info(f"--------- NuImages / {args.training_name} ----------")
    logger.info(f" DDP : {args.ddp}")
    logger.info(f" Num epoch : {args.num_epochs}")
    if args.ddp:
        logger.info(f" Gpu idx : {args.gpu_idx_ddp}")
        logger.info(f" Batch size * GPU num : {args.batch_size} * {len(args.gpu_idx_ddp.split(','))}" )
    else:
        logger.info(f" Gpu idx : {args.gpu_idx}")
        logger.info(f" Batch size : {args.batch_size}" )

    logger.info(f" Num workers : {args.num_workers}")
    logger.info(f" Random seed : {args.random_seed}")
    logger.info("----------------------------------")
    logger.info(f" Model : {args.model_name}")
    logger.info(f" Image size : {args.img_h}x{args.img_w}")
    logger.info(f" Optimizer type : {args.optimizer_type}")
    logger.info(f" Learning rate : {args.learning_rate:.5f}")
    logger.info(f" Weight decay : {args.weight_decay:.4f}")
    logger.info(f" Target class : {args.targets}")
    #logger.info(f" Visibility : {args.visibility}")
    logger.info("----------------------------------")
    logger.info(f" LR scheduling type : {args.lr_schd_type}")
    logger.info(f" div_factor : {args.div_factor:.1f}")
    logger.info(f" pct_start : {args.pct_start:.1f}")
    logger.info(f" final_div_factor : {args.final_div_factor}")
    logger.info("----------------------------------")

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    pixel_masks = torch.stack([item[1] for item in batch])
    #mask_labels = torch.stack([item[2] for item in batch])
    mask_labels = [item[2] for item in batch]
    class_labels = [item[3] for item in batch]  # 리스트로 유지
    
    # 딕셔너리로 반환
    batch_dict = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_masks,
        "mask_labels": mask_labels, 
        "class_labels": class_labels
    }
    
    return batch_dict



def augment_train_with_val(train_dataset, val_dataset, val_keep=0.5):
    # val_dataset에 남길 샘플 수
    num_val_samples_to_keep = int(len(val_dataset) * val_keep)

    # val_dataset에서 가져올 샘플 수 (전체에서 남길 샘플을 뺀 나머지)
    num_val_samples_to_transfer = len(val_dataset) - num_val_samples_to_keep

    # val_dataset의 인덱스를 랜덤으로 섞음
    val_indices = list(range(len(val_dataset)))
    random.shuffle(val_indices)

    # val_dataset을 분리: 일부는 train으로, 나머지는 val로 유지
    selected_indices = val_indices[:num_val_samples_to_transfer]  # train으로 추가할 데이터
    remaining_indices = val_indices[num_val_samples_to_transfer:]  # 여전히 val로 유지할 데이터

    # Subset으로 분리된 데이터셋 생성
    selected_val_subset = Subset(val_dataset, selected_indices)  # train에 추가할 val 데이터
    reduced_val_dataset = Subset(val_dataset, remaining_indices)  # 남은 val 데이터

    # train_dataset과 selected_val_subset을 합침
    augmented_train_dataset = ConcatDataset([train_dataset, selected_val_subset])

    return augmented_train_dataset, reduced_val_dataset







def compute_iou(mask_preds, mask_labels, threshold=0.5):
    """
    IoU 계산 함수

    Args:
        mask_preds (torch.Tensor): [batch_size, num_queries, height, width] - 모델 마스크 예측값 (0~1 확률값).
        mask_labels (torch.Tensor): [batch_size, num_queries, height, width] - 정답 마스크 (0 또는 1).
        threshold (float): 이진화 임계값 (default: 0.5).

    Returns:
        iou_per_query (torch.Tensor): [batch_size, num_queries] - 각 쿼리별 IoU 값.
    """
    # Step 1. 마스크 예측값 이진화 (threshold 적용)
    binary_mask_preds = (mask_preds > threshold).float()  # [batch_size, num_queries, height, width]

    # Step 2. Intersection (교집합) 계산
    intersection = (binary_mask_preds * mask_labels).sum(dim=(2, 3))  # [batch_size, num_queries]

    # Step 3. Union (합집합) 계산
    union = (binary_mask_preds + mask_labels - binary_mask_preds * mask_labels).sum(dim=(2, 3))  # [batch_size, num_queries]

    # Step 4. IoU 계산
    iou_per_query = intersection / (union + 1e-6)  # [batch_size, num_queries], 안정성을 위해 1e-6 추가

    return iou_per_query

def filter_valid_queries(class_preds, mask_preds, mask_labels, threshold=0.5, class_threshold=0.5):
    """
    유효한 쿼리만 필터링하여 IoU 계산.

    Args:
        class_preds (torch.Tensor): [batch_size, num_queries, num_classes] - 클래스 예측값 (확률).
        mask_preds (torch.Tensor): [batch_size, num_queries, height, width] - 마스크 예측값.
        mask_labels (torch.Tensor): [batch_size, num_queries, height, width] - 정답 마스크.
        threshold (float): 마스크 이진화 임계값.
        class_threshold (float): 클래스 확률 임계값.

    Returns:
        iou_per_query (torch.Tensor): [batch_size, num_queries] - IoU 값.
    """
    # Step 1. 클래스 필터링
    valid_queries = class_preds.max(dim=-1).values > class_threshold  # [batch_size, num_queries]
    
    # Step 2. IoU 계산
    iou_per_query = compute_iou(mask_preds, mask_labels, threshold)

    # Step 3. 유효한 쿼리만 필터링
    iou_per_query[~valid_queries] = 0.0  # 유효하지 않은 쿼리의 IoU를 0으로 설정

    return iou_per_query



from scipy.optimize import linear_sum_assignment
import torch

def hungarian_matching(class_preds, mask_preds, mask_labels, class_labels):
    """
    Hungarian Matching 수행.

    Args:
        class_preds (torch.Tensor): [batch_size, num_queries, num_classes] - 클래스 예측값.
        mask_preds (torch.Tensor): [batch_size, num_queries, height, width] - 확률 마스크 (sigmoid 적용됨).
        mask_labels (torch.Tensor): [batch_size, num_objects, height, width] - 정답 마스크.
        class_labels (torch.Tensor): [batch_size, num_objects] - 정답 클래스.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: 매칭된 (예측 쿼리, 정답 객체) 인덱스 쌍.
    """
    
    batch_size = class_preds.size(0)
    indices = []

    for b in range(batch_size):
        # 비용 행렬 계산
        num_queries = class_preds.size(1)
        num_objects = class_labels[b].size(0)

        # 마스크 비용: BCE 기반
        pred_masks_flat = mask_preds[b].view(num_queries, -1)  # [num_queries, height * width]
        target_masks_flat = mask_labels[b].view(num_objects, -1)  # [num_objects, height * width]
        mask_cost = torch.cdist(pred_masks_flat, target_masks_flat, p=1)  # [num_queries, num_objects]

        # 클래스 비용: Cross-Entropy 기반
        class_cost = -class_preds[b][:, class_labels[b]]  # [num_queries, num_objects]

        # 최종 비용 행렬
        cost_matrix = mask_cost + class_cost

        # Hungarian Matching
        pred_idx, target_idx = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
        indices.append((torch.tensor(pred_idx, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)))

    return indices

def compute_iou(binary_mask_preds, mask_labels, matching_indices):
    """
    IoU 계산.

    Args:
        binary_mask_preds (torch.Tensor): [batch_size, num_queries, height, width] - 이진화된 예측 마스크.
        mask_labels (torch.Tensor): [batch_size, num_objects, height, width] - 정답 마스크.
        matching_indices (List[Tuple[torch.Tensor, torch.Tensor]]): 매칭된 (예측 쿼리, 정답 객체) 인덱스.

    Returns:
        torch.Tensor: [batch_size, num_objects] - IoU 값.
    """
    batch_size = binary_mask_preds.size(0)
    iou_per_image = []

    for b in range(batch_size):
        pred_idx, target_idx = matching_indices[b]

        # 매칭된 마스크 선택
        matched_preds = binary_mask_preds[b, pred_idx]  # [num_objects, height, width]
        matched_targets = mask_labels[b, target_idx]  # [num_objects, height, width]

        # IoU 계산
        intersection = (matched_preds * matched_targets).sum(dim=(1, 2))  # [num_objects]
        union = (matched_preds + matched_targets - matched_preds * matched_targets).sum(dim=(1, 2))  # [num_objects]
        iou = intersection / (union + 1e-6)  # 안정성을 위해 1e-6 추가

        iou_per_image.append(iou)

    return torch.stack(iou_per_image)  # [batch_size, num_objects]


