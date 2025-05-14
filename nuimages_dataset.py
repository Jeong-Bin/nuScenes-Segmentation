import os
import torch
import numpy as np
from PIL import Image
from nuimages.utils.utils import mask_decode
from torchvision.transforms.functional import to_pil_image

class NuImagesDataset(torch.utils.data.Dataset):
    def __init__(self, args, nuimages, processor=None, transforms=None, remove_empty=True, mode='train', rank=0, logger=None):
        if len(nuimages.object_ann) == 0:
            self.has_object_ann = False # Check if the nuimages object contains the test set (no annotations)    
        else:
            self.has_object_ann = True  # Otherwise, dataset is the train or val split

        if len(nuimages.surface_ann) == 0:
            self.has_surface_ann = False # Check if the nuimages object contains the test set (no annotations)    
        else:
            self.has_surface_ann = True  # Otherwise, dataset is the train or val split

        self.args = args
        self.nuimages = nuimages
        self.processor = processor
        self.transforms = transforms
        self.rank = rank
        self.logger = logger
        self.sample_data = nuimages.sample_data
        self.object_annotations = nuimages.object_ann
        self.surface_annotations = nuimages.surface_ann

        if mode=='train': mode = 'Train'
        elif mode=='val' or mode=='valid': mode = 'Valid'
        elif mode=='test': mode = 'Test'

        # If training, remove any samples which contain no annotations
        if remove_empty and self.has_object_ann:
            sd_tokens_with_objects = set()
            for o in self.nuimages.object_ann:
                sd_tokens_with_objects.add(o['sample_data_token'])

            self.samples_with_objects = []
            for i, sample in enumerate(self.nuimages.sample):
                sd_token = sample['key_camera_token']
                if sd_token in sd_tokens_with_objects:
                    self.samples_with_objects.append(i)
            if self.rank==0: self.logger.info(f"[{mode}:objects] Number of samples (Original-Removed=Current): {len(self.nuimages.sample)} - {len(self.nuimages.sample)-len(self.samples_with_objects)} = {len(self.samples_with_objects)}")
        else:
            # Keep all samples if remove_empty set to false
            self.samples_with_objects = [i for i in range(len(self.nuimages.sample))]
            if self.rank==0: self.logger.info(f"[{mode}:objects] Number of samples: {len(self.samples_with_objects)}")

        if remove_empty and self.has_surface_ann:
            sd_tokens_with_surface = set()
            for s in self.nuimages.surface_ann:
                sd_tokens_with_surface.add(s['sample_data_token'])

            self.samples_with_surface = []
            for i, sample in enumerate(self.nuimages.sample):
                sd_token = sample['key_camera_token']
                if sd_token in sd_tokens_with_surface:
                    self.samples_with_surface.append(i)
            if self.rank==0: self.logger.info(f"[{mode}:surface] Number of samples (Original-Removed=Current): {len(self.nuimages.sample)} - {len(self.nuimages.sample)-len(self.samples_with_surface)} = {len(self.samples_with_surface)}")
        else:
            # Keep all samples if remove_empty set to false
            self.samples_with_surface = [i for i in range(len(self.nuimages.sample))]
            if self.rank==0: self.logger.info(f"[{mode}:surface] Number of samples: {len(self.samples_with_surface)}")
        
        if remove_empty:
            # object나 surface annotation이 하나라도 있는 샘플 사용
            if len(self.args.targets) == 1:
                if self.args.targets=='driveable':
                    self.valid_samples = self.samples_with_surface
                else:
                    self.valid_samples = self.samples_with_objects
            else:
                self.valid_samples = list(set(self.samples_with_objects) | set(self.samples_with_surface))
            if self.rank==0: self.logger.info(f"[{mode}] Number of samples with any annotation: {len(self.valid_samples)}")
            if self.rank==0: self.logger.info(f"    (Object annotations: {len(self.samples_with_objects)}, Surface annotations: {len(self.samples_with_surface)})")
        else:
            self.valid_samples = [i for i in range(len(self.nuimages.sample))]
            if self.rank==0: self.logger.info(f"[{mode}] Using all samples: {len(self.valid_samples)}")

    def __getitem__(self, idx):
        # 실제 샘플 인덱스 가져오기 (remove_empty=True인 경우를 위해)
        sample_idx = self.valid_samples[idx]
        sample = self.nuimages.sample[sample_idx]
        image_token = sample['key_camera_token']
        
        # 이미지 로드
        image_data = self.nuimages.get('sample_data', image_token)
        image_path = os.path.join(self.nuimages.dataroot, image_data['filename'])
        image = Image.open(image_path).convert('RGB')
        
        # segmentation mask 생성 (이미지 크기와 동일하게)
        #if len(self.args.targets) == 1:
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.int32) 
        # else:
        #     mask = np.full((image.size[1], image.size[0]), 
        #                     len(self.args.targets),  # background의 label(=class 개수-1+1)로 초기화 # 클래스가 3개라면 0~2은 유효 클래스, 3은 background
        #                     dtype=np.int32)
        
        if self.has_object_ann:
            # object annotations 처리
            object_anns = [a for a in self.object_annotations if a['sample_data_token'] == image_token]
            for ann in object_anns:
                # mask가 None인 경우 건너뛰기
                if ann['mask'] is None:
                    continue
                    
                category = self.nuimages.get('category', ann['category_token'])
                label_id = self._get_category_id(category['name'])
                
                # RLE 형식의 마스크를 디코딩
                m = mask_decode(ann['mask'])  # RLE를 바이너리 마스크로 디코딩
                mask[m.astype(bool)] = label_id
        
        if self.has_surface_ann:
            # surface annotations 처리 (주로 도로 관련)
            surface_anns = [a for a in self.surface_annotations if a['sample_data_token'] == image_token]
            for ann in surface_anns:
                # mask가 None인 경우 건너뛰기
                if ann['mask'] is None:
                    continue
                    
                category = self.nuimages.get('category', ann['category_token'])
                label_id = self._get_category_id(category['name'])
                
                # RLE 형식의 마스크를 디코딩
                m = mask_decode(ann['mask'])  # RLE를 바이너리 마스크로 디코딩
                mask[m.astype(bool)] = label_id

        if self.transforms:
            image, mask = self.transforms(image, mask) # torchvision.transforms

            # augmented = self.transforms(image=np.array(image), mask=mask) # albumentations
            # image, mask = augmented['image'], augmented['mask'] # albumentations

        # processor를 사용하여 이미지와 마스크 전처리
        if isinstance(mask, torch.Tensor): 
            mask = mask.numpy()
        encoded_inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        '''
        encoded_inputs['pixel_values'].shape : torch.Size([1, 3, 224, 480])
        encoded_inputs['pixel_mask'].shape : torch.Size([1, 224, 480]) # 입력 이미지의 패딩 마스크 (실제 이미지 영역은 1, 패딩된 영역은 0)
        encoded_inputs['mask_labels'] : [[tensor([[[1., 1.,  ..., 1., 1.],...]] | len : 1 | shape : [2, 224, 480] # semantic segmentation을 위한 ground truth 마스크
        encoded_inputs['class_labels'] : [tensor[0, 3]] | len : 1 # 이미지에 존재하는 클래스의 리스트
        '''

        # batch dimension 제거
        for k,v in encoded_inputs.items():
            if isinstance(v, torch.Tensor):
                encoded_inputs[k] = v.squeeze(0)  # batch dimension 제거
        
        # mask_labels를 고정된 크기로 변환 (클래스 수만큼의 채널을 가지도록)
        # num_classes = len(self.args.targets)+1  # background(0), vehicle(1), driveable(2), pedestrian(3), 
        # mask_labels = torch.zeros((num_classes, encoded_inputs['pixel_values'].shape[1], encoded_inputs['pixel_values'].shape[2]))
        
        assert encoded_inputs['mask_labels'][0].size(0) == len(encoded_inputs['class_labels'][0]), "mask_labels와 class_labels의 크기가 다릅니다."
        
        # 각 클래스별 binary mask 처리
        class_labels = encoded_inputs['class_labels'][0]  # 이미지에 존재하는 클래스 ID들
        # if len(self.args.targets) == 1:
        mask_labels = encoded_inputs['mask_labels'][0]
        #padded_class_labels = class_labels

        # if 0 in mask_labels and 0 not in class_labels :
        #     class_labels = torch.cat([torch.tensor([0]), class_labels])
        # else:
        #     original_masks = encoded_inputs['mask_labels'][0]  # [N, H, W] 형태
        
        #     for mask_idx, class_id in enumerate(class_ids):
        #         mask_labels[class_id] = original_masks[mask_idx]
            
        #     # background는 모든 클래스의 마스크가 없는 영역
        #     foreground = torch.zeros_like(mask_labels[0], dtype=torch.bool)  # bool 타입으로 초기화
        #     for i in range(len(self.args.targets)+1): 
        #         foreground = foreground | (mask_labels[i] > 0).bool()  # 명시적으로 bool로 변환
        #     mask_labels[0] = ~foreground  # background는 foreground가 없는 영역
            
        #     padded_class_labels = torch.zeros(num_classes, dtype=torch.long)
        #     padded_class_labels[:len(encoded_inputs['class_labels'][0])] = encoded_inputs['class_labels'][0]
        #     #padded_class_labels[0] = self.args.num_classes # background의 label(=class 개수-1+1)로 지정
        
        return (encoded_inputs['pixel_values'], 
                encoded_inputs['pixel_mask'], 
                mask_labels,
                class_labels)

    def __len__(self):
        return len(self.valid_samples)
    
    def _get_category_id(self, category_name):
        # NuImages의 세부 카테고리를 3개의 주요 카테고리로 매핑
        '''
        driveable = {"flat.driveable_surface"}

        vehicle = {"vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car", "vehicle.construction",
                   "vehicle.emergency.ambulance", "vehicle.emergency.police", "vehicle.trailer", "vehicle.truck",
                   "vehicle.bicycle", "vehicle.motocycle"}

        pedestrian = {"human.pedestrian.adult", "human.pedestrian.child", "human.pedestrian.construction_worker",
                      "human.pedestrian.personal_mobility", "human.pedestrian.police_officer", 
                      "human.pedestrian.stroller", "human.pedestrian.wheelchair"}
        
        background = {"animal", 
                      "movable_object.barrier", "movable_object.debris", "movable_object.pushable_pullable", "movable_object.trafficcone",
                      "static_object.bicycle_rack", "vehicle.ego"}
        '''

        if len(self.args.targets) == 1:
            if (self.args.targets[0] in category_name) and ('vehicle.ego' != category_name):
                return 1
            else : 
                return 0
            
        elif len(self.args.targets) == 2:
            if 'driveable_surface' in category_name:
                return 1
            elif ('vehicle.ego' != category_name) and ('vehicle' in category_name):
                return 2
            else:
                return 0

        elif len(self.args.targets) == 3:
            if 'driveable_surface' in category_name:
                return 1
            elif ('vehicle.ego' != category_name) and ('vehicle' in category_name):
                return 2
            elif 'pedestrian' in category_name:
                return 3
            else:
                return 0