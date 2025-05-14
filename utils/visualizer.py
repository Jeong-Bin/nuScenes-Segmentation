import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# def tensor_visualizer(tsr):
#     if type(tsr) == np.ndarray:
#         tsr = torch.tensor(tsr)
        
#     tensor_PIL = transforms.ToPILImage()
#     image = tensor_PIL(tsr)
#     plt.imshow(image)
#     plt.show()


# def label_to_color(label_map):
#     # (H, W) -> (H, W, 3) RGB
#     color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)
#     color_map[label_map == 0] = [0, 0, 255]    # 클래스 0 -> 파란색
#     color_map[label_map == 1] = [255, 0, 0]    # 클래스 1 -> 빨간색
#     return color_map
def label_to_color(args, maps):
    """
    multi_label : [C, H, W]
    binary_label: [H, W]
    """
    if len(maps.shape) == 3:
        h, w = maps.shape[-2:]
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        # background(0)
        color_map[maps[0] == 1] = [0, 0, 0]  # black
        # driveable(1)
        color_map[maps[1] == 1] = [0, 255, 0]  # G
        # vehicle(2)
        color_map[maps[2] == 1] = [0, 0, 255]  # B
        # pedestrian(3)
        color_map[maps[3] == 1] = [255, 0, 0]  # R
        return color_map
    
    else:
        h, w = maps.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        if len(args.targets) == 1:
            color_map[maps == 0] = [0, 0, 0]  # black
            color_map[maps == 1] = [255, 255, 255]  # white
        elif len(args.targets) == 2:
            # background(0)
            color_map[maps == 0] = [0, 0, 0]  # black
            # driveable(1)
            color_map[maps == 1] = [0, 255, 0]  # G
            # vehicle(2)
            color_map[maps == 2] = [0, 0, 255]  # B
        else:
            # background(0)
            color_map[maps == 0] = [0, 0, 0]  # black
            # driveable(1)
            color_map[maps == 1] = [0, 255, 0]  # G
            # vehicle(2)
            color_map[maps == 2] = [0, 0, 255]  # B
            # pedestrian(3)
            color_map[maps == 3] = [255, 0, 0]  # R
        return color_map
    
def save_images(args, output_dir, pixel_values, pred_masks, mask_labels, rank):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(pixel_values.size(0)):
        # 원본 이미지 복원
        original_image = pixel_values[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        # processor의 정규화 복원
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # 정규화 복원: x = (normalized * std) + mean
        original_image = (original_image * std + mean) * 255.0
        original_image = original_image.clip(0, 255).astype(np.uint8)
        
        original_path = os.path.join(output_dir, f"b{i}r{rank}_image.png")
        Image.fromarray(original_image).save(original_path)


        # 예측 마스크 저장
        pred_map = pred_masks[i].cpu().numpy()
        pred_color_map = label_to_color(args, pred_map)
        pred_path = os.path.join(output_dir, f"b{i}r{rank}_pred.png")
        Image.fromarray(pred_color_map).save(pred_path)

        # 레이블 마스크 저장
        label_map = mask_labels[i].cpu().numpy()
        label_color_map = label_to_color(args, label_map)
        label_path = os.path.join(output_dir, f"b{i}r{rank}_label.png")
        Image.fromarray(label_color_map).save(label_path)
