import numpy as np
import torch
import torch.nn.functional as F


# def iou_score(output, target):
#     smooth = 1e-5

#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()

#     return (intersection + smooth) / (union + smooth)
# def iou_score(output, target, num_classes):
#     smooth = 1e-6
#     with torch.no_grad():
#         output = torch.argmax(output, dim=1)  # 获取每个像素的预测类别
#         iou_per_class = []
#         for cls in range(num_classes):
#             true_class = (target == cls)
#             pred_class = (output == cls)
#             intersection = (true_class & pred_class).float().sum()
#             union = (true_class | pred_class).float().sum()
#             iou = (intersection + smooth) / (union + smooth)
#             iou_per_class.append(iou)
#         return sum(iou_per_class) / len(iou_per_class)  # 平均IoU


def iou_score(output, target, num_classes = 4):
    smooth = 1e-6
    with torch.no_grad():
        # print('11111',output.size())
        output = torch.argmax(output, dim=1)  # 获取每个像素的预测类别
        # print('55555555555555555,outputshape:',output.shape)
        # print('222222',output.size())
        
        iou_per_class = []
        
        for cls in range(num_classes):
            true_class = (target == cls).float()
            pred_class = (output == cls).float()
            # print('1111111',true_class.size(),pred_class.size())
            
            # 计算交集和并集
            # intersection = (true_class & pred_class).float().sum()
            # union = (true_class | pred_class).float().sum()
            # intersection = (true_class * pred_class).sum()  # 使用乘法计算交集
            # union = true_class.sum() + pred_class.sum() - intersection
            intersection = (true_class * pred_class).sum(dim=(1, 2))  # 每个图像的交集
            union = true_class.sum(dim=(1, 2)) + pred_class.sum(dim=(1, 2)) - intersection  # 每个图像的并集
            
            # 计算IoU并处理零除的情况
            iou = (intersection + smooth) / (union + smooth)
            iou_per_class.append(iou.cpu().numpy())
            # iou_per_class.append(iou)
                
            # iou_per_class.append(iou)
            # iou_per_class = torch.tensor(iou_per_class, device=output.device)  # 形状为 [num_classes, batch_size]
        iou_per_class = np.array(iou_per_class)  # 先转为单个 NumPy 数组
        iou_per_class = torch.tensor(iou_per_class, device=output.device)  # 形状为 [num_classes, batch_size]
            
            # iou_per_class = torch.stack(iou_per_class, dim=1)  # 形状为 [batch_size, num_classes]
        mean_iou = iou_per_class.mean(dim=1)
        
        # 返回平均IoU
        return mean_iou.mean(), mean_iou[1], mean_iou[2], mean_iou[3]



# def dice_coef(output, target):
#     smooth = 1e-5

#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()

#     return (2. * intersection + smooth) / \
#         (output.sum() + target.sum() + smooth)


def dice_coef(output, target, num_classes = 4):
    smooth = 1e-5
    output = torch.argmax(output, dim=1)  # 获取每个像素的预测类别
    dice_per_class = []
    
    with torch.no_grad():
        for cls in range(num_classes):
            true_class = (target == cls).float()
            pred_class = (output == cls).float()
            
            # 计算交集和总和
            intersection = (true_class * pred_class).sum()
            total = true_class.sum() + pred_class.sum()
            
            # 计算Dice系数并处理零除的情况
            if total == 0:
                dice = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
            else:
                dice = (2. * intersection + smooth) / (total + smooth)
                
            dice_per_class.append(dice)
        
        # 返回平均Dice系数
        return torch.mean(torch.stack(dice_per_class))
