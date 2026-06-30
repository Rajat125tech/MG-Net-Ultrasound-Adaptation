#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorboardX import SummaryWriter
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from networks.MG_Net_3D.network_backbone import MG_Net_2D
from monai.networks.nets import UNETR, SwinUNETR,VNet
from monai.metrics import DiceMetric, MeanIoU
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
import torch.nn as nn

from load_datasets_transforms import data_loader, data_transforms

import os
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='./dataset', help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='./output',  help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='ultrasound',  help='Datasets: {ultrasound, feta, flare, amos}')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MG_Net_2D', help='Network models: {TransBTS,VNet, nnFormer, UNETR, SwinUNETR, 3DUXNET,TransGCN,MG_Net_3D,MG_Net_2D}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=10000, help='Maximum iteration steps for training') 
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0,1', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1.0, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

if args.mode in ['train', 'eval']:
    train_samples, valid_samples, out_classes = data_loader(args)

    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
    ]

    val_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
    ]

    train_transforms, val_transforms = data_transforms(args)

    print('Start caching datasets!')
    if args.mode == 'train':
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=args.cache_rate, num_workers=args.num_workers)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

elif args.mode == 'test':
    test_samples, out_classes = data_loader(args)
    test_files = [
        {"image": image_name}
        for image_name in test_samples['images']
    ]

    test_transforms = data_transforms(args)

    print('Start caching test datasets!')
    test_ds = CacheDataset(
        data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)
## Load Networks
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=48,
        use_checkpoint=False,
    )
elif args.network == 'VNet':
    model = VNet(in_channels=1,out_channels=out_classes )
elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=out_classes,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
elif args.network == 'MG_Net_2D':
    model = MG_Net_2D(
        in_chans=1,
        out_chans=out_classes,
        depths=[1, 1, 1, 1],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=2,
    )

if len(args.gpu) > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)
print('Chosen Network Architecture: {}'.format(args.network))

if args.pretrain == 'True' or args.pretrain is True or (args.mode in ['eval', 'test'] and args.pretrained_weights):
    print('Start to load weights from: {}'.format(args.pretrained_weights))
    state_dict = torch.load(args.pretrained_weights, map_location=device)
    if next(iter(state_dict.keys())).startswith('module.') and not isinstance(model, nn.DataParallel):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)


root_dir = os.path.join(args.output)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)
    
t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            # val_outputs = model(val_inputs)
            val_outputs = sliding_window_inference(val_inputs, (96, 96), 2, model)
            # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    # model_feat.eval()
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        # with torch.no_grad():
        #     g_feat, dense_feat = model_feat(x)
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
        writer.add_scalar('Training Segmentation Loss', loss.item(), global_step)
        global_step += 1
    return global_step, dice_val_best, global_step_best


post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

if args.mode == 'eval':
    print("Starting evaluation on validation set...")
    model.eval()
    
    class_names = ["Background", "Brain", "CSP", "LV"]
    dice_metric_classwise = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    dice_metric_overall = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric_classwise = MeanIoU(include_background=True, reduction="mean_batch", get_not_nans=False)
    iou_metric_overall = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    
    total_correct_pixels = 0
    total_pixels = 0
    
    epoch_iterator_val = tqdm(
        val_loader, desc="Evaluating (dice=X.X)", dynamic_ncols=True
    )
    
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (96, 96), 2, model)
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            
            # Compute Dice
            dice_metric_classwise(y_pred=val_output_convert, y=val_labels_convert)
            dice_metric_overall(y_pred=val_output_convert, y=val_labels_convert)
            
            # Compute IoU
            iou_metric_classwise(y_pred=val_output_convert, y=val_labels_convert)
            iou_metric_overall(y_pred=val_output_convert, y=val_labels_convert)
            
            # Compute Pixel Accuracy
            val_preds_argmax = torch.argmax(val_outputs, dim=1, keepdim=True)
            correct = (val_preds_argmax == val_labels).sum().item()
            total = val_labels.numel()
            total_correct_pixels += correct
            total_pixels += total
            
            current_overall_dice = dice_metric_overall.aggregate().item()
            epoch_iterator_val.set_description(
                "Evaluating (dice=%2.5f)" % (current_overall_dice)
            )
            
        classwise_dice = dice_metric_classwise.aggregate().cpu().numpy()
        overall_dice = dice_metric_overall.aggregate().item()
        
        classwise_iou = iou_metric_classwise.aggregate().cpu().numpy()
        overall_iou = iou_metric_overall.aggregate().item()
        
        pixel_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        dice_metric_classwise.reset()
        dice_metric_overall.reset()
        iou_metric_classwise.reset()
        iou_metric_overall.reset()
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS ON VALIDATION SET:")
    print("="*50)
    print(f"Overall Mean Dice: {overall_dice:.5f}")
    print(f"Mean IoU (mIoU):   {overall_iou:.5f}")
    print(f"Pixel Accuracy:    {pixel_accuracy:.5f} ({pixel_accuracy * 100:.2f}%)")
    print("-"*50)
    print("PER-CLASS METRICS:")
    print("-"*50)
    for i, name in enumerate(class_names[:out_classes]):
        class_dice_val = classwise_dice[i] if i < len(classwise_dice) else float('nan')
        class_iou_val = classwise_iou[i] if i < len(classwise_iou) else float('nan')
        print(f"Class {i} ({name:10s}) | Dice: {class_dice_val:.5f} | IoU: {class_iou_val:.5f}")
    print("="*50 + "\n")
    
    import sys
    sys.exit(0)

elif args.mode == 'test':
    print("Starting inference on test set...")
    model.eval()
    
    from PIL import Image
    
    os.makedirs(args.output, exist_ok=True)
    predictions_dir = os.path.join(args.output, "predictions")
    visual_dir = os.path.join(args.output, "predictions_visual")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    
    epoch_iterator_test = tqdm(
        test_loader, desc="Inference", dynamic_ncols=True
    )
    
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_test):
            test_inputs = batch["image"].to(device)
            test_outputs = sliding_window_inference(test_inputs, (96, 96), 2, model)
            
            # Retrieve original image path
            orig_filepath = test_files[step]["image"]
            orig_filename = os.path.basename(orig_filepath)
            
            # Load original dimensions
            orig_img = Image.open(orig_filepath)
            orig_w, orig_h = orig_img.size
            
            # Get argmax class indices (96, 96)
            pred_mask = torch.argmax(test_outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Resize back to original dimensions using nearest neighbor interpolation
            pred_mask_img = Image.fromarray(pred_mask)
            pred_mask_resized = pred_mask_img.resize((orig_w, orig_h), resample=Image.NEAREST)
            
            # Save grayscale class mask (indices 0, 1, 2, 3)
            pred_mask_resized.save(os.path.join(predictions_dir, orig_filename))
            
            # Map classes to colors for visualization
            pred_np = np.array(pred_mask_resized)
            color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            color_mask[pred_np == 1] = [255, 0, 0]    # Brain (Red)
            color_mask[pred_np == 2] = [255, 255, 0]  # CSP (Yellow)
            color_mask[pred_np == 3] = [0, 0, 255]    # LV (Blue)
            
            # Save colored mask
            Image.fromarray(color_mask).save(os.path.join(visual_dir, orig_filename))
            
    print(f"\nInference complete!")
    print(f"Grayscale label masks saved in: {predictions_dir}")
    print(f"Colored visual masks saved in: {visual_dir}\n")
    import sys
    sys.exit(0)

# If mode is train
max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )





