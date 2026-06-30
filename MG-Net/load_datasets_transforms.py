import os
import glob
import numpy as np
import torch
from PIL import Image

from monai.transforms import (
    MapTransform,
    AsDiscreted,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    SaveImaged,
    Activationsd,
    Resized,
    ScaleIntensityd
)

try:
    from monai.transforms import AddChanneld
except ImportError:
    from monai.transforms import EnsureChannelFirstd as AddChanneld

class LoadPNGImaged(MapTransform):
    """
    Custom transform to load grayscale PNG image and map RGB PASCAL VOC label image
    to single-channel class indices (0: Background, 1: Brain, 2: CSP, 3: LV).
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key not in d:
                continue
            filepath = d[key]
            img = Image.open(filepath)
            if key == "image":
                # Convert ultrasound image to grayscale (1 channel)
                img = img.convert('L')
                img_np = np.array(img, dtype=np.float32)
                img_np = np.expand_dims(img_np, axis=0) # (1, H, W)
                d[key] = img_np
            elif key == "label":
                # PASCAL VOC mask image is RGB
                img = img.convert('RGB')
                img_np = np.array(img, dtype=np.uint8)
                r = img_np[..., 0]
                g = img_np[..., 1]
                b = img_np[..., 2]

                # Map colors:
                # Background: [0, 0, 0] -> 0
                # Brain: [255, 0, 0] -> 1
                # CSP: [255, 255, 0] -> 2
                # LV: [0, 0, 255] -> 3
                new_label = np.zeros(r.shape, dtype=np.int64)

                # Tolerance matching for solid colors
                brain_mask = (r > 150) & (g < 100) & (b < 100)
                csp_mask = (r > 150) & (g > 150) & (b < 100)
                lv_mask = (r < 100) & (g < 100) & (b > 150)

                new_label[brain_mask] = 1
                new_label[csp_mask] = 2
                new_label[lv_mask] = 3

                new_label = np.expand_dims(new_label, axis=0) # (1, H, W)
                d[key] = new_label
        return d

def data_loader(args):
    root_dir = args.root
    dataset = args.dataset

    print('Start to load data from directory: {}'.format(root_dir))

    if dataset == 'feta':
        out_classes = 8
    elif dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16
    elif dataset in ['ultrasound', 'prenatal_us']:
        out_classes = 4

    if args.mode in ['train', 'eval']:
        train_samples = {}
        valid_samples = {}

        if dataset in ['ultrasound', 'prenatal_us']:
            image_dir = os.path.join(root_dir, 'Orginal_train_images_to_959_661')
            mask_dir = os.path.join(root_dir, 'Mask', 'Test-Dataset-Segmentation', 'SegmentationClass')
            all_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))

            images_list = []
            labels_list = []
            for img_path in all_images:
                base_name = os.path.basename(img_path)
                lbl_path = os.path.join(mask_dir, base_name)
                if os.path.exists(lbl_path):
                    images_list.append(img_path)
                    labels_list.append(lbl_path)

            # Perform a deterministic 80/20 train/validation split
            train_img = []
            train_label = []
            valid_img = []
            valid_label = []
            for idx, (img, lbl) in enumerate(zip(images_list, labels_list)):
                if idx % 5 == 0:
                    valid_img.append(img)
                    valid_label.append(lbl)
                else:
                    train_img.append(img)
                    train_label.append(lbl)

            train_samples['images'] = train_img
            train_samples['labels'] = train_label
            valid_samples['images'] = valid_img
            valid_samples['labels'] = valid_label
        else:
            ## Input training data
            train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
            train_samples['images'] = train_img
            train_samples['labels'] = train_label

            ## Input validation data
            valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
            valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
            valid_samples['images'] = valid_img
            valid_samples['labels'] = valid_label

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        if dataset in ['ultrasound', 'prenatal_us']:
            image_dir = os.path.join(root_dir, 'Orginal_train_images_to_959_661')
            test_img = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            test_samples['images'] = test_img
        else:
            ## Input inference data
            test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
            test_samples['images'] = test_img

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes


def data_transforms(args):
    dataset = args.dataset
    if args.mode in ['train', 'eval']:
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset in ['ultrasound', 'prenatal_us']:
        train_transforms = Compose(
            [
                LoadPNGImaged(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=(96, 96), mode=("bilinear", "nearest")),
                ScaleIntensityd(keys=["image"]),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96),
                    rotate_range=(np.pi / 15,),
                    scale_range=(0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadPNGImaged(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=(96, 96), mode=("bilinear", "nearest")),
                ScaleIntensityd(keys=["image"]),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadPNGImaged(keys=["image"]),
                Resized(keys=["image"], spatial_size=(96, 96), mode="bilinear"),
                ScaleIntensityd(keys=["image"]),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'feta':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"],spatial_size=(96, 96, 96), mode="constant", method="end"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'flare':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'amos':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )


    if args.mode in ['train', 'eval']:
        if dataset in ['ultrasound', 'prenatal_us']:
            print('Performed Data Augmentations for all samples!')
        else:
            print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
            print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms


def infer_post_transforms(args, test_transforms, out_classes):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms
