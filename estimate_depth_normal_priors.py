import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys
from argparse import ArgumentParser


def run_metric3d(rgb_origin, model):
    #### adjust input size to fit pretrained model
    # keep ratio resize
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # remember to scale intrinsic, hold depth
    intrinsic = [2892.33 * scale, 2883.18 * scale, 777 * scale, 581 * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT,
                             value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    ###################### canonical camera space ######################
    # inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})

    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                 pad_info[2]: pred_depth.shape[1] - pad_info[3]]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2],
                                                 mode='bilinear').squeeze()
    ###################### canonical camera space ######################

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    #### you can now do anything with the metric depth
    # such as evaluate predicted depth

    #### normal are also available
    if 'prediction_normal' in output_dict:  # only available for Metric3Dv2, i.e. vit model
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        normal_confidence = output_dict['prediction_normal'][:, 3, :,
                            :]  # see https://arxiv.org/abs/2109.09881 for details
        # un pad and resize to some size if needed
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1],
                      pad_info[2]: pred_normal.shape[2] - pad_info[3]]

        # upsample to original size
        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :, :, :], rgb_origin.shape[:2],
                                                      mode='bilinear').squeeze()
        pred_normal = pred_normal.permute(1, 2, 0)

        # pred_normal_vis = pred_normal.cpu().numpy()
        # pred_normal_vis = (pred_normal_vis + 1) / 2
        # plt.imshow(pred_normal_vis)
        # plt.show()
        # you can now do anything with the normal
        # such as visualize pred_normal
        # pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        # pred_normal_vis = (pred_normal_vis + 1) / 2
        # print(pred_normal_vis.shape)

        return pred_depth, pred_normal

    return pred_depth, None



def estimate_scene_depth_normal_priors(path):
    images_dir = os.path.join(path, 'images')

    depth_dir = os.path.join(path, "estimated_dense_depth")
    os.makedirs(depth_dir, exist_ok=True)

    normal_dir = os.path.join(path, "estimated_dense_normal")
    os.makedirs(normal_dir, exist_ok=True)

    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)
    model.cuda().eval()

    for filename in os.listdir(images_dir):
        if os.path.isdir(os.path.join(images_dir, filename)):
            continue

        if filename.endswith('.png'):
            image_file = os.path.join(images_dir, filename)
            rgb_origin = cv2.imread(image_file)[:, :, ::-1]

            pred_depth, pred_normal = run_metric3d(rgb_origin, model)

            depth_file = filename.rsplit('.', 1)[0] + '.npy'
            depth_save_path = os.path.join(depth_dir, depth_file)
            np.save(depth_save_path, pred_depth.cpu().numpy())

            normal_file = filename.rsplit('.', 1)[0] + '.npy'
            normal_save_path = os.path.join(normal_dir, normal_file)
            np.save(normal_save_path, pred_normal.cpu().numpy())


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('-s', '--source-path', type=str, default='/home/prajwal_chagi/downloads/DTU', help='Path to the source data')
    args = parser.parse_args(sys.argv[1:])

    source_path = args.source_path
    estimate_scene_depth_normal_priors(source_path)




