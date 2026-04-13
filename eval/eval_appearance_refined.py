

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import torchvision
import torchvision.transforms.functional as tf


from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


class PipelineParams:
    """Pipeline parameters for 3DGS rendering."""
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class SimpleCamera:
    """Simple camera class compatible with 3DGS renderer."""
    def __init__(self, R, T, FoVx, FoVy, image, image_name, width, height,
                 uid=0, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        from sugar_utils.graphics_utils import getWorld2View2, getProjectionMatrix

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.uid = uid

        self.original_image = image
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_cameras_from_dtu(dtu_path, white_background=True):
    """Load camera information from DTU dataset in COLMAP format."""
    scene_info = readColmapSceneInfo(dtu_path, images="images", eval=True, llffhold=8)

    cameras = []
    # Use train cameras (not test) for evaluation
    for idx, cam_info in enumerate(scene_info.train_cameras):
        image = cam_info.image
        if image is not None:
            # Handle white background
            if white_background:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1, 1, 1])
                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).cuda()  # HWC -> CHW

            height, width = image_tensor.shape[1], image_tensor.shape[2]
        else:
            image_tensor = None
            height = cam_info.height
            width = cam_info.width

        camera = SimpleCamera(
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=image_tensor,
            image_name=cam_info.image_name,
            width=width,
            height=height,
            uid=idx,
        )
        cameras.append(camera)

    return cameras


def load_gaussian_model(ply_path, sh_degree=3):
    """Load a Gaussian model from PLY file."""
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


def evaluate_scan(scan_name, ply_dir, dtu_dir, output_dir, white_background=True):
    """Evaluate a single DTU scan."""
    print(f"\n{'='*60}")
    print(f"Evaluating {scan_name}")
    print(f"{'='*60}")

    # Paths
    dtu_scan_path = os.path.join(dtu_dir, "DTU", scan_name)
    ply_scan_path = os.path.join(ply_dir, scan_name)

    # Find the PLY file
    if not os.path.exists(ply_scan_path):
        print(f"PLY directory not found: {ply_scan_path}")
        return None

    ply_files = [f for f in os.listdir(ply_scan_path) if f.endswith('.ply')]
    if not ply_files:
        print(f"No PLY file found in {ply_scan_path}")
        return None

    ply_path = os.path.join(ply_scan_path, ply_files[0])
    print(f"Using PLY: {ply_files[0]}")

    # Check if DTU scan exists
    if not os.path.exists(dtu_scan_path):
        print(f"DTU scan not found: {dtu_scan_path}")
        return None

    # Load cameras
    print("Loading camera information...")
    cameras = load_cameras_from_dtu(dtu_scan_path, white_background=white_background)
    print(f"Number of evaluation views: {len(cameras)}")

    # Load Gaussian model
    print("Loading Gaussian model from PLY...")
    gaussians = load_gaussian_model(ply_path)
    print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

    # Setup pipeline and background
    pipe = PipelineParams()
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Create output directories
    scan_output_dir = os.path.join(output_dir, scan_name)
    render_path = os.path.join(scan_output_dir, "renders")
    gts_path = os.path.join(scan_output_dir, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    # Render images
    print("Rendering images...")
    with torch.no_grad():
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering")):
            # Render using 3DGS renderer
            rendering = render(camera, gaussians, pipe, background)["render"]
            gt = camera.original_image

            # Save images
            torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
            torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))

    # Compute metrics
    print("Computing metrics...")
    ssims, psnrs, lpipss = [], [], []
    image_names = []

    for fname in sorted(os.listdir(render_path)):
        render_img = Image.open(os.path.join(render_path, fname))
        gt_img = Image.open(os.path.join(gts_path, fname))

        render_tensor = tf.to_tensor(render_img).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()

        ssims.append(ssim(render_tensor, gt_tensor))
        psnrs.append(psnr(render_tensor, gt_tensor))
        lpipss.append(lpips(render_tensor, gt_tensor, net_type='vgg'))
        image_names.append(fname)

    # Compute averages
    avg_ssim = torch.tensor(ssims).mean().item()
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()

    print(f"\nResults for {scan_name}:")
    print(f"  SSIM : {avg_ssim:.7f}")
    print(f"  PSNR : {avg_psnr:.7f}")
    print(f"  LPIPS: {avg_lpips:.7f}")

    # Save results
    with open(os.path.join(scan_output_dir, 'results.json'), 'w') as f:
        json.dump({
            'SSIM': avg_ssim,
            'PSNR': avg_psnr,
            'LPIPS': avg_lpips
        }, f, indent=2)

    per_view = {
        'SSIM': {name: s.item() for s, name in zip(ssims, image_names)},
        'PSNR': {name: p.item() for p, name in zip(psnrs, image_names)},
        'LPIPS': {name: l.item() for l, name in zip(lpipss, image_names)},
    }
    with open(os.path.join(scan_output_dir, 'per_view.json'), 'w') as f:
        json.dump(per_view, f, indent=2)

    # Cleanup
    del gaussians
    torch.cuda.empty_cache()

    return {'SSIM': avg_ssim, 'PSNR': avg_psnr, 'LPIPS': avg_lpips}


def main():
    parser = ArgumentParser(description="Calculate metrics for refined SuGaR PLY files on DTU")
    parser.add_argument('--ply_dir', type=str,
                        required=True,
                        help='Path to refined PLY directory')
    parser.add_argument('--dtu_dir', type=str,
                        required=True,
                        help='Path to DTU dataset')
    parser.add_argument('--output_dir', type=str,
                        required=True,
                        help='Path to save metrics')
    parser.add_argument('--scans', type=str, nargs='+', default=None,
                        help='Specific scans to evaluate (e.g., scan24 scan37)')
    parser.add_argument('--white_background', action='store_true', default=True,
                        help='Use white background for DTU (default: True)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')

    args = parser.parse_args()

    # Set device
    torch.cuda.set_device(args.gpu)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of scans
    if args.scans:
        scans = args.scans
    else:
        scans = sorted([d for d in os.listdir(args.ply_dir)
                       if os.path.isdir(os.path.join(args.ply_dir, d))
                       and d.startswith('scan')])

    print(f"Scans to evaluate: {scans}")

    # Evaluate each scan
    all_results = {}
    for scan in scans:
        try:
            metrics = evaluate_scan(scan, args.ply_dir, args.dtu_dir,
                                   args.output_dir, args.white_background)
            if metrics:
                all_results[scan] = metrics
        except Exception as e:
            print(f"Error evaluating {scan}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        avg_psnr = np.mean([r['PSNR'] for r in all_results.values()])
        avg_ssim = np.mean([r['SSIM'] for r in all_results.values()])
        avg_lpips = np.mean([r['LPIPS'] for r in all_results.values()])

        print(f"\n{'Scan':<12} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
        print("-" * 44)
        for scan, metrics in sorted(all_results.items()):
            print(f"{scan:<12} {metrics['PSNR']:>10.4f} {metrics['SSIM']:>10.4f} {metrics['LPIPS']:>10.4f}")
        print("-" * 44)
        print(f"{'Average':<12} {avg_psnr:>10.4f} {avg_ssim:>10.4f} {avg_lpips:>10.4f}")

        # Save summary
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump({
                'per_scan': all_results,
                'average': {'PSNR': avg_psnr, 'SSIM': avg_ssim, 'LPIPS': avg_lpips}
            }, f, indent=2)

        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
