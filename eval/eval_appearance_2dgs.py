import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
from tqdm import tqdm


def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def calculate_lpips(img1, img2, lpips_model):
    """Calculate LPIPS between two images."""
    # Convert to tensor and normalize to [-1, 1]
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 127.5 - 1.0

    # Add batch dimension
    img1_tensor = img1_tensor.unsqueeze(0)
    img2_tensor = img2_tensor.unsqueeze(0)

    # Move to GPU if available
    device = next(lpips_model.parameters()).device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    with torch.no_grad():
        lpips_value = lpips_model(img1_tensor, img2_tensor)

    return lpips_value.item()


def process_scan(scan_path, lpips_model):
    """Process a single scan and calculate metrics."""
    gt_dir = scan_path / 'train' / 'ours_30000' / 'gt'
    renders_dir = scan_path / 'train' / 'ours_30000' / 'renders'

    if not gt_dir.exists() or not renders_dir.exists():
        print(f"Skipping {scan_path.name}: directories not found")
        return None

    # Get all image files
    gt_images = sorted(list(gt_dir.glob('*.png')))
    renders_images = sorted(list(renders_dir.glob('*.png')))

    if len(gt_images) == 0 or len(renders_images) == 0:
        print(f"Skipping {scan_path.name}: no images found")
        return None

    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Process each image pair
    for gt_img_path, render_img_path in zip(gt_images, renders_images):
        # Load images
        gt_img = load_image(gt_img_path)
        render_img = load_image(render_img_path)

        # Calculate metrics
        psnr_val = calculate_psnr(gt_img, render_img)
        ssim_val = calculate_ssim(gt_img, render_img)
        lpips_val = calculate_lpips(gt_img, render_img, lpips_model)

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)

    return {
        'psnr': {
            'mean': np.mean(psnr_values),
            'std': np.std(psnr_values),
            'values': psnr_values
        },
        'ssim': {
            'mean': np.mean(ssim_values),
            'std': np.std(ssim_values),
            'values': ssim_values
        },
        'lpips': {
            'mean': np.mean(lpips_values),
            'std': np.std(lpips_values),
            'values': lpips_values
        },
        'num_images': len(psnr_values)
    }


def main():
    dtu_path = 'eval/dtu'
    base_name = os.path.basename(dtu_path)
    base_dirs = [
        Path(dtu_path),
    ]

    # Initialize LPIPS model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    all_results = {}

    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"Directory {base_dir} does not exist, skipping...")
            continue

        print(f"\nProcessing {base_dir.name}...")

        # Get all scan directories
        scan_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('scan')])

        for scan_dir in tqdm(scan_dirs, desc=f"Processing {base_dir.name}"):
            scan_name = f"{base_dir.name}/{scan_dir.name}"
            print(f"\n  Processing {scan_name}...")

            results = process_scan(scan_dir, lpips_model)

            if results is not None:
                all_results[scan_name] = results
                print(f"    PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f}")
                print(f"    SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
                print(f"    LPIPS: {results['lpips']['mean']:.4f} ± {results['lpips']['std']:.4f}")
                print(f"    Images: {results['num_images']}")

    # Calculate overall statistics
    if all_results:
        all_psnr = []
        all_ssim = []
        all_lpips = []

        for scan_results in all_results.values():
            all_psnr.extend(scan_results['psnr']['values'])
            all_ssim.extend(scan_results['ssim']['values'])
            all_lpips.extend(scan_results['lpips']['values'])

        overall_stats = {
            'overall': {
                'psnr': {
                    'mean': np.mean(all_psnr),
                    'std': np.std(all_psnr)
                },
                'ssim': {
                    'mean': np.mean(all_ssim),
                    'std': np.std(all_ssim)
                },
                'lpips': {
                    'mean': np.mean(all_lpips),
                    'std': np.std(all_lpips)
                },
                'total_images': len(all_psnr)
            }
        }

        all_results['overall'] = overall_stats['overall']

        # Print overall statistics
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        print(f"PSNR:  {overall_stats['overall']['psnr']['mean']:.2f} ± {overall_stats['overall']['psnr']['std']:.2f}")
        print(f"SSIM:  {overall_stats['overall']['ssim']['mean']:.4f} ± {overall_stats['overall']['ssim']['std']:.4f}")
        print(f"LPIPS: {overall_stats['overall']['lpips']['mean']:.4f} ± {overall_stats['overall']['lpips']['std']:.4f}")
        print(f"Total images: {overall_stats['overall']['total_images']}")
        print("="*60)

    # Save results to JSON
    output_file = Path(f'results/metrics_results_{base_name}.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
