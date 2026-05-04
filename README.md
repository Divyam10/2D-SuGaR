<div align="center">

# 2D-SuGaR: Surface-Aware Gaussian Splatting for Geometrically Accurate Mesh Reconstruction

**Prajwal Gupta C.R.**$^{\dagger 1}$, **Divyam Sheth**$^{\dagger 1,2}$, **Jinjoo Ha**$^{1}$, **Mirela Ostrek**$^{1,3}$, **Justus Thies**$^{1,2,3}$

$^1$ TU Darmstadt &nbsp;&nbsp; $^2$ ELIZA &nbsp;&nbsp; $^3$ Max Planck Institute for Intelligent Systems &nbsp;&nbsp; $^\dagger$ equal contribution

*Eurographics 2026 (Short Papers)*

[![Project Page](https://img.shields.io/badge/Project_Page-Live-blue)](https://divyam10.github.io/2D-SuGaR-page/) &nbsp;&nbsp; [![arXiv](https://img.shields.io/badge/preprint-2605.00569-red)](https://arxiv.org/abs/2605.00569)

<br>
<br>
<img src="./assets/results1.jpg" alt="results1.jpg" width="900"/>

</div>

## Abstract

3D Gaussian Splatting (3DGS) has emerged as a powerful technique for generating photorealistic renderings of
a scene in real-time. However, the volumetric nature of
3DGS limits its ability to accurately capture surface geometry. To address this, 2D Gaussian Splatting (2DGS) was
proposed to enable view-consistent and geometrically accurate surface reconstruction from multi-view images. However, 2DGS can be sensitive to the initialization of the Gaussian primitives. Reliance on Structure-from-Motion (SfM)
initializations, which can produce poor estimates on challenging image sets, may lead to subpar results. In this work,
we enhance 2DGS by incorporating monocular depth and
normal priors to improve both geometric accuracy and robustness. We propose a depth-guided initialization strategy
for Gaussians and introduce a clustering-based technique
for pruning degenerate Gaussians. We evaluate our method
on the DTU dataset, where it achieves state-of-the-art results in mesh reconstruction while preserving high-quality
novel view synthesis.


## TODO

- [x] Release arXiv paper
- [x] Release project page
- [ ] Attach DTU reconstructions
- [x] Release training and evaluation code



## DTU Benchmark Results

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary><br>

Chamfer Distance (lower is better) on 15 scenes from the DTU dataset. **Bold** = best per column.

| Method | 24 | 37 | 40 | 55 | 63 | 65 | 69 | 83 | 97 | 105 | 106 | 110 | 114 | 118 | 122 | **Mean↓** | **Time↓** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2DGS | 0.48 | 0.91 | 0.39 | 0.39 | 1.01 | 0.83 | 0.81 | 1.36 | 1.27 | 0.76 | 0.70 | 1.40 | 0.40 | 0.76 | 0.52 | 0.80 | 10.9m |
| MILo | **0.43** | 0.74 | 0.34 | **0.37** | **0.80** | 0.74 | 0.70 | 1.21 | 1.22 | 0.66 | 0.62 | **0.80** | **0.37** | 0.76 | 0.48 | 0.68 | 35m |
| Ours w/o Refinement | 0.50 | 0.67 | 0.33 | 0.44 | 1.24 | 0.70 | 0.63 | 1.24 | 1.13 | 0.63 | 0.49 | 1.29 | 0.43 | 0.56 | 0.44 | 0.72 | **10.5m** |
| **Ours** | 0.48 | **0.65** | **0.29** | 0.39 | 1.24 | **0.69** | **0.61** | **1.17** | **1.10** | **0.56** | **0.45** | 1.19 | **0.37** | **0.50** | **0.40** | **0.67** | 18.5m |

</details>


<div align="center">
<h3>Novel View Synthesis</h3>
<table>
  <tr>
    <th>Color</th>
    <th>Depth</th>
    <th>Normal</th>
  </tr>
  <tr>
    <td><img src="./assets/55_render_traj_color.gif" alt="scan_55_color.gif" width="250"/></td>
    <td><img src="./assets/55_render_traj_depth.gif" alt="scan_55_depth.gif" width="250"/></td>
    <td><img src="./assets/55_render_traj_normal.gif" alt="scan_55_normal.gif" width="250"/></td>
  </tr>
  <tr>
    <td><img src="./assets/37_render_traj_color.gif" alt="scan_37_color.gif" width="250"/></td>
    <td><img src="./assets/37_render_traj_depth.gif" alt="scan_37_depth.gif" width="250"/></td>
    <td><img src="./assets/37_render_traj_normal.gif" alt="scan_37_normal.gif" width="250"/></td>
  </tr>
</table>

<br>
<br>

</div>



## Setup

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary>

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (recommended for full training and evaluation)

### Software Requirements

- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions
- CUDA toolkit 11.8 for PyTorch extensions
- C++ Compiler and CUDA SDK must be compatible

Please refer to the <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting</a> repository for additional details about the requirements and setup.

### Installation

#### 1. Clone the Repository

```shell
# HTTPS
git clone https://github.com/prajwalcr/2d-sugar.git --recursive
```

or

```shell
# SSH
git clone git@github.com:prajwalcr/2d-sugar.git --recursive
```

#### 2. Create the Conda Environment

To create and activate the Conda environment with all the required packages, go inside the `2d-sugar/` directory and run the following command:

```shell
python3 install.py
conda activate 2d-sugar
```

This script will automatically create a Conda environment named `2d-sugar` and install all the required packages. It will also automatically install the <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting</a> rasterizer, <a href="https://github.com/hbb1/2d-gaussian-splatting">2D Gaussian Splatting</a> rasterizer as well as the <a href="https://nvlabs.github.io/nvdiffrast/">Nvdiffrast</a> library for faster mesh rasterization.

If you encounter any issues with the installation, you can try to follow the detailed instructions below to install the required packages manually.

<details>
<summary><span style="font-weight: bold;">
Detailed instructions for manual installation
</span></summary>

##### a) Install the required Python packages
To install the required Python packages and activate the environment, go inside the `2d-sugar/` directory and run the following commands:

```shell
conda env create -f environment.yml
conda activate 2d-sugar
```

If this command fails to create a working environment, you can try to install the required packages manually by running the following commands:
```shell
conda create --name 2d-sugar -y python=3.9
conda activate 2d-sugar
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d==0.7.4 -c pytorch3d
conda install -c plotly plotly
conda install -c conda-forge rich
conda install -c conda-forge plyfile==0.8.1
conda install -c conda-forge jupyterlab
conda install -c conda-forge nodejs
conda install -c conda-forge ipywidgets
pip install open3d
pip install --upgrade PyMCubes
pip install imageio opencv-python scikit-image trimesh kornia
pip install mediapy==1.2.4
```

##### b) Install the Rasterizer

Run the following commands inside the `2d-sugar` directory to install the additional Python submodules required for Gaussian Splatting:

```shell
cd gaussian_splatting_2d/submodules/diff-surfel-rasterization/
pip install -e .
cd ../simple-knn/
pip install -e .
cd ../../../gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e .
cd ../../../
```

Please refer to the <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting</a> and the <a href="https://github.com/hbb1/2d-gaussian-splatting">2D Gaussian Splatting</a> repositories for more details.

#### c) (Optional) Install Nvdiffrast for faster Mesh Rasterization

Installing Nvdiffrast is optional but will greatly speed up the textured mesh extraction step, from a few minutes to less than 10 seconds.

```shell
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ../
```

</details>

</details>


## Usage

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary>

### Quick Start

You can run the following single script to optimize a full 2d-sugar model from scratch using a COLMAP dataset:

```shell
python3 train.py -s <path to COLMAP dataset>
```

### Sample Commands

#### Generate depth and normal priors (required before 2DGS training):
```shell
python3 estimate_depth_normal_priors.py -s <path to COLMAP dataset>
```
This uses [Metric3D](https://github.com/yvanyin/metric3d) to generate per-image depth and normal maps saved under `estimated_dense_depth/` and `estimated_dense_normal/` inside the scene directory. These are used for depth-guided Gaussian initialization and the normal prior loss during 2DGS training.

#### Train our 2dgs model:
```shell
python3 gaussian_splatting_2d/train.py -s <path to COLMAP dataset> -m <model output path> --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000 --save_iterations 1 7000 30000 --iterations 30000 --cluster_prune_iterations 7000 --initialization_prior monocular_depth_normal
```

#### Render the trained 2dgs model:
```shell
python3 gaussian_splatting_2d/render.py -s <path to COLMAP dataset> -m <model output path> --iteration 30000 --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0
```
The above command produces a mesh at a path such as: ```{model output path}/train/ours_{iteration_num}/fuse_post.ply ```


#### Run sugar refinement on the model (extracted mesh):
```shell
python3 train_sugar.py -s <path to COLMAP dataset> -c <model path> -m <mesh path> -o <refined model output path>
```
The above command produces a mesh at a path such as: ```{model output path}/sugarfine_fuse_post_normalconsistency01_gaussperface1.obj ```
The actual path may differ based on the provided arguments.


#### Run the full pipeline:
```shell
python3 train.py -s <path to COLMAP dataset> -m <model output path> --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000 --save_iterations 1 7000 30000 --iterations 30000 --cluster_prune_iterations 7000 --initialization_prior monocular_depth_normal --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0 -o <mesh output path>
```

<details>
<summary><span style="font-weight: bold;">Please click here to see the most important arguments for the `train.py` script.</span></summary>

|             Parameter              |  Type  |                                                                                                                 Description                                                                                                                 |
|:----------------------------------:|:------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      `--initialization_prior`      | `str`  |                                                                                             Defines the method used to initialize the Gaussians                                                                                             |
|      `--lambda_dist`      | `int`  |                                                                                           Regularization hyperparameter for depth distortion loss                                                                                           |
|      `--lambda_normal`      | `int`  |                                                                                          Regularization hyperparameter for normal consistency loss                                                                                          |
|      `--lambda_normal_prior`      | `int`  |                                                                                             Regularization hyperparameter for normal prior loss                                                                                             |
|      `--depth_ratio`      | `int`  |                                                                                             0 for mean depth and 1 for median depth, 0 works for most cases                                                                                             |
|       `--scene_path` / `-s`        | `str`  |                                                                                          Path to the source directory containing a COLMAP dataset.                                                                                          |
|              `--eval`              | `bool` |                                                                              If True, performs an evaluation split of the training images. Default is `True`.                                                                               |
|           `--export_ply`           | `bool` |                            If True, export a `.ply` file with the refined 3D Gaussians at the end of the training. This file can be large (+/- 500MB), but is needed for using 3DGS viewers. Default is `True`.                             |
| `--export_uv_textured_mesh` / `-t` | `bool` | If True, will optimize and export a traditional textured mesh as an `.obj` file from the refined SuGaR model, after refinement. Computing a traditional color UV texture should just take a few seconds with Nvdiffrast. Default is `True`. |
|          `--square_size`           | `int`  |                              Size of the square allocated to each pair of triangles in the UV texture. Increase for higher texture resolution. Please decrease if you encounter memory issues. Default is `8`.                              |
|        `--white_background`        | `bool` |                                                                               If True, the background of the images will be set to white. Default is `False`.                                                                               |


</details>
</details>


## Setting up custom dataset with Colmap
<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary>
In this section, we provide the steps to create a dataset for any scene/object captured using a camera/smartphone.

Refer to the <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting</a> repository for the directory structure expected by our data loaders.

1. Capture a video of the object by moving around the object with the camera focused at the center of the object at all times.
2. Convert the video to a set of image frames at an appropriate frame rate. You can use ffmpeg for this.
    ```
    ffmpeg -i <video file> -vf fps=6 ./%04d.png
    ```
   Move the extracted frames to a separate folder
3. Create the dataset using Colmap
   - Launch Colmap.
   - Select the workspace folder where you want to create the dataset.
   - Select the image folder containing the frames.
   - Choose ```Data type``` as ```Video frames```.
   - Select ```Shared intrinsics``` and ```Sparse model```.
   - Run Colmap.
4. Convert Colmap files to text format.
    ```
    colmap model_converter --input_path {workspace_folder}/sparse/0 --output_path {workspace_folder}/sparse/0 --output_type TXT
    ```
5. Run the ```convert.py``` script to extract undistorted images and SfM information from the input images.
   ```
    python3 convert.py -s <workspace folder>
    ```

</details>


## Visualize the 2d-sugar models in real-time

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary><br>

After optimizing a 2d-sugar model, you can visualize the model in real-time using the 3D Gaussian Splatting viewer of your choice.

Indeed, after optimization, we automatically export a `.ply` file in the `./output/refined_ply/` directory, containing the refined 3D Gaussians of 2d-sugar's hybrid representation and compatible with any 3DGS viewer. For instance, you can use the viewer provided in the original implementation of <a href="https://github.com/graphdeco-inria/gaussian-splatting">3D Gaussian Splatting</a>, or the awesome <a href="https://github.com/playcanvas/supersplat">SuperSplat viewer</a>. 

An online, <a href="https://playcanvas.com/supersplat/editor">in-browser version of SuperSplat</a> is also available.

<div align="center" >
<img src="./assets/viewer_example.png" alt="viewer_example.png" width="800"/>
</div><br>

Refer to the <a href="https://github.com/Anttwo/SuGaR">SuGaR</a> to find out more about the support provided for blender add-ons.

</details>


## Evaluation

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary><br>

You can find scripts to evaluate our method under ```gaussian_splatting_2d/scripts```.

### DTU Evaluation

For geometry reconstruction on DTU dataset, please download the preprocessed data from [Drive](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) or [Hugging Face](https://huggingface.co/datasets/dylanebert/2DGS). You also need to download the ground truth [DTU point cloud](https://roboimagedata.compute.dtu.dk/?page_id=36).

Then, simply run the ```dtu_eval.py``` file to get metrics for the mesh extracted from the 2dgs model that is trained. 
```
python3 gaussian_splatting_2d/scripts/dtu_eval.py --dtu <path to the preprocessed DTU dataset> --DTU_Official <path to the official DTU dataset>
```

To see a further improvement in results, refine the obtained mesh by running the following command:
```
python3 train_sugar.py -s <path to the preprocessed DTU dataset>/scan{scan_number}/ -c <model path> -m <mesh path> -o <output path>
```

You can obtain metrics on the refined mesh by running the following script:
```
python3 gaussian_splatting_2d/scripts/eval_dtu/evaluate_single_scene.py --input_mesh <path to refined mesh> --scan_id <scan number> --output_dir <output path> --mask_dir <path to the preprocessed DTU dataset> --DTU <path to the official DTU dataset>
```

### Appearance Metrics (DTU)

Three scripts compute appearance metrics (PSNR, SSIM, LPIPS) at different stages of the pipeline:

**Step 1 — 2DGS appearance metrics** (`eval_appearance_2dgs.py`)

Computes metrics from pre-rendered images produced by the 2DGS training step. Reads directly from the `train/ours_30000/renders/` and `train/ours_30000/gt/` directories — no model loading or re-rendering required.
```
python3 eval_appearance_2dgs.py
```

**Step 2 — Refined model appearance metrics** (`eval_appearance_refined.py`)

Computes appearance metrics for the post-refined SuGaR model. Loads the refined PLY Gaussian model, re-renders each view against the DTU cameras, and saves per-view and per-scan `results.json` files.
```
python3 eval_appearance_refined.py \
    --ply_dir <path to refined PLY directory> \
    --dtu_dir <path to DTU dataset> \
    --output_dir <path to save metrics> \
    [--scans scan24 scan37 ...]
```

**Aggregate metrics** (`aggregate_dtu_metrics.py`)

Reads the `results.json` files produced by either of the above scripts and computes averages across all scans. Prints a per-scene table and saves a `summary_metrics.json`.
```
python3 aggregate_dtu_metrics.py
```

</details>


## Sample Results on Custom Data

<details>
<summary><span style="font-weight: bold;">Click here to see content.</span></summary><br>

<div align="center">
<h3>Room Scene</h3>
<table>
  <tr>
    <th>Color</th>
    <th>Depth</th>
  </tr>
  <tr>
    <td><img src="./assets/room_scene_color.gif" alt="room_scene_color.gif" width="360"/></td>
    <td><img src="./assets/room_scene_depth.gif" alt="room_scene_depth.gif" width="360"/></td>
  </tr>
</table>
</div>

<br>
<br>

<div align="center">
<h3>Frankfurt Fridge Magnet</h3>
</div>
This example consists of a very small object on a flat and largely plain surface, which makes it hard to get good SfM initializations. Our depth prior based initialization method shows a clear improvement in reconstruction quality in such scenarios.

<div align="center">
<table align="center">
  <tr>
    <th colspan="2">2DGS</th>
    <th colspan="2">Ours</th>
  </tr>

  <tr>
    <td colspan="2" align="center">
      <img src="./assets/frankfurt_fridge_magnet_2dgs.gif" width="360"/>
    </td>
    <td colspan="2" align="center">
      <img src="./assets/frankfurt_fridge_magnet_ours.gif" width="360"/>
    </td>
  </tr>

  <tr>
    <td align="center">Original</td>
    <td align="center">Rendering</td>
    <td align="center">Original</td>
    <td align="center">Rendering</td>
  </tr>
</table>
</div>

</details>


## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{r20262dsugarsurfaceawaregaussiansplatting,
      title={2D-SuGaR: Surface-Aware Gaussian Splatting for Geometrically Accurate Mesh Reconstruction}, 
      author={Prajwal Gupta C. R. and Divyam Sheth and Jinjoo Ha and Mirela Ostrek and Justus Thies},
      year={2026},
      eprint={2605.00569},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.00569}, 
}
```


## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [SuGaR](https://github.com/Anttwo/SuGaR). The TSDF fusion for extracting mesh is based on [Open3D](https://github.com/isl-org/Open3D). The rendering script for MipNeRF360 is adopted from [Multinerf](https://github.com/google-research/multinerf/), while the evaluation scripts for DTU and Tanks and Temples dataset are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively. The fusing operation for accelerating the renderer is inspired by [Han's repodcue](https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce). We thank all the authors and acknowledge the valuable contributions from each of these wonderful repositories. 

