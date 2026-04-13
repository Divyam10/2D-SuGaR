import argparse
import json
import subprocess
import sys
from pathlib import Path

from sugar_utils.general_utils import str2bool
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar
from gaussian_splatting_2d.arguments import ModelParams, OptimizationParams, PipelineParams
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# Convert dict args to CLI-style arguments (e.g. "--lr 0.001")
def convert_dict_to_args(dict_args):
    cli_args = []
    for k, v in dict_args.items():
        if v is None:
            continue

        # Handle boolean flags
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
            continue

        # Handle list/tuple values (like nargs="+")
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            cli_args.append(f"--{k}")
            cli_args += [str(x) for x in v]
            continue

        # Default: single scalar value
        cli_args += [f"--{k}", str(v)]

    return cli_args


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full 2d-sugar model.')

    parser.add_argument('--skip_2dgs_train', action='store_true')
    parser.add_argument('--skip_2dgs_render', action='store_true')
    parser.add_argument('--skip_sugar_refine', action='store_true')

    # -- 2DGS Training Args --
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # -- 2DGS Render Args --
    parser.add_argument("--skip_render_train", action="store_true")
    parser.add_argument("--skip_render_test", action="store_true")
    parser.add_argument("--skip_render_mesh", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')

    # -- Sugar Refinement Args --
    parser.add_argument('-o', '--output_path',
                        type=str,
                        help='mesh.')
    parser.add_argument('--mesh_path',
                        type=str,
                        help='mesh.')

    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000,
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None,
                        help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None,
                        help='Max coordinates to use for foreground.')

    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1,
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000,
                        help='Number of refinement iterations.')

    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True,
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                             'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=8, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False,
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                             'This step takes a few minutes and is not needed in general, as it can also be risky. '
                             'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')

    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                             'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')

    # Evaluation split
    parser.add_argument('--eval_sugar_refine', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')


    # Parse arguments
    args = parser.parse_args()

    if not args.skip_2dgs_train:
        # -- 2DGS Train --
        train_args_dict = {
            'sh_degree': args.sh_degree,
            'source_path': args.source_path,
            'model_path': args.model_path,
            'images': args.images,
            'resolution': args.resolution,
            'white_background': args.white_background,
            'data_device': args.data_device,
            'eval': args.eval,
            'render_items': args.render_items,
            'initialization_prior': args.initialization_prior,
            'convert_SHs_python': args.convert_SHs_python,
            'compute_cov3D_python': args.compute_cov3D_python,
            'depth_ratio': args.depth_ratio,
            'debug': args.debug,
            'iterations': args.iterations,
            'position_lr_init': args.position_lr_init,
            'position_lr_final': args.position_lr_final,
            'position_lr_delay_mult': args.position_lr_delay_mult,
            'position_lr_max_steps': args.position_lr_max_steps,
            'feature_lr': args.feature_lr,
            'opacity_lr': args.opacity_lr,
            'scaling_lr': args.scaling_lr,
            'rotation_lr': args.rotation_lr,
            'percent_dense': args.percent_dense,
            'lambda_dssim': args.lambda_dssim,
            'lambda_dist': args.lambda_dist,
            'lambda_normal': args.lambda_normal,
            'lambda_normal_prior': args.lambda_normal_prior,
            'opacity_cull': args.opacity_cull,
            'densification_interval': args.densification_interval,
            'opacity_reset_interval': args.opacity_reset_interval,
            'densify_from_iter': args.densify_from_iter,
            'densify_until_iter': args.densify_until_iter,
            'densify_grad_threshold': args.densify_grad_threshold,
            'cluster_prune_iterations': args.cluster_prune_iterations,
            'dbscan_min_samples': args.dbscan_min_samples,
            'dbscan_knn_percentile': args.dbscan_knn_percentile,
            'ip': args.ip,
            'port': args.port,
            'detect_anomaly': args.detect_anomaly,
            'test_iterations': args.test_iterations,
            'save_iterations': args.save_iterations,
            'quiet': args.quiet,
            'checkpoint_iterations': args.checkpoint_iterations,
            'start_checkpoint': args.start_checkpoint,
        }

        train_script_2dgs = Path("./gaussian_splatting_2d/train.py").resolve()

        cli_args = convert_dict_to_args(train_args_dict)

        # Build command
        cmd = [sys.executable, str(train_script_2dgs)] + cli_args

        # Run and stream output live
        subprocess.run(cmd, check=True)

    if not args.skip_2dgs_render:
        ## -- 2DGS Render --
        render_args_dict = {
            'sh_degree': args.sh_degree,
            'source_path': args.source_path,
            'model_path': args.model_path,
            'images': args.images,
            'resolution': args.resolution,
            'white_background': args.white_background,
            'data_device': args.data_device,
            'eval': args.eval,
            'render_items': args.render_items,
            'initialization_prior': args.initialization_prior,
            'convert_SHs_python': args.convert_SHs_python,
            'compute_cov3D_python': args.compute_cov3D_python,
            'depth_ratio': args.depth_ratio,
            'debug': args.debug,
            'iteration': args.iterations,
            'skip_train': args.skip_render_train,
            'skip_test': args.skip_render_train,
            'skip_mesh': args.skip_render_train,
            'quiet': args.quiet,
            'render_path': args.render_path,
            'voxel_size': args.voxel_size,
            'depth_trunc': args.depth_trunc,
            'sdf_trunc': args.sdf_trunc,
            'num_cluster': args.num_cluster,
            'unbounded': args.unbounded,
            'mesh_res': args.mesh_res,
        }

        render_script_2dgs = Path("./gaussian_splatting_2d/render.py").resolve()

        cli_args = convert_dict_to_args(render_args_dict)

        # Build command
        cmd = [sys.executable, str(render_script_2dgs)] + cli_args

        # Run and stream output live
        subprocess.run(cmd, check=True)


    # -- Refine SuGaR --
    if not args.skip_sugar_refine:
        os.makedirs(args.output_path, exist_ok=True)

        mesh_path = args.mesh_path
        if not mesh_path:
            if args.unbounded:
                mesh_path = os.path.join(args.model_path, "train", f"ours_{args.iterations}", "fuse_unbounded_post.ply")
            else:
                mesh_path = os.path.join(args.model_path, "train", f"ours_{args.iterations}", "fuse_post.ply")

        refined_args = AttrDict({
            'scene_path': args.source_path,
            'checkpoint_path': args.model_path,
            'mesh_path': mesh_path,
            'output_dir': args.output_path,
            'iteration_to_load': args.iterations,
            'normal_consistency_factor': 0.1,
            'gaussians_per_triangle': args.gaussians_per_triangle,
            'n_vertices_in_fg': args.n_vertices_in_mesh,
            'refinement_iterations': args.refinement_iterations,
            'bboxmin': args.bboxmin,
            'bboxmax': args.bboxmax,
            'export_ply': args.export_ply,
            'eval': args.eval_sugar_refine,
            'gpu': args.gpu,
            'white_background': args.white_background,
        })
        refined_sugar_path = refined_training(refined_args)
        print(refined_sugar_path)

        # -- Extract mesh and texture from refined SuGaR --
        if args.export_uv_textured_mesh:
            refined_mesh_args = AttrDict({
                'scene_path': args.source_path,
                'iteration_to_load': args.iterations,
                'checkpoint_path': args.model_path,
                'refined_model_path': refined_sugar_path,
                'mesh_output_dir': args.output_path,
                'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
                'square_size': args.square_size,
                'eval': args.eval_sugar_refine,
                'gpu': args.gpu,
                'postprocess_mesh': args.postprocess_mesh,
                'postprocess_density_threshold': args.postprocess_density_threshold,
                'postprocess_iterations': args.postprocess_iterations,
                'mesh_path': mesh_path,
            })
            refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)
