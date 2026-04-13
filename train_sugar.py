import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to refine the mesh extracted from a trained 2dgs model.')

    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str,
                        help='(Required) path to the scene data to use.')
    parser.add_argument('-m', '--mesh_path',
                        type=str,
                        help='mesh.')
    parser.add_argument('-o', '--output_path',
                        type=str,
                        help='mesh.')
    parser.add_argument('-c', '--checkpoint_path',
                        type=str,
                        help='(Required) path to the model checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load',
                        type=int, default=30000,
                        help='iteration to load.')

    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000,
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None,
                        help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None,
                        help='Max coordinates to use for foreground.')

    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1,
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=8_000,
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
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False,
                        help='Use a white background instead of black.')

    # Mask directory for DTU dataset
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Path to the directory containing masks (e.g., for DTU dataset). If not specified, will look for a "mask" folder in the scene path.')

    # Parse arguments
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # ----- Refine SuGaR -----
    refined_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'mesh_path': args.mesh_path,
        'output_dir': args.output_path,
        'iteration_to_load': args.iteration_to_load,
        'normal_consistency_factor': 0.1,
        'gaussians_per_triangle': args.gaussians_per_triangle,
        'n_vertices_in_fg': args.n_vertices_in_mesh,
        'refinement_iterations': args.refinement_iterations,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'export_ply': args.export_ply,
        'eval': args.eval,
        'gpu': args.gpu,
        'white_background': args.white_background,
        'mask_dir': args.mask_dir,
    })
    refined_sugar_path = refined_training(refined_args)
    print(refined_sugar_path)

    # ----- Extract mesh and texture from refined SuGaR -----
    if args.export_uv_textured_mesh:
        refined_mesh_args = AttrDict({
            'scene_path': args.scene_path,
            'iteration_to_load': args.iteration_to_load,
            'checkpoint_path': args.checkpoint_path,
            'refined_model_path': refined_sugar_path,
            'mesh_output_dir': args.output_path,
            'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
            'square_size': args.square_size,
            'eval': args.eval,
            'gpu': args.gpu,
            'postprocess_mesh': args.postprocess_mesh,
            'postprocess_density_threshold': args.postprocess_density_threshold,
            'postprocess_iterations': args.postprocess_iterations,
            'mesh_path': args.mesh_path
        })
        refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)