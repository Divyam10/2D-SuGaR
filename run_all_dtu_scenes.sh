#!/bin/bash

# DTU Multi-Scene Training and Evaluation Script
# This script trains and evaluates SuGaR on multiple DTU scenes with mask support

# Configuration
BASE_DATA_DIR="/home/divyam_sheth/data/dtu/DTU"
BASE_RESULTS_DIR="/home/divyam_sheth/results/dtu_exp"
BASE_OUTPUT_DIR="/home/divyam_sheth/results/dtu_mask_refine_15k_bgreg_exp1_better_lr_densify"
DTU_SAMPLE_DIR="/home/divyam_sheth/data/dtu/SampleSet/SampleSet/MVS_Data"
MASK_BASE_DIR="/home/yes/data/dtu/DTU"

# List of DTU scenes to process
SCENES=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)

# Training parameters
ITERATION=30000
N_VERTICES=1000000
GAUSSIANS_PER_TRIANGLE=1
REFINEMENT_ITERATIONS=15000

# GPU to use (change if needed)
GPU=0

# Function to run training for a single scene
train_scene() {
    local scan_id=$1
    local scene_path="${BASE_DATA_DIR}/scan${scan_id}"
    local checkpoint_path="${BASE_RESULTS_DIR}/scan${scan_id}/"
    local mesh_path="${BASE_RESULTS_DIR}/scan${scan_id}/train/ours_${ITERATION}/fuse_post.ply"
    local output_path="${BASE_OUTPUT_DIR}/scan${scan_id}"
    local mask_dir="${MASK_BASE_DIR}/scan${scan_id}/mask"

    echo "========================================"
    echo "Training Scene: scan${scan_id}"
    echo "========================================"

    # Check if required files exist
    if [ ! -d "$scene_path" ]; then
        echo "Error: Scene path does not exist: $scene_path"
        return 1
    fi

    if [ ! -d "$checkpoint_path" ]; then
        echo "Error: Checkpoint path does not exist: $checkpoint_path"
        return 1
    fi

    if [ ! -f "$mesh_path" ]; then
        echo "Error: Mesh file does not exist: $mesh_path"
        return 1
    fi

    # Create output directory
    mkdir -p "$output_path"

    # Run training
    echo "Starting training for scan${scan_id}..."
    python3 train_sugar.py \
        -s "$scene_path" \
        -c "$checkpoint_path" \
        -m "$mesh_path" \
        -o "$output_path" \
        -i "$ITERATION" \
        -v "$N_VERTICES" \
        -g "$GAUSSIANS_PER_TRIANGLE" \
        -f "$REFINEMENT_ITERATIONS" \
        --gpu "$GPU" \
        --mask_dir "$mask_dir" \
        --eval True

    if [ $? -eq 0 ]; then
        echo "Training completed successfully for scan${scan_id}"
        return 0
    else
        echo "Training failed for scan${scan_id}"
        return 1
    fi
}

# Function to run evaluation for a single scene
evaluate_scene() {
    local scan_id=$1
    local output_path="${BASE_OUTPUT_DIR}/scan${scan_id}"
    local input_mesh="${output_path}/sugarfine_fuse_post_normalconsistency01_gaussperface1.obj"

    echo "========================================"
    echo "Evaluating Scene: scan${scan_id}"
    echo "========================================"

    # Check if mesh exists
    if [ ! -f "$input_mesh" ]; then
        echo "Error: Output mesh does not exist: $input_mesh"
        echo "Skipping evaluation for scan${scan_id}"
        return 1
    fi

    # Run evaluation
    echo "Starting evaluation for scan${scan_id}..."
    python3 gaussian_splatting_2d/scripts/eval_dtu/evaluate_single_scene.py \
        --input_mesh "$input_mesh" \
        --scan_id "$scan_id" \
        --output_dir "$output_path" \
        --mask_dir "$MASK_BASE_DIR" \
        --DTU "$DTU_SAMPLE_DIR"

    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for scan${scan_id}"
        return 0
    else
        echo "Evaluation failed for scan${scan_id}"
        return 1
    fi
}

# Main execution
main() {
    local mode=${1:-"both"}  # Options: train, eval, both
    local start_scene=${2:-0}  # Optional: start from specific scene index

    echo "=========================================="
    echo "DTU Multi-Scene Processing Script"
    echo "Mode: $mode"
    echo "Total scenes: ${#SCENES[@]}"
    echo "Starting from scene index: $start_scene"
    echo "=========================================="

    local success_count=0
    local failure_count=0
    local skipped_count=0

    for idx in "${!SCENES[@]}"; do
        if [ $idx -lt $start_scene ]; then
            continue
        fi

        scan_id="${SCENES[$idx]}"

        echo ""
        echo "=========================================="
        echo "Processing scene $((idx + 1))/${#SCENES[@]}: scan${scan_id}"
        echo "=========================================="

        # Training
        if [ "$mode" = "train" ] || [ "$mode" = "both" ]; then
            train_scene "$scan_id"
            train_status=$?

            if [ $train_status -ne 0 ]; then
                ((failure_count++))
                echo "Skipping evaluation for scan${scan_id} due to training failure"
                continue
            fi
            ((success_count++))
        fi

        # Evaluation
        if [ "$mode" = "eval" ] || [ "$mode" = "both" ]; then
            evaluate_scene "$scan_id"
            eval_status=$?

            if [ $eval_status -ne 0 ]; then
                ((failure_count++))
            else
                ((success_count++))
            fi
        fi
    done

    # Print summary
    echo ""
    echo "=========================================="
    echo "Processing Complete!"
    echo "=========================================="
    echo "Successful: $success_count"
    echo "Failed: $failure_count"
    echo "Skipped: $skipped_count"
    echo "=========================================="
}

# Parse command line arguments
MODE=${1:-"both"}
START_SCENE=${2:-0}

# Validate mode
if [ "$MODE" != "train" ] && [ "$MODE" != "eval" ] && [ "$MODE" != "both" ]; then
    echo "Usage: $0 [train|eval|both] [start_scene_index]"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run both training and evaluation for all scenes"
    echo "  $0 train              # Run only training for all scenes"
    echo "  $0 eval               # Run only evaluation for all scenes"
    echo "  $0 both 5             # Run both, starting from scene index 5"
    exit 1
fi

# Run main function
main "$MODE" "$START_SCENE"
