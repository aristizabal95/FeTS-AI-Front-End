
import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
import SimpleITK as sitk
from .PrepareDataset import (
    generate_tumor_segmentation_fused_images,
    save_screenshot,
)
import subprocess
import time


MODALITY_MAPPING = {
    "t1c": "t1c",
    "t1ce": "t1c",
    "t1": "t1n",
    "t1n": "t1n",
    "t2": "t2w",
    "t2w": "t2w",
    "t2f": "t2f",
    "flair": "t2f",
}

MODALITY_VARIANTS = {
    "t1c": "T1GD",
    "t1ce": "T1GD",
    "t1": "T1",
    "t1n": "T1",
    "t2": "T2",
    "t2w": "T2",
    "t2f": "FLAIR",
    "flair": "FLAIR",
}

def setup_argparser():
    parser = argparse.ArgumentParser("MedPerf Brain Extraction")
    parser.add_argument(
        "--input-csv", dest="input_csv", type=str, help="Path to input csv"
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", type=str, help="Output path"
    )
    parser.add_argument(
        "--subject-id", dest="id", type=str, help="subject id"
    )
    parser.add_argument(
        "--timepoint", dest="tp", type=str, help="timepoint"
    )
    parser.add_argument(
        "--exec", dest="nnunet_executable", type=str, help="path to the nnunet executable"
    )

    return parser

def get_models():
    models_path = os.path.join(os.environ["RESULTS_FOLDER"], "nnUNet", "3d_fullres")
    return os.listdir(models_path)

def get_mod_order(model):
    order_path = os.path.join(os.environ["RESULTS_FOLDER"], os.pardir, "nnUNet_modality_order", model, "order")
    with open(order_path, "r") as f:
        order_str = f.readline()
    # remove 'order = ' from the splitted list
    modalities = order_str.split()[2:]
    modalities = [MODALITY_MAPPING[mod] for mod in modalities]
    return modalities

def prepare_case(path, id, tp, order):
    tmp_subject = f"{id}-{tp}"
    tmp_path = os.path.join(path, "tmp-data")
    tmp_subject_path = os.path.join(tmp_path, tmp_subject)
    tmp_out_path = os.path.join(path, "tmp-out")
    shutil.rmtree(tmp_path, ignore_errors=True)
    shutil.rmtree(tmp_out_path, ignore_errors=True)
    os.makedirs(tmp_subject_path)
    os.makedirs(tmp_out_path)
    in_modalities_path = os.path.join(path, "DataForFeTS", id, tp)
    input_modalities = {}
    for modality_file in os.listdir(in_modalities_path):
        if not modality_file.endswith(".nii.gz"):
            continue
        modality = modality_file[:-7].split("_")[-1]
        norm_mod = MODALITY_MAPPING[modality]
        mod_idx = order.index(norm_mod)
        mod_idx = str(mod_idx).zfill(4)

        out_modality_file = f"{tmp_subject}_{mod_idx}.nii.gz"
        in_file = os.path.join(in_modalities_path, modality_file)
        out_file = os.path.join(tmp_subject_path, out_modality_file)
        input_modalities[MODALITY_VARIANTS[modality]] = in_file
        shutil.copyfile(in_file, out_file)

    return tmp_subject_path, tmp_out_path, input_modalities

def run_model(model, data_path, out_path, nnunet_executable):
    # models are named Task<ID>_..., where <ID> is always 3 numbers
    task_id = model[4:7]
    cmd = f"{nnunet_executable} -i {data_path} -o {out_path} -t {task_id}"
    print(cmd)
    print(os.listdir(data_path))
    start = time.time()
    subprocess.call(cmd, shell=True)
    end = time.time()
    total_time = end - start
    print(f"Total time elapsed is {total_time} seconds")

def finalize_pred(tmp_out_path, out_pred_filepath):
    # We assume there's only one file in out_path
    pred = None
    for file in os.listdir(tmp_out_path):
        if file.endswith(".nii.gz"):
            pred = file

    if pred is None:
        raise RuntimeError("No tumor segmentation was found")

    pred_filepath = os.path.join(tmp_out_path, pred)
    shutil.move(pred_filepath, out_pred_filepath)
    return out_pred_filepath

def main():
    args = setup_argparser().parse_args()
    id, tp = args.id, args.tp
    subject_id = f"{id}_{tp}"
    models = get_models()
    outputs = []
    images_for_fusion = []
    out_path = os.path.join(args.output_dir, "DataForQC", id, tp)
    out_pred_path = os.path.join(out_path, "TumorMasksForQC")
    os.makedirs(out_pred_path, exist_ok=True)
    for i, model in enumerate(models):
        order = get_mod_order(model)
        tmp_data_path, tmp_out_path, input_modalities = prepare_case(
            args.output_dir, id, tp, order
        )
        out_pred_filepath = os.path.join(
            out_pred_path, f"{id}_{tp}_tumorMask_model_{i}.nii.gz"
        )
        run_model(model, tmp_data_path, tmp_out_path, args.nnunet_executable)
        output = finalize_pred(tmp_out_path, out_pred_filepath)
        outputs.append(output)
        images_for_fusion.append(sitk.ReadImage(output, sitk.sitkUInt8))

        # cleanup
        shutil.rmtree(tmp_data_path, ignore_errors=True)
        shutil.rmtree(tmp_out_path, ignore_errors=True)

    fused_outputs = generate_tumor_segmentation_fused_images(
        images_for_fusion, out_pred_path, subject_id
    )
    outputs += fused_outputs

    for output in outputs:
        # save the screenshot
        tumor_mask_id = os.path.basename(output).replace(".nii.gz", "")
        save_screenshot(
            input_modalities,
            os.path.join(
                out_path,
                f"{tumor_mask_id}_summary.png",
            ),
            output,
        )

if __name__ == "__main__":
    main()