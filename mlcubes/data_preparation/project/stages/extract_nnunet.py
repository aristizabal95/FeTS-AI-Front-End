from typing import Union, List, Tuple
from tqdm import tqdm
import pandas as pd
import os
from os.path import realpath, dirname, join
import shutil
import time
import SimpleITK as sitk
import subprocess
import traceback
from LabelFusion.wrapper import fuse_images

from .extract import Extract
from .PrepareDataset import (
    FINAL_FOLDER,
    generate_tumor_segmentation_fused_images,
    save_screenshot,
)
from .utils import update_row_with_dict, get_id_tp, MockTqdm

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


class ExtractNnUNet(Extract):
    def __init__(
        self,
        data_csv: str,
        out_path: str,
        subpath: str,
        prev_stage_path: str,
        prev_subpath: str,
        status_code: int,
        extra_labels_path=[],
        nnunet_executable: str = "/nnunet_env/bin/nnUNet_predict"
    ):
        self.data_csv = data_csv
        self.out_path = out_path
        self.subpath = subpath
        self.data_subpath = FINAL_FOLDER
        self.prev_path = prev_stage_path
        self.prev_subpath = prev_subpath
        os.makedirs(self.out_path, exist_ok=True)
        self.pbar = tqdm()
        self.failed = False
        self.exception = None
        self.__status_code = status_code
        self.extra_labels_path = extra_labels_path
        self.nnunet_executable = nnunet_executable

    @property
    def name(self) -> str:
        return "nnUNet Tumor Extraction"

    @property
    def status_code(self) -> str:
        return self.__status_code

    def _process_case(self, index: Union[str, int]):
        id, tp = get_id_tp(index)
        cmd = f"python3 -m project.stages.extract_nnunet_sp --input-csv={self.data_csv} --output-dir={self.out_path} --subject-id={id} --timepoint={tp} --exec={self.nnunet_executable}"
        subprocess.run(cmd.split(), check=True)