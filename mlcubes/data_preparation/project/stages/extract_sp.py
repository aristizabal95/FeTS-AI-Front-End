import argparse
import pandas as pd
from tqdm import tqdm
from .PrepareDataset import Preparator

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

    return parser

def main():
    args = setup_argparser().parse_args()
    prep = Preparator(args.input_csv, args.output_dir, "BratsPipeLine")
    
    # Reset the file contents for errors
    open(prep.stderr_log, "w").close()

    # Update the out dataframes to current state
    prep.read()

    df = prep.subjects_df
    row_search = df[(df["SubjectID"] == args.id) & (df["Timepoint"] == args.tp)]
    if len(row_search) > 0:
        row = row_search.iloc[0]
    else:
        # Most probably this case was semi-prepared. Mock a row
        row = pd.Series({"SubjectID": args.id, "Timepoint": args.tp, "T1": "", "T1GD": "", "T2": "", "FLAIR": ""})
        
    # recreate row
    prep.extract_brain(row, tqdm())

    prep.write()

if __name__ == "__main__":
    print("test")
    main()