import argparse
import os
import sys


from Marker.f1_marker import calc_f1_score
from Marker.rougel_marker import calc_rl_score
from Marker.ctrl_marker import calc_ctrl_score
from Marker.med_nli_marker import calc_med_nli_score


if __name__ == "__main__":
    current_directory = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str)
    parser.add_argument("--output_file",type=str)
    parser.add_argument("--root_path", type=str, default=f"{current_directory}")
    args = parser.parse_args()

    # print(f"The average f1 score in {args.input_file}: {calc_f1_score(args.input_file)}")
    # print(f"The average rouge-l score in {args.input_file}: {calc_rl_score(args.input_file)}")
    print(f"The average ctrl score in {args.input_file}: {calc_ctrl_score(args.input_file, args.root_path)}")
    # med_nli_score = calc_med_nli_score(args.input_file, args.output_file)
    # print(f"The average med-nli score by sample or sentence in {args.input_file}: {med_nli_score[0]} {med_nli_score[1]}")