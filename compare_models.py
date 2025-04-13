import dotenv

dotenv.load_dotenv()

from model_comparison_utils import compare_models, export_model_csv_data
from utils import setup_log
import argparse

setup_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_filename_prefix",
        type=str,
        help="Prefix of file on where results will be stored.",
        required=True,
    )
    parser.add_argument(
        "--model_results_file_prefix",
        type=str,
        help="Which models we want to compare.",
        required=True,
    )
    parser.add_argument(
        "--model_results_file_suffix",
        type=str,
        help="Which models we want to compare.",
        required=True,
    )
    parser.add_argument(
        "--verbose",
        help="Verbose",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    compare_models(
        results_filename_prefix=args.results_filename_prefix,
        model_results_file_prefix=args.model_results_file_prefix,
        model_results_file_suffix=args.model_results_file_suffix,
        verbose=args.verbose,
    )
    export_model_csv_data(
        results_filename_prefix=args.results_filename_prefix,
        model_results_file_prefix=args.model_results_file_prefix,
        model_results_file_suffix=args.model_results_file_suffix,
        verbose=args.verbose,
    )
