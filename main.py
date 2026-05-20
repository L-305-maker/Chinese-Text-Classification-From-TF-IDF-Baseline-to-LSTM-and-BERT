import argparse
import sys
from importlib import import_module


ENTRYPOINTS = {
    "Log_TF_IDF": ("src.train_lr", "main", True),
    "LR": ("src.train_lr", "main", True),
    "BERT": ("src.train_bert", "main", True),
    "LSTM": ("src.train_lstm", "main", True),
    "data_processor": ("src.data_processor", "process_raw_to_csv", False),
    "visualize": ("src.visualize", "main", True),
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Chinese Text Classification Project",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=list(ENTRYPOINTS),
        help="choose which task to run"
    )

    return parser


def parse_args(argv=None):
    parser = build_parser()
    args, remaining_args = parser.parse_known_args(argv)

    if args.help and args.mode is None:
        parser.print_help()
        sys.exit(0)

    if args.mode is None:
        parser.error("the following arguments are required: --mode")

    if args.help:
        remaining_args.insert(0, "--help")

    return args, remaining_args


def main():
    args, remaining_args = parse_args()
    module_name, function_name, accepts_args = ENTRYPOINTS[args.mode]
    runner = getattr(import_module(module_name), function_name)
    return runner(remaining_args) if accepts_args else runner()


if __name__ == "__main__":
    main()
