import argparse
from importlib import import_module


ENTRYPOINTS = {
    "Log_TF_IDF": ("src.train_lr", "main", True),
    "LR": ("src.train_lr", "main", True),
    "BERT": ("src.train_bert", "main", True),
    "LSTM": ("src.train_lstm", "main", True),
    "data_processor": ("src.data_processor", "process_raw_to_csv", False),
    "visualize": ("src.visualize", "main", True),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chinese Text Classification Project"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(ENTRYPOINTS),
        help="choose which task to run"
    )

    args, remaining_args = parser.parse_known_args()

    return args, remaining_args

def main():
    args, remaining_args = parse_args()
    module_name, function_name, accepts_args = ENTRYPOINTS[args.mode]
    runner = getattr(import_module(module_name), function_name)
    return runner(remaining_args) if accepts_args else runner()


if __name__ == "__main__":
    main()
