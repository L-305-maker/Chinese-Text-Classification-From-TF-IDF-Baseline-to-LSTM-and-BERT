import argparse
from src.lr_train import main as lr_train_main
from src.train_bert import main as bert_train_main
from src.train_lstm import main as lstm_train_main
from src.data_process import process_raw_to_csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chinese Text Classification Project"
    )
    parser.add_argument(
        "--mode",
        type = str,
        required=True,
        choices=["Log_TF_IDF","BERT","LSTM","data_processor"],
        help="choose which task to run"
    )

    args, remaining_args = parser.parse_known_args()

    return args, remaining_args

def main():
    args, remaining_args = parse_args()
    if args.mode == "data_processor":
        process_raw_to_csv()
    elif args.mode == "BERT":
        bert_train_main(remaining_args)
    elif args.mode == "LSTM":
        lstm_train_main(remaining_args)
    elif args.mode == "Log_TF_IDF":
        lr_train_main(remaining_args)
    

if __name__ == "__main__":
    main()
