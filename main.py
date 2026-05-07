import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chinese Text Classification Project"
    )
    parser.add_argument(
        "--mode",
        type = str,
        required=True,
        choices=["Log_TF_IDF","BERT","LSTM","data_processor","visualize"],
        help="choose which task to run"
    )

    args, remaining_args = parser.parse_known_args()

    return args, remaining_args

def main():
    args, remaining_args = parse_args()
    if args.mode == "data_processor":
        from src.data_process import process_raw_to_csv

        process_raw_to_csv()
    elif args.mode == "BERT":
        from src.train_bert import main as bert_train_main

        bert_train_main(remaining_args)
    elif args.mode == "LSTM":
        from src.train_lstm import main as lstm_train_main

        lstm_train_main(remaining_args)
    elif args.mode == "Log_TF_IDF":
        from src.lr_train import main as lr_train_main

        lr_train_main(remaining_args)
    elif args.mode == "visualize":
        from src.visualize import main as visualize_main

        visualize_main(remaining_args)
    

if __name__ == "__main__":
    main()
