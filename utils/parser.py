import argparse

def create_parser():

    parser = argparse.ArgumentParser()

    #INFERENCE ARGS
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_file", type=bool, default=True)
    #TRAINIGN ARGS
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", type=str, default="resnet_18")
    parser.add_argument("--metric", type=str, default="ACCURACY")

    args = parser.parse_args()
    return args