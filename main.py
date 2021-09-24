from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame

DATA_PATH = Path(__file__).parent / "data"
DatasetType = int
TEST, TRAIN = 0, 1


def load_dataset(type: DatasetType) -> DataFrame:
    path = DATA_PATH
    if type == TEST:
        path = path / "test_public.csv"
    elif type == TRAIN:
        path = path / "train_public.csv"
    else:
        raise Exception("训练集类型是0或1")
    return pd.read_csv(path)


def main():
    test_dataset = load_dataset(TEST)
    print(test_dataset)


if __name__ == "__main__":
    main()
