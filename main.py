import re
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).parent / "data"
DatasetType = int
TEST, TRAIN, INTERNET = 0, 1, 2


class Dataloader:

    def load_dataset(self, type: DatasetType) -> DataFrame:
        if type == TEST:
            return self.load("test_public")
        elif type == TRAIN:
            return self.load("train_public")
        elif type == INTERNET:
            return self.load("train_internet")
        else:
            raise Exception("训练集类型是0,1或2")

    def load(self, name: str):
        if (DATA_PATH / (name + ".feather")).exists():
            # 直接读取feather格式
            return pd.read_feather(DATA_PATH / (name + ".feather"))
        elif (DATA_PATH / (name + ".csv")).exists():
            # 没有存储为feather格式
            data = pd.read_csv(DATA_PATH / (name + ".csv"))
            data.to_feather(DATA_PATH / (name + ".feather"))
            return data
        else:
            raise Exception("训练集不存在")


class Preprocessor:
    internet_drop = ['user_id', 'sub_class', 'work_type', 'house_loan_status', 'marriage', 'offsprings', 'f5']
    data_drop = ['user_id', 'known_outstanding_loan', 'known_dero', 'app_type']

    def clean_mon(self, x: str):
        mons = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        year_group = re.search('(\d+)', x)
        if year_group:
            year = int(year_group.group(1))
            if year < 22:
                year += 2000
            elif 100 > year > 22:
                year += 1900
            else:
                year = 2022
        else:
            year = 2022

        month_group = re.search('([a-zA-Z]+)', x)
        if month_group:
            mon = month_group.group(1).lower()
            month = mons[mon]
        else:
            month = 0

        return year * 100 + month

    def process_column(self, train_data: DataFrame, test_data: DataFrame, train_internet: DataFrame):
        """属性对齐"""
        train_internet.drop(self.internet_drop, axis=1, inplace=True)
        # 将isDefault改为is_default
        train_data["is_default"] = train_data["isDefault"]
        train_data.drop(self.data_drop + ["isDefault"], axis=1, inplace=True)
        test_data.drop(self.data_drop, axis=1, inplace=True)

    def process_column_type(self, data: DataFrame):
        """处理数据类型"""
        # 处理日期
        data['issue_date'] = pd.to_datetime(data['issue_date'])
        data['issue_mon'] = data['issue_date'].dt.year * 100 + data['issue_date'].dt.month
        data.drop(['issue_date'], axis=1, inplace=True)
        data['work_year'] = data['work_year'].map({
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        })
        data['class'] = data['class'].map({
            'A': 0, 'B': 1, 'C': 2, 'D': 3,
            'E': 4, 'F': 5, 'G': 6
        })
        data['work_year'].fillna(-1, inplace=True)
        # 处理earlies_credit_mon中的月份
        data['earlies_credit_mon'] = data['earlies_credit_mon'].apply(self.clean_mon)

    def label_encode(self, data: DataFrame):
        """字符串编码"""
        employer_type_encoder = LabelEncoder()
        data['employer_type'] = employer_type_encoder.fit_transform(data['employer_type'])
        industry_encoder = LabelEncoder()
        data['industry'] = industry_encoder.fit_transform(data['industry'])

    def process(self, train_data: DataFrame, test_data: DataFrame, train_internet: DataFrame) -> DataFrame:
        self.process_column(train_data, test_data, train_internet)
        # 连接三个表
        data = pd.concat([train_data, train_internet, test_data]).reset_index(drop=True)
        self.process_column_type(data)
        # 编码字符串格式的属性
        self.label_encode(data)
        return data


class Trainer:

    def train(self, data: DataFrame):
        # 训练集
        train = data[data['is_default'].notna()]
        # 测试集
        test = data[data['is_default'].isna()]

        feature_names = list(filter(lambda x: x not in ['is_default', 'loan_id'], train.columns))

        prediction = test[['loan_id']].copy()
        prediction["isDefault"] = 0

        model = lgb.LGBMClassifier(objective='binary',
                                   boosting_type='gbdt',
                                   tree_learner='serial',
                                   num_leaves=32,
                                   max_depth=6,
                                   learning_rate=0.1,
                                   n_estimators=10000,
                                   subsample=0.8,
                                   colsample_bytree=0.6,
                                   reg_alpha=0.5,
                                   reg_lambda=0.5,
                                   random_state=2021,
                                   is_unbalance=True,
                                   metric='auc')
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
        for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train['is_default'])):
            X_train = train.iloc[trn_idx][feature_names]
            Y_train = train.iloc[trn_idx]['is_default']

            X_val = train.iloc[val_idx][feature_names]
            Y_val = train.iloc[val_idx]['is_default']

            lgb_model = model.fit(X_train,
                                  Y_train,
                                  eval_names=['train', 'valid'],
                                  eval_set=[(X_train, Y_train), (X_val, Y_val)],
                                  verbose=500,
                                  eval_metric='auc',
                                  early_stopping_rounds=50)

            pred_test = lgb_model.predict_proba(test[feature_names], num_iteration=lgb_model.best_iteration_)
            prediction["isDefault"] += pred_test[:, 1] / kfold.n_splits

        prediction.columns = ['id', 'isDefault']
        prediction.to_csv("submit.csv", index=False)


def main():
    dataloader = Dataloader()
    preprocessor = Preprocessor()
    trainer = Trainer()

    train_data = dataloader.load_dataset(TRAIN)
    test_data = dataloader.load_dataset(TEST)
    train_internet = dataloader.load_dataset(INTERNET)
    data = preprocessor.process(train_data, test_data, train_internet)
    trainer.train(data)


if __name__ == "__main__":
    main()
