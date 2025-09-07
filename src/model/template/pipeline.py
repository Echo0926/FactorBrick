import os,tqdm
import pandas as pd
import numpy as np
import dolphindb as ddb
from typing import List, Dict
from src.ReturnModel_mr import ReturnModel_Backtest_mr as ReturnModel_Backtest
from src.model_func.template.torch_model import *
from src.utils import init_path,get_glob_list,save_model,load_model,find_closest_lower,parallel_read_pqt
import torch
from sklearn.ensemble import AdaBoostRegressor  # AdaBoost
from sklearn.ensemble import RandomForestRegressor  # RandomForest
from sklearn.ensemble import GradientBoostingRegressor  # GBDT回归器
from sklearn.neural_network import MLPRegressor # MLP(sklearn版)
# from catboost import CatBoostRegressor  # CatBoost
from lightgbm import LGBMRegressor      # LGBM
from xgboost import XGBRegressor        # XGB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator
from typing import List,Tuple
import warnings
warnings.filterwarnings("ignore")
# torch.autograd.set_detect_anomaly(True) # 启用异常检测

"""
由于FactorBrick中自定义模型始终预测的是收益率
因而这里用的全部是Regressor而非Classifier
"""

# 硬编码: (模型名称全部小写)
model_sklearn = {
    "adaboost": AdaBoostRegressor,      # AdaBoost
    # "catboost": CatBoostRegressor,      # CatBoost
    "dnn": CustomDNN,                   # DNN Wrapper(torch)
    "resnet": CustomResNet,             # ResNet Wrapper(torch)
    "gbdt": GradientBoostingRegressor,  # GBDT回归器
    "lightgbm": LGBMRegressor,          # LightGBM
    "mlp": MLPRegressor,                # MLP(sklearn)
    "randomforest": RandomForestRegressor, # RandomForest
    "xgboost": XGBRegressor             # XGBoost
}
early_stop_list = ["catboost","lightgbm","xgboost"]

def sklearn_model_pick(x, y, model_config: Dict,
                       model_name: str, model_method,
                       cv:int = 5,
                       eval_set:List[Tuple]=None  # 早停验证集(可选)
                       ) -> BaseEstimator:
    """
    Sklearn网格搜索模型最佳参数
    """
    # 参数准备
    default_params, grid_params = model_config["default_params"], model_config["grid_params"]
    input_dim = x.shape[1]
    torch.manual_seed(42)

    # 特殊模型
    if model_name in ["dnn","resnet"]:
        if model_name == "dnn":
            model_method = get_DNN(input_dim=input_dim, **default_params)
        elif model_name == "resnet":
             model_method = get_RESNET(input_dim=input_dim, **default_params)
        grid = GridSearchCV(estimator=model_method,
                            param_grid=grid_params,
                            cv=cv,
                            n_jobs=-1)
    else:
        grid = GridSearchCV(estimator=model_method(**default_params),
                            param_grid=grid_params,
                            cv=cv,
                            n_jobs=-1)

    # 是否早停(可以早停+设置了早停验证集)
    if model_name in early_stop_list and eval_set:
        grid.fit(x, y, eval_set=eval_set)
    else:
        grid.fit(x, y)

    # 返回最佳模型
    best_params, best_scores = grid.best_params_, grid.best_score_
    if model_name in ["dnn","resnet"]:
        if model_name =="dnn":
            best_model = get_DNN(input_dim=input_dim, **best_params)
        elif model_name == "resnet":
            best_model = get_RESNET(input_dim=input_dim, **best_params)
    else:
        best_model = model_method(**best_params)

    return best_model.fit(x, y)

def get_fixed_data_from_dataframe(self: ReturnModel_Backtest,
                                  factor_list: List,
                                  call_back_interval:int = 1,
                                  current_period:int = 0,
                                  split_test_size:float=0.2,
                                  random_state:int = 42,
                                  split_train: bool = True,  # 将训练集继续二比八分割
                                  latest_test: bool = True,  # 是否最新一期period作为测试集
                                  ) -> Dict:
    """获取固定全部因子的训练集与测试集"""    # Combine Data
    # Configuration
    combine_path = self.combine_database
    factor_list = self.factor_list
    label = self.label_pred
    x_train_train = None
    x_train_test = None
    y_train_train = None
    y_train_test = None

    # Data Prepare
    data = parallel_read_pqt(file_path=combine_path,
                             columns=["period","date","minute","symbol",label]+factor_list,
                             start_date=self.start_date,    # 这里的start_date & end_date可以更精确一点
                             end_date=self.end_date,
                             desc="Reading Combine Data For Training...")
    data = data.sort_values(by=["symbol","period"]).reset_index(drop=True).fillna(0.0)

    # 数据清洗
    slice_df = data.query(f'{current_period-call_back_interval} <= period <= {current_period}').reset_index(drop=True)
    slice_df.replace([np.inf, -np.inf], np.nan, inplace=True)    # 用每个period每列的均值填充空值
    slice_df[factor_list] = slice_df.groupby('period')[factor_list].transform(lambda x: x.fillna(np.nanmean(x)))
    slice_df.fillna(0.0,inplace=True)

    # 分period进行特征标准化
    x_df = slice_df[factor_list]
    for period in slice_df['period'].unique():
        period_mask = slice_df['period'] == period
        scaler = StandardScaler()  # 每个period使用独立的StandardScaler实例
        x_df.loc[period_mask, factor_list] = scaler.fit_transform(x_df.loc[period_mask, factor_list])

    slice_df[factor_list] = x_df

    # 数据分割
    x, y = slice_df[factor_list], slice_df[f"{label}"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=split_test_size,
                                                        random_state=random_state)
    if latest_test:
        latest_df = slice_df[slice_df["period"]==current_period].reset_index(drop=True)
        hist_df = slice_df[slice_df["period"]<current_period].reset_index(drop=True)
        x_train, y_train, x_test, y_test = hist_df[factor_list], hist_df[f"{label}"],  latest_df[factor_list], latest_df[f"{label}"]
        x_train = x_train.to_numpy("float32")
        y_train = y_train.to_numpy("float32")
        x_test = x_test.to_numpy("float32")
        y_test = y_test.to_numpy("float32")

    if split_train: # 如果对训练集继续train
        x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train,
                                                                                    test_size=split_test_size,
                                                                                    random_state=random_state)

    return {"data": slice_df, "x": x.to_numpy("float32"), "y": y.to_numpy("float32"),
            "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test,
            "x_train_train": x_train_train, "x_train_test": x_train_test,
            "y_train_train":y_train_train, "y_train_test":y_train_test}

def get_fixed_data_from_dolphindb(self,
                                  factor_list: List,
                                  call_back_interval:int = 1,
                                  current_period:int = 0,
                                  split_test_size: float = 0.2,
                                  random_state: int = 42,
                                  split_train: bool = True,  # 将训练集继续二比八分割
                                  latest_test: bool = True,  # 是否最新一期period作为测试集
                                  ) -> Dict:
    # Configuration
    combine_path = (self.combine_database, self.combine_table)
    factor_list = self.factor_list
    label = self.label_pred
    x_train_train = None
    x_train_test = None
    y_train_train = None
    y_train_test = None

    slice_df = self.session.run(rf"""
    select * from loadTable("{combine_path[0]}","{combine_path[1]}") where {current_period}-{call_back_interval} <= period <= {current_period};
    """).reset_index(drop=True)
    slice_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    slice_df[factor_list] = slice_df.groupby('period')[factor_list].transform(lambda x: x.fillna(np.nanmean(x)))
    slice_df.fillna(0.0,inplace=True)

    # 分period进行特征标准化
    x_df = slice_df[factor_list]
    for period in slice_df['period'].unique():
        period_mask = slice_df['period'] == period
        scaler = StandardScaler()  # 每个period使用独立的StandardScaler实例
        x_df.loc[period_mask, factor_list] = scaler.fit_transform(x_df.loc[period_mask, factor_list])

    slice_df[factor_list] = x_df

    # 数据分割
    x, y = slice_df[factor_list], slice_df[f"{label}"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=split_test_size,
                                                        random_state=random_state)
    if latest_test:
        latest_df = slice_df[slice_df["period"]==current_period].reset_index(drop=True)
        hist_df = slice_df[slice_df["period"]<current_period].reset_index(drop=True)
        x_train, y_train, x_test, y_test = hist_df[factor_list], hist_df[f"{label}"],  latest_df[factor_list], latest_df[f"{label}"]
        x_train = x_train.to_numpy("float32")
        x_test = x_test.to_numpy("float32")
        y_train = y_train.to_numpy("float32")
        y_test = y_test.to_numpy("float32")

    if split_train: # 如果对训练集继续train
        x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train,
                                                                                    test_size=split_test_size,
                                                                                    random_state=random_state)


    return {"data": slice_df, "x": x.to_numpy("float32"), "y": y.to_numpy("float32"),
            "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test,
            "x_train_train": x_train_train, "x_train_test": x_train_test,
            "y_train_train":y_train_train, "y_train_test":y_train_test}


def get_flexible_data_from_dataframe(self: ReturnModel_Backtest,
                                     factor_list: List, # 这里传入的是全部因子池
                                     call_back_interval:int = 1,
                                     current_period:int = 0,
                                     split_test_size: float = 0.2,
                                     random_state: int = 42,
                                    split_train: bool = True,  # 将训练集继续二比八分割
                                     latest_test: bool = True,  # 是否最新一期period作为测试集
                                     ic_threshold = None,   # IC绝对值阈值
                                     ic_percentile = None   # IC绝对值百分比
                                     ) -> Dict:
    """获取指定因子List -> 传参到"""
    # Configuration
    if not ic_threshold and not ic_percentile:
        raise ValueError("ic_threshold or ic_percentile must be provided.")
    if ic_threshold and not ic_percentile:    # 说明是按照IC绝对值的阈值进行筛选的
        factor_slice = self.session.run(f"""
        current_period = {current_period};
        factor_return = select * from loadTable("{self.result_database}","{self.summary_table}") where period == current_period and class == "IC" and indicator in {factor_list};
        factor_return = select * from factor_return where abs(value) >= percentile(abs(value), {ic_percentile}, "linear");
        select distinct(indicator) as factor from factor_return;
        """)
    elif ic_percentile and not ic_threshold:   # 说明是是按照IC绝对值的百分比进行筛选的
        factor_slice = self.session.run(f"""
        current_period = {current_period};
        factor_return = select * from loadTable("{self.result_database}","{self.summary_table}") where period == current_period and class == "IC" and indicator in {factor_list};
        factor_return = select * from factor_return where abs(value) >= double({ic_threshold});
        select distinct(indicator) as factor from factor_return;
        """)
    else:   # 双重标准
        factor_slice = self.session.run(f"""
        current_period = {current_period};
        factor_return = select * from loadTable("{self.result_database}","{self.summary_table}") where period == current_period and class == "IC" and indicator in {factor_list};
        factor_return = select * from factor_return where abs(value) >= double({ic_threshold});
        factor_return = select * from factor_return where abs(value) >= percentile(abs(value), {ic_percentile}, "linear");
        select distinct(indicator) as factor from factor_return;
        """)
    factor_list = factor_slice["factor"].tolist()

    return get_fixed_data_from_dataframe(
        self=self,
        factor_list=factor_list,
        call_back_interval=call_back_interval,
        current_period=current_period,
        split_test_size=split_test_size,
        random_state=random_state,
        split_train= split_train,  # 将训练集继续二比八分割
        latest_test= latest_test   # 是否最新一期period作为测试集
    )

def get_flexible_data_from_dolphindb(self: ReturnModel_Backtest,
                                     factor_list: List, # 这里传入的是全部因子池
                                     call_back_interval:int = 1,
                                     current_period:int = 0,
                                     split_test_size: float = 0.2,
                                     random_state: int = 42,
                                     split_train: bool = True,  # 将训练集继续二比八分割
                                     latest_test: bool = True,  # 是否最新一期period作为测试集
                                     ic_threshold = None,   # IC绝对值阈值
                                     ic_percentile = None   # IC绝对值百分比
                                     ) -> Dict:
    """获取指定因子List -> 传参到"""
    # Configuration
    if not ic_threshold and not ic_percentile:
        raise ValueError("ic_threshold or ic_percentile must be provided.")
    if ic_percentile and not ic_threshold:    # 说明是是按照IC绝对值的百分比进行筛选的
        factor_slice = self.session.run(f"""
        current_period = {current_period};
        factor_return = select * from loadTable("{self.result_database}","{self.summary_table}") where period == current_period and class == "IC" and indicator in {factor_list};
        factor_return = select * from factor_return where abs(value) >= percentile(abs(value), {ic_percentile}, "linear");
        select distinct(indicator) as factor from factor_return;
        """)
    elif ic_threshold and not ic_percentile:   # 说明是按照IC绝对值的阈值进行筛选的
        factor_slice = self.session.run(f"""
        current_period = {current_period};
        factor_return = select * from loadTable("{self.result_database}","{self.summary_table}") where period == current_period and class == "IC" and indicator in {factor_list};
        factor_return = select * from factor_return where abs(value) >= double({ic_threshold});
        select distinct(indicator) as factor from factor_return;
        """)
    else:   # 双重标准
        factor_slice = self.session.run(f"""
        current_period = {current_period};
        factor_return = select * from loadTable("{self.result_database}","{self.summary_table}") where period == current_period and class == "IC" and indicator in {factor_list};
        factor_return = select * from factor_return where abs(value) >= double({ic_threshold});
        factor_return = select * from factor_return where abs(value) >= percentile(abs(value), {ic_percentile}, "linear");
        select distinct(indicator) as factor from factor_return;
        """)
    factor_list = factor_slice["factor"].tolist()

    return get_fixed_data_from_dolphindb(
        self=self,
        factor_list=factor_list,
        call_back_interval=call_back_interval,
        current_period=current_period,
        split_test_size=split_test_size,
        random_state=random_state,
        split_train=split_train,  # 将训练集继续二比八分割
        latest_test=latest_test   # 是否最新一期period作为测试集
    )

def from_sklearn_to_dataframe(self: ReturnModel_Backtest,
                 model_list: List,  # 所有待评估的模型List
                 model_config: Dict,    # 所有待评估的模型配置
                 init_period:int = 3,   # 初始化所需样本
                 call_back_interval:int =1,  # 回看周期
                 cross_validation:int = 5,   # K折交叉验证
                 split_test_size: float = 0.2,
                 random_state: int = 42,
                 early_stopping:bool=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor:bool=False,   # 是否固定因子池
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=None  # IC绝对值百分比
                 ):
    """
    每次选择所有因子进行训练
    """
    # Model Comprehension
    model_name = [str(i).lower() for i in model_list]
    model_list = [model_sklearn[i] for i in model_name if i in model_sklearn]
    if not model_list:  # 说明没有使用任何sklearn相关的ML模型
        return None

    # Config
    factor_list = self.factor_list
    combine_path = self.combine_path
    template_path = self.template_path
    template_table = self.template_table
    template_df = pd.read_parquet(rf"{template_path}\{template_table}.pqt")
    period_list= list(template_df["period"])
    label = self.label_pred    # target label

    # Model
    model_save_path = self.model_save_path      # 模型保存路径
    model_return_path = self.model_path         # 模型预测收益率保存路径
    init_path(path_dir=model_save_path)
    init_path(path_dir=model_return_path)
    save_path_dict = {}
    for model in model_name:
        init_path(path_dir=rf"{model_save_path}/{model}")
        save_path_dict[model] = rf"{model_save_path}/{model}"

    # First Period
    if fixed_factor:
        data_dict = get_fixed_data_from_dataframe(self, factor_list=factor_list,
                                                  call_back_interval=call_back_interval,
                                                  current_period=init_period-1,
                                                  split_test_size=split_test_size,
                                                  random_state=random_state,
                                                  split_train= False,  # 将训练集继续二比八分割
                                                  latest_test= False,   # 是否最新一期period作为测试集
                                                  ) # 因为是从0开始index的
        x, y = data_dict["x"], data_dict["y"]
        for name, model_method in zip(model_name, model_list):
            if name in early_stop_list and early_stopping:  # 说明可以进行早停
                x_train, x_test, y_train, y_test = data_dict["x_train"], data_dict["x_test"], data_dict["y_train"], data_dict["y_test"]
                best_model = sklearn_model_pick(x_train, y_train,
                                                model_config=model_config[name],
                                                model_name=name,
                                                model_method=model_method,
                                                cv=cross_validation,
                                                eval_set=[(x_test, y_test)],  # 早停验证集
                                                )
            else:
                best_model = sklearn_model_pick(x, y,
                                                model_config=model_config[name],
                                                model_name=name,
                                                model_method=model_method,
                                                cv=cross_validation)
            save_model(best_model,
                       save_path=save_path_dict[name],
                       file_name=str(int(call_back_interval - 1)),
                       target_format="bin")
    else:
        pass    # 动态因子池不需要初始化

    # Iteration Period
    for period in tqdm.tqdm(period_list[init_period:], desc="Model BackTesting..."):
        # 先获取本期数据
        if fixed_factor:    # 动态因子
            data_dict = get_flexible_data_from_dataframe(self, factor_list=factor_list,
                                                         call_back_interval=call_back_interval,
                                                         current_period=period,
                                                         split_test_size=split_test_size,
                                                         random_state=random_state,
                                                         ic_threshold=ic_threshold,
                                                         ic_percentile=ic_percentile,
                                                         split_train=True,
                                                         latest_test=True  # 最新一期作为验证集
                                                         )
        else: # 静态因子
            data_dict = get_fixed_data_from_dataframe(self, factor_list=factor_list,
                                                      call_back_interval=call_back_interval,
                                                      current_period=period,
                                                      split_test_size=split_test_size,
                                                      random_state=random_state,
                                                      split_train=False,
                                                      latest_test=True  # 最新一期作为验证集
                                                      )
        slice_df, x, y = data_dict["data"], data_dict["x"], data_dict["y"]
        x_train, x_test, y_train, y_test = (data_dict["x_train"], data_dict["x_test"],
                                                data_dict["y_train"], data_dict["y_test"])
        x_train_train, x_train_test, y_train_train, y_train_test = (data_dict["x_train_train"], data_dict["x_train_test"],
                                                                    data_dict["y_train_train"], data_dict["y_train_test"])
        res_df = slice_df[["period", "date", "minute", "symbol", label]]
        res_df = res_df[res_df["period"]==period].reset_index(drop = True)  # 仅仅保留最新一期预测

        if not fixed_factor:    # 动态因子
            for name, model_method in zip(model_name, model_list):
                if name in early_stop_list and early_stopping:  # 说明可以进行早停
                    best_model = sklearn_model_pick(x_train_train, y_train_train,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation,
                                                    eval_set=[(x_train_test, y_train_test)],  # 早停验证集
                                                    )
                else:
                    best_model = sklearn_model_pick(x_train, y_train,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation)
                save_model(best_model,
                           save_path=save_path_dict[name],
                           file_name=str(int(period)),
                           target_format="bin")

            # 调用本期模型(利用历史数据训练)预测本期收益率
            for name, model_method in zip(model_name, model_list):
                model = load_model(load_path=save_path_dict[name],
                                   file_name=str(int(period)),
                                   target_format="bin")
                return_pred = model.predict(x_test)
                res_df[f"return_pred_{name}"]=return_pred

            # 保存收益率预测数据
            res_df = res_df[["period", "date", "minute", "symbol", label] + [f"return_pred_{name}" for name in model_name]]
            res_df[res_df["period"] == period].to_parquet(rf"{model_return_path}\{period}.pqt", index=False)

        else:   # 静态因子
            # 调用上一期模型预测本期收益率
            for name, model_method in zip(model_name, model_list):
                current_model_list = sorted([int(str(i).replace(".bin","")) for i in
                                             get_glob_list(path_dir=f"{save_path_dict[name]}\*.bin")])
                current_model_idx = find_closest_lower(period, current_model_list)
                model = load_model(load_path=save_path_dict[name],
                                   file_name=str(int(current_model_idx)),
                                   target_format="bin")
                return_pred = model.predict(x)
                res_df[f"return_pred_{name}"]=return_pred

            # 保存收益率预测数据
            res_df = res_df[["period", "date", "minute", "symbol", label] + [f"return_pred_{name}" for name in model_name]]
            res_df[res_df["period"]==period].to_parquet(rf"{model_return_path}\{period}.pqt",index=False)

            # 再将本期数据添加至模型进行训练
            for name, model_method in zip(model_name, model_list):
                if name in early_stop_list and early_stopping:  # 说明可以进行早停
                    best_model = sklearn_model_pick(x_train, y_train,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation,
                                                    eval_set=[(x_test, y_test)],  # 早停验证集
                                                    )
                else:
                    best_model = sklearn_model_pick(x, y,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation)
                save_model(best_model,
                           save_path=save_path_dict[name],
                           file_name=str(int(period)),
                           target_format="bin")

def from_sklearn_to_dolphindb(self: ReturnModel_Backtest,
                 model_list: List,  # 所有待评估的模型List
                 model_config: Dict,    # 所有待评估的模型配置
                 init_period:int = 3,   # 初始化所需样本
                 call_back_interval:int =1,  # 回看周期
                 cross_validation:int = 5,   # K折交叉验证
                 split_test_size:float = 0.2,   # 测试集比例
                 random_state:int = 42,  # 随机种子
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor: bool = False,  # 是否固定因子池
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=None  # IC绝对值百分比
                 ):
    """
    每次选择所有因子进行训练
    """
    # Model Comprehension
    model_name = [str(i).lower() for i in model_list]
    model_list = [model_sklearn[i] for i in model_name if i in model_sklearn]
    if not model_list:  # 说明没有使用任何sklearn相关的ML模型
        return None

    # Config
    factor_list = self.factor_list
    combine_path = (self.combine_database, self.combine_table)
    template_path = (self.combine_database, self.template_table)
    template_df = self.session.run(rf"""
    select * from loadTable("{template_path[0]}","{template_path[1]}")
    """)
    period_list= list(template_df["period"])
    label = self.label_pred    # target label

    # Model
    model_save_path = self.Model_save_path      # 模型保存路径
    model_return_path = (self.model_database, self.ModelIndividualR_table)    # 模型预测收益率保存路径
    init_path(path_dir=model_save_path)
    save_path_dict = {}
    for model in model_name:
        init_path(path_dir=rf"{model_save_path}/{model}")
        save_path_dict[model] = rf"{model_save_path}/{model}"
    appender = ddb.PartitionedTableAppender(dbPath=model_return_path[0],
                                            tableName=model_return_path[1],
                                            partitionColName="date",
                                            dbConnectionPool=self.pool)  # 写入数据的appender

    # First Period
    if fixed_factor:
        data_dict = get_fixed_data_from_dolphindb(self, factor_list=factor_list,
                                                  call_back_interval=call_back_interval,
                                                  current_period=int(init_period - 1),
                                                  split_test_size=split_test_size,
                                                  random_state=random_state,
                                                  split_train=False,  # 将训练集继续二比八分割
                                                  latest_test=False,  # 是否最新一期period作为测试集
                                                  )  # 因为是从0开始index的
        x, y = data_dict["x"], data_dict["y"]
        for name, model_method in zip(model_name, model_list):
            if name in early_stop_list and early_stopping:  # 说明可以进行早停
                x_train, x_test, y_train, y_test = data_dict["x_train"], data_dict["x_test"], data_dict["y_train"], data_dict["y_test"]
                best_model = sklearn_model_pick(x_train, y_train,
                                                model_config=model_config[name],
                                                model_name=name,
                                                model_method=model_method,
                                                cv=cross_validation,
                                                eval_set=[(x_test, y_test)],  # 早停验证集
                                                )
            else:
                best_model = sklearn_model_pick(x, y,
                                                model_config=model_config[name],
                                                model_name=name,
                                                model_method=model_method,
                                                cv=cross_validation)
            save_model(best_model,
                       save_path=save_path_dict[name],
                       file_name=str(int(call_back_interval - 1)),
                       target_format="bin")
    else:
        pass  # 动态因子池不需要初始化

    # Iteration Period
    for period in tqdm.tqdm(period_list[init_period:], desc="Model BackTesting..."):
        period = int(period)
        # 先获取本期数据
        if not fixed_factor:  # 动态因子
            data_dict = get_flexible_data_from_dolphindb(self, factor_list=factor_list,
                                                         call_back_interval=call_back_interval,
                                                         current_period=period,
                                                         split_test_size=split_test_size,
                                                         random_state=random_state,
                                                         ic_threshold=ic_threshold,
                                                         ic_percentile=ic_percentile,
                                                         split_train=True,
                                                         latest_test=True  # 最新一期作为验证集
                                                         )
        else:  # 静态因子
            data_dict = get_fixed_data_from_dolphindb(self, factor_list=factor_list,
                                                      call_back_interval=call_back_interval,
                                                      current_period=period,
                                                      split_test_size=split_test_size,
                                                      random_state=random_state,
                                                      split_train=False,
                                                      latest_test=True  # 最新一期作为验证集
                                                      )
        slice_df, x, y = data_dict["data"], data_dict["x"], data_dict["y"]
        x_train, x_test, y_train, y_test = (data_dict["x_train"], data_dict["x_test"],
                                            data_dict["y_train"], data_dict["y_test"])
        x_train_train, x_train_test, y_train_train, y_train_test = (
            data_dict["x_train_train"], data_dict["x_train_test"],
            data_dict["y_train_train"], data_dict["y_train_test"])
        res_df = slice_df[["period", "date", "minute", "symbol", label]]
        res_df = res_df[res_df["period"] == period].reset_index(drop=True)  # 仅仅保留最新一期预测

        if not fixed_factor:  # 动态因子
            for name, model_method in zip(model_name, model_list):
                if name in early_stop_list and early_stopping:  # 说明可以进行早停
                    best_model = sklearn_model_pick(x_train_train, y_train_train,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation,
                                                    eval_set=[(x_train_test, y_train_test)],  # 早停验证集
                                                    )
                else:
                    best_model = sklearn_model_pick(x_train, y_train,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation)
                save_model(best_model,
                           save_path=save_path_dict[name],
                           file_name=str(int(period)),
                           target_format="bin")

            # 调用本期模型(利用历史数据训练)预测本期收益率
            for name, model_method in zip(model_name, model_list):
                model = load_model(load_path=save_path_dict[name],
                                   file_name=str(int(period)),
                                   target_format="bin")
                return_pred = model.predict(x_test)
                res_df[f"return_pred_{name}"] = return_pred

            # 保存收益率预测数据
            res_df = res_df[["period", "date", "minute", "symbol", label] + [f"return_pred_{name}" for name in model_name]]
            res_df.insert(0,"Benchmark",self.benchmark)
            appender.append(res_df[res_df["period"] == period].reset_index(drop=True))


        else:  # 静态因子
            # 调用上一期模型预测本期收益率
            for name, model_method in zip(model_name, model_list):
                current_model_list = sorted([int(str(i).replace(".bin", "")) for i in
                                             get_glob_list(path_dir=f"{save_path_dict[name]}\*.bin")])
                current_model_idx = find_closest_lower(period, current_model_list)
                model = load_model(load_path=save_path_dict[name],
                                   file_name=str(int(current_model_idx)),
                                   target_format="bin")
                return_pred = model.predict(x)
                res_df[f"return_pred_{name}"] = return_pred

            # 保存收益率预测数据
            res_df = res_df[["period", "date", "minute", "symbol", label] + [f"return_pred_{name}" for name in model_name]]
            res_df.insert(0,"Benchmark",self.benchmark)
            appender.append(res_df[res_df["period"] == period].reset_index(drop=True))

            # 再将本期数据添加至模型进行训练
            for name, model_method in zip(model_name, model_list):
                if name in early_stop_list and early_stopping:  # 说明可以进行早停
                    best_model = sklearn_model_pick(x_train, y_train,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation,
                                                    eval_set=[(x_test, y_test)],  # 早停验证集
                                                    )
                else:
                    best_model = sklearn_model_pick(x, y,
                                                    model_config=model_config[name],
                                                    model_name=name,
                                                    model_method=model_method,
                                                    cv=cross_validation)
                save_model(best_model,
                           save_path=save_path_dict[name],
                           file_name=str(int(period)),
                           target_format="bin")

