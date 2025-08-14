import json5, json5
from src.ReturnModel_mr import ReturnModel_Backtest_mr as ReturnModel_Backtest
from src.model_func.template.pipeline import from_sklearn_to_dolphindb, from_sklearn_to_dataframe
from src.utils import *
pd.set_option("display.max_columns",None)

# model config
with open(r"E:\factorbrick\src\config\model_config.json5","rb") as file:
    model_cfg = json5.load(file)

def ModelBackTest(self: ReturnModel_Backtest):
    """
    ML&DL模型预测收益率
    """
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=1,  # 回看周期
                 cross_validation=5,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=70  # IC绝对值百分比(这里选前30%的因子进行训练)
                 )

def ModelBackTest_20250422(self: ReturnModel_Backtest):
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=1,  # 回看周期
                 cross_validation=5,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=90  # IC绝对值百分比(这里选前10%的因子进行训练)
                 )

def ModelBackTest_20250423(self: ReturnModel_Backtest):
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=1,  # 回看周期
                 cross_validation=5,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=70  # IC绝对值百分比(这里选前30%的因子进行训练)
                 )

def ModelBackTest_20250424(self: ReturnModel_Backtest):
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=1,  # 回看周期
                 cross_validation=5,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=60  # IC绝对值百分比(这里选前40%的因子进行训练)
                 )

def ModelBackTest_20250428(self: ReturnModel_Backtest):
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=3,  # 回看周期
                 cross_validation=5,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=65  # IC绝对值百分比(这里选前35%的因子进行训练)
                 )

def ModelBackTest_20250505(self: ReturnModel_Backtest):
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=30,  # 回看周期
                 cross_validation=5,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=65  # IC绝对值百分比(这里选前35%的因子进行训练)
                 )


def ModelBackTest_20250719(self: ReturnModel_Backtest):
    from_sklearn_to_dolphindb(self,
                 model_list=self.Model_list,  # 所有待评估的模型List
                 model_config=model_cfg,  # 所有待评估的模型配置
                 init_period=1,  # 初始化所需样本
                 call_back_interval=10,  # 回看周期
                 cross_validation=10,  # K折交叉验证
                 split_test_size=0.2,   # 测试集比例
                 random_state=42,
                 early_stopping=True,    # 是否对能够使用早停的模型使用早停
                 fixed_factor=False,    # 采用动态因子池方法进行训练
                 ic_threshold=None,  # IC绝对值阈值
                 ic_percentile=70  # IC绝对值百分比(这里选前30%的因子进行训练)
                 )