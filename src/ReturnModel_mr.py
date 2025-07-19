import os,sys
import json5
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
import matplotlib.pyplot as plt
from src.utils import *
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
plt.rcParams['font.sans-serif']=['KaiTi'] # 显示中文
plt.rcParams['axes.unicode_minus']=False # 显示负号

class ReturnModel_Backtest:
    def __init__(self, session, pool, config,
        Symbol_prepareFunc=None,
        Benchmark_prepareFunc=None,
        Factor_prepareFunc=None,
        Combine_prepareFunc=None,
        FactorR_predictFunc=None,
        FactorIC_predictFunc=None,
        MultiFactorR_predictFunc=None,
        ModelR_predictFunc=None,
        Factor_sliceFunc=None,
        Asset_sliceFunc=None,
        Optimize_func=None,
                 ):
        # 基本信息
        self.strategy_name="strategy"
        self.session=session
        self.pool=pool
        self.src_path=config["src_path"]  # 源代码路径
        self.data_path=config["data_path"] # 数据路径

        # 库表类
        self.symbol_database=config["symbol_database"]
        self.symbol_table=config["symbol_table"]
        self.factor_database=config["factor_database"]
        self.factor_table=config["factor_table"]
        self.factor_list=config["factor_list"]
        self.benchmark_database=config["benchmark_database"]
        self.benchmark_table=config["benchmark_table"]
        self.benchmark_list=config["benchmark_list"]
        self.combine_database=config["combine_database"]  # symbol date
        self.combine_table=config["combine_table"]
        self.result_database=config["result_database"]
        self.model_database=config["model_database"]
        self.optimize_database=config["optimize_database"]

        # 函数类
        self.Symbol_prepareFunc=Symbol_prepareFunc
        self.Factor_prepareFunc=Factor_prepareFunc
        self.Benchmark_prepareFunc=Benchmark_prepareFunc
        self.Combine_prepareFunc=Combine_prepareFunc        # 数据合并函数
        self.FactorR_predictFunc=FactorR_predictFunc        # 因子收益率预测函数
        self.FactorIC_predictFunc=FactorIC_predictFunc      # 因子IC&RankIC预测函数
        self.MultiFactorR_predictFunc=MultiFactorR_predictFunc  # 因子收益率预测函数(Multi)
        self.ModelIndividualR_predictFunc=ModelR_predictFunc    # 资产收益率预测函数(ML&DL)
        self.Factor_sliceFunc=Factor_sliceFunc  # 因子分组函数
        self.Asset_sliceFunc=Asset_sliceFunc    # 资产分组函数
        self.OptimizeFunc=Optimize_func         # 组合优化函数

        # 变量类
        self.benchmark = self.benchmark_list[0] # 基准收益率
        self.label_pred = config["label_pred"]  # Y的标签
        self.posPeriod=config["posPeriod"]     # 调仓周期
        self.t = self.posPeriod
        self.callBackPeriod=config["callBackPeriod"] # 回看周期
        self.start_date=config["start_date"]
        self.end_date=config["end_date"]
        self.start_dot_date=pd.Timestamp(self.start_date).strftime('%Y.%m.%d')
        self.end_dot_date=pd.Timestamp(self.end_date).strftime('%Y.%m.%d')
        self.Period_return_Algo=config["period_return_Algo"]        # 资产收益率计算公式-1(每个period内共享该收益率)
        # self.Daily_return_Algo=config["daily_return_Algo"]          # 资产收益率计算公式-2(每个period的每个交易日顺序对应下一个period的交易日)
        self.SingleFactor_estimation=config["SingleFactor_estimation"]    # 是否进行单因子估计
        # self.DailySingleFactor_estimation=config["DailySingleFactor_estimation"]
        self.MultiFactor_estimation=config["MultiFactor_estimation"]      # 是否进行多因子估计
        self.Multi_Intercept=config["Multi_Intercept"]                    # 多因子模型是否添加截距项
        self.Ridge_estimation=config["Ridge_estimation"]
        self.Lasso_estimation=config["Lasso_estimation"]
        self.ElasticNet_estimation=config["ElatsicNet_estimation"]
        self.Ridge_lamdas=[0.001,0.01,0.1,1,10,100,1000] if not config["Ridge_lamdas"] else config["Ridge_lamdas"]
        self.Lasso_lamdas=[0.001,0.01,0.1,1,10,100,1000] if not config["Lasso_lamdas"] else config["Lasso_lamdas"]
        self.ElasticNet_lamdas=[0.001,0.01,0.1,1,10,100,1000] if not config["ElasticNet_lamdas"] else config["ElasticNet_lamdas"]
        self.Group_list=["group"] if not config["Group_list"] else config["Group_list"]                      # 默认只有一个分组维度

        # 中间计算+最终结果类
        self.template_table="template"                  # 包含 start_date end_date period,为回测结果的基石
        self.template_daily_table="template_daily"      # 包含 symbol date 两列
        self.template_minute_table="template_minute"    # 包含 symbol date minute 三列
        self.template_individual_table="template_individual"    # 包含 symbol start_date end_date period,为回测的所有结果
        self.individualF_table="individual" # 统计每个标的区间因子+下个区间的收益率
        # self.individualF_daily_table="individual_daily" # 统计每日的因子+下个区间的收益率

        # 单因子结果
        self.summary_table="summary"        # 向量化回测得到因子统计量
        # self.summary_daily_table="summary_daily"
        self.factorR_table="factor_return"   # t-1期数据WFA预测t期因子收益率
        self.factorIC_table="factor_IC"
        # self.factorR_daily_table="factor_return_daily"
        self.individualR_table="individual_return"  # 合并因子收益率,从而得到个股各阶段的real_return(t-t+1时刻真实收益),expect_return(t+1时刻对其拟合),return_pred(根据t-1期数据WFA预测出的因子收益率计算出的个股收益率)

        # 多因子结果
        self.Multisummary_table="Multisummary"  # 多因子向量化回测得到因子统计量
        # self.Multisummary_daily_table="Multisummary_daily"
        self.MultifactorR_table="Multifactor_return" # t-1期数据WFA预测t期因子收益率
        self.MultiIndividualR_table="MultiIndividual_return"    # 多因子模型预测的个股收益率

        # 模型预测收益率结果
        self.Model_save_path = config["Model_save_path"]
        self.ModelIndividualR_table="ModelIndividual_return"    # 自定义模型(ML/DL)预测的个股收益率
        self.Model_list = config["Model_list"]

        # 资产选择结果
        self.factor_slice_table="factor_slice"      # 因子选择结果表
        self.asset_slice_table="asset_slice"    # 资产选择结果
        self.portfolio_table="portfolio"
        self.signal_table="signal"

        # 组合优化结果
        self.SingleReturn_add = config["SingleReturn_add"]
        self.SingleReturn_method =config["SingleReturn_method"]
        self.MultiReturn_add = config["MultiReturn_add"]
        self.ModelReturn_add = config["ModelReturn_add"]
        self.optimize_data_table="optimize_data"   # 投资组合优化信息(用于计算)
        self.optimize_result_table="optimize_result" # 投资组合优化结果
        self.optstrategy_list = config["optstrategy_list"]

    def init_SymbolDatabase(self,dropDatabase=False):
        """[Optional]第一次运行,初始化行情数据库"""
        if dropDatabase:
            if self.session.existsDatabase(dbUrl=self.symbol_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.symbol_database)
            else:
                pass
        if not self.session.existsTable(dbUrl=self.symbol_database,tableName=self.symbol_table):
            self.session.run(f"""
            db1 = database(, RANGE,2000.01M+(0..30)*12)
            db2 = database(, HASH,[SYMBOL,50])
            db=database("{self.symbol_database}",COMPO,[db1,db2],engine="TSDB")
            schemaTb=table(1:0,`symbol`date`minute`open`close`marketvalue`state`industry,[SYMBOL,DATE,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,SYMBOL]);
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.symbol_table}",partitionColumns=`date`symbol,sortColumns=["symbol","industry","minute","date"],keepDuplicates=LAST)
            """)
        else:
            pass

    def init_BenchmarkDatabase(self):
        """[Optional]第一次运行,初始化基准收益数据库"""
        if not self.session.existsTable(dbUrl=self.benchmark_database,tableName=self.benchmark_table):
            self.session.run(f"""
            db1 = database(, RANGE,2000.01M+(0..30)*12)
            db2 = database(, HASH,[SYMBOL,50])
            db=database("{self.benchmark_database}",COMPO,[db1,db2],engine="TSDB")
            schemaTb=table(1:0,`symbol`date`minute`open`close,[SYMBOL,DATE,INT,DOUBLE,DOUBLE]);
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.benchmark_table}",partitionColumns=`date`symbol,sortColumns=["symbol","date","minute"],keepDuplicates=LAST)
            """)
        else:
            pass

    def init_FactorDatabase_long(self, dropDatabase=False):
        """
        [Optional]第一次运行,初始化因子数据库(宽表形式)
        """
        if dropDatabase:
            if self.session.existsDatabase(dbUrl=self.factor_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.factor_database)

        if not self.session.existsTable(dbUrl=self.factor_database,tableName=self.factor_table):
            self.session.run(f"""
            // 创建因子数据库(宽表形式)
            db=database("{self.factor_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
            schemaTb=table(1:0,{["symbol","date","minute"]+self.factor_list},{["SYMBOL","DATE","INT"]+["DOUBLE"]*len(self.factor_list)});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.factor_table}",partitionColumns="date",sortColumns=["symbol","minute","date"],keepDuplicates=LAST)
            """)

    def init_short_FactorDatabase(self,dropDatabase=False):
        """[Optional]第一次运行,初始化因子数据库(窄表形式)
        【新增】state表示该票当日是否能够交易"""
        if dropDatabase:
            if self.session.existsDatabase(dbUrl=self.factor_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.factor_database)

        if not self.session.existsTable(dbUrl=self.factor_database,tableName=self.factor_table):
            self.session.run(f"""
            // 创建因子数据库(窄表形式)
            create database "{self.factor_database}" 
            partitioned by RANGE(date(datetimeAdd(2000.01M,0..30*12,"M"))), VALUE(`f1`f2),   // 默认两个因子分区,后面再加
            engine='TSDB'
            
            create table "{self.factor_database}"."{self.factor_table}"(
                symbol SYMBOL, 
                date DATE[comment="日期列", compress="delta"], 
                minute INT[comment="分钟列"],
                factor_name SYMBOL, 
                factor_value DOUBLE
            )
            partitioned by date, factor_name,
            sortColumns=[`symbol,`date,`minute], 
            keepDuplicates=LAST, 
            sortKeyMappingFunction=[hashBucket{{, 500}}];
            
            // 添加数据库分区
            for (factor in {self.factor_list}){{
                addValuePartitions(database("{self.factor_database}"),string(factor),1);  // DolphinDB会自动判断是否存在现有数据分区
            }};
            """)
        else:
            pass

    def init_CombineDataBase(self):
        """[Necessary]初始化合并数据库+模板数据库"""
        # Combine Table 默认每次回测前删除上次的因子库（因为因子个数名称&调仓周期可能不一样）
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.combine_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.combine_table)
        columns_name=["symbol","date","minute","open","close","marketvalue","state","industry"]+["period"]+[f"{i}_open" for i in self.benchmark_list]+[f"{i}_close" for i in self.benchmark_list]+self.factor_list+[self.label_pred]
        columns_type=["SYMBOL","DATE","INT","DOUBLE","DOUBLE","DOUBLE","DOUBLE","SYMBOL"]+["DOUBLE"]+["DOUBLE"]*len(self.benchmark_list)+["DOUBLE"]*len(self.benchmark_list)+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.combine_table}",partitionColumns="date",sortColumns=["symbol","date","minute"],keepDuplicates=LAST)
        """)

        # Template Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_table)
        columns_name=["start_date","end_date","period"]
        columns_type=["DATE","DATE","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_table}",partitionColumns="start_date",sortColumns=["start_date"])
        """)

        # Template Individual Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_individual_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_individual_table)
        columns_name=["symbol","start_date","end_date","period"]
        columns_type=["SYMBOL","DATE","DATE","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_individual_table}",partitionColumns="start_date",sortColumns=["symbol","start_date"],keepDuplicates=LAST)
        """)

        # Template Daily Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_daily_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_daily_table)
        columns_name=["symbol","date","period"]
        columns_type=["SYMBOL","DATE","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_daily_table}",partitionColumns="date",sortColumns=["symbol","date"],keepDuplicates=LAST)
        """)

        # Template Minute Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_minute_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_minute_table)
        columns_name=["symbol","date","minute","period"]
        columns_type=["SYMBOL","DATE","INT","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_minute_table}",partitionColumns="date",sortColumns=["symbol","date","minute"],keepDuplicates=LAST)
        """)

    def init_ResultDataBase(self,dropDatabase=False):
        """单因子&多因子结果&Structured Data数据库"""
        if dropDatabase:
            if self.session.existsDatabase(dbUrl=self.result_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.result_database)
            else:
                pass

        # individual_factor(current_factor_value+current_period_return)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualF_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.individualF_table)
        columns_name=["Benchmark","period","symbol"]+self.factor_list+["real_return"]
        columns_type=["SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}",LIST,{self.benchmark_list},engine="OLAP");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualF_table}",partitionColumns=["Benchmark"])
        """)

        # # individual_factor(daily_factor_value+next_period_return)
        # if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualF_daily_table):
        #     self.session.dropTable(dbPath=self.result_database,tableName=self.individualF_daily_table)
        # columns_name=["Benchmark","date","minute","symbol"]+self.factor_list+["real_return"]
        # columns_type=["SYMBOL","DATE","INT","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]
        # self.session.run(f"""
        # db=database("{self.result_database}");
        # schemaTb=table(1:0,{columns_name},{columns_type});
        # t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualF_daily_table}",partitionColumns=["Benchmark"])
        # """)

        if self.SingleFactor_estimation:    # 如果需要进行单因子测试
            # 单因子模型数据库
            # summary_result(model eval+Factor Return+Factor IC+Factor t)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.summary_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.summary_table)
            columns_name=["Benchmark","period","class","indicator","value"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.summary_table}",partitionColumns=["Benchmark"])
            """)

            # # summary_result(daily)(model eval+Factor Return+Factor IC+Factor t)
            # if self.session.existsTable(dbUrl=self.result_database,tableName=self.summary_daily_table):
            #     self.session.dropTable(dbPath=self.result_database,tableName=self.summary_daily_table)
            # columns_name=["Benchmark","date","class","indicator","value"]
            # columns_type=["SYMBOL","DATE","SYMBOL","SYMBOL","DOUBLE"]
            # self.session.run(f"""
            # db=database("{self.result_database}");
            # schemaTb=table(1:0,{columns_name},{columns_type});
            # t=db.createPartitionedTable(table=schemaTb,tableName="{self.summary_daily_table}",partitionColumns=["Benchmark"])
            # """)

            # factorR_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.factorR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.factorR_table)
            columns_name=["Benchmark","period","class","indicator","value","value_pred"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.factorR_table}",partitionColumns=["Benchmark"])
            """)

            # factorIC_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.factorIC_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.factorIC_table)
            columns_name=["Benchmark","period","class","indicator","value","value_pred"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.factorIC_table}",partitionColumns=["Benchmark"])
            """)

            # individualR_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.individualR_table)
            columns_name=["Benchmark","period","symbol","real_return","method"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE","SYMBOL"]
            for col in self.factor_list:
                columns_name=columns_name+[str(col)+"_return_pred"]
                columns_type=columns_type+["DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualR_table}",partitionColumns=["Benchmark"])
            """)
        else:
            pass

        if self.MultiFactor_estimation: # 如果需要进行多因子测试
            # 多因子模型数据库
            # Multisummary_result(model eval+Factor Return+Factor IC+Factor t)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.Multisummary_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.Multisummary_table)
            columns_name=["Benchmark","period","class","indicator","value"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.Multisummary_table}",partitionColumns=["Benchmark"])
            """)

            # MultifactorR_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.MultifactorR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.MultifactorR_table)
            columns_name=["Benchmark","period","class","indicator","value","value_pred"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.MultifactorR_table}",partitionColumns=["Benchmark"])
            """)

            # MultiIndividualR_result(相比IndividualR_result少了一列indicator,因为是所有因子合在一起预测的结果)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.MultiIndividualR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.MultiIndividualR_table)
            L=[]
            for i,j in zip([self.Lasso_estimation,self.Ridge_estimation,self.ElasticNet_estimation],
                            ["Lasso","Ridge","ElasticNet"]):
                if i:
                    L.append(j)
            columns_name=["Benchmark","period","symbol","real_return"]+["return_pred_OLS"]+[f"return_pred_{i}" for i in L]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]+["DOUBLE"]+["DOUBLE"]*len(L)
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.MultiIndividualR_table}",partitionColumns=["Benchmark"])
            """)
        else:
            pass

    def init_ModelDatabase(self):
        """自定义模型(ML/DL)资产收益率预测数据库"""
        # 默认删除原来的数据库
        if self.session.existsTable(dbUrl=self.model_database,tableName=self.ModelIndividualR_table):
            self.session.dropTable(dbPath=self.model_database,tableName=self.ModelIndividualR_table)
        if not self.session.existsTable(dbUrl=self.model_database,tableName=self.ModelIndividualR_table):
            columns_name=["Benchmark","period","date","minute","symbol","real_return"]+[str(i)+"_return_pred" for i in self.Model_list]
            columns_type=["SYMBOL","DOUBLE","DATE","INT","SYMBOL","DOUBLE"]+["DOUBLE"]*len(self.Model_list)
            self.session.run(f"""
            db=database("{self.model_database}",RANGE,2000.01M+(0..30)*30,engine="OLAP");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.ModelIndividualR_table}",partitionColumns=["date"])
            """)

    def init_SliceDatabase(self):
        """因子选择&资产选择数据库"""
        # factor_slice 因子选择(不适用于多因子模型)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.factor_slice_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.factor_slice_table)
        columns_name=["Benchmark","period","indicator","target"]
        columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.factor_slice_table}",partitionColumns=["Benchmark"])
        """)

        # asset_slice 资产标的选择
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.asset_slice_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.asset_slice_table)
        columns_name=["Benchmark","period","symbol"]+self.Group_list
        columns_type=["SYMBOL","DOUBLE","SYMBOL"]+len(self.Group_list)*["DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.asset_slice_table}",partitionColumns=["Benchmark"])
        """)

    def init_OptimizeDatabase(self,dropDatabase=False):
        """【新增】initOptimizeDatabase(COMPO Database)"""
        if dropDatabase:
            if session.existsDatabase(dbUrl=self.optimize_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.optimize_database)
            else:
                pass
        # optimize_data(用于组合优化的数据库)
        if self.session.existsTable(dbUrl=self.optimize_database,tableName=self.optimize_data_table):
            self.session.dropTable(dbPath=self.optimize_database,tableName=self.optimize_data_table)
        columns_name = ["Benchmark","period","symbol","real_return"]
        columns_type = ["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]
        if self.SingleReturn_add:   # 说明需要添加单因子模型的收益率预测结果
            columns_name += [f"{i}_return_pred" for i in self.factor_list]
            columns_type += [f"DOUBLE"]*len(self.factor_list)
        if self.MultiReturn_add:    # 说明需要添加多因子模型的收益率预测结果
            columns_name += ["Multi_return_pred"]
            columns_type += ["DOUBLE"]
        if self.ModelReturn_add:    # 说明需要添加自定义机器学习模型收益率预测结果
            columns_name += [f"{i}_return_pred" for i in self.Model_list]
            columns_type += [f"DOUBLE"]*len(self.Model_list)
        columns_name += ["marketvalue","industry"]+self.Group_list
        columns_type += ["DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.Group_list)
        self.session.run(f"""
        db=database("{self.optimize_database}",LIST,{self.benchmark_list},engine="OLAP");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.optimize_data_table}",partitionColumns=["Benchmark"])
        """)

        # optimize_result(组合优化结果数据库)
        if self.session.existsTable(dbUrl=self.optimize_database,tableName=self.optimize_result_table):
            self.session.dropTable(dbPath=self.optimize_database,tableName=self.optimize_result_table)
        if not session.existsTable(dbUrl=self.optimize_database,tableName=self.optimize_result_table):
            columns_name=["Benchmark","period","symbol"]+self.optstrategy_list  # 优化投资组合
            columns_type=["SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.optstrategy_list)
            self.session.run(f"""
            db=database("{self.optimize_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.optimize_result_table}",partitionColumns=["Benchmark"])
            """)

    def add_SymbolData(self):
        self.Symbol_prepareFunc(self)

    def add_FactorData(self):
        self.Factor_prepareFunc(self)

    def add_BenchmarkData(self):
        self.Benchmark_prepareFunc(self)

    def add_CombineData(self):
        self.Combine_prepareFunc(self)

    def pred_FactorR(self):
        """[单因子]利用t-1期因子收益率预测t期因子收益率函数"""
        self.FactorR_predictFunc(self)

    def pred_MultiFactorR(self):
        """[多因子]利用t-1期因子收益率预测t期因子收益率的函数"""
        self.MultiFactorR_predictFunc(self)

    def pred_FactorIC(self):
        """因子IC&RankIC预测方法"""
        self.FactorIC_predictFunc(self)

    def pred_ModelIndividualR(self,**params):
        """[多因子]利用t-1期因子值训练模型并预测t期资产收益率的函数"""
        self.ModelIndividualR_predictFunc(self,**params)

    def slice_Factor(self):
        """因子筛选函数"""
        self.Factor_sliceFunc(self)

    def slice_Asset(self):
        """资产标的筛选函数"""
        self.Asset_sliceFunc(self)

    def summary_command(self):
        """[单因子回测]individual_return(period_return)&summary_result&summary_daily_result"""
        return rf"""
        // 单因子回测框架
        factor_list={self.factor_list}; // 因子列表
        period_list=sort(distinct(period) from loadTable("{self.combine_database}","{self.combine_table}"), true);
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // 保存Individual结果
        for (benchmark_str in ["{self.benchmark}"]){{
            individual_return=select firstNot(R) as R,{','.join(f"first({item}) as {item}" for item in self.factor_list)} from pt group by symbol,period;
            individual_return=select benchmark_str as Benchmark,period,symbol,{','.join(self.factor_list)},R from individual_return;
            loadTable('{self.result_database}','{self.individualF_table}').append!(individual_return);
            undef(`individual_return);
        }}

        // Summary Map Reduce
        def Summary_mr(p,benchmark_str){{
            pt=select *,{self.Period_return_Algo} as R from loadTable("{self.combine_database}","{self.combine_table}") where period<=p and period>p-{self.callBackPeriod} order by symbol,period,date,minute;
            pt[`benchmark_open]=pt[benchmark_str+"_open"];   // 当前benchmark对应的开盘价
            pt[`benchmark_close]=pt[benchmark_str+"_close"]; // 当前benchmark对应的收盘价
            
            // Data
            reg_df=select * from pt where not isNull(R); 
            func=def(X):countNanInf(X,true); // 【新增】去除了收益率空缺值的样本进行回归
            reg_df[`naninf_count]=byRow(func,reg_df[factor_list]);
            reg_df=select * from reg_df where naninf_count=0;
                    
            if (count(reg_df)>0){{
                // IC&RankIC
                IC=[];
                RankIC=[];
                for (col in factor_list){{
                    append!(IC,corr(reg_df[col],reg_df[`R]));
                    append!(RankIC,corr(rank(reg_df[col]),rank(reg_df[`R])));
                }};
                IC_df=table(factor_list as `indicator,IC as `value);
                IC_df=select `IC as class, indicator, value from IC_df;
                RankIC_df=table(factor_list as `indicator,RankIC as `value);
                RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                        
                //OLS
                counter=0;
                for (col in factor_list){{
                    reg_df[col+"_alpha"]=1.0; // 添加alpha
                    result_OLS=ols(reg_df[`R],reg_df[[col+"_alpha",col]],intercept=false,mode=2);  // OLS回归结果
                                
                    // 统计结果(summary_result,OLS)
                    beta_df=select "R_OLS" as class,factor as indicator,beta as value from result_OLS[`Coefficient];
                    tstat_df=select "tstat_OLS" as class,factor as indicator,tstat as value from result_OLS[`Coefficient];
                    RegDict=dict(result_OLS[`RegressionStat][`item],result_OLS[`RegressionStat][`statistics]);
                    R_square=RegDict[`R2];
                    Adj_square=RegDict[`AdjustedR2];
                    Std_error=RegDict[`StdError];
                    Obs=RegDict['Observations'];
                            
                    // 添加至summary_table的数据行
                    if (counter==0){{
                        summary_result=table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class, [col,col,col,col] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);}};
                    else{{
                        summary_result.append!(table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class, [col,col,col,col] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value))}};
                    summary_result.append!(beta_df);
                    summary_result.append!(tstat_df);
                                
                    // 统计结果(summary_result,Lasso)
                    if (Lasso_estimation==1){{
                        result_Lasso=lassoCV(reg_df,`R,[col+"_alpha",col],alphas={self.Lasso_lamdas},intercept=false); // Lasso回归结果
                        beta_df=table(result_Lasso[`xColNames] as indicator,result_Lasso[`coefficients] as value);
                        beta_df=select "R_Lasso" as class,* from beta_df;
                        summary_result.append!(beta_df);
                    }};
                            
                    // 统计结果(summary_result,Ridge)
                    if (Ridge_estimation==1){{
                        result_Ridge=ridgeCV(reg_df,`R,[col+"_alpha",col],alphas={self.Ridge_lamdas},intercept=false); // Ridge回归结果
                        beta_df=table(result_Ridge[`xColNames] as indicator,result_Ridge[`coefficients] as value);
                        beta_df=select "R_Ridge" as class,* from beta_df;
                        summary_result.append!(beta_df);
                    }};
                            
                    // 统计结果(summary_result,ElasticNet)
                    if (ElasticNet_estimation==1){{
                        result_ElasticNet=elasticNetCV(reg_df,`R,[col+"_alpha",col],alphas={self.ElasticNet_lamdas},intercept=false); // Ridge回归结果
                        beta_df=table(result_ElasticNet[`xColNames] as indicator,result_ElasticNet[`coefficients] as value);
                        beta_df=select "R_ElasticNet" as class,* from beta_df;
                        summary_result.append!(beta_df);
                    }};
                    counter=counter+1;
                }};
                summary_result.append!(IC_df);
                summary_result.append!(RankIC_df);
                summary_result=select p as period,* from summary_result;  // 最后添加日期
            }}; // period循环END
            final_pos_result=select benchmark_str as Benchmark,period,class,indicator,value from summary_result;
            return final_pos_result;
        }}
        if (int({int(self.SingleFactor_estimation)})==1){{  // 说明需要进行单因子测试  
            for (benchmark_str in {self.benchmark_list}){{
                summary_func = Summary_mr{{,benchmark_str}}; // DolphinDB函数部分应用
                total_res = peach(summary_func,period_list);
            }}
            loadTable('{self.result_database}','{self.summary_table}').append!(total_res);
            undef(`total_res);
        }};  //单因子测试部分END
        """

    def Multisummary_command(self):
        """[多因子回测]"""
        return rf"""
        // 多因子回测框架
        factor_list={self.factor_list}; // 多因子列表
        period_list=sort(exec distinct(period) from loadTable("{self.combine_database}","{self.combine_table}"), true);
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        add_Intercept=int({int(self.Multi_Intercept)});      // 多因子模型是否添加截距项
       
        // MultiSummary Map Reduce
        def MultiSummary_mr(p, benchmark_str){{
            pt=select *,{self.Period_return_Algo} as R from loadTable("{self.combine_database}","{self.combine_table}") where period<=p and period>p-{self.callBackPeriod} order by symbol,period,date,minute;
            pt[`benchmark_open]=pt[benchmark_str+"_open"];   // 当前benchmark对应的开盘价
            pt[`benchmark_close]=pt[benchmark_str+"_close"]; // 当前benchmark对应的收盘价
            counter=0;
            print("current_period: "+string(p));
            // Data
            reg_df=select * from pt where not isNull(R); 
            func=def(X):countNanInf(X,true); // 【新增】去除了收益率为空的样本
            reg_df[`naninf_count]=byRow(func,reg_df[factor_list]);
            reg_df=select * from reg_df where naninf_count=0;
            
            if (count(reg_df)>0){{
                // IC&RankIC
                IC=[];
                RankIC=[];
                for (col in factor_list){{
                    append!(IC,corr(reg_df[col],reg_df[`R]));
                    append!(RankIC,corr(rank(reg_df[col]),rank(reg_df[`R])));
                }};
                IC_df=table(factor_list as `indicator,IC as `value);
                IC_df=select `IC as class, indicator, value from IC_df;
                RankIC_df=table(factor_list as `indicator,RankIC as `value);
                RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                    
                //OLS(多因子回测)
                result_OLS=ols(reg_df[`R],reg_df[factor_list],intercept=true,mode=2);  // 这里添加截距项,所以多因子模型factor_list不能有截距项
                        
                // 统计结果(Multisummary_result,OLS)
                beta_df=select "R_OLS" as class,factor as indicator,beta as value from result_OLS[`Coefficient];
                tstat_df=select "tstat_OLS" as class,factor as indicator,tstat as value from result_OLS[`Coefficient];
                RegDict=dict(result_OLS[`RegressionStat][`item],result_OLS[`RegressionStat][`statistics]);
                R_square=RegDict[`R2];
                Adj_square=RegDict[`AdjustedR2];
                Std_error=RegDict[`StdError];
                Obs=RegDict['Observations'];
                        
                // 添加至summary_table的数据行
                summary_result=table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class,[`R_square,`Adj_square,`Std_Error,`Obs]  as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);
                summary_result.append!(beta_df);
                summary_result.append!(tstat_df);
                        
                // 统计结果(Multisummary_result,Lasso)
                if (Lasso_estimation==1){{
                    if (add_Intercept==1){{  // 添加截距项
                        result_Lasso=lassoCV(reg_df,`R,factor_list,alphas={self.Lasso_lamdas},intercept=true);
                        result_Lasso=dict([`intercept].append!(result_Lasso[`xColNames]),[result_Lasso[`intercept]].append!(result_Lasso[`coefficients]));
                    }};else{{ // 不添加截距项
                        result_Lasso=lassoCV(reg_df,`R,factor_list,alphas={self.Lasso_lamdas},intercept=false);
                        result_Lasso=dict(result_Lasso[`xColNames],result_Lasso[`coefficients]);
                    }};
                    beta_df=table(result_Lasso.keys() as indicator,result_Lasso.values() as value);
                    beta_df=select "R_Lasso" as class,* from beta_df;
                    summary_result.append!(beta_df);
                }};
                        
                // 统计结果(Multisummary_result,Ridge)
                if (Ridge_estimation==1){{
                    if (add_Intercept==1){{  // 添加截距项
                        result_Ridge=ridgeCV(reg_df,`R,factor_list,alphas={self.Ridge_lamdas},intercept=true);
                        result_Ridge=dict([`intercept].append!(result_Ridge[`xColNames]),[result_Ridge[`intercept]].append!(result_Ridge[`coefficients]));
                    }};else{{ // 不添加截距项
                        result_Ridge=ridgeCV(reg_df,`R,factor_list,alphas={self.Ridge_lamdas},intercept=false);
                        result_Ridge=dict(result_Ridge[`xColNames],result_Ridge[`coefficients]);
                    }};
                    beta_df=table(result_Ridge.keys() as indicator,result_Ridge.values() as value);
                    beta_df=select "R_Ridge" as class,* from beta_df;
                    summary_result.append!(beta_df);
                }};
                    
                // 统计结果(Multisummary_reuslt,ElasticNet)
                if (ElasticNet_estimation==1){{
                    if (add_Intercept==1){{  // 添加截距项
                        result_ElasticNet=elasticNetCV(reg_df,`R,factor_list,alphas={self.ElasticNet_lamdas},intercept=true);
                        result_ElasticNet=dict([`intercept].append!(result_ElasticNet[`xColNames]),[result_ElasticNet[`intercept]].append!(result_ElasticNet[`coefficients]));
                    }};else{{ // 不添加截距项
                        result_ElasticNet=elasticNetCV(reg_df,`R,factor_list,alphas={self.ElasticNet_lamdas},intercept=false);
                        result_ElasticNet=dict(result_ElasticNet[`xColNames],result_ElasticNet[`coefficients]);
                    }};
                    beta_df=table(result_ElasticNet.keys() as indicator,result_ElasticNet.values() as value);
                    beta_df=select "R_ElasticNet" as class,* from beta_df;
                    summary_result.append!(beta_df);
                }};
                    
                summary_result.append!(IC_df);
                summary_result.append!(RankIC_df);
                summary_result=select p as period,* from summary_result;  // 最后添加日期
                return summary_result
            }} // MultiSummary END
        }}
        for (benchmark_str in {self.benchmark_list}){{
            multisummary_func = MultiSummary_mr{{,benchmark_str}}; // DolphinDB函数部分应用
            total_res = peach(multisummary_func,period_list);
            // 格式调整!!
            total_res=select benchmark_str as Benchmark,period,class,indicator,value from total_res order by Benchmark,period;
            loadTable('{self.result_database}','{self.Multisummary_table}').append!(total_res);
            undef(`total_res); // 释放内存
        }};
        """

    def individual_command(self):
        """合并资产区间real_return+区间return_pred"""
        return rf"""
        benchmark_str="{self.benchmark}";
        factor_list={self.factor_list};
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // 个股因子值+区间收益率数据
        basic_select=["Benchmark","period","symbol","real_return"];
        current_select=factor_list.copy();
        slice_pt=sql(select=sqlCol(basic_select.copy().append!(current_select)),from=loadTable("{self.result_database}","{self.individualF_table}"),where=[<Benchmark=benchmark_str>]).eval();
        undef(`individual_pt); // 释放内存
        for (col in factor_list){{
            rename!(slice_pt,col,string(col)+"_factor_value");
        }};
    
        // 因子收益率数据
        factor_pt=select * from loadTable("{self.result_database}","{self.factorR_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
        rename!(factor_pt,`value,`factor_return);
        rename!(factor_pt,`value_pred,`factor_return_pred);
        pred_factor_OLS=select factor_return_pred from factor_pt where class="R_OLS" pivot by period,indicator; // 预测因子收益率(OLS)
        
        // OLS预期收益率计算
        OLS_pt=select * from slice_pt left join pred_factor_OLS on pred_factor_OLS.period=slice_pt.period;
        OLS_pt[`method]="OLS"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
        undef(`pred_factor_OLS); // 内存释放
        for (col in factor_list){{
            OLS_pt[string(col)+"_return_pred"]=OLS_pt[col+"_alpha"]+OLS_pt[col+"_factor_value"]*OLS_pt[col]; // α+因子值*β
            dropColumns!(OLS_pt,string(col)+"_factor_value"); // 删了因子值
            dropColumns!(OLS_pt,string(col)+"_alpha");    // 删了alpha
            dropColumns!(OLS_pt,string(col)); // 删了β
        }};
    
        // Lasso预期收益率计算
        if (Lasso_estimation==1){{
            pred_factor_Lasso=select factor_return_pred from factor_pt where class="R_Lasso" pivot by period,indicator; // 预测因子收益率(Lasso)
            Lasso_pt=select * from slice_pt left join pred_factor_Lasso on pred_factor_Lasso.period=slice_pt.period;
            Lasso_pt[`method]="Lasso"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
            undef(`pred_factor_Lasso); // 内存释放
            for (col in factor_list){{
                Lasso_pt[string(col)+"_return_pred"]=Lasso_pt[col+"_alpha"]+Lasso_pt[col+"_factor_value"]*Lasso_pt[col]; // α+因子值*β
                dropColumns!(Lasso_pt,string(col)+"_factor_value"); // 删了因子值
                dropColumns!(Lasso_pt,string(col)+"_alpha");    // 删了alpha
                dropColumns!(Lasso_pt,string(col)); // 删了β
            }};
            OLS_pt.append!(Lasso_pt);  // 合并数据
            undef(`Lasso_pt); // 内存释放
        }};
        
        // Ridge预期收益率计算
        if (Ridge_estimation==1){{
            pred_factor_Ridge=select factor_return_pred from factor_pt where class="R_Ridge" pivot by period,indicator; // 预测因子收益率(Ridge)
            Ridge_pt=select * from slice_pt left join pred_factor_Ridge on pred_factor_Ridge.period=slice_pt.period;
            Ridge_pt[`method]="Ridge"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
            undef(`pred_factor_Ridge); // 内存释放
            for (col in factor_list){{
                Ridge_pt[string(col)+"_return_pred"]=Ridge_pt[col+"_alpha"]+Ridge_pt[col+"_factor_value"]*Ridge_pt[col]; // α+因子值*β
                dropColumns!(Ridge_pt,string(col)+"_factor_value"); // 删了因子值
                dropColumns!(Ridge_pt,string(col)+"_alpha");    // 删了alpha
                dropColumns!(Ridge_pt,string(col)); // 删了β
            }};
            OLS_pt.append!(Ridge_pt);  // 合并数据
            undef(`Ridge_pt); // 内存释放
        }};
        
        // ElasticNet预期收益率计算
        if (ElasticNet_estimation==1){{
            pred_factor_ElasticNet=select factor_return_pred from factor_pt where class="R_ElasticNet" pivot by period,indicator; // 预测因子收益率(ElasticNet)
            ElasticNet_pt=select * from slice_pt left join pred_factor_ElasticNet on pred_factor_ElasticNet.period=slice_pt.period;
            ElasticNet_pt[`method]="ElasticNet"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
            undef(`pred_factor_ElasticNet); // 内存释放
            for (col in factor_list){{
                ElasticNet_pt[string(col)+"_return_pred"]=ElasticNet_pt[col+"_alpha"]+ElasticNet_pt[col+"_factor_value"]*ElasticNet_pt[col]; // α+因子值*β
                dropColumns!(ElasticNet_pt,string(col)+"_factor_value"); // 删了因子值
                dropColumns!(ElasticNet_pt,string(col)+"_alpha");    // 删了alpha
                dropColumns!(ElasticNet_pt,string(col)); // 删了β
            }};
            OLS_pt.append!(ElasticNet_pt);  // 合并数据
            undef(`ElasticNet_pt); // 内存释放
        }};
        
        // 添加至数据库
        loadTable("{self.result_database}","{self.individualR_table}").append!(OLS_pt);
        undef(`OLS_pt`slice_pt`factor_pt); // 内存释放
        """

    def MultiIndividual_command(self):
        """[多因子]根据FactorR计算资产的预测收益率"""
        L=[]
        for i,j in zip([self.Lasso_estimation,self.Ridge_estimation,self.ElasticNet_estimation],["Lasso","Ridge","ElasticNet"]):
            if i:
                L.append(j)
        if len(L)==0:   # 说明只估计OLS,需要把最后的,删掉
            MultiString="""
            final_result=select firstNot(Benchmark) as Benchmark,firstNot(real_return) as real_return,factor_value**factor_return_pred_OLS as return_pred_OLS from individual_pt group by period,symbol;
            undef(`individual_pt);
            final_result=select Benchmark,period,symbol,real_return,return_pred_OLS from final_result;
            """
        else:
            MultiString=f"""
            final_result=select firstNot(Benchmark) as Benchmark,firstNot(real_return) as real_return,factor_value**factor_return_pred_OLS as return_pred_OLS,{','.join([f"factor_value**factor_return_pred_{k} as return_pred_{k}" for k in L])} from individual_pt group by period,symbol;
            undef(`individual_pt);
            final_result=select Benchmark,period,symbol,real_return,return_pred_OLS,{','.join([f"return_pred_{k}" for k in L])} from final_result;
            """
        return rf"""
        benchmark_str="{self.benchmark}"
        add_Intercept=int({int(self.Multi_Intercept)}); // 多因子模型是否添加截距项
        Intercept=`intercept;   // DolphinDB SQL ols命令默认的截距项factor name
        factor_list={self.factor_list};
        total_factor_list=factor_list.copy();
        total_factor_list.append!(Intercept); // 添加了截距项的factor_list
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // 个股因子值+区间收益率数据
        individual_pt=select * from loadTable("{self.result_database}","{self.individualF_table}") where Benchmark=benchmark_str;
        
        // 添加截距项(DOUBLE format)
        if (add_Intercept==1){{
            individual_pt[Intercept]=1.0;
            individual_pt=unpivot(individual_pt,keyColNames=`Benchmark`period`symbol`real_return,valueColNames=total_factor_list);
        }}else{{
            individual_pt=unpivot(individual_pt,keyColNames=`Benchmark`period`symbol`real_return,valueColNames=factor_list);
        }};
        rename!(individual_pt,`valueType,`indicator);
        rename!(individual_pt,`value,`factor_value);
        
        // 因子收益率数据
        factor_pt=select * from loadTable("{self.result_database}","{self.MultifactorR_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
        rename!(factor_pt,`value,`factor_return);
        rename!(factor_pt,`value_pred,`factor_return_pred);
        // dropColumns!(factor_pt,`start_date`end_date);
        
        OLS_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_OLS";
        rename!(OLS_pt,`factor_return_pred,`factor_return_pred_OLS); // 预测因子收益率(OLS)
        
        if (Lasso_estimation==1){{
            Lasso_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_Lasso";
            rename!(Lasso_pt,`factor_return_pred,`factor_return_pred_Lasso); // 预测因子收益率(Lasso)
        }};
        
        if (Ridge_estimation==1){{
            Ridge_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_Ridge";
            rename!(Ridge_pt,`factor_return_pred,`factor_return_pred_Ridge); // 预测因子收益率(Ridge)
        }};
        
        if (ElasticNet_estimation==1){{
            ElasticNet_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_ElasticNet";
            rename!(ElasticNet_pt,`factor_return_pred,`factor_return_pred_ElasticNet); // 预测因子收益率(ElasticNet)
        }};
        undef(`factor_pt); // 释放内存
        
        // Combine因子收益率(OLS/Lasso/Ridge/ElasticNet)
        individual_pt=select * from individual_pt left join OLS_pt on individual_pt.Benchmark=OLS_pt.Benchmark and individual_pt.period=OLS_pt.period and OLS_pt.indicator=individual_pt.indicator;
        undef(`OLS_pt);
        
        if (Lasso_estimation==1){{
            individual_pt=select * from individual_pt left join Lasso_pt on individual_pt.Benchmark=Lasso_pt.Benchmark and individual_pt.period=Lasso_pt.period and Lasso_pt.indicator=individual_pt.indicator;
            undef(`Lasso_pt)
        }};
        
        if (Ridge_estimation==1){{
            individual_pt=select * from individual_pt left join Ridge_pt on individual_pt.Benchmark=Ridge_pt.Benchmark and individual_pt.period=Ridge_pt.period and Ridge_pt.indicator=individual_pt.indicator;
            undef(`Ridge_pt);
        }};
        
        if (ElasticNet_estimation==1){{
            individual_pt=select * from individual_pt left join ElasticNet_pt on individual_pt.Benchmark=ElasticNet_pt.Benchmark and individual_pt.period=ElasticNet_pt.period and ElasticNet_pt.indicator=individual_pt.indicator;
            undef(`ElasticNet_pt); 
        }};
        
        // 计算预测收益率(OLS)
        {MultiString}
        loadTable("{self.result_database}","{self.MultiIndividualR_table}").append!(final_result);
        undef(`final_result);
        """

    def OptimizeData_Func(self):
        """投资组合优化框架
        [新增]: 由于添加了SingleFactor_estimation &MultiFactor_estimation两个参数,导致只能合并一个表
        当前是SingleFactor Estimation & MultiFactor Estimation两者必选其一的 Model Estimation 可选
        [新增]: 运行逻辑优化,由于之前的函数都是for loop中取数据,因而在社区版DolphinDB 8G内存下很容易Out of Memory, 所以这里改成pandas操作
        """
        L=[]
        for i,j in zip([self.Lasso_estimation,self.Ridge_estimation,self.ElasticNet_estimation],["Lasso","Ridge","ElasticNet"]):
            if i:
                L.append(j)
        appender = ddb.PartitionedTableAppender(dbPath=self.optimize_database,
                                                tableName=self.optimize_data_table,
                                                partitionColName="Benchmark",
                                                dbConnectionPool=self.pool)  # 写入数据的appender
        info_pt = self.session.run(f"""
            info_pt=select firstNot(marketvalue) as marketvalue,first(industry) as industry from loadTable("{self.combine_database}","{self.combine_table}") group by symbol,date;
            template_pt=select symbol,start_date as date,period from loadTable("{self.combine_database}","{self.template_individual_table}");
            info_pt = select * from info_pt left join template_pt on template_pt.symbol=info_pt.symbol and template_pt.date = info_pt.date;
            info_pt = select symbol,period,marketvalue,industry from info_pt;
            info_pt
        """)    # 资产信息情况
        asset_pt = self.session.run(f"""
            select period,symbol,{','.join(self.Group_list)} from loadTable("{self.result_database}","{self.asset_slice_table}") where Benchmark="{self.benchmark}"
        """)    # 资产分组情况

        total_df = pd.DataFrame()
        if self.SingleFactor_estimation and self.SingleReturn_add:    # 说明估计单因子收益率,有对应的结果存储在表中
            total_period_list = sorted(set(self.session.run(f"""select distinct(period) as period from loadTable("{self.result_database}","{self.individualR_table}")""")["period"]))
            total_df = pd.concat([self.session.run(f"""
                pt = select * from loadTable("{self.result_database}","{self.individualR_table}") where Benchmark="{self.benchmark}" and period=int({period}) and method="{self.SingleReturn_method}";  // OLS/Ridge/Lasso/ElasticNet
                dropColumns!(pt,`Benchmark`method);
                pt
            """) for period in tqdm.tqdm(total_period_list,desc="Getting SingleFactor ReturnPred")],axis=0,ignore_index=True)

        if self.MultiFactor_estimation and self.MultiReturn_add:   # 说明估计多因子收益率,有对应的结果存储在表中
            total_period_list = sorted(set(self.session.run(f"""select distinct(period) as period from loadTable("{self.result_database}","{self.MultiIndividualR_table}")""")["period"]))
            multi_return = pd.concat([self.session.run(f"""
                pt = select * from loadTable("{self.result_database}","{self.MultiIndividualR_table}") where Benchmark="{self.benchmark}" and period=int({period});
                dropColumns!(pt,`Benchmark);
                pt
            """) for period in tqdm.tqdm(total_period_list,desc="Getting MultiFactor ReturnPred")],axis=0,ignore_index=True)
            if not total_df.empty:
                del multi_return["real_return"]
            total_df = pd.merge(total_df, multi_return, on=["symbol","period"], how="left").reset_index(drop=True) if not total_df.empty else multi_return
            del multi_return    # 释放内存

        if self.ModelIndividualR_predictFunc is not None and self.ModelReturn_add:    # 说明估计模型收益率,有对应的结果存储在表中
            model_return = self.session.run(f"""
                Model_pt=select firstNot(real_return) as real_return, {",".join([f"firstNot({model}_return_pred) as {model}_return_pred" for model in self.Model_list])} from loadTable("{self.model_database}","{self.ModelIndividualR_table}") where Benchmark="{self.benchmark}" group by symbol,period,date;
                dropColumns!(Model_pt,`date);
                Model_pt
            """)
            if not total_df.empty:
                del model_return["real_return"]
            total_df = pd.merge(total_df, model_return, on=["symbol","period"], how="left").reset_index(drop=True) if not total_df.empty else model_return
            del model_return   # 释放内存

        total_df = pd.merge(total_df,info_pt, on=["symbol","period"], how="left").reset_index(drop=True)
        total_df = pd.merge(total_df,asset_pt,on=["symbol","period"], how="left").reset_index(drop=True)
        total_df.insert(0,"Benchmark",self.benchmark)   # 插入一列Benchmark

        # 批量添加数据
        chunk_size = 100000  # 每个 chunk 的大小
        for start in tqdm.tqdm(range(0, len(total_df), chunk_size), desc="Adding Optimize Data"):
            end = start + chunk_size
            chunk = total_df[start:end]
            appender.append(chunk)


    def BackTest(self):
        # 因子模型部分
        self.init_ResultDataBase(dropDatabase=True)
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating SingleFactor result"):
            self.benchmark=benchmark
            self.session.run(self.summary_command()) # Step1.向量化生成单因子回测的IC/RankIC/等可向量化运算指标(即使采用WalkForward框架估计也能得到相同的内容的部分)
            # summary_command计算数据的功能,因而必须执行,单因子测试的部分在执行命令时判断,MultiSummary_command则没有这个功能
            if self.SingleFactor_estimation:
                 self.pred_FactorIC()
                 self.pred_FactorR()                      # Step1.5 [自定义] 利用t-1期WalkForward分别对t期的因子收益率+alpha进行预测，结合t期已知的因子值，给出个股预测的收益率
                 self.session.run(self.individual_command()) # Step2. 根据FactorR(单因子)代入模型得到个股预期收益率expect_return,返回symbol date period start_ts end_ts expect_return real_return return_pred
        if self.MultiFactor_estimation:     # 说明需要估计多因子收益率
            for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating MultiFactor result"):
                self.benchmark=benchmark
                self.session.run(self.Multisummary_command()) # Step1.向量化生成单因子回测的IC/RankIC/等可向量化运算指标(即使采用WalkForward框架估计也能得到相同的内容的部分)
                self.pred_FactorIC()
                self.pred_MultiFactorR()                      # Step1.5 [自定义] 利用t-1期WalkForward分别对t期的因子收益率+alpha进行预测，结合t期已知的因子值，给出个股预测的收益率
                self.session.run(self.MultiIndividual_command()) # Step2.根据MultiFactorR代入模型得到个股预期收益率expect_return

    def ModelTest(self):
        self.init_ModelDatabase()
        # ML/DL模型部分
        if self.ModelIndividualR_predictFunc:   # 说明传入了自定义函数进行ModelPredict
            # Init ModelTable()
            for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating ModelReturn result"):
                self.benchmark=benchmark
                self.pred_ModelIndividualR()    # Step2. [自定义] 应用其他ML/DL模型预测资产预期收益率

    def Slice(self):
        # 因子&资产选择部分
        self.init_SliceDatabase()
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Slicing Factor&Asset"):
            self.benchmark=benchmark
            if self.Factor_sliceFunc:
                self.Factor_sliceFunc(self)
            if self.Asset_sliceFunc:
                self.Asset_sliceFunc(self) # Step3. [自定义]根据因子IC/RankIC/其他指标选择因子,再根据所选因子对个股构建等权组合

    def Optimize(self):
        # 组合优化部分
        self.init_OptimizeDatabase(dropDatabase=True)    # init_optimize_database
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Optimizing result"):
            self.benchmark=benchmark
            self.OptimizeData_Func() # 准备投资组合优化的Data
            self.OptimizeFunc(self)         # Step4. [自定义]根据个股预期收益率+其他约束条件+(可选:风险模型)构建优化器优化该组合


if __name__=="__main__":
    from src.factor_func.Data_func import ReturnModel_Data as R
    # from src.factor_func.Data_func_mr import ReturnModel_Data as R
    from src.factor_func.ReturnModel_func import FactorIC_pred,FactorR_pred,MultiFactorR_pred,Factor_slice,Asset_slice
    from src.factor_func.Optimize_func_riskfolio import execute_optimize
    from src.model_func.Model import *
    session=ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")

    with open(r".\config\returnmodel_config.json5", mode="r", encoding="UTF-8") as file:
        cfg = json5.load(file)
    F=ReturnModel_Backtest(
        session=session,pool=pool,config=cfg,
        Symbol_prepareFunc=R.prepare_symbol_data,
        Benchmark_prepareFunc=R.prepare_benchmark_data,
        Factor_prepareFunc=R.prepare_factor_data_long,
        Combine_prepareFunc=R.prepare_combine_data,
        FactorR_predictFunc=FactorR_pred,
        FactorIC_predictFunc=FactorIC_pred,
        MultiFactorR_predictFunc=MultiFactorR_pred,
        ModelR_predictFunc=ModelBackTest_20250719,
        Factor_sliceFunc=Factor_slice,
        Asset_sliceFunc=Asset_slice,
        Optimize_func=execute_optimize,
    )
    # F.init_SymbolDatabase(dropDatabase=True)
    # F.add_SymbolData()
    # F.init_BenchmarkDatabase()
    # F.add_BenchmarkData()
    # F.init_FactorDatabase_long(dropDatabase=True)   # 如果是long_factor_database: 因子池变动的时候都要重新跑
    # F.add_FactorData()

    # # 如果原始数据没有变化，那么不用运行init_CombineDatabase()与add_CombineData()
    # F.init_CombineDataBase()
    # F.add_CombineData()
    # F.BackTest()
    F.ModelTest()
    # F.Slice()
    # F.Optimize()