import pandas as pd
import numpy as np
# import riskfolio as rp
import dolphindb as ddb
import tqdm
import warnings
warnings.filterwarnings("ignore")
from src.factor_func.opt_utils import *

def execute_optimize(self):
    # Preparation
    pt=self.session.run(f"""
    // 假设return_pred=avg(MultiFactor-Method,ML/DL-Method)
    pt=select Benchmark,period,symbol,nullFill(real_return,0) as real_return,marketvalue,industry,Composition as group,nullFill(AdaBoost_return_pred,0) as return_pred from loadTable("{self.optimize_database}","{self.optimize_data_table}") where method="OLS";
    pt=select * from pt where not isNull(return_pred) and return_pred<10 and return_pred>-10;
                                
    // 生成收益率Rank
    update pt set rank_false=rank(return_pred,ascending=false,tiesMethod='first') context by period; // 倒序排序+相同值按原来数据的顺序排序
    update pt set rank_true=rank(return_pred,ascending=true,tiesMethod="first") context by period; // 正序排序+相同值按照原来的数据顺序排序
    pt
    """)
    total_period_list=sorted(self.session.run(rf"""exec distinct(period) from loadTable("{self.optimize_database}","{self.optimize_data_table}") where method="OLS" """))
    appender=ddb.PartitionedTableAppender(dbPath=self.optimize_database,
                                          tableName=self.optimize_result_table,
                                          partitionColName="Benchmark",
                                          dbConnectionPool=self.pool)  # 写入数据的appender

    # 假设period为周频,回看周期为252Day(50个period),初始收益率为20Day
    # 同时由于sample_period为4,因而从period=4+5+1=10开始回测
    for i in tqdm.tqdm(range(10,len(total_period_list)),desc="Optimizing..."):
        # 基础信息
        p=total_period_list[i]  # 最新一期的收益率(预测收益率所在的区间)
        start_period,end_period=max(p-50,sorted(total_period_list)[4]),p-1   # 历史收益率period区间
        current_pt=pt.query(f"(period>={start_period})&(period<={p})").reset_index(drop=True)

        # 类型1：等权多空
        long_list = sorted(pt.query(f"(period=={p})&(rank_false<=5)")["symbol"]) # 预期收益率排名前5的资产
        short_list = sorted(pt.query(f"(period=={p})&(rank_true<=5)")["symbol"]) # 预期收益率排名后5的资产
        slice_pt =current_pt[current_pt["symbol"].isin(long_list+short_list)].reset_index(drop=True)
        mv_Dict = dict(zip(slice_pt["symbol"], slice_pt["marketvalue"]))
        L5S5_eq_Dict = {symbol: 1/len(long_list) for symbol in long_list}
        L5S5_eq_Dict.update({
            symbol: -1/len(short_list) for symbol in short_list}
        )
        L5S5_mv_Dict = {symbol: mv_Dict[symbol]/sum(mv_Dict.values()) for symbol in long_list}
        L5S5_mv_Dict.update({
            symbol: -mv_Dict[symbol]/sum(mv_Dict.values()) for symbol in short_list}
        )

        long_list = sorted(pt.query(f"(period=={p})&(rank_false<=3)")["symbol"]) # 预期收益率排名前5的资产
        short_list = sorted(pt.query(f"(period=={p})&(rank_true<=3)")["symbol"]) # 预期收益率排名后5的资产
        slice_pt =current_pt[current_pt["symbol"].isin(long_list+short_list)].reset_index(drop=True)
        mv_Dict = dict(zip(slice_pt["symbol"], slice_pt["marketvalue"]))
        L3S3_eq_Dict = {symbol: 1/len(long_list) for symbol in long_list}
        L3S3_eq_Dict.update({
            symbol: -1/len(short_list) for symbol in short_list}
        )
        L3S3_mv_Dict = {symbol: mv_Dict[symbol]/sum(mv_Dict.values()) for symbol in long_list}
        L3S3_mv_Dict.update({
                        symbol: -mv_Dict[symbol]/sum(mv_Dict.values()) for symbol in short_list}
        )

        long_list = sorted(pt.query(f"(period=={p})&(rank_false<=1)")["symbol"]) # 预期收益率排名前5的资产
        short_list = sorted(pt.query(f"(period=={p})&(rank_true<=1)")["symbol"]) # 预期收益率排名后5的资产
        slice_pt =current_pt[current_pt["symbol"].isin(long_list+short_list)].reset_index(drop=True)
        mv_Dict = dict(zip(slice_pt["symbol"], slice_pt["marketvalue"]))
        L1S1_eq_Dict = {symbol: 1/len(long_list+short_list) for symbol in long_list}
        L1S1_eq_Dict.update({
            symbol: -1/len(long_list+short_list) for symbol in short_list}
        )
        L1S1_mv_Dict = {symbol: mv_Dict[symbol]/sum(mv_Dict.values()) for symbol in long_list}
        L1S1_mv_Dict.update({
                        symbol: -mv_Dict[symbol]/sum(mv_Dict.values()) for symbol in short_list}
        )

        # 类型2：TOP组合优化
        # TOP5
        current_pt=pt.query(f"(period>={start_period})&(period<={p})").reset_index(drop=True)  # Structure: symbol period marketvalue industry return_pred
        symbol_list=sorted(pt.query(f"(period=={p})")["symbol"])  # 最新一期的symbol_list
        current_pt=current_pt[current_pt["symbol"].isin(symbol_list)].reset_index(drop=True)
        return_hist=pd.pivot_table(current_pt,index="period",columns="symbol",values="real_return")
        return_pred=pd.pivot_table(current_pt,index="period",columns="symbol",values="return_pred")   # period symbol1...symbolN
        return_matrix=pd.DataFrame(return_hist.loc[start_period:end_period]).dropna(axis=1)   # 历史收益率矩阵[start_period:end_period]
        symbol_list=return_matrix.columns.tolist()  # 资产代码向量
        expect_return=return_pred.loc[p,symbol_list]  # 预期收益率向量
        cov_matrix=return_matrix.tail(min(50,return_matrix.shape[0])).cov() # 历史协方差矩阵

        """Solve Optimization"""
        # 效用函数+个股约束+预期收益率约束
        Total_opt_Dict=opt_weights(return_matrix=return_matrix,
                                   cov_matrix=cov_matrix,
                                   expect_returns=expect_return,
                                   HCP=False,
                                   objective="Utility",
                                   lamda=0.5,
                                   sht=False,
                                   lowerret=0.01,
                                   w_min=-0.1,
                                   w_max=0.1,
                                   budget=1.0,
                                   long_max=0.75,
                                   short_max=-0.25,
                                )
        Total_opt_Dict=dict(zip(symbol_list,Total_opt_Dict["weight"]))

        # 结果汇总
        total_symbol_list=sorted(set(list(Total_opt_Dict.keys())+
                                     list(L5S5_eq_Dict.keys())+
                                     list(L3S3_eq_Dict.keys())+
                                     list(L1S1_eq_Dict.keys())))
        result_df=pd.DataFrame({"symbol":total_symbol_list,
                                "period":[p]*len(total_symbol_list)})
        result_df["L5S5_eq"] = result_df["symbol"].map(L5S5_eq_Dict)
        result_df["L3S3_eq"] = result_df["symbol"].map(L3S3_eq_Dict)
        result_df["L1S1_eq"] = result_df["symbol"].map(L1S1_eq_Dict)
        result_df["L5S5_mv"] = result_df["symbol"].map(L5S5_mv_Dict)
        result_df["L3S3_mv"] = result_df["symbol"].map(L3S3_mv_Dict)
        result_df["L1S1_mv"] = result_df["symbol"].map(L1S1_mv_Dict)
        result_df["Total_OPT"] = result_df["symbol"].map(Total_opt_Dict)


        # 汇总结果+添加至数据库
        result_df["Benchmark"]=self.benchmark
        result_df=result_df[["Benchmark","period","symbol"]+
                            ["L5S5_eq",
                            "L3S3_eq",
                            "L1S1_eq",
                            "L5S5_mv",
                            "L3S3_mv",
                            "L1S1_mv",
                            "Total_OPT"]
        ]
        appender.append(result_df)  # 添加至数据库


