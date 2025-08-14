import tqdm
import pandas as pd
import numpy as np
import dolphindb as ddb
from src.ReturnModel_mr import ReturnModel_Backtest_mr as ReturnModel_Backtest


# 以下是自定义功能函数
def FactorIC_pred(self: ReturnModel_Backtest):
    self.session.run(rf"""
    benchmark_str="{self.benchmark}";
    factor_pt=select * from loadTable("{self.result_database}","{self.summary_table}") where Benchmark=benchmark_str and class in ["IC","RankIC"] order by Benchmark,returnInterval,period;  // IC & RankIC
    sortBy!(factor_pt,[`Benchmark,`ReturnInterval,`period],[1,1,1]);
    // 加一期
    for (i in seq(1,1)){{
        last_pt=select Benchmark,ReturnInterval,period+1 as period,class,indicator,NULL as value from factor_pt where period=max(period);
        append!(factor_pt,last_pt);
    }};
    undef(`last_pt);
    
     // [自定义,IC预测方法]
    update factor_pt set value_pred=prev(ma(value,3,1)) context by Benchmark,ReturnInterval,class,indicator; //EMA(IC值,5)作为本期IC的预测值

    // 添加预测数据至数据库(单因子IC数据库)
    loadTable("{self.result_database}","{self.factorIC_table}").append!(factor_pt);
    undef(`factor_pt); // 释放内存
    """)

def FactorR_pred(self: ReturnModel_Backtest):
    """[自定义]t-1期因子收益率预测t期因子收益率的逻辑"""
    self.session.run(rf"""
    benchmark_str="{self.benchmark}";
    factor_pt=select * from loadTable("{self.result_database}","{self.summary_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];  // 【更新:对R_OLS/R_Lasso/R_Ridge/R_ElasticNet四者同时预测】
    sortBy!(factor_pt,[`Benchmark,`ReturnInterval,`period],[1,1,1]);
    
    // 加一期
    for (i in seq(1,1)){{
        last_pt=select Benchmark,ReturnInterval,period+1 as period,class,indicator,NULL as value from factor_pt where period=max(period);
        append!(factor_pt,last_pt);
    }};
    undef(`last_pt);
    
    // [自定义,单因子收益率预测]
    update factor_pt set value_pred=prev(ma(value,3,1)) context by Benchmark,ReturnInterval,class,indicator; //EMA移动平均方法预测值

    // 添加预测数据至数据库(单因子收益率数据库)
    loadTable("{self.result_database}","{self.factorR_table}").append!(factor_pt);
    undef(`factor_pt); // 释放内存
    """)

def MultiFactorR_pred(self: ReturnModel_Backtest):
    """[自定义]t-1期多因子收益率预测t期多因子收益率的逻辑"""
    self.session.run(rf"""
    benchmark_str="{self.benchmark}";
    factor_pt=select * from loadTable("{self.result_database}","{self.Multisummary_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
    sortBy!(factor_pt,[`Benchmark,`ReturnInterval,`period],[1,1,1]);
    
    // 加一期
    for (i in seq(1,1)){{
        last_pt=select Benchmark,ReturnInterval,period+1 as period,class,indicator,NULL as value from factor_pt where period=max(period);
        append!(factor_pt,last_pt);
    }};
    undef(`last_pt);
    
    // [自定义,t期因子收益率估计方法]
    update factor_pt set value_pred=prev(ma(value,3,1)) context by Benchmark,ReturnInterval,class,indicator; //EMA移动平均方法预测值
    
    // 添加预测数据至数据库(多因子收益率数据库)
    loadTable("{self.result_database}","{self.MultifactorR_table}").append!(factor_pt);
    undef(`factor_pt); //释放内存
    """)

def Factor_slice(self: ReturnModel_Backtest):
    """
    [自定义]筛选/分组因子逻辑
    """
    self.session.run(rf"""
    """)

def Asset_slice(self: ReturnModel_Backtest):
    """[自定义]根据因子选取资产"""
    self.session.run(rf"""
       benchmark_str="{self.benchmark}";
       factor_list={self.factor_list};
       
        for (interval in {self.returnIntervals}){{
            print("Slicing Asset with interval"+string(interval))
            factor_IC=select value from loadTable("{self.result_database}","{self.Multisummary_table}") where Benchmark=benchmark_str and ReturnInterval==interval and class="IC" pivot by period,indicator; // 因子回测统计结果
            info_pt=select first(marketvalue) as marketvalue,first(industry) as industry from loadTable("{self.combine_database}","{self.combine_table}") group by symbol,period;
            asset_pt=select * from asset_pt left join info_pt on info_pt.symbol=asset_pt.symbol and info_pt.period=asset_pt.period; // 合并添加行业+市值
            undef(`info_pt);

            // IC加权的因子合成
            S=sum(factor_IC[factor_list]);
            for (col in factor_list){{
                factor_IC[col]=factor_IC[col]/S;
            }};

            total_period_list=sort(exec distinct(period) as period from asset_pt,true);
            update asset_pt set 辅助列_industry=string(industry)+string(period);
           // 选股结果
           asset_pt[`group1]=0; // 第一种分组方式
           update asset_pt set group1=NULL;
           asset_pt[`group2]=0; // 第二种分组方式
           update asset_pt set group2=NULL;
           asset_pt[`group3]=0; // 第三种分组方式
           update asset_pt set group3=NULL;
           asset_pt[`group4]=0; // 第四种分组方式
           update asset_pt set group4=NULL;
           asset_pt[`group5]=0; // 第五种分组方式
           update asset_pt set group5=NULL;

           counter=0;
           for (p_int in total_period_list){{
               factorIC_slice=select * from factor_IC where period=p_int;
               asset_pt_slice=select * from asset_pt where period=p_int;
               IC_Dict=transpose(factorIC_slice);
               // 已知每个因子乘一个常数并不影响单因子选股结果,但方便合成因子的计算
               for (factor in factor_list){{
                    asset_pt_slice[factor]=asset_pt_slice[factor]*(IC_Dict[factor][0]);
               }};
               asset_pt_slice["Composition"]=sum(asset_pt_slice[factor_list]);
    
               // 分组方式1：按单因子分组
               asset_pt_slice["合成因子"]=asset_pt_slice["Composition"];
               factor_name="合成因子";
               asset_pt_slice["factor"]=asset_pt_slice[factor_name];
               update asset_pt_slice set group1=1 where factor<=quantile(factor,0.1,"nearest") context by period;
               update asset_pt_slice set group1=2 where quantile(factor,0.1,"nearest")<factor and factor<=quantile(factor,0.2,"nearest") context by period;
               update asset_pt_slice set group1=3 where quantile(factor,0.2,"nearest")<factor and factor<=quantile(factor,0.3,"nearest") context by period;
               update asset_pt_slice set group1=4 where quantile(factor,0.3,"nearest")<factor and factor<=quantile(factor,0.4,"nearest") context by period;
               update asset_pt_slice set group1=5 where quantile(factor,0.4,"nearest")<factor and factor<=quantile(factor,0.5,"nearest") context by period;
               update asset_pt_slice set group1=6 where quantile(factor,0.5,"nearest")<factor and factor<=quantile(factor,0.6,"nearest") context by period;
               update asset_pt_slice set group1=7 where quantile(factor,0.6,"nearest")<factor and factor<=quantile(factor,0.7,"nearest") context by period;
               update asset_pt_slice set group1=8 where quantile(factor,0.7,"nearest")<factor and factor<=quantile(factor,0.8,"nearest") context by period;
               update asset_pt_slice set group1=9 where quantile(factor,0.8,"nearest")<factor and factor<=quantile(factor,0.9,"nearest") context by period;
               update asset_pt_slice set group1=10 where isNull(group1);
    
               asset_pt_slice["合成因子"]=asset_pt_slice["TurnoverRate"];
               factor_name="合成因子";
               asset_pt_slice["factor"]=asset_pt_slice[factor_name];
               update asset_pt_slice set group2=1 where factor<=quantile(factor,0.2,"nearest") context by period;
               update asset_pt_slice set group2=1 where factor<=quantile(factor,0.1,"nearest") context by period;
               update asset_pt_slice set group2=2 where quantile(factor,0.1,"nearest")<factor and factor<=quantile(factor,0.2,"nearest") context by period;
               update asset_pt_slice set group2=3 where quantile(factor,0.2,"nearest")<factor and factor<=quantile(factor,0.3,"nearest") context by period;
               update asset_pt_slice set group2=4 where quantile(factor,0.3,"nearest")<factor and factor<=quantile(factor,0.4,"nearest") context by period;
               update asset_pt_slice set group2=5 where quantile(factor,0.4,"nearest")<factor and factor<=quantile(factor,0.5,"nearest") context by period;
               update asset_pt_slice set group2=6 where quantile(factor,0.5,"nearest")<factor and factor<=quantile(factor,0.6,"nearest") context by period;
               update asset_pt_slice set group2=7 where quantile(factor,0.6,"nearest")<factor and factor<=quantile(factor,0.7,"nearest") context by period;
               update asset_pt_slice set group2=8 where quantile(factor,0.7,"nearest")<factor and factor<=quantile(factor,0.8,"nearest") context by period;
               update asset_pt_slice set group2=9 where quantile(factor,0.8,"nearest")<factor and factor<=quantile(factor,0.9,"nearest") context by period;
               update asset_pt_slice set group2=10 where isNull(group1);
    
               asset_pt_slice["合成因子"]=asset_pt_slice["Momentum5"];
               factor_name="合成因子";
               asset_pt_slice["factor"]=asset_pt_slice[factor_name];
               update asset_pt_slice set group3=1 where factor<=quantile(factor,0.2,"nearest") context by period;
               update asset_pt_slice set group3=1 where factor<=quantile(factor,0.1,"nearest") context by period;
               update asset_pt_slice set group3=2 where quantile(factor,0.1,"nearest")<factor and factor<=quantile(factor,0.2,"nearest") context by period;
               update asset_pt_slice set group3=3 where quantile(factor,0.2,"nearest")<factor and factor<=quantile(factor,0.3,"nearest") context by period;
               update asset_pt_slice set group3=4 where quantile(factor,0.3,"nearest")<factor and factor<=quantile(factor,0.4,"nearest") context by period;
               update asset_pt_slice set group3=5 where quantile(factor,0.4,"nearest")<factor and factor<=quantile(factor,0.5,"nearest") context by period;
               update asset_pt_slice set group3=6 where quantile(factor,0.5,"nearest")<factor and factor<=quantile(factor,0.6,"nearest") context by period;
               update asset_pt_slice set group3=7 where quantile(factor,0.6,"nearest")<factor and factor<=quantile(factor,0.7,"nearest") context by period;
               update asset_pt_slice set group3=8 where quantile(factor,0.7,"nearest")<factor and factor<=quantile(factor,0.8,"nearest") context by period;
               update asset_pt_slice set group3=9 where quantile(factor,0.8,"nearest")<factor and factor<=quantile(factor,0.9,"nearest") context by period;
               update asset_pt_slice set group3=10 where isNull(group1);
    
               if (counter==0){{
                   result=select Benchmark,period,symbol,group1,group2,group3 from asset_pt_slice;
               }};else{{
                   result.append!(select Benchmark,period,symbol,group1,group2,group3 from asset_pt_slice);
               }};
               counter=counter+1;
           }};

            // 存储至数据库中
            result=select Benchmark,interval as `ReturnInterval,period,symbol,group1,group2,group3 from result;
            loadTable("{self.result_database}","{self.asset_slice_table}").append!(result); 
       }}
       undef(`result`factorIC_slice`asset_pt_slice`factor_IC`asset_pt);   // 释放内存
       """)


