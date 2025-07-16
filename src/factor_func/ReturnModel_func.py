import tqdm
import pandas as pd
import numpy as np
import dolphindb as ddb
from src.ReturnModel import ReturnModel_Backtest


# 以下是自定义功能函数
def FactorIC_pred(self: ReturnModel_Backtest):
    self.session.run(rf"""
     benchmark_str="{self.benchmark}";
    factor_pt=select * from loadTable("{self.result_database}","{self.summary_table}") where Benchmark=benchmark_str and class in ["IC","RankIC"];  // IC & RankIC
    
    // 加一期
    for (i in seq(1,1)){{
        last_pt=select Benchmark,period+1 as period,class,indicator,NULL as value from factor_pt where period=max(period);
        append!(factor_pt,last_pt);
    }};
    undef(`last_pt);
    sortBy!(factor_pt,[`period],[1]);
    
     // [自定义,预留训练期个数]
    k={self.callBackPeriod};    // 预留K期样本
    total_period_list=sort(exec distinct(period) as period from factor_pt,true); // 如果采用Walkforward估计所需要的total_period_list;
    // schema:...辅助列_period indicator value, 需要得到value_pred
    for (period_int in total_period_list){{
        // Preparation
        slice_pt=select * from factor_pt where period<=period_int;
        sortBy!(slice_pt,[`Benchmark,`period],[1,1]);  // 这里一定要按时间排序
        update slice_pt set 辅助列=string(Benchmark)+string(class)+string(indicator);
        
        // [自定义,t期IC&RankIC估计方法]
        // update slice_pt set value_pred=prev(value) context by 辅助列; //上一期IC值作为本期IC的预测值
        update slice_pt set value_pred=prev(ma(value,3,1)) context by 辅助列; //EMA(IC值,5)作为本期IC的预测值
        dropColumns!(slice_pt,`辅助列);         // 添加至sample_pt
    }};

    // 添加预测数据至数据库(单因子IC数据库)
    loadTable("{self.result_database}","{self.factorIC_table}").append!(slice_pt);
    undef(`slice_pt); // 释放内存
    
    
    """)

def FactorR_pred(self: ReturnModel_Backtest):
    """[自定义]t-1期因子收益率预测t期因子收益率的逻辑"""
    self.session.run(rf"""
    benchmark_str="{self.benchmark}";
    factor_pt=select * from loadTable("{self.result_database}","{self.summary_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];  // 【更新:对R_OLS/R_Lasso/R_Ridge/R_ElasticNet四者同时预测】
    
    // 加一期
    for (i in seq(1,1)){{
        last_pt=select Benchmark,period+1 as period,class,indicator,NULL as value from factor_pt where period=max(period);
        append!(factor_pt,last_pt);
    }};
    undef(`last_pt);
    sortBy!(factor_pt,[`period],[1]);
    
    // [自定义,预留训练期个数]
    k={self.callBackPeriod};    // 预留K期样本
    total_period_list=sort(exec distinct(period) as period from factor_pt,true); // 如果采用Walkforward估计所需要的total_period_list;
    // schema:...辅助列_period indicator value, 需要得到value_pred
    for (period_int in total_period_list){{
        // Preparation
        slice_pt=select * from factor_pt where period<=period_int;
        sortBy!(slice_pt,[`Benchmark,`period],[1,1]);  // 这里一定要按时间排序
        update slice_pt set 辅助列=string(Benchmark)+string(class)+string(indicator);
        
        // [自定义,t期因子收益率估计方法]
        update slice_pt set value_pred=prev(ma(value,3,1)) context by 辅助列; //EMA移动平均方法预测值
        dropColumns!(slice_pt,`辅助列);         // 添加至sample_pt
    }};

    // 添加预测数据至数据库(单因子收益率数据库)
    loadTable("{self.result_database}","{self.factorR_table}").append!(slice_pt);
    undef(`slice_pt); // 释放内存
    """)

def MultiFactorR_pred(self: ReturnModel_Backtest):
    """[自定义]t-1期多因子收益率预测t期多因子收益率的逻辑"""
    self.session.run(rf"""
    benchmark_str="{self.benchmark}";
    factor_pt=select * from loadTable("{self.result_database}","{self.Multisummary_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
    sortBy!(factor_pt,[`period],[1]);
    
    // 加一期
    for (i in seq(1,1)){{
        last_pt=select Benchmark,period+1 as period,class,indicator,NULL as value from factor_pt where period=max(period);
        append!(factor_pt,last_pt);
    }};
    undef(`last_pt);
    sortBy!(factor_pt,[`period],[1]);
    
    // [自定义,预留训练期个数]
    k=3;    // 预留K期样本
    total_period_list=sort(exec distinct(period) as period from factor_pt,true); // 如果采用Walkforward估计所需要的total_period_list;
    // schema:...辅助列_period indicator value, 需要得到value_pred
    for (period_int in total_period_list){{
        // Preparation
        slice_pt=select * from factor_pt where period<=period_int;

        // 这里一定要按时间排序,同时class不能进入排序(因为样本期有独特标识)
        sortBy!(slice_pt,[`Benchmark,`period],[1,1]);
        update slice_pt set 辅助列=string(Benchmark)+string(class)+string(indicator);

        // [自定义,t期因子收益率估计方法]
        update slice_pt set value_pred=prev(ma(value,3,1)) context by 辅助列; //EMA移动平均方法预测值

        // 添加至sample_pt
        dropColumns!(slice_pt,`辅助列);
    }};
    // 添加预测数据至数据库(多因子收益率数据库)
    loadTable("{self.result_database}","{self.MultifactorR_table}").append!(slice_pt);
    undef(`slice_pt); //释放内存
    """)

def Factor_slice(self: ReturnModel_Backtest):
    """
    [自定义]筛选/分组因子逻辑
    """
    self.session.run(rf"""
    // 信息准备
    benchmark_str="{self.benchmark}";
    factor_pt=select Benchmark,period,class,indicator,value_pred as value from loadTable("{self.result_database}","{self.MultifactorR_table}") where Benchmark=benchmark_str;
    summary_pt=select Benchmark,period,class,indicator,value from loadTable("{self.result_database}","{self.Multisummary_table}") where Benchmark=benchmark_str;
    sortBy!(summary_pt,`period,1);     // 注:保险起见还是按照period sortBy!一下,因为后续context by 操作需要顺序
    
    // 利用R/IC/RankIC对各因子进行评价
    factor_list={self.factor_list};
    R_pt=select * from summary_pt where class="R_OLS";   // 这里可以选R_OLS/R_Lasso/R_Ridge/R_ElasticNet
    // 添加最新一期的period
    unique_period_list=sort(exec distinct(period) from R_pt,true);
    factor_pt=select * from factor_pt where period not in unique_period_list and class="R_OLS";
    R_pt.append!(factor_pt);
    undef(`factor_pt);
    sortBy!(R_pt,`period`indicator,[1,1]);
    IC_pt=select * from summary_pt where class="IC";
    sortBy!(IC_pt,`period`indicator,[1,1]);
    RankIC_pt=select * from summary_pt where class="RankIC";
    sortBy!(RankIC_pt,`period`indicator,[1,1]);
    
    // 对各因子各阶段进行打分(注意,R/IC/RankIC在t时刻知道过去t-1时刻的值)
    // PartI: 因子收益率打分
    R_pt[`score]=0;
    update R_pt set total_indicator=strReplace(string(indicator),"_alpha",""); // 相当于因子收益率+因子alpha
    update R_pt set 判断列1=prev(value)-prev(ma(value,3,1)) context by indicator;
    update R_pt set 判断列2=prev(value) context by indicator;
    update R_pt set score=score+1 where 判断列1>0;
    update R_pt set score=score+1 where 判断列2>0;
    update R_pt set 辅助列=string(total_indicator)+"period"+string(period);
    update R_pt set score=sum(score) context by 辅助列; // 将因子收益率+因子alpha得分汇总起来得到因子得分
    
    // PartII:因子IC值打分
    IC_pt[`score]=0;
    update IC_pt set 判断列1=prev(value) context by indicator;
    update IC_pt set 判断列2=move(value,2) context by indicator;
    update IC_pt set score=score+1 where 判断列1>0;
    update IC_pt set score=score+1 where 判断列1>0 and 判断列2>0;
    
    // PartIII:因子RankIC值打分
    RankIC_pt[`score]=0;
    update RankIC_pt set 判断列1=prev(value) context by indicator;
    update RankIC_pt set 判断列2=move(value,2) context by indicator;
    update RankIC_pt set score=score+1 where 判断列1>0;
    update RankIC_pt set score=score+1 where 判断列1>0 and 判断列2>0;
    
    // PartIV:合成总得分
    R_score=select score from R_pt pivot by period,total_indicator;
    R_score=unpivot(R_score,keyColNames="period",
    valueColNames=factor_list);
    R_score[`class]="R";
    IC_score=select score from IC_pt pivot by period,indicator;
    IC_score=unpivot(IC_score,keyColNames="period",valueColNames=factor_list);
    IC_score["class"]="IC";
    RankIC_score=select score from RankIC_pt pivot by period,indicator;
    RankIC_score=unpivot(RankIC_score,keyColNames="period",valueColNames=factor_list);
    RankIC_score["class"]="RankIC";
    factor_score=R_score.copy();
    factor_score.append!(IC_score).append!(RankIC_score);
    rename!(factor_score,`period`indicator`score`class);
    
    // PartV:根据各阶段中因子总得分进行选择
    update factor_score set 辅助列=string(period)+string(indicator);
    factor_score=select first(period) as period,first(indicator) as indicator,sum(score) as total_score from factor_score group by 辅助列;
    dropColumns!(factor_score,`辅助列);
    update factor_score set 判断列=rank(total_score,false) context by period;
    
    // 是否选择 (-1-0-1,其中-1表示负向因子,1表示正向因子,0表示放弃该因子)
    factor_score[`target]=0;
    update factor_score set target=1 where 判断列<=2;  // 正向因子
    update factor_score set target=-1 where 判断列>2;  // 负向因子
    result_pt=select period,indicator,target from factor_score;
    result_pt=select benchmark_str as Benchmark,period,indicator,target from result_pt; // 最终结果
    
    // 存储至数据库
    loadTable('{self.result_database}','{self.factor_slice_table}').append!(result_pt);
    """)

def Asset_slice(self: ReturnModel_Backtest):
    """[自定义]根据因子选取资产"""
    self.session.run(rf"""
       benchmark_str="{self.benchmark}";
       factor_list={self.factor_list};
       asset_pt=select * from loadTable("{self.result_database}","{self.individualF_table}") where Benchmark=benchmark_str;  // 资产因子值结果
       factor_IC=select value from loadTable("{self.result_database}","{self.Multisummary_table}") where Benchmark=benchmark_str and class="IC" pivot by period,indicator; // 因子回测统计结果
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
       result=select Benchmark,period,symbol,group1,group2,group3 from result;
       loadTable("{self.result_database}","{self.asset_slice_table}").append!(result);
       undef(`result`factorIC_slice`asset_pt_slice`factor_IC`asset_pt);   // 释放内存
       """)


