import pandas as pd
from src.BackTest import Backtest
from src.strategy_func.template import future_strategy
from src.utils import init_path
from typing import Dict

def get_future_signal_to_dolphindb(session, BackTest_config: Dict, ReturnModel_config: Dict):
    """
    CTA策略信号值生成(to:symbol date buy_signal sell_signal)
    """
    # Step0. Configuration
    start_date, end_date = BackTest_config["start_date"], BackTest_config["end_date"]
    start_dot_date, end_dot_date = pd.Timestamp(start_date).strftime('%Y.%m.%d'), pd.Timestamp(end_date).strftime('%Y.%m.%d')

    # Step1.创建信号数据库
    save_database, save_table = BackTest_config["future_signal_database"], BackTest_config["future_signal_table"]
    if session.existsTable(save_database,save_table):
        session.dropTable(dbPath=save_database,tableName=save_table)
    if not session.existsTable(save_database,save_table):
        session.run(f"""
         db=database("{save_database}",RANGE, 2000.01M+(0..30)*12, engine="TSDB")
                    schemaTb=table(1:0,`date`minute`timestamp`contract`pre_settle`open`high`low`close`settle`start_date`end_date`multi`margin`open_long`open_short`close_long`close_short,
                    [DATE,INT,DATETIME,SYMBOL,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DATE,DATE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE])
                    t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns="date",sortColumns=["date","contract","minute","timestamp"],keepDuplicates=LAST)
        """)

    # Step2.生成信号并存储到数据库中`
    res_signal = session.run(rf"""
    // config
    start_ts, end_ts = {start_dot_date}, {end_dot_date};
    estimation_method = "Ridge";
    init_period = 2; // 因子模型初始化period周期长度, 与returnModel保持一致
     
    // 行情数据(filter主连合约)
    k_data = select * from loadTable("dfs://future_cn/combination","base") where date between start_ts and end_ts and isMainContract=1;
    
    // 【方式1】预期收益率数据(From 单因子)
    template_df = select start_date as date,period from loadTable("dfs://asset_cn/ReturnModel","template");
    r_data = select * from loadTable("dfs://asset_cn/ReturnModel_result","individual_return") where method = estimation_method and period > init_period;
    r_data = select * from r_data left join template_df on r_data.period=template_df.period order by date; // mapping date
    undef(`template_df);
    
    // 选择CumsumIC表现最好的因子
    factor_name= "LnCumSlopeDiff_0_reg10_mean_return_pred";
    r_data = sql(sqlCol(["date","symbol",factor_name]),r_data).eval();
    rename!(r_data, factor_name,"return_pred");
    r_data = select "IH" as product, mean(return_pred) as return_pred from r_data group by date order by date;// 截面均值
    
    // // 【方式2】预期收益率数据(From ML)
    // r_data = select * from loadTable("dfs://asset_cn/ReturnModel_MLModel","ModelIndividual_return") where date between start_ts and end_ts;
    // r_data = select firstNot(adaboost_return_pred) as return_pred from r_data where Benchmark = "IH" group by date, symbol;
    // r_data = select "IH" as product, mean(return_pred) as return_pred from r_data group by date order by date;// 截面均值
    
    // 生成日频信号
    r_data[`daily_long_signal] = 0.0;
    r_data[`daily_short_signal] = 0.0;
    update r_data set daily_long_signal = 1.0 where return_pred > 0;
    update r_data set daily_short_signal = 1.0 where return_pred < 0;
    daily_signal = select date,daily_long_signal,daily_short_signal from r_data;
    undef(`r_data);
    k_data = select * from k_data left join daily_signal on k_data.date = daily_signal.date;
    update k_data set daily_long_signal = nullFill(daily_long_signal,0);
    update k_data set daily_short_signal = nullFill(daily_short_signal,0);
    undef(`daily_signal);
    
    // 生成分钟频进场信号(执行顺序: close_long -> close_short -> open_long -> open_short)
    // 信号生成
    k_data[`open_long_signal]=0.0;      // 期货开多信号
    k_data[`open_short_signal]=0.0;     // 期货开空信号
    k_data[`close_long_signal]=0.0;     // 期货平多信号
    k_data[`close_short_signal]=0.0;    // 期货平空信号
    k_data[`long_signal] = 0.0;
    k_data[`short_signal] = 0.0; 
    update k_data set long_signal = 1.0 where prev(ma(close,5,1)>pre_settle) and 930<minute<=945 and daily_long_signal = 1.0 context by contract+string(date); // 早盘后第15分钟下单，条件: 上一分钟: 5分钟EMA均线>pre_settle
    update k_data set short_signal = 1.0 where prev(ma(close,5,1)<pre_settle) and 930<minute<=945 and daily_short_signal = 1.0 context by contract+string(date); // 早盘后前15分钟下单, 条件: 上一分钟: 5分钟EMA均线<pre_settle
    update k_data set long_signal = 0.0 where prev(long_signal) = 1.0 and long_signal = 1.0 and prev(date)=date context by date; // 取第一个信号下多单
    update k_data set short_signal = 0.0 where prev(short_signal) = 1.0 and short_signal = 1.0 and prev(date)=date context by date; // 取第一个信号下空单
    
    // 转化为仓位操作(这里比较简单, 把判断任务交给了柜台来完成)
    update k_data set open_long_signal = long_signal;
    update k_data set open_short_signal = short_signal;
    
    // 保存信号至数据库中
    pt = select date,minute,timestamp,contract,pre_settle,open,high,low,close,settle,start_date,end_date,multi,margin,open_long_signal,open_short_signal,close_long_signal,close_short_signal from k_data;
    undef(`k_data);
    loadTable("{save_database}","{save_table}").append!(pt);
    undef(`pt);
    """)
    return res_signal

def get_future_signal_to_dataframe(session, BackTest_config: Dict, ReturnModel_config: Dict):
    """
    CTA策略信号值生成(to:symbol date open_long_signal open_short_signal close_long_signal close_short_signal)
    """
    # Step0. Configuration
    start_date, end_date = BackTest_config["start_date"], BackTest_config["end_date"]
    start_dot_date, end_dot_date = pd.Timestamp(start_date).strftime('%Y.%m.%d'), pd.Timestamp(end_date).strftime(
        '%Y.%m.%d')

    # Step1.创建信号数据库
    save_database, save_table = BackTest_config["future_signal_database"], BackTest_config["future_signal_table"]

    # Step2.生成信号并存储到数据库中`
    res_signal = session.run(rf"""
       // config
       start_ts, end_ts = {start_dot_date}, {end_dot_date};
       estimation_method = "Ridge";
       init_period = 2; // 因子模型初始化period周期长度, 与returnModel保持一致

       // 行情数据(filter主连合约)
       k_data = select * from loadTable("dfs://future_cn/combination","base") where date between start_ts and end_ts and isMainContract=1;

       // 【方式1】预期收益率数据(From 单因子)
       template_df = select start_date as date,period from loadTable("dfs://asset_cn/ReturnModel","template");
       r_data = select * from loadTable("dfs://asset_cn/ReturnModel_result","individual_return") where method = estimation_method and period > init_period;
       r_data = select * from r_data left join template_df on r_data.period=template_df.period order by date; // mapping date
       undef(`template_df);

       // 选择CumsumIC表现最好的因子
       // factor_name= "ArgMinVolSpreadRatio_0_5_mean_return_pred";
       // r_data = sql(sqlCol(["date","symbol",factor_name]),r_data).eval();
       // rename!(r_data, factor_name,"return_pred");
       // r_data = select "IH" as product, mean(return_pred) as return_pred from r_data group by date order by date;// 截面均值

       // // 【方式2】预期收益率数据(From ML)
       r_data = select * from loadTable("dfs://asset_cn/ReturnModel_MLModel_20250423","ModelIndividual_return") where date between start_ts and end_ts;
       r_data = select firstNot(lightgbm_return_pred) as return_pred from r_data where Benchmark = "IH" group by date, symbol;
       r_data = select "IH" as product, mean(return_pred) as return_pred from r_data group by date order by date;// 截面均值

       // 生成日频信号
       r_data[`daily_long_signal] = 0.0;
       r_data[`daily_short_signal] = 0.0;
       update r_data set daily_long_signal = 1.0 where return_pred > 0;
       update r_data set daily_short_signal = 1.0 where return_pred < 0;
       daily_signal = select date,daily_long_signal,daily_short_signal from r_data;
       undef(`r_data);
       k_data = select * from k_data left join daily_signal on k_data.date = daily_signal.date;
       update k_data set daily_long_signal = nullFill(daily_long_signal,0);
       update k_data set daily_short_signal = nullFill(daily_short_signal,0);
       undef(`daily_signal);

       // 生成分钟频进场信号(执行顺序: close_long -> close_short -> open_long -> open_short)
       // 信号生成
       k_data[`open_long_signal]=0.0;      // 期货开多信号
       k_data[`open_short_signal]=0.0;     // 期货开空信号
       k_data[`close_long_signal]=0.0;     // 期货平多信号
       k_data[`close_short_signal]=0.0;    // 期货平空信号
       k_data[`long_signal] = 0.0;
       k_data[`short_signal] = 0.0; 
       update k_data set long_signal = 1.0 where prev(ma(close,5,1)>pre_settle) and 930<minute<=945 and daily_long_signal = 1.0 context by contract+string(date); 
       // 早盘后第15分钟下单，条件: 上一分钟: 5分钟EMA均线>pre_settle
       update k_data set short_signal = 1.0 where prev(ma(close,5,1)<pre_settle) and 930<minute<=945 and daily_short_signal = 1.0 context by contract+string(date); 
       // 早盘后前15分钟下单, 条件: 上一分钟: 5分钟EMA均线<pre_settle
       update k_data set long_signal = 0.0 where prev(long_signal) = 1.0 and long_signal = 1.0 and prev(date)=date context by date; // 取第一个信号下多单
       update k_data set short_signal = 0.0 where prev(short_signal) = 1.0 and short_signal = 1.0 and prev(date)=date context by date; // 取第一个信号下空单

       // 转化为仓位操作(这里指定为后五分钟的open的均值)
       update k_data set open_long_signal = 0.2 * long_signal*(move(open,-1)+move(open,-2)+move(open,-3)+move(open,-4)+move(open,-5)) context by contract;
       update k_data set open_short_signal = 0.2 * short_signal*(move(open,-1)+move(open,-2)+move(open,-3)+move(open,-4)+move(open,-5)) context by contract;
        
       // 筛除信号为0的记录
       update k_data set signal_sum = open_long_signal+open_short_signal+close_long_signal+close_short_signal;
       k_data = select * from k_data where signal_sum>0;
        
       // 保存信号至数据库中
       pt = select date,minute,timestamp,contract,pre_settle,open,high,low,close,settle,start_date,end_date,multi,margin,open_long_signal as open_long,open_short_signal as open_short,close_long_signal as close_long,close_short_signal as close_short from k_data;
      
       pt
       """)
    init_path(rf"{save_database}")
    res_signal.to_parquet(f"{save_database}/{save_table}.pqt",index=False)      # 保存为DataFrame格式

def CTA_strategy(self: Backtest):
    """
    纯CTA策略执行Pipeline
    """
    res_strategy = future_strategy.CTA_strategy(self,
                                                pos_limit=1,              # 每次的杠杆水平
                                                static_loss=0.015,  # [多单/空单]单笔交易保证金止损比例(静态)
                                                static_profit=0.045,  # [多单/空单]单笔交易保证金止盈比例(静态)
                                                dynamic_loss=0.02,  # [多单/空单]单笔交易保证金止损比例(动态)
                                                dynamic_profit=0.06,  # [多单/空单]单笔交易保证金止盈比例(动态)                                                                    short_profit_limit: float = 0.045  # [空单]单笔交易保证金止盈比例
    )

    return res_strategy

