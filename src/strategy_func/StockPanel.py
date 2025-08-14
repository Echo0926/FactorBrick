import pandas as pd
from src.BackTest import Backtest
from src.strategy_func.template import stock_strategy
from src.utils import init_path
from typing import Dict

def get_stock_signal_to_dataframe(session, BackTest_config: Dict, ReturnModel_config: Dict):
    """
    股票策略信号值生成(to:symbol date buy_signal sell_signal)
    """
    # Step0. Configuration
    start_date, end_date = BackTest_config["start_date"], BackTest_config["end_date"]
    start_dot_date, end_dot_date = pd.Timestamp(start_date).strftime('%Y.%m.%d'), pd.Timestamp(end_date).strftime(
        '%Y.%m.%d')

    # Step1.创建信号数据库
    save_database, save_table = BackTest_config["stock_signal_database"], BackTest_config["stock_signal_table"]

    # Step2.生成信号并存储到数据库中
    res_signal = session.run(rf"""
        // config
       start_ts, end_ts = {start_dot_date},temporalAdd({end_dot_date}stock_cn/value","market_hfq") where date between start_ts and end_ts;
       update k_data set minute = 15,1,"XSHG");
       estimation_method = "Ridge";
       init_period = 2; // 因子模型初始化period周期长度, 与returnModel保持一致
        
        // 行情数据
       k_data = select * from loadTable("dfs://00;
       update k_data set timestamp = datetime(string(date)+"T15:00:00"); 
        
       // 预期收益率数据
       // r_data = select symbol,period,turnoverrate_return_pred as return_pred from loadTable("dfs://asset_cn/ReturnModel_result_20250505","individual_return");
       template_pt = select symbol,start_date as date,period from loadTable("dfs://asset_cn/ReturnModel_20250505","template_individual");
       r_data = select * from r_data left join template_pt on template_pt.period = r_data.period;
       r_data = select date,minute,symbol,xgboost_return_pred as return_pred from loadTable("dfs://asset_cn/ReturnModel_MLModel_20250710","ModelIndividual_return");
       r_data = select firstNot(return_pred) as return_pred from r_data group by date,symbol;
    
       // 生成日频信号
       r_data[`daily_long_signal] = 0.0;
       r_data[`daily_short_signal] = 0.0;
       r_data[`long_rank] = 0.0
       update r_data set positive_rank = rank(return_pred,false) context by date;
       update r_data set negative_rank = rank(return_pred,true) context by date;
       update r_data set daily_long_signal = 1.0 where positive_rank <= percentile(positive_rank,15) context by date;  // 交易预期收益率前20%的票
       update r_data set daily_short_signal = 1.0 where negative_rank <= percentile(negative_rank,15) context by date;
       daily_signal = select symbol,date,daily_long_signal,daily_short_signal from r_data;
       undef(`r_data);
       k_data = select * from k_data left join daily_signal on k_data.date = daily_signal.date and k_data.symbol = daily_signal.symbol;
       sortBy!(k_data,`symbol`date,[1,1]);    // 这里一定要排序!
       update k_data set daily_long_signal = nullFill(daily_long_signal,0);
       update k_data set daily_short_signal = nullFill(daily_short_signal,0);
       undef(`daily_signal);
       
       // 生成分钟频进场信号(执行顺序: sell -> buy)
       // 信号生成
       k_data[`buy_signal]=0.0;      // 股票开多信号
       k_data[`sell_signal]=0.0;     // 股票开空信号
       
       // 这里指定为后五分钟的open的均值
       update k_data set next_price = next(open) context by symbol;
       update k_data set buy_signal = next_price;
       update k_data set sell_signal = next_price;
       update k_data set buy_signal = buy_signal * daily_long_signal; 
       update k_data set sell_signal = sell_signal * daily_short_signal; 
       
       // 筛除信号为0的记录
       update k_data set signal_sum = buy_signal+sell_signal;
       k_data = select * from k_data where signal_sum>0;
       
        // 保存信号至数据库中
       pt = select date,minute,timestamp,symbol,open,high,low,close,volume,buy_signal,sell_signal from k_data;
      
       pt
       
    """)
    init_path(rf"{save_database}")
    res_signal.to_parquet(f"{save_database}/{save_table}.pqt",index=False)      # 保存为DataFrame格式


def Stock_strategy(self: Backtest):
    """
    指数增强/空气指增策略执行Pipeline
    """
    res_strategy = stock_strategy.Stock_strategy(self,
                                                pos_limit=0.5,          # 每次的杠杆水平
                                                static_loss=0.05,       # [多单]单笔交易止损比例(按价格)
                                                static_profit=0.06,      # [多单]单笔交易止盈比例(按价格)
                                                dynamic_profit=0.05,
                                                dynamic_loss=0.05,
    )

    return res_strategy