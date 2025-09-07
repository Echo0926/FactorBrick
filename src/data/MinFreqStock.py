import pandas as pd
import numpy as np
import dolphindb as ddb
from ReturnModel_mr import ReturnModel_Backtest_mr as ReturnModel_Backtest

def addMinFreqSymbol1000(self: ReturnModel_Backtest):
    """分钟频1000行情+基本面信息
    """
    self.session.run(rf"""
       idx_code = "000852.SH" // 指数名称
       start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
       code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                    where index_code == idx_code and (trade_date between start_ts and end_ts); 
       // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
       pt = select SecurityID,TradeDate,
       int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,open,close,1.0 as state 
       from loadTable("dfs://MinKDB","Min1K") where (tradeDate between start_ts and end_ts) and code in code_list;

       // 市值数据+行业数据
       mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
       from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where (trade_date between start_ts and end_ts) and ts_code in code_list;
       pt = lj(pt,mv_pt, `TradeDate`SecurityID);
       undef(`mv_pt);

       // 添加至数据库中
       pt = select SecurityID,TradeDate,minute,open,close,cmarketvalue,state,industry from pt;
       loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
    """)

def addMinFreqSymbol500(self: ReturnModel_Backtest):
    """分钟频500行情+基本面信息
    """
    self.session.run(rf"""
       idx_code = "000905.SH" // 指数名称
       start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
       code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                    where index_code == idx_code and (trade_date between start_ts and end_ts); 
       // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
       pt = select SecurityID,TradeDate,
       int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,open,close,1.0 as state 
       from loadTable("dfs://MinKDB","Min1K") where (tradeDate between start_ts and end_ts) and code in code_list;

       // 市值数据+行业数据
       mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
       from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where (trade_date between start_ts and end_ts) and ts_code in code_list;
       pt = lj(pt,mv_pt, `TradeDate`SecurityID);
       undef(`mv_pt);

       // 添加至数据库中
       pt = select SecurityID,TradeDate,minute,open,close,cmarketvalue,state,industry from pt;
       loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
    """)

def addMinFreqSymbol300(self: ReturnModel_Backtest):
    """分钟频300行情+基本面信息
    """
    self.session.run(rf"""
       idx_code = "399300.SZ" // 指数名称
       start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
       code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                    where index_code == idx_code and (trade_date between start_ts and end_ts); 
       // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
       pt = select SecurityID,TradeDate,
       int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,open,close,1.0 as state 
       from loadTable("dfs://MinKDB","Min1K") where (tradeDate between start_ts and end_ts) and code in code_list;

       // 市值数据+行业数据
       mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
       from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where (trade_date between start_ts and end_ts) and ts_code in code_list;
       pt = lj(pt,mv_pt, `TradeDate`SecurityID);
       undef(`mv_pt);

       // 添加至数据库中
       pt = select SecurityID,TradeDate,minute,open,close,cmarketvalue,state,industry from pt;
       loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
    """)


