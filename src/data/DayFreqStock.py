import pandas as pd
import numpy as np
import dolphindb as ddb
from ReturnModel_light import SingleFactorBackTest

def addDayFreqSymbol1000(self: SingleFactorBackTest):
    self.session.run(rf"""
    idx_code = "000852.SH" // 指数名称
    start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
    code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                where index_code == idx_code; 
    // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
    pt = select ts_code as SecurityID, trade_date as TradeDate,15:00:00.000 as TradeTime,open * adj_factor as open,close * adj_factor as close,1.0 as state 
    from loadTable("dfs://DayKDB","o_tushare_a_stock_daily") where trade_date between start_ts and end_ts and ts_code in code_list; 

    // 市值数据+行业数据
    mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
    from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts;
    pt = lj(pt,mv_pt, `TradeDate`SecurityID);
    undef(`mv_pt);

    // 添加至数据库中
    pt = select SecurityID,TradeDate,TradeTime,open,close,cmarketvalue,state,industry from pt;
    loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
    """)

def addDayFreqBenchmark1000(self: SingleFactorBackTest):
    """[自定义]基准收益池选取及计算
    注：`000001这种不能直接作为列名，因而在前面加上b以示区分
    """
    self.session.run(rf"""
    // 指数行情数据  
    idx_code = "000852.SH"
    pt=select idx_code as symbol,trade_date as TradeDate,15:00:00.000 as TradeTime,open,close 
        from loadTable("dfs://DayKDB","o_tushare_index_kline_daily") where ts_code = idx_code ;
    loadTable('{self.benchmark_database}','{self.benchmark_table}').append!(pt);
    undef(`pt);
    """)

def addDayFreqSymbol500(self: SingleFactorBackTest):
    self.session.run(rf"""
    idx_code = "000905.SH" // 指数名称
    start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
    code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                where index_code == idx_code; 
    // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
    pt = select ts_code as SecurityID, trade_date as TradeDate,15:00:00.000 as TradeTime,
                open * adj_factor as open,close * adj_factor as close,1.0 as state 
                from loadTable("dfs://DayKDB","o_tushare_a_stock_daily") 
                where trade_date between start_ts and end_ts and ts_code in code_list; 

    // 市值数据+行业数据
    mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
    from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts;
    pt = lj(pt,mv_pt, `TradeDate`SecurityID);
    undef(`mv_pt);

    // 添加至数据库中
    pt = select SecurityID,TradeDate,TradeTime,open,close,cmarketvalue,state,industry from pt;
    loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
    """)

def addDayFreqBenchmark500(self: SingleFactorBackTest):
    """[自定义]基准收益池选取及计算
    注：`000001这种不能直接作为列名，因而在前面加上b以示区分
    """
    self.session.run(rf"""
    // 指数行情数据  
    idx_code = "000905.SH"
    pt=select idx_code as symbol,trade_date as TradeDate,15:00:00.000 as TradeTime,
        open,close from loadTable("dfs://DayKDB","o_tushare_index_kline_daily") where ts_code = idx_code ;
    loadTable('{self.benchmark_database}','{self.benchmark_table}').append!(pt);
    undef(`pt);
    """)

def addDayFreqSymbol300(self: SingleFactorBackTest):
    self.session.run(rf"""
       idx_code = "399300.SZ" // 指数名称
       start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
       code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") 
                   where index_code == idx_code; 
       // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
       pt = select ts_code as SecurityID, trade_date as TradeDate,1500 as TradeTime,
            open * adj_factor as open,close * adj_factor as close,1.0 as state 
            from loadTable("dfs://DayKDB","o_tushare_a_stock_daily") 
            where trade_date between start_ts and end_ts and ts_code in code_list; 

       // 市值数据+行业数据
       mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
       from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts;
       pt = lj(pt,mv_pt, `TradeDate`SecurityID);
       undef(`mv_pt);

       // 添加至数据库中
       pt = select SecurityID,TradeDate,TradeTime,open,close,cmarketvalue,state,industry from pt;
       loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
    """)

def addDayFreqBenchmark300(self: SingleFactorBackTest):
    """[自定义]基准收益池选取及计算
    注：`000001这种不能直接作为列名，因而在前面加上b以示区分
    """
    self.session.run(rf"""
    // 指数行情数据  
    idx_code = "399300.SZ"
    pt = select idx_code as symbol,trade_date as TradeDate,15:00:00.000 as TradeTime,
            open,close from loadTable("dfs://DayKDB","o_tushare_index_kline_daily") where ts_code = idx_code;
    loadTable('{self.benchmark_database}','{self.benchmark_table}').append!(pt);
    undef(`pt);
    """)