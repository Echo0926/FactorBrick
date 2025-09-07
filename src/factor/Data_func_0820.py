import random
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
from src.ReturnModel_mr import ReturnModel_Backtest_mr as ReturnModel_Backtest
from src.utils import get_ts_list, parallel_read_pqt
import json

"""数据Preparation函数"""
"""
ReturnModel:
symbol date price(标的价格) marketvalue(市值) state(交易状态) industry benchmark1...N(基准指数) period(调仓周期) Factor1...FactorN(因子)
"""
def modify_code(code_list):
    "将1补全为000001"
    L=[]
    for code in code_list:
        if len(str(code))<6:
            code="0"*(6-len(str(code)))+str(code)   # 1→000001
        L.append(str(code))
    return L

class ReturnModel_Data:
    """准备ReturnModel的数据"""

    def addDayFreqSymbol(self: ReturnModel_Backtest):
        self.session.run(rf"""
        idx_code = "000905.SH"//"399300.SZ" // 指数名称
        start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
         code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") where index_code == idx_code; 
        // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
        pt = select ts_code as SecurityID, trade_date as TradeDate,1500 as minute,open * adj_factor as open,close * adj_factor as close,1.0 as state 
        from loadTable("dfs://DayKDB","o_tushare_a_stock_daily") where trade_date between start_ts and end_ts and ts_code in code_list; 
        
        // 市值数据+行业数据
        mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
        from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts;
        pt = lj(pt,mv_pt, `TradeDate`SecurityID);
        undef(`mv_pt);

        // 添加至数据库中
        pt = select SecurityID,TradeDate,minute,open,close,cmarketvalue,state,industry from pt;
        loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
        """)

    def addMinFreqSymbol(self: ReturnModel_Backtest):
        """[自定义]标的价格选取及计算(必须包括state列表示当天该标的能否交易)
        symbol date price state industry
        """
        self.session.run(rf"""
        idx_code = "000905.SH"//"399300.SZ" // 指数名称
        start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
        code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") where index_code == idx_code; 
        // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
        pt = select SecurityID,TradeDate,
        int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,open,close,1.0 as state 
        from loadTable("dfs://MinKDB","Min1K") where tradeDate between start_ts and end_ts and code in code_list;
            
        // 市值数据+行业数据
        mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
        from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts and ts_code in code_list;
        pt = lj(pt,mv_pt, `TradeDate`SecurityID);
        undef(`mv_pt);
            
        // 添加至数据库中
        pt = select SecurityID,TradeDate,minute,open,close,cmarketvalue,state,industry from pt;
        loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
        """)

    def addDayFreqBenchmark(self: ReturnModel_Backtest):
        """[自定义]基准收益池选取及计算
        [注] `000001这种不能直接作为列名，因而在前面加上b以示区分
        """
        self.session.run(rf"""
        // 指数行情数据,   
        idx_code = "000905.SH"//"399300.SZ"
        pt=select idx_code as symbol,trade_date as date,1500 as minute,open,close from loadTable("dfs://DayKDB","o_tushare_index_kline_daily") where ts_code = idx_code ;
        loadTable('{self.benchmark_database}','{self.benchmark_table}').append!(pt);
        undef(`pt);
        """)

    def addLongFactorFromMinFreqData(self: ReturnModel_Backtest):
        """[自定义]Alpha因子计算并存储至因子数据库(宽表形式)"""
        self.session.run(f"""        
        idx_code = "000905.SH"//"399300.SZ";
        code_list = exec distinct(con_code) as component from loadTable("dfs://DayKDB","o_tushare_index_weight") where index_code == idx_code;        
        data = select code as symbol,tradeDate as date,int(string(tradeTime.hourOfDay())+lpad(string(tradeTime.minuteOfHour()),2,"0")) as minute,
        open,high,low,close,volume,amount as turnover from loadTable("dfs://MinKDB","Min1K") where (tradeDate between temporalAdd(date({self.start_dot_date}),-1,"d") and date({self.end_dot_date})) and code in code_list;
        factor_list = {self.factor_list};
        
        // 分钟频时序因子
        defg shioFunc(mvol, price){{ // defg 聚合函数声明
            idx_max = imax(mvol);
            priceList_m = price[:idx_max]
            idx_m = imin(priceList_m) 
            priceList_n = price[idx_max+1:]
            idx_n = imin(priceList_n)
            Cm = price[idx_m]
            Cn = price[idx_n]
            res = (Cn-Cm)\(Cm)\(idx_n-idx_m)
            return res
        }}
        
        // 涨潮半潮汐(推动涨潮的力量更大)
        defg shioStrongFunc(mvol, price){{ // defg聚合函数声明
            idx_max = imax(mvol);
            Cmax = price[idx_max];
            priceList_m = price[:idx_max]
            idx_m = imin(priceList_m) 
            priceList_n = price[idx_max+1:]
            idx_n = imin(priceList_n)
            Cm = price[idx_m]
            Cn = price[idx_n]
            Vm = mvol[idx_m]
            Vn = mvol[idx_n]
            res = iif(Vm<Vn, (Cmax-Cm)\Cm\(idx_max-idx_m), (Cn-Cmax)\Cmax\(idx_n-idx_max))
            return res
        }}
        
        // 退潮半潮汐(推动退潮的力量更大)
        defg shioWeakFunc(mvol, price){{ // defg聚合函数声明
            idx_max = imax(mvol);
            Cmax = price[idx_max];
            priceList_m = price[:idx_max]
            idx_m = imin(priceList_m) 
            priceList_n = price[idx_max+1:]
            idx_n = imin(priceList_n)
            Cm = price[idx_m]
            Cn = price[idx_n]
            Vm = mvol[idx_m]
            Vn = mvol[idx_n]
            res = iif(Vm>Vn, (Cmax-Cm)\Cm\(idx_max-idx_m), (Cn-Cmax)\Cmax\(idx_n-idx_max))
            return res
        }}
        
        // MapFunc
        def FactorMap(data){{
            factor_list = {self.factor_list};
            pt = select * from data order by symbol,date,minute  // 正序排序
            // 新增最新一个交易日的frame[用于日频决策]
            last_pt=select temporalAdd(lastNot(date),1,"XSHG") as date,1500 as minute,NULL as open,NULL as high,NULL as low,NULL as close,NULL as volume,NULL as turnover from pt group by symbol; // 选取最后一个日期加一的形式作为决策的形式
            append!(pt,last_pt);
            undef(`last_pt);
            update pt set vwap = nullFill!(turnover/volume,0);
            update pt set mVol = msum(volume,9) context by symbol, date
            update pt set mVol = move(mVol,4) context by symbol, date
            pt = select shioFunc(mVol, close) as shio, 
                        shioStrongFunc(mVol, close) as shioStrong,
                        shioWeakFunc(mVol, close) as shioWeak
                 from pt group by date, symbol order by date;
            // 市值 + 行业
            info_pt = select ts_code as symbol,trade_date as date,total_mv,circ_mv from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic")
            pt = lj(pt, info_pt, `date`symbol);
            undef(`info_pt);
            
            update pt set minute = 1500; // 日频因子随便添加一个分钟列
            update pt set shio = prev(shio) context by symbol;
            update pt set shioStrong = prev(shioStrong) context by symbol;
            update pt set shioWeak = prev(shioWeak) context by symbol;
            update pt set shio_avg20 = prev(mavg(shio,20)) context by symbol; 
            update pt set shioStrong_avg20 = prev(mavg(shioStrong,20)) context by symbol;
            update pt set shioWeak_avg20 = prev(mavg(shioWeak,20)) context by symbol;
            update pt set shio_std20 = prev(mstd(shio,20)) context by symbol; 
            update pt set shioStrong_std20 = prev(mstd(shioStrong,20)) context by symbol;
            update pt set shioWeak_std20 = prev(mstd(shioWeak,20)) context by symbol;
            
            // zscore
            pt = sql(select = [sqlCol(`symbol),sqlCol(`date),sqlCol(`minute),sqlCol(`circ_mv)].append!(sqlCol(factor_list)), from=pt).eval()
            
            // 截面空缺值填充
            for (col in {self.factor_list}){{
                pt[`draft] = pt[col]
                update pt set draft = nullFill(draft, avg(draft)) context by date,minute; // 取日频截面的均值对因子进行填充
                // update pt set draft = residual(draft,circ_mv,ols(draft,circ_mv)) context by date;
                pt[col] = pt[`draft]
                dropColumns!(pt,`draft);
            }}
            dropColumns!(pt,`circ_mv)
            return pt
        }}
        
        // pt = mr(ds, component).unionAll(false) // MapReduce操作
        print("start calculating factor")
        pt = FactorMap(data);
        print("end calculating factor")
        
        // 添加至数据库
        def InsertData(DBName, TBName, data, batchsize){{
            // 预防Out of Memory，分批插入数据，batchsize为每次数据的记录数
            start_idx = 0
            end_idx = batchsize
            krow = rows(data)
            do{{ 
                slice_data = data[start_idx:min(end_idx,krow),]
                if (rows(slice_data)>0){{
                loadTable(DBName, TBName).append!(slice_data);
                print(start_idx);
                }}
                start_idx = start_idx + batchsize
                end_idx = end_idx + batchsize
            }}while(start_idx < krow)
        }}
        InsertData(DBName="{self.factor_database}",TBName="{self.factor_table}",data=pt,batchsize=1000000); // batch设为1000000条记录
        """)

    def prepare_combine_data(self: ReturnModel_Backtest):
        """[自定义period,其余标准化]合并symbol+benchmark+factor→提供标准化的数据"""
        self.session.run(rf"""
        // 清理缓存
        clearAllCache();
        
        // 区间收益率计算函数 -> FutureReturn1..30
        def ReturnCal(data, idCol, openCol, closeCol, periodSize){{
            df = data;
            df[`open] = df[openCol];
            df[`close] = df[closeCol];
            df[`id] = df[idCol]
            update df set FutureReturn = (move(close,-periodSize)-close)\close context by id;
            // res = sql(select=sqlColAlias(<(move(close,-periodSize)-close)\close>,`FutureReturn),
                // from=df,groupBy=sqlCol(idCol),groupFlag=0).eval()
            return exec FutureReturn from df
        }}
        // 区间超额收益率计算函数 -> // FutureReturn1..30
        def ReturnOverCal(data, idCol, openCol, closeCol, benchmarkOpenCol, benchmarkCloseCol, periodSize){{
            df = data;
            df[`open] = df[openCol];
            df[`close] = df[closeCol];
            df[`benchmarkOpen] = df[benchmarkOpenCol]
            df[`benchmarkClose] = df[benchmarkCloseCol]
            update df set FutureReturn = (move(close,-periodSize)-close)/close - (move(benchmarkClose,-periodSize)-benchmarkClose)/benchmarkClose context by id
            // res = sql(select=sqlColAlias(<(move(close,-periodSize)-close)/close - (move(benchmarkClose,-periodSize)-benchmarkClose)/benchmarkClose>,`FutureReturn+string(periodSize)),
                // from=df,groupBy=sqlCol(idCol),groupFlag=0).eval()
            return exec FutureReturn from res
        }}
        
        // Index Component Constraint
        idx_code = "000905.SH"//"399300.SZ";
        index_df = select con_code as symbol,trade_date as date from loadTable("dfs://DayKDB","o_tushare_index_weight") where index_code == idx_code and trade_date between date({self.start_dot_date}) and temporalAdd(date({self.end_dot_date}), 1 ,"XSHG"); // 成分股信息
        symbol_list = exec distinct(symbol) as component from index_df;
        
        // 补全index_df
        total_date_list = getMarketCalendar("XSHG", date({self.start_dot_date}), temporalAdd(date({self.end_dot_date}), 1 ,"XSHG"))
        current_date_list = sort(exec distinct(date) from index_df);
        last_date = current_date_list[0]
        for (i in 1..size(total_date_list)-1){{
            ts = total_date_list[i]
            if (!(ts in current_date_list)){{
                // 离他最近比它小的date
                index_df.append!(select symbol, ts as `date from index_df where date == last_date)      
            }}else{{
                last_date = ts
            }}
        }}
        sortBy!(index_df,`date`symbol,[1,1]);
        
        // Symbol df processing
        symbol_df=select symbol,date,minute,open,close,marketvalue,state,industry from loadTable('{self.symbol_database}','{self.symbol_table}') where date between date({self.start_dot_date}) and temporalAdd(date({self.end_dot_date}), 1 ,"XSHG") and symbol in symbol_list;        
        // bar_return & Period_return
        update symbol_df set bar_return = nullFill((next(close)-close)/close,0.0) context by symbol;
        update symbol_df set bar_return = clip(bar_return, -0.1, 0.1);
        idCol = `symbol;
        openCol = `open;
        closeCol = `close;
        returnIntervals = {self.returnIntervals}
        returnFunc = ReturnCal{{symbol_df, idCol, openCol, closeCol}}   // TODO: 价格参数确定计算的区间收益率函数类型
        returnRes = each(returnFunc, returnIntervals)
        for (i in 0..(size(returnIntervals)-1)){{
            interval = returnIntervals[i]
            symbol_df[`FutureReturn+string(interval)] = returnRes[i]
        }}
        print("未来区间收益率计算完毕") // for ICIR & 回归法
        symbol_df = lj(index_df,symbol_df,`symbol`date);
        undef(`index_df);
        
         // 新增一个period的frame[日频决策,与因子部分保持一致]
        sortBy!(symbol_df,[`symbol,`date,`minute],[1,1,1]); // 正序排序
        for (i in seq(1,int(1))){{  // 这里后续可以根据调仓周期去灵活变换
            pt=select temporalAdd(lastNot(date),1,"XSHG") as date,1500 as minute,NULL as open,NULL as close,NULL as marketvalue,nullFill(last(state),1) as state,"NULL" as industry,0.0 as bar_return,{",".join([f"0.0 as FutureReturn{i}" for i in self.returnIntervals])} from symbol_df group by symbol; // 选取最后一个日期加一的形式作为决策的形式
            symbol_df.append!(pt);
        }};
        undef(`pt);
        update symbol_df set industry = industry.ffill() context by symbol;  // 假设industry不变
        print("Symbol添加完毕")
        
        // Period of processing
        symbol_df[`monthidx]=0;
        symbol_df[`weekidx]=0;
        df=select first(monthidx) as monthidx,first(weekidx) as weekidx, 1 as dateidx from symbol_df group by date;
        sortBy!(df,`date,[1]);
        update df set monthdelta=date-monthBegin(date);
        update df set weekdelta=date-weekBegin(date);
        update df set monthidx=1 where date=min(date) or monthdelta<=prev(monthdelta);
        update df set weekidx=1 where date=min(date) or weekdelta<=prev(weekdelta);
        update df set period_month=cumsum(monthidx); // 月频period
        update df set period_week=cumsum(weekidx); // 周频period
        update df set period_date=cumsum(dateidx); // 日频period
        
        // 月频调仓/K月频
        // month_Dict=dict(df[`date],df[`period_month]);
        // update symbol_df set period=month_Dict[date];
        
        // 周频调仓/K周频
        // week_Dict=dict(df[`date],df[`period_week]);
        // update symbol_df set period=week_Dict[date];
        // update symbol_df set period=int(double(period)/2)

        // 日频调仓/K日频 (暂不支持分钟频调仓)
        day_Dict=dict(df[`date],df[`period_date]);
        update symbol_df set period=day_Dict[date];
        
        dropColumns!(symbol_df,`monthidx`weekidx);
        undef(`df);
        
        // Benchmark df processing
        benchmark_df=select date,minute,open as {self.benchmark}_open,close as {self.benchmark}_close from loadTable("{self.benchmark_database}","{self.benchmark_table}"); // 这里只设了一个benchmark
        symbol_df=lj(symbol_df,benchmark_df,`date`minute);
        undef(`benchmark_df); 
        print("Benchmark 添加完毕")
        
        // Factor df processing
        factor_list={self.factor_list}; 
        factor_pt=select symbol,date,minute,{",".join(self.factor_list)} from loadTable("{self.factor_database}","{self.factor_table}") where date >= {self.start_dot_date} and symbol in symbol_list;
        symbol_df=lj(symbol_df,factor_pt,`symbol`date`minute);
        undef(`factor_pt)
        print("Factor 添加完毕")

        // 添加至数据库
        def InsertData(DBName, TBName, data, batchsize){{
            // 预防Out of Memory，分批插入数据，batchsize为每次数据的记录数
            start_idx = 0
            end_idx = batchsize
            krow = rows(data)
            do{{ 
                slice_data = data[start_idx:min(end_idx,krow),]
                if (rows(slice_data)>0){{
                loadTable(DBName, TBName).append!(slice_data);
                }}
                start_idx = start_idx + batchsize
                end_idx = end_idx + batchsize
            }}while(start_idx < krow)
        }}
        InsertData(DBName="{self.combine_database}",TBName="{self.combine_table}",data=nullFill(symbol_df,0),batchsize=1000000); 
        """)
        # TODO 截面均值填充因子nan值

        # 添加Template数据库(由于symbol_df已经添加过了，所以这里不再添加下一个交易日的structure)
        # Template Daily+Template Individual+Template
        self.session.run(rf"""
        pt=select symbol,date,minute,period from loadTable("{self.combine_database}","{self.combine_table}");
        sortBy!(pt,`symbol`period`date`minute,[1,1,1,1]); // 正序排序

        // 添加Template Minute数据库
        loadTable("{self.combine_database}","{self.template_minute_table}").append!(pt);

        // 添加Template Daily数据库
        pt = select firstNot(minute) as minute from pt group by symbol,period,date;
        pt = select symbol,date,period from pt;
        loadTable("{self.combine_database}","{self.template_daily_table}").append!(pt);

        // 添加Template Individual数据库
        pt=select firstNot(date) as start_date,lastNot(date) as end_date from pt group by symbol,period;
        sortBy!(pt,[`symbol,`period],[1,1]);
        pt=select symbol,start_date,end_date,period from pt;
        loadTable("{self.combine_database}","{self.template_individual_table}").append!(pt);

        // 添加Template 数据库
        pt=select firstNot(start_date) as start_date,firstNot(end_date) as end_date from pt group by period;
        pt=select start_date,end_date,period from pt;
        sortBy!(pt,`period,1);
        loadTable("{self.combine_database}","{self.template_table}").append!(pt);
        undef(`pt); // 释放内存
        """)
