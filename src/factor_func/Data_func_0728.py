import random
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
from src.ReturnModel import ReturnModel_Backtest
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

    def addMinFreqSymbol(self: ReturnModel_Backtest):
        """[自定义]标的价格选取及计算(必须包括state列表示当天该标的能否交易)
        symbol date price state industry
        """
        self.session.run(rf"""
        start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
        code_list = exec distinct(component) as component from loadTable("dfs://component","component_cn") where index == "000016"; 
        // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
        pt = select SecurityID,TradeDate,
        int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,open,close,1.0 as state 
        from loadTable("dfs://MinuteKDB","stock_bar") where TradeDate between start_ts and end_ts and SecurityID in code_list;
            
        // 市值数据+行业数据
        mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
        from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts;
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
        // 指数行情数据,由于这里只有日频指数数据,所以随便填     
        pt=select "b"+left(string(ts_code),6) as symbol,trade_date as date,1500 as minute,open,close from loadTable("dfs://DayKDB","o_tushare_index_kline_daily") where ts_code = "000001.SZ" ;
        loadTable('{self.benchmark_database}','{self.benchmark_table}').append!(pt);
        undef(`pt);
        """)

    def addLongFactor(self: ReturnModel_Backtest):
        """[自定义]Alpha因子计算并存储至因子数据库(宽表形式)"""
        self.session.run(f"""
        code_list = exec distinct(component) as component from loadTable("dfs://component","component_cn") where index == "000016" and date between temporalAdd(date({self.start_dot_date}),-1,"d") and date({self.end_dot_date});  
        pt = select SecurityID as symbol,TradeDate as date,int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,
        open,high,low,close,volume,turnover from loadTable("dfs://MinuteKDB","stock_bar") where SecurityID in code_list and (TradeDate between temporalAdd(date({self.start_dot_date}),-1,"d") and date({self.end_dot_date})); // 这里根据因子情况修改-1为状态函数的最大时间跨度
        
        pt = select * from pt order by symbol,date,minute  // 正序排序
        // 新增最新一个交易日的frame[用于日频决策]
        last_pt=select temporalAdd(lastNot(date),1,"XSHG") as date,1500 as minute,NULL as open,NULL as high,NULL as low,NULL as close,NULL as volume,NULL as turnover from pt group by symbol; // 选取最后一个日期加一的形式作为决策的形式
        append!(pt,last_pt);
        undef(`last_pt);
        update pt set vwap = nullFill!(turnover/volume,0);

        // 分钟频时序因子
        update pt set ret240 = (close-move(close,-240))/close context by symbol;
        update pt set ret240 = nullFill(ret240.ffill(),0) context by symbol;
        update pt set ret240 = clip(ret240,-0.99,1) 
        varFunc = valueAtRisk{{,'normal',0.95}};
        cvarFunc = condValueAtRisk{{, 'normal',0.95}};
        factor_list = {self.factor_list};
        update pt set vaR120_240 = moving(varFunc,ret240,120,1) context by symbol;
        update pt set vaR180_240 = moving(varFunc,ret240,180,1) context by symbol;
        update pt set vaR240_240 = moving(varFunc,ret240,240,1) context by symbol;
        update pt set cvaR120_240 = moving(cvarFunc,ret240,120,1) context by symbol;
        update pt set cvaR180_240 = moving(cvarFunc,ret240,180,1) context by symbol;
        update pt set cvaR240_240 = moving(cvarFunc,ret240,240,1) context by symbol;

        // zscore
        pt = sql(select = [sqlCol(`symbol),sqlCol(`date),sqlCol(`minute)].append!(sqlCol(factor_list)), from=pt).eval()

        // 截面空缺值填充
        for (col in {self.factor_list}){{
            pt[`draft] = pt[col]
            update pt set draft = nullFill(draft, avg(draft)) context by minute; // 取分钟频截面的均值对因子进行填充
            dropColumns!(pt,`draft);
        }}
         
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
        InsertData(DBName="{self.factor_database}",TBName="{self.factor_table}",data=pt,batchsize=1000000); // batch设为1000000条记录
        """)

    def addLongFactor_mr(self: ReturnModel_Backtest):
        """[自定义]Alpha因子计算并存储至因子数据库(宽表形式)"""
        self.session.run(f"""
        code_list = exec distinct(component) as component from loadTable("dfs://component","component_cn") where index == "000016" and date between temporalAdd(date({self.start_dot_date}),-1,"d") and date({self.end_dot_date});  
        ds = repartitionDS(<select SecurityID as symbol,TradeDate as date,int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,
        open,high,low,close,volume,turnover from loadTable("dfs://MinuteKDB","stock_bar") where SecurityID in code_list and (TradeDate between temporalAdd(date({self.start_dot_date}),-1,"d") and date({self.end_dot_date}))>,`SecurityID); // 这里根据因子情况修改-1为状态函数的最大时间跨度
        
        // MapFunc
        def FactorMap(data){{
            pt = select * from data order by symbol,date,minute  // 正序排序
            // 新增最新一个交易日的frame[用于日频决策]
            last_pt=select temporalAdd(lastNot(date),1,"XSHG") as date,1500 as minute,NULL as open,NULL as high,NULL as low,NULL as close,NULL as volume,NULL as turnover from pt group by symbol; // 选取最后一个日期加一的形式作为决策的形式
            append!(pt,last_pt);
            undef(`last_pt);
            update pt set vwap = nullFill!(turnover/volume,0);
            
            // 分钟频时序因子
            update pt set ret240 = (close-move(close,-240))/close context by symbol;
            update pt set ret240 = nullFill(ret240.ffill(),0) context by symbol;
            update pt set ret240 = clip(ret240,-0.99,1) 
            varFunc = valueAtRisk{{,'normal',0.95}};
            cvarFunc = condValueAtRisk{{, 'normal',0.95}};
            factor_list = ["vaR120_240","vaR180_240","vaR240_240",
            "cvaR120_240","cvaR180_240","cvaR240_240"];
            update pt set vaR120_240 = moving(varFunc,ret240,120,1) context by symbol;
            update pt set vaR180_240 = moving(varFunc,ret240,180,1) context by symbol;
            update pt set vaR240_240 = moving(varFunc,ret240,240,1) context by symbol;
            update pt set cvaR120_240 = moving(cvarFunc,ret240,120,1) context by symbol;
            update pt set cvaR180_240 = moving(cvarFunc,ret240,180,1) context by symbol;
            update pt set cvaR240_240 = moving(cvarFunc,ret240,240,1) context by symbol;
                        
            // zscore
            pt = sql(select = [sqlCol(`symbol),sqlCol(`date),sqlCol(`minute)].append!(sqlCol(factor_list)), from=pt).eval()
            
            // 截面空缺值填充
            for (col in {self.factor_list}){{
                pt[`draft] = pt[col]
                update pt set draft = nullFill(draft, avg(draft)) context by minute; // 取分钟频截面的均值对因子进行填充
                dropColumns!(pt,`draft);
            }}
            return nullFill(pt,0.0)
        }}
        
        pt = mr(ds, FactorMap).unionAll(false) // MapReduce操作
        
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
        InsertData(DBName="{self.factor_database}",TBName="{self.factor_table}",data=pt,batchsize=1000000); // batch设为1000000条记录
        """)

    def prepare_combine_data(self: ReturnModel_Backtest):
        """[自定义period,其余标准化]合并symbol+benchmark+factor→提供标准化的数据"""
        self.session.run(rf"""
        // 清理缓存
        clearAllCache();
        
        // Index Component Constraint
        index_df = select component as symbol,date from loadTable("dfs://component","component_cn") where index == "000016" and date between date({self.start_dot_date}) and temporalAdd(date({self.end_dot_date}),int({self.t}),"XSHG"); // 上证50成分股信息
        symbol_list = exec distinct(symbol) from index_df;
                
        // Symbol df processing
        symbol_df=select symbol,date,minute,open,close,marketvalue,state,industry from loadTable('{self.symbol_database}','{self.symbol_table}') where date between date({self.start_dot_date}) and temporalAdd(date({self.end_dot_date}),int({self.t}),"XSHG") and symbol in symbol_list;        
        symbol_df = lj(index_df,symbol_df,`symbol`date);
        undef(`index_df);
        
         // 新增一个period的frame[日频决策,与因子部分保持一致]
        sortBy!(symbol_df,[`symbol,`date,`minute],[1,1,1]); // 正序排序
        for (i in seq(1,int({self.t}))){{
            pt=select temporalAdd(lastNot(date),1,"XSHG") as date,1500 as minute,NULL as open,NULL as close,NULL as marketvalue,nullFill(last(state),1) as state,"AA" as industry from symbol_df group by symbol; // 选取最后一个日期加一的形式作为决策的形式
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
        benchmark_df=select date,minute,open as b000001_open,close as b000001_close from loadTable("{self.benchmark_database}","{self.benchmark_table}"); // 这里只设了一个benchmark
        symbol_df=lj(symbol_df,benchmark_df,`date`minute);
        undef(`benchmark_df); 
        print("Benchmark 添加完毕")
        
        // Factor df processing
        factor_list={self.factor_list}; 
        factor_pt=select symbol,date,minute,{",".join(self.factor_list)} from loadTable("{self.factor_database}","{self.factor_table}") where date >= {self.start_dot_date} and symbol in symbol_list;
        symbol_df=lj(symbol_df,factor_pt,`symbol`date`minute);
        undef(`factor_pt)
        print("Factor 添加完毕")
        
        // NBR240
        update symbol_df set NBR240 = nullFill((move(close,-240)-move(open,-1))/move(open,-1),0) context by symbol;
        
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
