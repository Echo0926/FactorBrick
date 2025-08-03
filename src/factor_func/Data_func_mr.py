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

    def prepare_symbol_data(self: ReturnModel_Backtest):
        """[自定义]标的价格选取及计算(必须包括state列表示当天该标的能否交易)
        symbol date price state industry
        """
        self.session.run(rf"""
        start_ts, end_ts = date({self.start_dot_date}), date({self.end_dot_date});
        // 行情数据+是否停牌数据(0/1,该字段当前未使用,之后需要加入回测框架中)
        pt = select SecurityID,TradeDate,
        int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,open,close,1.0 as state 
        from loadTable("dfs://MinuteKDB","stock_bar") where TradeDate between start_ts and end_ts;
            
        // 市值数据+行业数据
        mv_pt = select ts_code as SecurityID,trade_date as TradeDate,circ_mv as cmarketvalue, ts_code_exchange as industry 
        from loadTable("dfs://DayKDB","o_tushare_a_stock_daily_basic") where trade_date between start_ts and end_ts;
        pt = lj(pt,mv_pt, `TradeDate`SecurityID);
        undef(`mv_pt);
            
        // 添加至数据库中
        pt = select SecurityID,TradeDate,minute,open,close,cmarketvalue,state,industry from pt;
        loadTable("{self.symbol_database}","{self.symbol_table}").append!(pt);
        """)

    def prepare_benchmark_data(self: ReturnModel_Backtest):
        """[自定义]基准收益池选取及计算
        [注] `000001这种不能直接作为列名，因而在前面加上b以示区分
        """
        self.session.run(rf"""
        // 指数行情数据,由于这里只有日频指数数据,所以随便填     
        pt=select "b"+left(string(ts_code),6) as symbol,trade_date as date,1500 as minute,open,close from loadTable("dfs://DayKDB","o_tushare_index_kline_daily") where ts_code = "000001.SZ" ;
        loadTable('{self.benchmark_database}','{self.benchmark_table}').append!(pt);
        undef(`pt);
        """)

    def prepare_factor_data_short(self: ReturnModel_Backtest):
        """[自定义]Alpha因子计算并存储至因子数据库(窄表形式)"""

        # 主连数据→品种因子
        nullFill_string="\n".join([f"update wide_df set {factor}= nullFill({factor},avg({factor})) context by date;" for factor in self.factor_list])
        self.session.run(rf"""
        pt=select product as symbol,date,settle as price,volume,open_interest from loadTable("dfs://future_cn/combination","base") where date>=2020.01.01 and isMainContract=1;
        
        // 新增最新一个交易日的frame
        sortBy!(pt,[`symbol,`date],[1,1]); // 正序排序
        last_pt=select temporalAdd(lastNot(date),1,"XSHG") as date,NULL as price,NULL as volume,NULL as open_interest from pt group by symbol; // 选取最后一个日期加一的形式作为决策的形式
        append!(pt,last_pt);
        undef(`last_pt);
        
        // 定义每日因子
        // Level-1 (通过OHLC数据即可计算)
        wide_df=select symbol,date,            
            prev(nullFill((price-prev(price))/prev(price),0)) as Momentum1,
            prev(nullFill((price-move(price,5))/move(price,5),0)) as Momentum5,
            prev(nullFill((price-move(price,10))/move(price,10),0)) as Momentum10,
            prev(nullFill((price-move(price,15))/move(price,15),0)) as Momentum15,
            prev(nullFill((price-move(price,30))/move(price,30),0)) as Momentum30,
            
            prev(nullFill(prev(volume)/prev(open_interest),0)) as TurnoverRate,  
            prev(nullFill(prev(ma(volume,5,1))/prev(ma(open_interest,5,1)),0)) as TurnoverRate5,  
            prev(nullFill(prev(ma(volume,10,1))/prev(ma(open_interest,10,1)),0)) as TurnoverRate10,
            prev(nullFill(prev(ma(volume,15,1))/prev(ma(open_interest,15,1)),0)) as TurnoverRate15,
            prev(nullFill(prev(ma(volume,30,1))/prev(ma(open_interest,30,1)),0)) as TurnoverRate30 from pt context by symbol;
        undef(`pt);
                                
        // Level-2 (AggFunc)
        update wide_df set Momentum5_skew=mskew(Momentum1,window=5,biased=true,minPeriods=5) context by symbol;
        update wide_df set Momentum10_skew=mskew(Momentum1,window=10,biased=true,minPeriods=10) context by symbol;
        update wide_df set Momentum15_skew=mskew(Momentum1,window=15,biased=true,minPeriods=10) context by symbol;
        update wide_df set Momentum30_skew=mskew(Momentum1,window=30,biased=true,minPeriods=10) context by symbol;
        
        update wide_df set Momentum5_kurt=mkurtosis(Momentum1,window=5,biased=true,minPeriods=5) context by symbol;
        update wide_df set Momentum10_kurt=mkurtosis(Momentum1,window=10,biased=true,minPeriods=10) context by symbol;
        update wide_df set Momentum15_kurt=mkurtosis(Momentum1,window=15,biased=true,minPeriods=10) context by symbol;
        update wide_df set Momentum30_kurt=mkurtosis(Momentum1,window=30,biased=true,minPeriods=10) context by symbol;
        
        update wide_df set Momentum5_std=mstd(Momentum1,window=5,minPeriods=5) context by symbol;
        update wide_df set Momentum10_std=mstd(Momentum1,window=10,minPeriods=10) context by symbol;
        update wide_df set Momentum15_std=mstd(Momentum1,window=15,minPeriods=10) context by symbol;
        update wide_df set Momentum30_std=mstd(Momentum1,window=30,minPeriods=10) context by symbol;

        update wide_df set TurnoverRate10_std=mstd(TurnoverRate,window=10,minPeriods=10) context by symbol;
        update wide_df set TurnoverRate20_std=mstd(TurnoverRate,window=20,minPeriods=10) context by symbol;
        update wide_df set TurnoverRate30_std=mstd(TurnoverRate,window=30,minPeriods=10) context by symbol;
        update wide_df set TurnoverRate60_std=mstd(TurnoverRate,window=60,minPeriods=10) context by symbol;
        
        
        // Level-2 填充空缺值
        {nullFill_string}
        narrow_df=wide_df.unpivot(keyColNames=`symbol`date,valueColNames={self.factor_list});
        undef(`wide_df);
        
        // 窄表存储到因子数据库中
        counter=0;
        total_month_list=exec distinct(month(date)) as month from narrow_df;  
        total_month_list.sort();
        for (i in total_month_list){{
            slice_df=select * from narrow_df where month(date)=i;
            loadTable('{self.factor_database}','{self.factor_table}').append!(slice_df);
            counter=counter+1;
        }};
        undef(`narrow_df); // 填充空缺值
        """)


    def prepare_factor_data_long(self: ReturnModel_Backtest):
        """[自定义]Alpha因子计算并存储至因子数据库(宽表形式)"""
        self.session.run(f"""
        ds = repartitionDS(<select SecurityID as symbol,TradeDate as date,int(string(TradeTime.hourOfDay())+lpad(string(TradeTime.minuteOfHour()),2,"0")) as minute,
        open,high,low,close,volume,turnover from loadTable("dfs://MinuteKDB","stock_bar") where TradeDate between temporalAdd(date({self.start_dot_date}),-1,"d") and date({self.end_dot_date})>,`SecurityID); // 这里根据因子情况修改-1为状态函数的最大时间跨度
        
        // MapFunc
        def FactorMap(data){{
            pt = select * from data order by symbol,date,minute  // 正序排序
            // 新增最新一个交易日的frame[用于日频决策]
            last_pt=select temporalAdd(lastNot(date),1,"XSHG") as date,1500 as minute,NULL as open,NULL as high,NULL as low,NULL as close,NULL as volume,NULL as turnover from pt group by symbol; // 选取最后一个日期加一的形式作为决策的形式
            append!(pt,last_pt);
            undef(`last_pt);
            update pt set vwap = nullFill!(turnover/volume,0);
            
            // 分钟频时序因子
            update pt set momentum_5 = nullFill(prev((close - prev(close))/prev(close)),0) context by symbol;   
            //update pt set turnoverrate = prev(turnover_rate) context by symbol;
            //update pt set turnoverrate_std_20 = prev(mstd(turnover_rate,20,1)) context by symbol;
            update pt set volatility_20 = prev(mstd((close-prev(close))/prev(close),20,1)) context by symbol;
            update pt set high_break_5 = prev(close/mmax(high,5,1)-1) context by symbol;
            update pt set volume_spike_20 = prev(volume/mavg(volume,20)-1) context by symbol;
            update pt set volamount_corr_10 = prev(mcorr(turnover,volume,10)) context by symbol;
            // update pt set big_order_ratio = prev(turnover/(marketvalue*turnover_rate)) context by symbol;
            update pt set alpha_3 = prev(mcorr(rank(open),rank(close),10)) context by symbol;
            update pt set alpha_12 = prev(sign((volume - mfirst(volume, 2))) * (-1 * (close - mfirst(close, 2)))) context by symbol;
            update pt set alpha_14 = prev(ratios(close)-1-mfirst(ratios(close)-1,4)) context by symbol;
            update pt set draft = prev(high - mfirst(high, 3)) context by symbol;
            update pt set alpha_23 = prev(iif((msum(high, 20) \ 20 < high), -draft, 0)) context by symbol;
            dropColumns!(pt,`draft);
            update pt set alpha_26 = prev(-mmax(mcorr(mrank(volume, true, 5), mrank(high, true, 5), 5), 3)) context by symbol;
            update pt set alpha_41 = prev(pow(high * low, 0.5) - vwap) context by symbol;
            update pt set draft = prev((mfirst(close, 21) - mfirst(close, 11)) \ 10 - (mfirst(close, 11) - close) \ 10) context by symbol;
            update pt set alpha_46 = prev(iif(0.25 < draft, -1, iif(draft < 0, 1, (mfirst(close, 2) - close)))) context by symbol;
            dropColumns!(pt,`draft);
            update pt set draft = prev(((mfirst(close, 21) - mfirst(close, 11)) \ 10 - (mfirst(close, 11) - close) \ 10) < -0.1) context by symbol;
            update pt set alpha_49 = prev(iif(draft, 1, mfirst(close, 2) - close)) context by symbol;
            dropColumns!(pt,`draft);
            update pt set draft = (mfirst(close, 21) - mfirst(close, 11)) \ 10 - (mfirst(close, 11) - close) \ 10 < -0.05 context by symbol;
            update pt set alpha_51 = prev(iif(draft, 1, -(close - mfirst(close, 2)))) context by symbol;
            dropColumns!(pt,`draft);
            update pt set alpha_53 =  prev(-(((close - low) - (high - close)) \ (close - low) - mfirst(((close - low) - (high - close)) \ (close - low), 10))) context by symbol;
            update pt set alpha_54 = prev(-(low - close) * pow(open, 5) \ ((low - high) * pow(close, 5))) context by symbol;       
            
            // 分钟频截面因子(待补充)
            dropColumns!(pt,["open","high","low","close","volume","turnover","vwap"])
            
            // zscore
            
            // 截面空缺值填充
            for (col in {self.factor_list}){{
                pt[`draft] = pt[col]
                update pt set draft = nullFill(draft, avg(draft)) context by minute; // 取分钟频截面的均值对因子进行填充
                dropColumns!(pt,`draft);
            }}
            return pt
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

        // Symbol df processing
        symbol_df=select symbol,date,minute,open,close,marketvalue,state,industry from loadTable('{self.symbol_database}','{self.symbol_table}') where date between date({self.start_dot_date}) and temporalAdd(date({self.end_dot_date}),int({self.t}),"XSHG");
        
        // Index Component Constraint
        index_df = select component as symbol,date from loadTable("dfs://component","component_cn") where index == "000016" and date between date({self.start_dot_date}) and temporalAdd(date({self.end_dot_date}),int({self.t}),"XSHG"); // 上证50成分股信息
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
        
        // Factor df processing
        factor_list={self.factor_list}; 
        factor_pt=select symbol,date,{",".join(self.factor_list)} from loadTable("{self.factor_database}","{self.factor_table}") where date >= {self.start_dot_date};
        update factor_pt set minute = 1500;
        
        symbol_df=lj(symbol_df,factor_pt,`symbol`date`minute);
        undef(`factor_pt)
        
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
        InsertData(DBName="{self.combine_database}",TBName="{self.combine_table}",data=symbol_df,batchsize=1000000); 
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
