import os
import dolphindb as ddb
from ReturnModel_light import SingleFactorBackTest

def Combine(self: SingleFactorBackTest):
    # 定义函数
    self.session.run(rf"""
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
    }};
        
    def ReturnCalFunc(df, periodSize, idCol, timeCol, openCol, closeCol){{
        /* 不同周期的收益率计算函数 */
        expr = "(move(close,-periodSize)-open)/open"
        expr = strReplace(
                    strReplace(
                        strReplace(expr, "close", closeCol),
                    "open", openCol),
                "periodSize",
            string(periodSize))
        return sql(select=[sqlCol(idCol),sqlCol(timeCol),
            sqlColAlias(parseExpr(expr),`Ret+string(periodSize))],
            from=df, groupBy=sqlCol(idCol), groupFlag=0).eval()
    }};
    """)

    # 拿数据
    self.session.run(rf"""
    // 最后的数据格式: symbol,TradeDate,TradeTime,BarReturn,FutureReturn,factor_list
    
    // 行情数据(若需要Benchmark可自行添加)
    symbol_df = select symbol,TradeDate,TradeTime,
                concatDateTime(TradeDate,TradeTime) as timeForCal,
                open,close // ,marketvalue,industry 
                from loadTable("{self.symbol_database}","{self.symbol_table}") 
                where TradeDate between date({self.start_dot_date}) and date({self.end_dot_date});
    sortBy!(symbol_df,`timeForCal);
    symbol_list = exec distinct(symbol) from symbol_df;
    
    idCol = `symbol;
    timeCol = `timeForCal;
    openCol = `open;
    closeCol = `close;
    returnIntervals = {self.returnIntervals};
    ReturnCalFunc_ = ReturnCalFunc{{symbol_df, , idCol, timeCol, openCol, closeCol}};
    res_list = peach(ReturnCalFunc_, returnIntervals) // peach并行计算, 列名为Ret+returnInterval
    returnCol = `Ret+string(returnIntervals) // 未来区间收益率列名列表['Ret1',Ret5...]
    matchingCols = [idCol, timeCol]
    for (i in 0..(size(returnIntervals)-1)){{
        interval = returnIntervals[i]
        symbol_df = lj(symbol_df, res_list[i], matchingCols)
    }}
    update symbol_df set barReturn = next(close)-close\close context by symbol;
    
    // 因子数据
    factor_list = {self.factor_list};
    if ({int(self.dailyFreq)}==1){{
        factor_df = select value from loadTable("{self.factor_database}","{self.factor_table}") 
            where (date between {self.start_dot_date} and {self.end_dot_date}) 
            and factor in {self.factor_list} and symbol in symbol_list
            pivot by date as TradeDate,symbol,factor;
        update factor_df set TradeTime = 15:00:00.000;
        matchingCols = ["symbol","TradeDate"];
    }}
    else{{
        factor_df = select value from loadTable("{self.factor_database}","{self.factor_table}") 
            where (date between {self.start_dot_date} and {self.end_dot_date}) 
            and factor in {self.factor_list} and symbol in symbol_list
            pivot by date as TradeDate, time as TradeTime,symbol,factor;
        matchingCols = ["symbol","TradeDate","TradeTime"]
    }}
    symbol_df = lsj(symbol_df, factor_df, matchingCols);
    
    // 添加period
    time_list = sort(exec distinct(timeForCal) from symbol_df, true)
    time_dict = dict(time_list, cumsum(1..size(time_list)))
    symbol_df[`period] = int(time_dict[symbol_df[`timeForCal]]);
    
    // 最终数据
    totalData = sql(select=sqlCol(`symbol`TradeDate`TradeTime`barReturn).append!(sqlCol(returnCol)).append!(sqlCol(factor_list)).append!(sqlCol(`period)),
                    from=symbol_df).eval()
    InsertData("{self.combine_database}", "{self.combine_table}", totalData, batchsize=1000000);
        
    // 添加至模板数据库
    template_pt = select first(timeForCal.date()) as startDate,
                        first(timeForCal.time()) as startTime,
                        last(timeForCal.date()) as endDate,
                        last(timeForCal.time()) as endTime
                from symbol_df group by period;
    loadTable("{self.combine_database}","{self.template_table}").append!(template_pt);
    template_ind = select first(timeForCal.date()) as TradeDate,
                        first(timeForCal.time()) as TradeTime
                    from symbol_df group by symbol, period;
    loadTable("{self.combine_database}","{self.template_individual_table}").append!(template_ind);
    """)
