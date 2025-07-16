import pandas as pd
import numpy as np
import dolphindb as ddb
import sys
import tqdm
sys.path.append(r"E:\苗欣奕的东西\行研宝\func\future_cn_func")
from future_cn_basic import *

def get_signal(session,K_database,K_table,
               save_database,save_table,params=None):
    """
    信号值生成
    """
    if session.existsTable(save_database,save_table):
        session.dropTable(dbPath=save_database,tableName=save_table)
    if not session.existsTable(save_database,save_table):
        session.run(f"""
         db=database("{save_database}",RANGE, 2000.01M+(0..30)*12, engine="TSDB")
                    schemaTb=table(1:0,`product`date`contract`long_signal`short_signal,[SYMBOL,DATE,SYMBOL,DOUBLE,DOUBLE])
                    t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns="date",sortColumns=["product","contract","date"],keepDuplicates=LAST)
        """)
    K_data=session.run(f"""
    // 选择品种(剔除股指期货和利率期货)
    pt=select product,date,contract,pre_settle,open,high,low,close,settle,南华价格指数 as nh_index,多单持仓总计_top5.ffill() as long5,多单持仓变动_top5 as long5_chg,空单持仓总计_top5.ffill() as short5,空单持仓变动_top5 as short5_chg,isMainContract from loadTable('{K_database}','{K_table}');
    df=select date,即期汇率_美元兑人民币 as ex_settle,nullFill(即期汇率_美元兑人民币/prev(即期汇率_美元兑人民币)-1,0) as ex_return from loadTable('dfs://macro_cn/value','monetary') where date>=2010.01.01;
    pt=select * from pt where product in [`AU];
    pt=select * from df left join pt on pt.date=df.date; 
    
    // 构建品种多空情绪指标(前20大多单多还是空单多)
    update pt set emotion=mavg(long5,20)-mavg(short5,20)    
    
    // K线数据合成信号
    update pt set isMainContract=0 where isMainContract=2;  // 这里信号限定为主力合约
    update pt set pct_chg=(settle-pre_settle)/pre_settle*signum(isMainContract) context by contract;  //日度涨跌幅
    update pt set pct_chg5=(settle-move(settle,5))/move(settle,5)*signum(isMainContract) context by contract; //周度涨跌幅
    update pt set nh_chg=nh_index-ma(nh_index,5,1) context by contract;   // 南华指数是否破了周均值(采用简单MA计算)
    update pt set ex_chg=ex_settle-ma(ex_settle,15,1) context by contract;   // 汇率是否破了半个月均值(采用简单MA计算)
    update pt set ex_chg=0 where isNull(ex_chg)   // fillna

    update pt set short_signal=0    // 超涨会跌
    update pt set short_signal=1 where date=weekBegin(date) and emotion<0 and prev(ex_chg)<0 and move(ex_chg,2)<0 and signum(isMainContract)=1 context by contract; // 上一个交易日的数据判断(避免未来函数)
    update pt set long_signal=0     // 超跌会涨
    update pt set long_signal=1 where date=weekBegin(date) and emotion>0 and prev(ex_chg)>0 and move(ex_chg,2)>0 and signum(isMainContract)=1 context by contract; // 上一个交易日的数据判断(避免未来函数)
    pt=select product,date,contract,long_signal,short_signal from pt;
    append!(loadTable('{save_database}','{save_table}'),pt);
    pt
    """)
    return K_data

def strategy(self):
    # 0.Prepare Time【固定】
    K_ts_list=trans_time(self.session.run(f"""select distinct(date) as date from loadTable('{self.K_database}','{self.K_table}')""")['date'].tolist(),"timestamp")    # 所有交易时间
    total_ts_list=trans_time(get_ts_list(start_date=self.start_date,end_date=self.end_date,to_current=False,freq='D',cut_weekend=True),"timestamp")
    current_ts_list=[i for i in K_ts_list if i in total_ts_list]  # pd.Timestamp('20240101')
    current_ts_list.sort()
    str_ts_list=[i.strftime('%Y.%m.%d') for i in current_ts_list]   # 20240101
    dot_ts_list=[f"{i[:4]}{i[4:6]}{i[6:]}" for i in str_ts_list]   # 2024.01.01

    # 1.Prepare Data
    def get_last(contract_str):
        """设置合约的最后一个交易日(最大最大为10号,再往后可能没有行情平不了单)"""
        contract_str=''.join(filter(str.isdigit,contract_str))
        year,month,date=contract_str[:2],contract_str[2:],"05"
        return pd.Timestamp(f"20{year}{month}{date}")

    # 1.Iteration
    for current_ts,str_ts,dot_ts in tqdm.tqdm(zip(current_ts_list,str_ts_list,dot_ts_list),total=len(current_ts_list)):
        self.current_date=current_ts
        self.current_str_date=str_ts
        self.current_dot_date=dot_ts
        decision_df=self.session.run(f"""
        Kdata_df=select contract,open,high,low,close,pre_settle,nullFill(交易保证金率/100,0) as margin,合约乘数	as multi from loadTable('{self.K_database}','{self.K_table}') where date=date({self.current_dot_date}) and not isNull(open) and not isNull(close);
        signal_df=select contract,long_signal,short_signal from loadTable('{self.signal_database}','{self.signal_table}') where date=date({self.current_dot_date});
        df=select * from Kdata_df left join signal_df on Kdata_df.contract==signal_df.contract;
        df""")
        # 每日监控限价仓单
        self.monitor(order_type='long',order_sequence=True)
        self.monitor(order_type='short',order_sequence=False)

        for idx,row in decision_df.iterrows():
            contract,long_signal,short_signal,open_price,high_price,low_price,close_price,margin,multi=\
                row['contract'],row['long_signal'],row['short_signal'],row['open'],row['high'],row['low'],row['close'],row['margin'],row['multi']

            """每日开多/开空"""
            if long_signal==1 or short_signal==1:
                if long_signal==1:  # 开多
                    if contract not in self.long_position.keys():
                        self.execute(order_type='long',contract=contract,price=open_price,vol=5*multi,margin=0,
                                    max_price=1.045*open_price,min_price=0.985*open_price,max_date=min(pd.Timestamp(self.current_date)+pd.Timedelta(days=15),get_last(contract_str=contract)-pd.Timedelta(days=15)))    # 盈亏比3:1
                if short_signal==1: # 开空
                    if contract not in self.short_position.keys():
                        self.execute(order_type='short',contract=contract,price=open_price,vol=5*multi,margin=0,
                                     max_price=1.015*open_price,min_price=0.955*open_price,max_date=min(pd.Timestamp(self.current_date)+pd.Timedelta(days=15),get_last(contract_str=contract)-pd.Timedelta(days=15)))    # 盈亏比3:1
        """统计"""
        self.profit_Dict[self.current_date]=self.profit
        self.cash_Dict[self.current_date]=self.cash
    return self
