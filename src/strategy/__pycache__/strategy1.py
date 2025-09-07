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
    if session.existsDatabase(save_database):
        session.dropDatabase(save_database)
    if not session.existsTable(save_database,save_table):
        session.run(f"""
         db=database("{save_database}",RANGE, 2000.01M+(0..30)*12, engine="TSDB")
                    schemaTb=table(1:0,`product`date`contract`long_signal`short_signal,[SYMBOL,DATE,SYMBOL,DOUBLE,DOUBLE])
                    t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns="date",sortColumns=["product","contract","date"],keepDuplicates=LAST)
        """)
    K_data=session.run(f"""
    // 选择品种(剔除股指期货和利率期货)
    pt=select product,date,contract,pre_settle,open,high,low,close,settle,南华价格指数 as nh_index,isMainContract from loadTable('{K_database}','{K_table}');
    pt=select * from pt where product!=`IM and product!=`IH and product!=`IF and product!=`IC and product in [`EG`SC`TA`MA`PX`V`PP`EB`PG`SC`BC`LU`BU`BR]
    
    // K线数据合成信号
    update pt set isMainContract=0 where isMainContract=2;  // 这里信号限定为主力合约
    update pt set pct_chg=(settle-pre_settle)/pre_settle*signum(isMainContract) context by contract;  //日度涨跌幅
    update pt set nh_chg=ma((nh_index-move(nh_index,1))/nh_index,10,1)   // 南华指数日度涨跌幅10日均值(采用简单MA计算)
    update pt set nh_chg=0 where isNull(nh_chg)   // fillna
    update pt set rank_True=rank(pct_chg,true) from pt context by date;   // 日度涨幅
    update pt set rank_False=rank(pct_chg,false) from pt context by date;   // 日度跌幅
    update pt set pct_chg5=(settle-move(settle,5))/move(settle,5)*signum(isMainContract) context by contract; //周度涨跌幅
    update pt set rank5_True=rank(pct_chg5,true) from pt context by date;   // 涨幅排行榜
    update pt set rank5_False=rank(pct_chg5,false) from pt context by date;   // 跌幅排行榜
    update pt set short_signal=0    // 超涨会跌
    update pt set short_signal=1 where prev(rank5_True)<=5 and date=weekBegin(date) and prev(nh_chg)>0 and signum(isMainContract)=1 context by contract; // 上一个交易日的数据判断(避免未来函数)
    update pt set long_signal=0     // 超跌会涨
    update pt set long_signal=1 where prev(rank5_False)<=5 and date=weekBegin(date) and prev(nh_chg)<0 and signum(isMainContract)=1 context by contract; // 上一个交易日的数据判断(避免未来函数)
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
        Kdata_df=select contract,open,high,low,close,settle,nullFill(交易保证金率/100,0) as margin,合约乘数	as multi from loadTable('{self.K_database}','{self.K_table}') where date=date({self.current_dot_date}) and not isNull(open) and not isNull(close);
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
                        self.execute(order_type='long',contract=contract,price=open_price,vol=2*multi,margin=0,
                                    max_price=1.15*open_price,min_price=0.95*open_price,max_date=pd.Timestamp(self.current_date)+pd.Timedelta(days=5))    # 盈亏比3:1
                if short_signal==1:
                    if contract not in self.short_position.keys():
                        self.execute(order_type='short',contract=contract,price=open_price,vol=2*multi,margin=0,
                                     max_price=1.05*open_price,min_price=0.85*open_price,max_date=pd.Timestamp(self.current_date)+pd.Timedelta(days=5))    # 盈亏比3:1
        """统计"""
        self.profit_Dict[self.current_date]=self.profit
        # print(self.current_date,self.profit,self.cash)
    return self
