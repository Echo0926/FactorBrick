import re
import pandas as pd
import dolphindb as ddb
import msgpack as mp
import numpy as np
import tqdm
from joblib import Parallel,delayed
from src.utils import *
from typing import List

def gen_bar_data(snapshot_path:str, product: str, save_path:str, start_date=None, end_date=None):
    product_list = ["IC","IH","IF"]
    if product not in product_list:
        raise ValueError(f"product must be one of {product_list}")
    if not start_date:
        start_date="20100101"
    if not end_date:
        end_date="20300101"
    start_date = int(start_date)
    end_date = int(end_date)
    total_date_list = get_glob_list(path_dir=rf"{snapshot_path}\*")
    filter_date_list = [int(i) for i in total_date_list if start_date<=int(i)<=end_date]

    def processing(msg_list):
        total_df_list = []
        df_list = []
        prev_minute = 0
        for idx in range(len(msg_list)):
            # 字段信息
            msg = msg_list[idx]
            contract_name = msg[0]
            product_name = re.findall(r'[a-zA-Z]+', contract_name)[0]    # 从合约中提取品种名称
            minute = julian_time_convert(msg[1][-1])
            date = msg[2]
            price = msg[3]
            settle = msg[4]
            volume = msg[-4]
            turnover = msg[-3]
            high_limit = msg[-2]
            low_limit = msg[-1]

            df_dict = {"product": product_name,
                       "contract": contract_name,
                       "date": date,
                       "minute": minute,
                       "price": price,
                       "settle": settle,
                       "volume": volume,
                       "turnover": turnover,
                       "high_limit": high_limit,
                       "low_limit": low_limit
            }
            if idx == 0:
                prev_minute = minute

            # 合成Bar
            if minute != prev_minute:   # 说明切换到下一分钟的Bar了
                prev_minute = minute
                bar_df = pd.DataFrame(df_list)
                if not bar_df.empty:
                    total_df_list.append(
                        {
                            "product": bar_df["product"].iloc[0],
                            "contract": bar_df["contract"].iloc[0],
                            "date": bar_df["date"].iloc[0],
                            "minute": bar_df["minute"].iloc[0],
                            "open": bar_df["price"].iloc[0],
                            "high": bar_df["price"].max(),
                            "low": bar_df["price"].min(),
                            "close": bar_df["price"].iloc[-1],
                            "settle": bar_df["settle"].iloc[-1],
                            "volume": bar_df["volume"].iloc[-1] - bar_df["volume"].iloc[0],
                            "turnover": bar_df["turnover"].iloc[-1] - bar_df["turnover"].iloc[0],
                            "high_limit": bar_df["high_limit"].iloc[0],
                            "low_limit": bar_df["low_limit"].iloc[0],
                        }
                    )
                df_list = []    # 清空当前的df_list
            else:
                df_list.append(df_dict)
                prev_minute = minute

        return pd.DataFrame(total_df_list)
    def pipeline(date_str):
        target_file_list = get_glob_list(rf"{snapshot_path}\{date_str}\*{product}*")    # 对应的品种snapshot数据文件
        data_list = []
        for file in target_file_list:
            with open(rf"{snapshot_path}\{date_str}\{file}","rb+") as f:
                data = mp.unpack(f)
            data = processing(data)
            data_list.append(data)
        init_path(path_dir=rf"{save_path}\{date_str}")
        if data_list:   # 如果当前有该品种的Bar Data
            data = pd.concat(data_list,ignore_index=True,axis=0)
            data.to_parquet(rf"{save_path}\{date_str}\{product}.pqt")


    Parallel(n_jobs=-1)(delayed(pipeline)(date_str) for date_str in tqdm.tqdm(
        filter_date_list,desc="processing"))

def add_bar_data(session:ddb.session, pool, bar_path, start_date, end_date, save_database, save_table):
    """
    添加Bar Data至DolphinDB数据库中
    """
    # 创建数据库
    if not session.existsTable(save_database, save_table):
        session.run(f"""
        db=database("{save_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
        schemaTb=table(1:0,`product`contract`date`minute`open`high`low`close`settle`volume`turnover`high_limit`low_limit,[SYMBOL,SYMBOL,DATE,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE]);
        t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns="date",sortColumns=["product","contract","date","minute"],keepDuplicates=LAST)
        """)
    appender = ddb.PartitionedTableAppender(dbPath=save_database,
                                            tableName=save_table,
                                            partitionColName="date",
                                            dbConnectionPool=pool)  # 写入数据的appender

    current_ts_list= get_date_list(path=bar_path,begin_date=start_date,end_date=end_date)
    for date in tqdm.tqdm(current_ts_list, desc="appending bar data"):
        df =pd.read_parquet(rf"{bar_path}\{date}")
        if not df.empty:
            df=df[["product","contract","date","minute","open","high","low","close","settle","volume","turnover","high_limit","low_limit"]]
            df["date"]=df["date"].apply(str).apply(pd.Timestamp)
            df["minute"]=df["minute"].apply(int)
            for col in ["open","high","low","close","settle","volume","turnover","high_limit","low_limit"]:
                df[col] = df[col].astype(float)
            appender.append(df)




if __name__== "__main__":
    session=ddb.session()
    session.connect("localhost",8848,"admin","123456")
    pool=ddb.DBConnectionPool("localhost",8848,10,"admin","123456")

    gen_bar_data(
        snapshot_path=r"D:\MXY\t0_ih_backtest\factorbrick\data\snapshot",
        product="IH",
        save_path=r"D:\MXY\t0_ih_backtest\factorbrick\data\bar",
        start_date="20230101",
        end_date="20250416"
    )

    gen_bar_data(
        snapshot_path=r"D:\MXY\t0_ih_backtest\factorbrick\data\snapshot",
        product="IC",
        save_path=r"D:\MXY\t0_ih_backtest\factorbrick\data\bar",
        start_date="20230101",
        end_date="20250416"
    )

    add_bar_data(session, pool, r"D:\MXY\t0_ih_backtest\factorbrick\data\bar","20200101","20300101",
                 save_database="dfs://future_cn/value",save_table="market")


