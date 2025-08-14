import os
import pandas as pd
import numpy as np
from src.utils import *

def config_time(self):
    """
    自动检测回测时间并配置参数
    """
    def get_time(k_time, k_date_list):
        k_time = k_time.drop_duplicates(subset=['date', 'timestamp'], keep="first").sort_values(
            by=['timestamp']).reset_index(drop=True)

        # 确定最终的回测日期
        total_date_list = trans_time(
            get_ts_list(start_date=self.start_date,
                        end_date=self.end_date,
                        to_current=False,
                        freq='D',
                        cut_weekend=True), "timestamp"
        )  # 所选回测周期的所有的交易时间
        date_list = sorted([i for i in k_date_list if i in total_date_list])
        dot_date_list = [i.strftime('%Y.%m.%d') for i in date_list]  # 2020.01.01
        str_date_list = [f"{i.replace('.', '')}" for i in dot_date_list]  # 20200101
        return date_list, dot_date_list, str_date_list

    # 配置时间 -> 确定stock_date/future_date/option_date
    if self.run_stock:
        if self.data_type == "dolphindb":
            k_date_list = trans_time(self.session.run(
                f"""select distinct(date) as date from loadTable('{self.stock_K_database}','{self.stock_K_table}')"""
            )['date'].tolist(), "timestamp")  # 行情数据库中所有的交易时间
            k_time = self.session.run(
                f"""select date,timestamp from loadTable('{self.stock_K_database}','{self.stock_K_table}')"""
            )
        else:
            k_date_list = sorted(set(
                pd.read_parquet(f"{self.stock_K_database}/{self.stock_K_table}.pqt", columns=["date"])[
            "date"].tolist()))
            k_date_list = trans_time(k_date_list, "timestamp")
            k_time = pd.read_parquet(f"{self.stock_K_database}/{self.stock_K_table}.pqt",
                                     columns=["date", "timestamp"])

        date_list, dot_date_list, str_date_list = get_time(k_time, k_date_list)
        self.stock_date_list = date_list
        self.stock_dot_date_list = dot_date_list
        self.stock_str_date_list = str_date_list

        # 设置每天的日内分钟(笛卡尔积)
        k_timestamp_dict = {}
        for date in date_list:
            slice_df = k_time[k_time['date'] == date].reset_index(drop=True)
            k_timestamp_dict[date] = sorted(set(slice_df["timestamp"].tolist()))
        self.stock_timestamp_dict = k_timestamp_dict

    if self.run_future:
        if self.data_type == "dolphindb":
            k_date_list = trans_time(self.session.run(
                f"""select distinct(date) as date from loadTable('{self.future_K_database}','{self.future_K_table}')"""
            )['date'].tolist(), "timestamp")  # 行情数据库中所有的交易时间
            k_time = self.session.run(
                f"""select date,timestamp from loadTable('{self.future_K_database}','{self.future_K_table}')"""
            )
        else:
            k_date_list = sorted(set(pd.read_parquet(f"{self.future_K_database}/{self.future_K_table}.pqt",columns=["date"])["date"].tolist()))
            k_date_list = trans_time(k_date_list, "timestamp")
            k_time = pd.read_parquet(f"{self.future_K_database}/{self.future_K_table}.pqt",columns=["date","timestamp"])

        date_list, dot_date_list, str_date_list = get_time(k_time, k_date_list)

        self.future_date_list = date_list
        self.future_dot_date_list = dot_date_list
        self.future_str_date_list = str_date_list

        # 设置每天的日内分钟(笛卡尔积)
        k_timestamp_dict = {}
        for date in date_list:
            slice_df=k_time[k_time['date']==date].reset_index(drop=True)
            k_timestamp_dict[date] = sorted(set(slice_df["timestamp"].tolist()))
        self.future_timestamp_dict = k_timestamp_dict

    if self.run_option:
        if self.data_type == "dolphindb":
            k_date_list = trans_time(self.session.run(
                f"""select distinct(date) as date from loadTable('{self.option_K_database}','{self.option_K_table}')"""
            )['date'].tolist(), "timestamp")  # 行情数据库中所有的交易时间
            k_time = self.session.run(
                f"""select date,timestamp from loadTable('{self.option_K_database}','{self.option_K_table}')"""
            )
        else:
            k_date_list = sorted(set(
                pd.read_parquet(f"{self.option_K_database}/{self.option_K_table}.pqt", columns=["date"])[
                    "date"].tolist()))
            k_date_list = trans_time(k_date_list, "timestamp")
            k_time = pd.read_parquet(f"{self.option_K_database}/{self.option_K_table}.pqt",
                                     columns=["date", "timestamp"])

        date_list, dot_date_list, str_date_list = get_time(k_time, k_date_list)

        self.option_date_list = date_list
        self.option_dot_date_list = dot_date_list
        self.option_str_date_list = str_date_list

        # 设置每天的日内分钟(笛卡尔积)
        k_timestamp_dict = {}
        for date in date_list:
            slice_df = k_time[k_time['date'] == date].reset_index(drop=True)
            k_timestamp_dict[date] = sorted(set(slice_df["timestamp"].tolist()))
        self.option_timestamp_dict = k_timestamp_dict

    # 合并不同标的的date_list并进行排序
    self.date_list = sorted(list(set(self.stock_date_list + self.future_date_list + self.option_date_list)))
    self.dot_date_list = [i.strftime('%Y.%m.%d') for i in self.date_list]
    self.str_date_list = [f"{i.replace('.', '')}" for i in self.dot_date_list]

def counter_init(self):
    """
    初始化柜台
    """
    def del_json(path):
        if os.path.exists(path):
            date_list = get_glob_list(path_dir=rf"{path}\*")
            for date in date_list:
                file_list = get_glob_list(rf"{path}\{date}\*")
                if not file_list:
                    os.removedirs(rf"{path}\{date}")
                else:
                    for file in file_list:
                        os.remove(rf"{path}\{date}\{file}")

    if self.run_stock: # 如果运行股票策略【默认删除上次的柜台json文件夹】
        # del_json(self.stock_counter_json)
        del_json(self.stock_signal_json)

    if self.run_future: # 如果运行期货策略【默认删除上次的柜台json文件夹】
        del_json(self.future_counter_json)
        del_json(self.future_signal_json)

    if self.run_option:  # 如果运行期权策略【默认删除上次的柜台json文件夹】
        del_json(self.option_counter_json)
        del_json(self.option_signal_jsoni)

def counter_start(self):
    """
    准备行情数据与信号数据parquet,输出为json
    """
    # Step1.准备数据
    if self.run_stock:
        if self.data_type == "dataframe":
            # # # 行情数据
            # df = pd.read_parquet(rf"{self.stock_K_database}\{self.stock_K_table}.pqt")
            # df["date"] = df["date"].apply(pd.Timestamp)
            # df = df[(df["date"]>=pd.Timestamp(self.start_dot_date)) & (df["date"]<=pd.Timestamp(self.end_dot_date))].reset_index(drop=True)
            # df = df[["date","minute","timestamp","symbol","open","high","low","close","volume"]]
            # write_k_json(df,date_col="date",symbol_col="symbol",save_path=self.stock_counter_json,index_col="minute")

            # 信号数据
            df = pd.read_parquet(rf"{self.stock_signal_database}\{self.stock_signal_table}.pqt")
            df["date"] = df["date"].apply(pd.Timestamp)
            df = df[(df["date"]>=pd.Timestamp(self.start_dot_date)) & (df["date"]<=pd.Timestamp(self.end_dot_date))].reset_index(drop=True)
            for col in ["open","high","low"]:
                df[col] = df[col].fillna(df["close"])
            df = df[["date","minute","timestamp","symbol","open","high","low","close","volume","buy_signal","sell_signal"]]
            write_k_json(df,date_col="date",symbol_col="symbol",save_path=self.stock_signal_json,index_col="minute")

        elif self.data_type == "dolphindb":
            # 行情数据
            df = self.session.run(f"""
                pt=select date,minute,timestamp,symbol,open,high,low,close,volume from loadTable("{self.stock_K_database}","{self.stock_K_table}") where date between date({self.start_dot_date}) and date({self.end_dot_date}); 
                update pt set open = nullFill(open, close);
                update pt set low = nullFill(low, close);
                update pt set high = nullFill(high, close);       
                pt         
            """)
            write_k_json(df,date_col="date",symbol_col="symbol",save_path=self.stock_counter_json,index_col="minute")

            # 信号数据
            df = self.session.run(f"""
                pt=select date,minute,timestamp,symbol,open,high,low,close,volume,buy_signal,sell_signal from loadTable("{self.stock_K_database}","{self.stock_K_table}") where date between date({self.start_dot_date}) and date({self.end_dot_date}); 
                update pt set open = nullFill(open, close);
                update pt set low = nullFill(low, close);
                update pt set high = nullFill(high, close);   
                pt             
            """)
            write_k_json(df,date_col="date",symbol_col="symbol",save_path=self.stock_signal_json,index_col="minute")

    if self.run_future:
        if self.data_type == "dataframe":
            pass
            # 行情数据
            df = pd.read_parquet(rf"{self.future_K_database}\{self.future_K_table}.pqt")
            df["date"] = df["date"].apply(pd.Timestamp)
            df = df[(df["date"]>=pd.Timestamp(self.start_dot_date)) & (df["date"]<=pd.Timestamp(self.end_dot_date))].reset_index(drop=True)
            for col in ["open","high","low","close"]:
                df[col] = df[col].fillna(df["settle"])
            df = df[["date","minute","timestamp","contract","open","high","low","close","volume"]]
            write_k_json(df,date_col="date",symbol_col="contract",save_path=self.future_counter_json,index_col="minute")

            # 信号数据
            df = pd.read_parquet(rf"{self.future_signal_database}\{self.future_signal_table}.pqt")
            df["date"] = df["date"].apply(pd.Timestamp)
            df = df[(df["date"]>=pd.Timestamp(self.start_dot_date)) & (df["date"]<=pd.Timestamp(self.end_dot_date))].reset_index(drop=True)
            for col in ["open","high","low","close"]:
                df[col] = df[col].fillna(df["settle"])
            df = df[["date","minute","timestamp","contract","open","high","low","close","open_long","open_short","close_long","close_short"]]
            write_k_json(df,date_col="date",symbol_col="contract",save_path=self.future_signal_json,index_col="minute")

        elif self.data_type == "dolphindb":
            # 行情数据 並ぶ
            df = self.session.run(f"""
                pt=select date,minute,timestamp,contract,open,high,low,close,volume from loadTable("{self.future_K_database}","{self.future_K_table}") where date between date({self.start_dot_date}) and date({self.end_dot_date}); 
                update pt set open = nullFill(open, settle);
                update pt set low = nullFill(low, settle);
                update pt set high = nullFill(high, settle); 
                update pt set settle = nullFill(close, settle);
                pt               
            """)
            write_k_json(df,date_col="date",symbol_col="contract",save_path=self.future_counter_json,index_col="minute")

            # 信号数据　
            df = self.session.run(f"""
                pt=select date,minute,timestamp,contract,open,high,low,close,volume,open_long,open_short,close_long,close_short from loadTable("{self.future_signal_database}","{self.future_signal_table}") where date between date({self.start_dot_date}) and date({self.end_dot_date}); 
                update pt set open = nullFill(open, settle);
                update pt set low = nullFill(low, settle);
                update pt set high = nullFill(high, settle); 
                update pt set settle = nullFill(close, settle);    
                pt                  
            """)
            write_k_json(df,date_col="date",symbol_col="contract",save_path=self.future_signal_json,index_col="minute")

    if self.run_option:
        if self.data_type == "dataframe":
            # 行情数据
            df = pd.read_parquet(rf"{self.option_K_database}\{self.option_K_table}.pqt")
            df["date"] = df["date"].apply(pd.Timestamp)
            df = df[(df["date"]>=pd.Timestamp(self.start_dot_date)) & (df["date"]<=pd.Timestamp(self.end_dot_date))].reset_index(drop=True)
            for col in ["open","high","low","close"]:
                df[col] = df[col].fillna(df["settle"])
            df = df[["date","minute","timestamp","option","open","high","low","close","settle","volume"]]
            write_k_json(df,date_col="date",symbol_col="option",save_path=self.option_counter_json, index_col="minute")

            # 信号数据
            df = pd.read_parquet(rf"{self.option_signal_database}\{self.option_signal_table}.pqt")
            df["date"] = df["date"].apply(pd.Timestamp)
            df = df[(df["date"]>=pd.Timestamp(self.start_dot_date)) & (df["date"]<=pd.Timestamp(self.end_dot_date))].reset_index(drop=True)
            for col in ["open","high","low","close"]:
                df[col] = df[col].fillna(df["settle"])
            df = df[["date","minute","timestamp","contract","open","high","low","close","buy_call","buy_put","sell_call","sell_put"]]
            write_k_json(df,date_col="date",symbol_col="contract",save_path=self.future_signal_json,index_col="minute")

        elif self.data_type == "dolphindb":
            self.session.run(f"""
                pt=select date,minute,timestamp,option,open,high,low,close,volume,open_long,open_short,close_long,close_short from loadTable("{self.option_signal_database}","{self.option_signal_table}") where date between date({self.start_dot_date}) and date({self.end_dot_date}); 
                update pt set open = nullFill(open, settle);
                update pt set low = nullFill(low, settle);
                update pt set high = nullFill(high, settle); 
                update pt set settle = nullFill(close, settle);    
                pt                            
            """)

