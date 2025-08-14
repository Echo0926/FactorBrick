import json, json5
import pandas as pd
import numpy as np
import msgpack as mp
import os
from typing import Dict,List
from src.utils import write_json, julian_time_convert, init_path
import tqdm
from joblib import Parallel,delayed

def get_ts_list( file_path, start_date, end_date) ->List[int]:
    date_list = os.listdir(path=file_path)
    res_list = sorted([int(i) for i in date_list if int(start_date)<=int(i)<=int(end_date)])
    return res_list

def get_symbol(df: pd.DataFrame, global_config:Dict, default_max_date: str = "20300101",
               ) -> Dict:
    """将成分股DataFrame的转化为成分股json信息
    字段说明：
    S_INFO_WINDCODE：?
    S_CON_WINDCODE： 成分股代码
    S_CON_INDATE：进入日期
    S_CON_OUTDATE：调出日期
    CUR_SIGN：是否当前是成分股的0-1虚拟变量
    输出格式:
    {
        symbol:[start_date,end_date]
    } -> {date: [symbol_list]}
    """
    # Dict comprehension
    info_dict = {}
    for _,row in df.iterrows():
        symbol = row["S_CON_WINDCODE"]
        start_date, end_date = row["S_CON_INDATE"], row["S_CON_OUTDATE"]
        start_date = int(start_date)
        end_date = int(end_date) if not np.isnan(end_date) else default_max_date
        if symbol not in info_dict:
            info_dict[symbol] = [(start_date,end_date)]
        else:
            info_dict[symbol].append((start_date,end_date))

    # 生成对应时间戳
    remote_path = global_config["data_params"]["k_data_remote_path"]
    start_date, end_date = int(global_config["data_params"]["start_date"]), int(global_config["data_params"][
        "end_date"])
    total_ts_list = sorted(get_ts_list(remote_path, start_date, end_date))

    res_dict = {}
    for date in total_ts_list:
        res_dict[date] = []
        for symbol,date_list in info_dict.items():
            for start_date,end_date in date_list:
                if int(start_date) <= int(date) <= int(end_date):
                    res_dict[date].append(symbol)

    return res_dict

def gen_k_data(symbol_cfg: Dict, global_config: Dict, date_str:str):
    """生成K线并存放在本地
    msgpack文件 minute symbol [], pre_close open high low close volume turnover
    """
    remote_path = global_config["data_params"]["k_data_remote_path"]
    if date_str not in symbol_cfg.keys():   # 说明缺失指数成分股
        return None
    symbol_list = symbol_cfg[date_str]

    total_df_list = []
    for symbol in symbol_list:
        symbol = symbol.replace(".SH","").replace(".SZ","")
        msg_path = rf"{remote_path}{date_str}\BAR_1M_{symbol}_{date_str}.msgPack"
        if os.path.exists(msg_path):
            total_data = mp.unpack(open(msg_path, 'rb'))
            for data in total_data:
                total_df_list.append({"date":date_str,
                                      "minute":julian_time_convert(data[1][-1]),
                                      "symbol":data[0],
                                      "pre_close":data[2],
                                      "open":data[3],
                                      "close": data[4],
                                      "high":data[5],
                                      "low":data[6],
                                      "volume":data[7],
                                      "turnover":data[8]
                                      })
    res_df= pd.DataFrame(total_df_list).reset_index(drop=True)

    # 保存pqt数据至本地
    save_path = global_config["data_params"]["k_data_save_path"]
    init_path(path_dir=save_path)
    init_path(path_dir=fr"{save_path}\{date_str}")
    res_df.to_parquet(rf"{save_path}\{date_str}\data.pqt",index=False)

def gen_nbr_data(symbol_config:Dict, global_config: Dict, date_str:str):
    # 获取指定symbol_list
    symbol_list = symbol_config["symbol_list"]

    # 读取remote nbr data
    nbr_path = global_config["data_params"]["nbr_data_remote_path"]
    res_df = pd.read_parquet(fr"{nbr_path}\{date_str}").rename(columns={"curTime":"minute"})
    res_df["minute"] = (res_df["minute"]/100000).astype(str)

    # filter`
    res_df = res_df[res_df["symbol"].isin(symbol_list)].reset_index(drop=True)

    # 保存pqt数据至本地
    save_path = global_config["data_params"]["nbr_data_save_path"]
    init_path(path_dir=fr"{save_path}\{date_str}")
    res_df.to_parquet(rf"{save_path}\{date_str}\data.pqt",index=False)

if __name__ == "__main__":
    # Step1. 给定成分股与配置, 生成symbol.json
    with open(r"global.json5", encoding="utf-8") as file:
        global_cfg = json5.load(file)
    symbol_data = pd.read_csv(r"D:\MXY\t0_ih_backtest\factorbrick\src\data_func\index50_cons.csv",
                              index_col=None, header=0)
    symbol_json = get_symbol(symbol_data,global_cfg,default_max_date="20250501")
    from itertools import chain
    total_symbol_list = sorted(set(list(chain.from_iterable(symbol_json.values()))))
    for key in symbol_json.keys():
        symbol_json[key] = total_symbol_list    # 笛卡尔积
    write_json(symbol_json,r"D:\MXY\t0_ih_backtest\factorbrick\src\data_func/","index50_symbol.json")

    # Step2. 更新symbol_list至 symbol.json中
    from itertools import chain
    symbol_cfg = {"symbol_list": sorted(set(list(chain.from_iterable(symbol_json.values()))))}
    write_json(symbol_cfg,r"D:\MXY\t0_ih_backtest\factorbrick\src\data_func/","symbol.json")

    Parallel(n_jobs=8)(delayed(gen_k_data)(symbol_json,global_cfg,date_str) for date_str in tqdm.tqdm(
            get_ts_list(global_cfg["data_params"]["k_data_remote_path"],
                        global_cfg["data_params"]["start_date"],
                        global_cfg["data_params"]["end_date"]),desc="Generating k data")
    )

    # Parallel(n_jobs=8)(delayed(gen_nbr_data)(symbol_cfg,global_cfg,date_str) for date_str in tqdm.tqdm(
    #         get_ts_list(global_cfg["data_params"]["nbr_data_remote_path"],
    #                     global_cfg["data_params"]["start_date"],
    #                     global_cfg["data_params"]["end_date"]),desc="Generating nbr data")
    # )

