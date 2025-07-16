import os,re,glob
import datetime
import numpy as np
import pandas as pd
from pandas_market_calendars import get_calendar
from typing import List
import pickle
import bisect
from typing import Dict,Tuple,List
import matplotlib.pyplot as plt

def init_path(path_dir):
    "创建当前.py目录下的文件夹"
    if os.path.exists(path=path_dir)==bool(False):
        os.mkdir(path=path_dir)
    return None

def get_glob_list(path_dir):
    "返回符合条件的文件名列表"
    # return glob.glob(pathname=path_dir)
    return [os.path.basename(i) for i in glob.iglob(pathname=path_dir,recursive=False)]

def trans_time(ts_list,target_format):
    """
    将ts_list按照任意格式互相转换 target_format: choice of {"string","date","datetime","timestamp"}
    """
    def trans(string):
        "月份/日期转换(01→1)"
        if string[0]!=0:
            result=string
        elif string[0]==0:
            result=string[-1]
        return int(result)

    def str_to_datetime(ts_list):
        formats_8=['%d/%m/%Y','%Y-%m-%d','%Y%m%d','%Y/%m/%d']
        formats_6=["%Y/%m","%Y%m","%Y-%m"]
        formats_4=["%Y"]

        s=ts_list[0]
        s=s.replace("-","")
        s=s.replace("/","")
        if len(s)==8:
            for fmt in formats_8:
                try:
                    L=[datetime.datetime.strptime(i,fmt) for i in ts_list]
                    return L
                except ValueError:
                    pass
            raise ValueError("Invalid date format")
        elif len(s)==6:
            if len(s)==6:
                for fmt in formats_6:
                    try:
                        L=[datetime.datetime.strptime(i, fmt) for i in ts_list]
                        return L
                    except ValueError:
                        pass
                raise ValueError("Invalid date format")
        elif len(s)==4:
            for fmt in formats_4:
                try:
                    L=[datetime.datetime.strptime(i, fmt) for i in ts_list]
                    return L
                except ValueError:
                    pass
            raise ValueError("Invalid date format")

    def trans_from_string(ts_list,target_format):
        # 将字符串转化为其他时间格式
        if target_format=="datetime":
            L=str_to_datetime(ts_list=ts_list)
        elif target_format=="date":
            L=[i.date() for i in str_to_datetime(ts_list=ts_list)]
        elif target_format=="timestamp":
            L=[pd.to_datetime(i) for i in ts_list]
        elif target_format=="string":
            L=ts_list
        return L

    def trans_from_datetime(ts_list,target_format):
        if target_format=="string":
            L=[i.strftime("%Y-%m-%d") for i in ts_list]
        elif target_format=="date":
            L=[i.date() for i in ts_list]
        elif target_format=="timestamp":
            L=[pd.to_datetime(i) for i in ts_list]
        elif target_format=="datetime":
            L=ts_list
        return L

    # step1. target_format为timestamp的直接输出
    if target_format=="timestamp":
        if type(ts_list[0])==type(np.datetime64('2023-12-29T00:00:00.000000000')):   # DolphinDB导出的date格式通常为numpy.datetime64格式
            L=[pd.Timestamp(i) for i in ts_list]
        else:
            L=[pd.to_datetime(i) for i in ts_list]

    # step2. target_format为其他的间接输出
    else:
        """
        int型→string型
        """
        if type(ts_list[0])==type(1):
            ts_list=[str(i) for i in ts_list]
        sample=ts_list[0]
        if type(sample)==type("string"):    # 字符串格式时间{"20200101","2020-01-01"}
            L=trans_from_string(ts_list=ts_list,target_format=target_format)

        elif type(sample)==type(datetime.datetime(2020,1,1,12,0,0)):  # datetime.datetime格式
            L=trans_from_datetime(ts_list=ts_list,target_format=target_format)

        elif type(sample)==type(datetime.date(2020,1,1)):   # datetime.date格式
            datetime_list=[datetime.datetime.strptime(str(i),'%Y-%m-%d') for i in ts_list]  # date→datetime格式→others
            L=trans_from_datetime(ts_list=datetime_list,target_format=target_format)

        elif type(sample)==type(pd.Timestamp('2020-01-01 12:34:56')):   # pandas timestamp格式→string格式→others
            string_list=[i.strftime('%Y-%m-%d %H:%M:%S')[:10] for i in ts_list]
            L=trans_from_string(ts_list=string_list,target_format=target_format)

        else:
            L=[pd.Timestamp(i) for i in ts_list]

    return L

def get_ts_list(start_date="20200101",end_date="21000101",freq="D",to_current=False,cut_weekend=False):
    "生成时间序列list"

    # 转化格式
    start_date=start_date.replace("-","")
    end_date=end_date.replace("-","")
    import datetime
    def trans(string):
        "月份/日期转换(01→1)"
        if string[0]!=0:
            result=string
        elif string[0]==0:
            result=string[-1]
        return int(result)

    if freq=="D":
        if type(start_date)==type("string"):    # 字符串格式时间
            start_date=datetime.date(trans(start_date[:4]),trans(start_date[4:6]),trans(start_date[6:8]))
            end_date=datetime.date(trans(end_date[:4]),trans(end_date[4:6]),trans(end_date[6:8]))
        elif type(start_date)==type(datetime.date(2020,1,2)): # datetme.date时间戳格式
            start_date=start_date
            end_date=end_date
        else:
            print("请输入string格式的时间戳(Exp:20200101)")

        if bool(to_current)==bool(False):
            if bool(cut_weekend)==bool(True):
                date_list=[i.strftime('%Y%m%d') for i in get_calendar('XSHG').valid_days(start_date=start_date,end_date=end_date)]
            else:
                date_list=[i for i in list(pd.date_range(start=start_date,end=end_date,freq="D"))]
        elif bool(to_current)==bool(True):
            if bool(cut_weekend)==bool(True):
                date_list=[i.strftime('%Y%m%d') for i in get_calendar('XSHG').valid_days(start_date=start_date,end_date=datetime.date.today())]
            else:
                date_list=[i for i in list(pd.date_range(start=start_date,end=datetime.date.today(),freq="D"))]
        return date_list

    elif freq=="M":
        if bool(to_current)==bool(False):
            month_list=[str(i)[:6] for i in list(pd.date_range(start=start_date,end=end_date,freq="M").strftime("%Y%m%d"))]
        elif bool(to_current)==bool(True):
            month_list=[str(i)[:6] for i in list(pd.date_range(start=start_date,end=datetime.date.today(),freq="M").strftime("%Y%m%d"))]
        return month_list

    elif freq=="Q":
        if bool(to_current)==bool(False):
            quarter_list=[date.strftime("%Y%m%d") for date in pd.date_range(start=start_date,end=end_date, freq="Q")]
        elif bool(to_current)==bool(True):
            quarter_list=[date.strftime("%Y%m%d") for date in pd.date_range(start=start_date,end=datetime.date.today(), freq="Q")]
        return quarter_list

    elif freq=="Y":
        current_month=datetime.datetime.now().month
        current_year=datetime.datetime.now().year
        if bool(to_current)==bool(False):
            year_list=[date.strftime("%Y%m%d") for date in pd.date_range(start=start_date,end=end_date, freq="Y")]
            if current_month>=7:
                year_list.append("{}0630".format(current_year))
            return year_list
        elif bool(to_current)==bool(True):
            year_list=[date.strftime("%Y%m%d") for date in pd.date_range(start=start_date,end=datetime.date.today(), freq="Y")]
            if current_month>=7:
                year_list.append("{}0630".format(current_year))
            return year_list

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :file_utils.py
# @Time :2025/1/17 16:03
# @Author :Usami Renko

import bisect
import os, glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import importlib
import sys
from pathlib import Path
import pickle, joblib
from joblib import Parallel,delayed
import tqdm

def julian_time_convert(julian_time):
    """
    将朱利安时间转换为小时和分钟的组合。

    朱利安时间是一种天文学中使用的时间系统，这里用于表示从一个固定时间点开始的毫秒数。
    此函数的目的是将这种时间格式转换为更容易读取的小时和分钟格式。

    Returns:
        int: 以小时和分钟表示的时间，格式为HHMM。
    """
    # 计算小时数
    hour = julian_time // 3600 // 1000

    # 计算剩余的毫秒数，用于后续分钟的计算
    julian_min = julian_time - hour * 3600 * 1000

    # 计算分钟数
    minute = julian_min // 60 // 1000

    """
    julian time 的转化删掉了多余的毫秒 所以数据在计算时需要min + 1以保证数据不泄露
    +1 后切换julian_time从开始时间切换为
    """
    if minute == 59:
        minute = 0
        hour += 1
    else:
        minute += 1

    # 将小时和分钟合并为一个整数返回
    return hour * 100 + minute

def julian_day_convert(julian_day):
    """
    将朱利安日转换为日期。
    """

def julian_seconds_convert(julian_time):
    """
    将儒略时间转换为秒。

    儒略时间是一种天文学中使用的连续时间刻度。这个函数的目的是将儒略时间转换为一天中的秒数。

    参数:
    julian_time (int): 儒略时间，表示从某一固定时刻起的毫秒数。

    返回:
    int: 一天中的秒数，考虑到儒略时间的整数部分。
    """
    # 计算小时数，每小时3600秒，但这里以1000的倍数进行整除是为了处理毫秒
    hour = julian_time // 3600 // 1000

    # 计算剩余的毫秒数，用于后续分钟和秒的计算
    julian_min = julian_time - hour * 3600 * 1000

    # 从剩余的毫秒数中计算出分钟数，每分钟60秒，同样这里以1000的倍数进行整除以处理毫秒
    minute = julian_min // 60 // 1000

    # 计算秒数，这里通过小时和分钟计算出一个综合的秒数值，同时考虑剩余的毫秒数
    # 这里的计算方式确保了即使在小时和分钟转换时有毫秒损失，也能准确地转换为秒
    second = 100 * (hour * 100 + minute) + (julian_min - minute * 60 * 1000) // 1000

    # 返回计算出的秒数
    return second






# def filter_corr(corr_info: pd.DataFrame, ic_info: pd.DataFrame, corr_threshold=0.8) -> List[str]:
#     """
#     根据相关性系数和IC值筛选因子。
#
#     该函数首先处理相关性系数数据，设置因子名称为索引。然后筛选出相关性系数大于给定阈值的因子，
#     并根据IC值的绝对值选择保留的因子，移除相关性高但IC值较低的因子。
#
#     参数:
#     corr_info: 包含因子相关性系数信息的DataFrame。
#     ic_info: 包含因子IC值信息的DataFrame。
#     corr_threshold: 相关性系数的阈值，默认为0.8。
#
#     返回:
#     一个字符串列表，包含筛选后的因子名称。
#     """
#
#     # 检查是否存在未命名列，并进行重命名及设置索引
#     if "Unnamed: 0" in corr_info.columns:
#         corr_info.rename(columns={"Unnamed: 0": "factor_name"}, inplace=True)
#         df = corr_info.set_index('factor_name')
#     else:
#         df = corr_info.copy()
#
#     # 筛选大于阈值的部分数据
#     filter_df = df[df > corr_threshold]
#     filter_df.fillna(-2, inplace=True)  # corr值域在（-1， 1）之间，-2必为填充值
#
#     corr_dict = {}
#     columns = df.columns
#
#     # 逐行遍历
#     for row in filter_df.itertuples():
#         row_name = row.Index
#         new_set = []
#         # new_set.append(row_name)
#         for i in range(0, len(columns)):
#             if row[i + 1] != -2.0:
#                 print(row[i + 1], columns[i], i)
#                 new_set.append(columns[i])
#                 print()
#         if len(new_set) > 1:
#             # 由于自身的cor == 1 所以有2个的情况下 才需要筛选
#             corr_dict[row_name] = new_set
#
#     # 选出相关性最高的
#     remove_factors = []
#     ic_info['abs_ic'] = np.abs(ic_info['ic_avg'])
#     for key in corr_dict.keys():
#         values = corr_dict[key]
#         bundle = ic_info[ic_info['factor'].isin(values)]
#         top_factor = bundle.sort_values(by=['abs_ic'], ascending=False).iloc[0].factor
#         values.remove(top_factor)
#         remove_factors += values
#
#     passed_factors = list(set(columns) - set(remove_factors))
#     return passed_factors


def get_ic_daily(nbr_path: str, feat_list: List[str], label: str):
    """计算每日特征信息系数(IC)并格式化输出

    从邻域数据文件中读取数据，计算特征与目标变量的相关系数矩阵，提取目标变量的相关系数，
    按特征列表顺序重组数据，并添加日期列

    :param nbr_path: 邻域数据文件路径（Parquet格式）
    :param feat_list: 需要计算的特征列名列表
    :param label: 目标变量列名
    :return: 包含特征IC值和日期的DataFrame，格式为：
             [feature1_ic, feature2_ic, ..., date]

    """
    # 读取邻域数据并选择指定列
    nbr_df = pd.read_parquet(nbr_path)

    # 计算相关系数矩阵并提取目标变量相关性
    cols = feat_list + [label]
    corr_matrix = nbr_df[cols].corr()

    # 重组数据格式：转置矩阵并按特征列表排序列
    df = corr_matrix[label].reset_index().set_index('index').T[feat_list]

    # 添加数据日期（取数据中的最大日期）
    df['date'] = nbr_df['date'].max()
    return df


def simulate_symbols(folder_path: str, sample_num: int,
                     begin_date: str, end_date: str,
                     output_path: str,
                     file_name: str = "/150000000.pqt"):
    """模拟生成并保存指定数量在时间范围内存在的证券代码

    1. 读取起始日期和结束日期的Parquet文件
    2. 找出两个日期共有的证券代码
    3. 随机抽取指定数量的证券代码
    4. 将结果保存为JSON文件

    :param folder_path: Parquet文件存储根路径
    :param sample_num: 需要抽取的证券代码数量
    :param begin_date: 起始日期（格式示例：'20230101'）
    :param end_date: 结束日期（格式示例：'20231231'）
    :param output_path: 输出JSON文件路径
    :param file_name: Parquet文件名格式，默认为"/150000000.pqt"
    """
    # 读取首尾日期数据
    df_begin = pd.read_parquet(folder_path + begin_date + file_name)
    df_end = pd.read_parquet(folder_path + end_date + file_name)

    # 计算共有证券代码并抽样
    common_symbol = list(set(df_begin.symbol) & set(df_end.symbol))
    res_symbols = list(df_end[df_end['symbol'].isin(common_symbol)].symbol.sample(sample_num))

    # 构建JSON数据结构并写入文件
    json_data = json.dumps({"symbol_list": res_symbols}, skipkeys=True, indent=4)
    with open(output_path + "/symbol.json", 'w') as f:
        f.write(json_data)


def get_date_list(path: str, begin_date: str, end_date: str) -> List[str]:
    """
    获取指定日期范围内的文件夹列表。

    该函数会根据给定的路径和日期范围，筛选出路径下所有在指定日期范围内的文件夹名称。

    参数:
    path: str - 文件路径，表示需要筛选的文件夹所在的目录。
    begin_date: str - 开始日期，表示日期范围的起始点。
    end_date: str - 结束日期，表示日期范围的结束点。

    返回值:
    List[str] - 一个包含所有符合条件的日期文件夹名称的列表。
    """
    # 获取路径下的所有文件夹列表
    folders = os.listdir(path)

    # 筛选出在指定日期范围内的文件夹名称
    date_list = [x for x in folders if begin_date <= x <= end_date]

    # 返回筛选后的日期列表
    return date_list

def get_recent_date_list(path: str,end_date: str, recent_days: int = 14) -> List[str]:
    folders = os.listdir(path)
    date_list = [x for x in folders if x <= end_date][-recent_days:]
    return date_list



def call_func(module_path, function_name: str, args: List[Any] = [], kwargs: Dict[str, Any] = {}):
    """

    :param module_path:
    :param function_name:
    :param args:
    :param kwargs:
    :return:
    """
    module = importlib.import_module(module_path)
    # 获取函数对象
    if not hasattr(module, function_name):
        raise AttributeError(f"Function {function_name} not found in {module_path}")
    func = getattr(module, function_name)
    return func(*args, **kwargs)


def call_function_from_json(json_path: str):
    """
    通过 JSON 配置文件动态调用 Python 函数
    :param json_path: JSON 配置文件路径
    :return: 函数执行结果
    """
    try:
        # 读取 JSON 配置
        with open(json_path, 'r') as f:
            config = json.load(f)

        # 解析必要参数
        module_path = config['module']  # 示例值: "my_package.my_module"
        function_name = config['function']
        args = config.get('args', [])
        kwargs = config.get('kwargs', {})

        # 添加自定义模块路径（如果需要）
        if 'module_path' in config:
            sys.path.insert(0, str(Path(config['module_path']).resolve()))

        # 动态导入模块
        module = importlib.import_module(module_path)

        # 获取函数对象
        if not hasattr(module, function_name):
            raise AttributeError(f"Function {function_name} not found in {module_path}")
        func = getattr(module, function_name)

        # 执行函数
        return func(*args, **kwargs)

    except KeyError as e:
        raise ValueError(f"Missing required config key: {e}") from e
    except ModuleNotFoundError as e:
        raise ImportError(f"Module not found: {e.name}") from e
    except Exception as e:
        raise RuntimeError(f"Execution failed: {str(e)}") from e


def get_limited(date_dir, daily_symbol_folder) -> pd.DataFrame:
    """
    读取数据，并提取标的代码跌停和涨停信息。

    参数:
    date_dir (str): 日期目录，用于构成JSON文件名。
    daily_symbol_folder (str): 每日股票标的信息存储的文件夹路径。

    返回:
    pd.DataFrame:  df[['symbol', 'lowLimit', 'highLimit']]。
    """
    # 打开并读取指定日期的JSON文件
    with open(daily_symbol_folder + '/' + date_dir + '.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        symbol_data = data.get('symbol', {}).get('symbolList', {})

    # 创建一个列表用于存储每个symbol的数据
    rows = []

    # 遍历symbol数据，提取相关信息
    for symbol, info in symbol_data.items():
        info['symbol'] = symbol
        rows.append(info)

    # 将列表转换为DataFrame
    df = pd.DataFrame(rows)
    # 提取并设置DataFrame的列，然后将symbol设置为索引
    limited_df = df[['symbol', 'lowLimit', 'highLimit']]
    limited_df.set_index('symbol', inplace=True)
    return limited_df


def get_paused(date_dir: str, daily_symbol_folder: str, paused_folder: str):
    """
    读取指定路径下的JSON文件，提取每个symbol的信息并找出停牌的股票。
    将停牌股票存储到对应date_dir文件夹中的.pqt文件中。

    该函数从指定文件夹下的JSON文件中读取数据，提取所有symbol的信息，
    将其存储到一个DataFrame中，然后找出其中停牌（paused值为1.0）的股票，
    并将这些股票信息保存到对应date_dir的文件夹中的.pqt文件中。

    Args:
        :param date_dir: JSON文件的日期文件夹路径。
        :param daily_symbol_folder:
        :param paused_folder:

    Returns:
        list: 一个包含所有停牌股票symbol的列表。

    """
    paused = []
    try:
        # 读取JSON文件
        with open(daily_symbol_folder + '/' + date_dir + '.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            symbol_data = data.get('symbol', {}).get('symbolList', {})
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或JSON解析错误，返回空列表
        return paused

    # 创建一个列表用于存储每个symbol的数据
    rows = []

    for symbol, info in symbol_data.items():
        info['symbol'] = symbol
        rows.append(info)

    # 如果没停牌的就直接返回
    if not rows:
        return paused

    # 将列表转换为DataFrame
    df = pd.DataFrame(rows)

    # 找出停牌股票
    if 'paused' in df.columns:
        filtered_df = df[df['paused'] == 1.0]
        paused = filtered_df['symbol'].tolist()

        # 保存停牌股票信息到.pqt文件
        output_folder = os.path.join(paused_folder, date_dir)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{date_dir}_paused_stocks.pqt')
        filtered_df.to_parquet(output_path, index=False)

    return paused


def get_trade_size_threshold(date_dir, paused_list, trade_flow_path: str,
                             symbol_list: List[str]) -> pd.DataFrame:
    trade_flow_path = trade_flow_path + f'{date_dir}/{date_dir}.json'
    with open(trade_flow_path, 'r') as trade_flow_file:
        trade_flow_threshold = pd.DataFrame(json.load(trade_flow_file))
        # print(trade_flow_threshold)

        filter_symbol_list = [symbol for symbol in symbol_list if symbol not in paused_list]

        selected_trade_flow_threshold = trade_flow_threshold[filter_symbol_list]
        return selected_trade_flow_threshold


def write_json(data, output_path: str, file_name: str):
    """将Python对象数据写入JSON文件

    Args:
        data (dict/list): 需要序列化的Python对象，支持字典或列表类型
        output_path (str): 输出目录路径，需要以路径分隔符结尾
        file_name (str): 输出的JSON文件名（需包含.json扩展名）

    Returns:
        None: 无返回值，直接生成文件到指定路径
    """

    # 序列化数据为格式化的JSON字符串（UTF-8编码）

    json_data = json.dumps(data, default=lambda obj: obj.__dict__, skipkeys=True, indent=4)

    # 打开文件并写入数据（自动覆盖已有内容）
    f = open(output_path + file_name, 'w')
    f.write(json_data)
    f.close()


def write_factor_name(logic_name: str, data_type: int, param=None):
    """生成因子名称字符串

    参数:
        logic_name (str): 逻辑名称部分
        data_type (int): 数据类型标识
        param (any, optional): 参数部分. 支持列表类型参数, 默认值为0

    返回:
        str: 格式为 "{logic_name}_{data_type}_{param}" 的拼接字符串
    """

    # 如果param是列表，将其转换为字符串
    if isinstance(param, list):
        param = ''.join([str(x) for x in param])

    # 如果param为空值，则设置为默认值0
    if not param:
        param = 0

    return f'{logic_name}_{data_type}_{param}'

def find_closest_lower(target: int, sorted_list: List[int]) -> Tuple[int,None]:
    """
    在有序列表中找到最近且小于目标值的元素，若不存在则返回None。

    参数：
        target: 需要比较的整数
        sorted_list: 已排序的升序整数列表

    返回：
        最接近且小于target的元素，或None（若无符合条件的元素）
    """
    # 查找第一个 >= target 的位置
    pos = bisect.bisect_left(sorted_list, target)

    # 如果位置为0，说明所有元素都 >= target，返回None
    if pos == 0:
        return sorted_list[0]

    # 返回前一个元素（即最大的 < target 的元素）
    return sorted_list[pos - 1]

def get_data_type_code(data_type_str: str) -> int:
    """根据数据类型字符串返回对应的标识码

    Args:
        data_type_str (str): 数据类型名称，支持 "Snapshot", "Order", "Trade"

    Returns:
        int: 对应的标识码（0, 1, 2）

    Raises:
        ValueError: 当传入无效类型时抛出异常
    """
    mapping = {
        "Snapshot": 0,
        "Order": 1,
        "Trade": 2
    }
    if data_type_str not in mapping:
        raise ValueError(f"无效的数据类型: {data_type_str}")
    return mapping[data_type_str]

def save_pqt(output_file_path, save_df):
    # print()
    if not os.path.exists(output_file_path):
        # 如果文件不存在，先将res_df转换成.pqt文件存放在FEATURE_PATH中
        save_df.to_parquet(output_file_path, index=False)

    else:
        # 如果文件已存在，则进行合并操作
        try:
            basic_df = pd.read_parquet(output_file_path)
            merged_res_df = pd.merge(basic_df, save_df, on=['symbol', 'time'], how='left',
                                     suffixes=('_left', '_right'))  # 宽表合并
            # 删除带有 '_left' 后缀的列
            columns_to_drop = [col for col in merged_res_df.columns if col.endswith('_left')]
            merged_res_df = merged_res_df.drop(columns=columns_to_drop)

            # 去除 '_right' 后缀
            merged_res_df.columns = merged_res_df.columns.str.replace('_right', '')
            # 这样也不用怕弄错了，可以直接更新

            # 将合并后的DataFrame存储到OUTPUT_FOLDER中
            merged_res_df.to_parquet(output_file_path, index=False)

        except Exception as e:
            print(f"Error reading Parquet file {output_file_path}: {e}")

def init_path(path_dir):
    "创建当前path_dir目录下的文件夹"
    if not os.path.exists(path=path_dir):
        os.mkdir(path=path_dir)


def get_glob_list(path_dir):
    "返回符合条件的文件名列表"
    # return glob.glob(pathname=path_dir)
    return [os.path.basename(i) for i in glob.iglob(pathname=path_dir,recursive=False)]


def save_model(model_obj, save_path:str, file_name:str, target_format:str):
    """
    保存模型为指定格式到本地指定路径
    target_format: ['pickle','joblib','npy','npz','bin']
    """
    init_path(path_dir=save_path)

    # 合并save_path：save_path\file_name.target_format
    save_path = rf"{save_path}\{file_name}.{target_format}"
    total_format_list = ["pickle","npy","npz","bin"]
    if target_format not in total_format_list:
        raise ValueError(f"target format must in {total_format_list}")

    if target_format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(model_obj, f)

    elif target_format == 'joblib':  # 通常比pickle更适合scikit-learn模型
        joblib.dump(model_obj, save_path)

    elif target_format == 'npy':
        np.save(save_path, model_obj, allow_pickle=True)

    elif target_format == 'npz':
        np.savez(save_path, model=model_obj)

    elif target_format == 'bin':  # 自定义二进制格式
        with open(save_path, 'wb') as f:
            f.write(pickle.dumps(model_obj))


def load_model(load_path: str, file_name: str, target_format: str):
    """
    从指定路径加载模型

    Args:
        load_path: 加载路径（目录）
        file_name: 文件名（不含后缀）
        target_format: 文件格式，可选 ['pickle','joblib','npy','npz','bin']

    Returns:
        加载的模型对象

    Raises:
        ValueError: 当target_format不被支持时
        FileNotFoundError: 当文件不存在时
    """
    # 构建完整路径
    full_path = f"{load_path}/{file_name}.{target_format}"

    total_format_list = ["pickle", "joblib", "npy", "npz", "bin"]
    if target_format not in total_format_list:
        raise ValueError(f"target format must in {total_format_list}")

    if target_format == 'pickle':
        with open(full_path, 'rb') as f:
            return pickle.load(f)

    elif target_format == 'joblib':
        return joblib.load(full_path)

    elif target_format == 'npy':
        return np.load(full_path, allow_pickle=True)

    elif target_format == 'npz':
        return np.load(full_path)['model']

    elif target_format == 'bin':
        with open(full_path, 'rb') as f:
            return pickle.loads(f.read())

def parallel_read_pqt(file_path, columns=None, start_date=None, end_date=None, desc:str="parallel reading", n_jobs:int = -1):
    """Joblib Parallel+delayed 读取指定目录下的所有"""
    if not start_date:
        start_date = 20000101
    if not end_date:
        end_date = 20300101
    start_date = int(start_date)
    end_date = int(end_date)
    current_file_list = get_glob_list(path_dir=rf"{file_path}\*")
    filter_file_list = [i for i in current_file_list if int(start_date)<=int(i)<=int(end_date) and
                        get_glob_list(path_dir=rf"{file_path}\{i}\*")]

    def read_pqt(path, col):
        file_list = get_glob_list(path_dir=f"{path}\*")

        if not col:
            return pd.concat([pd.read_parquet(rf"{path}\{file}") for file in file_list],ignore_index=True,axis=0)
        try:
            return pd.concat([pd.read_parquet(rf"{path}\{file}",columns=col) for file in file_list],ignore_index=True,axis=0)
        except Exception as e:
            print(e)
            df = pd.concat([pd.read_parquet(rf"{path}\{file}") for file in file_list],ignore_index=True,axis=0)
            filter_col = [i for i in col if i in df.columns]
            return df[filter_col]

    row_list = Parallel(n_jobs=n_jobs)(
        delayed(read_pqt)(rf"{file_path}\{d}", columns)
            for d in tqdm.tqdm(filter_file_list,desc=desc)
    )

    return pd.concat(row_list, axis=0, ignore_index=True)

def get_nan_time_dict(res_df):
    nan_time_dict = {}
    for col in res_df.columns:
        if res_df[col].isna().any():
            nan_times = set(res_df[res_df[col].isna()]['time'])
            nan_time_dict[col] = nan_times
    return nan_time_dict

def write_k_json(data, date_col, symbol_col, save_path, index_col: str="timestamp"):
    """
    输出到指定目录下的json文件
    {
        "index_col":{
            "open":,
            "high":
            "low":
            "close":
            "volume":
            ... (除了symbol_col与date_col传进来的所有列)
        }
    }
    """
    init_path(save_path)
    filter_col = [i for i in data.columns.tolist() if i not in [date_col, symbol_col]]
    total_date_list = sorted(set(data[date_col]))
    data[date_col] = data[date_col].apply(pd.Timestamp)
    if "minute" in filter_col:
        data["minute"] = data["minute"].apply(int)
    if "timestamp" in filter_col:
        data["timestamp"] = data["timestamp"].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    if "start_date" in filter_col:
        data["start_date"] = data["start_date"].apply(lambda x: pd.Timestamp(x).strftime('%Y%m%d'))
    if "end_date" in filter_col:
        data["end_date"] = data["end_date"].apply(lambda x: pd.Timestamp(x).strftime('%Y%m%d'))

    def processing(date_str):
        date_str = pd.Timestamp(date_str).strftime('%Y%m%d')
        df = data[data[date_col] == date_str].reset_index(drop=True)
        init_path(path_dir=rf"{save_path}\{date_str}")

        symbol_list = sorted(set(data[symbol_col].tolist()))
        for symbol in symbol_list:
            slice_df = df[df[symbol_col] == symbol]
            res_dict = {}

            # 如果文件已存在，读取现有数据
            if os.path.exists(rf"{save_path}\{date_str}\{symbol}.json"):
                with open(rf"{save_path}\{date_str}\{symbol}.json", "r",encoding='utf-8') as f:
                    res_dict = json.load(f)

            # 更新数据
            for _, row in slice_df.iterrows():
                res_dict[row[index_col]] = {col: row[col] for col in filter_col}

            # 写入更新后的数据
            with open(rf"{save_path}\{date_str}\{symbol}.json", "w",encoding='utf-8') as f:
                json.dump(res_dict, f, indent=4)

    Parallel(n_jobs=-1)(delayed(processing)(date_str) for date_str in
                        tqdm.tqdm(total_date_list, desc="writing k json"))
    # for date_str in tqdm.tqdm(total_date_list, desc="writing k json"):
    #     processing(date_str)


def write_info_json(data, date_col, symbol_col, save_path, index_col: str="timestamp"):
    """
    输出到指定目录下的json文件
    {
        "index_col":{
            "open":,
            "high":
            "low":
            "close":
            "volume":
            ... (除了symbol_col与date_col传进来的所有列)
        }
    }
    """
    init_path(save_path)
    filter_col = [i for i in data.columns.tolist() if i not in [date_col, symbol_col]]
    total_date_list = sorted(set(data[date_col]))
    data[date_col] = data[date_col].apply(pd.Timestamp)
    if "timestamp" in filter_col:
        data["timestamp"] = data["timestamp"].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    if "start_date" in filter_col:
        data["start_date"] = data["start_date"].apply(lambda x: pd.Timestamp(x).strftime('%Y%m%d'))
    if "end_date" in filter_col:
        data["end_date"] = data["end_date"].apply(lambda x: pd.Timestamp(x).strftime('%Y%m%d'))

    def processing(date_str):
        date_str = pd.Timestamp(date_str).strftime('%Y%m%d')
        df = data[data[date_col] == date_str].reset_index(drop=True)
        res_dict = {}

        # 如果文件已存在，读取现有数据
        if os.path.exists(rf"{save_path}\{date_str}.json"):
            with open(rf"{save_path}\{date_str}.json", "r", encoding='utf-8') as f:
                res_dict = json.load(f)

        symbol_list = sorted(set(data[symbol_col].tolist()))
        for symbol in symbol_list:
            slice_df = df[df[symbol_col] == symbol]

            # 更新数据
            for _, row in slice_df.iterrows():
                res_dict[row[index_col]] = {col: row[col] for col in filter_col}

            # 写入更新后的数据
            with open(rf"{save_path}\{date_str}.json", "w",encoding='utf-8') as f:
                json.dump(res_dict, f, indent=4)

    Parallel(n_jobs=-1)(delayed(processing)(date_str) for date_str in
                        tqdm.tqdm(total_date_list, desc="writing info json"))
    # for date_str in tqdm.tqdm(total_date_list, desc="writing info json"):
    #     processing(date_str)

def load_info_json(path, date_str):
    if not os.path.exists(rf"{path}\{date_str}.json"):
        return {}
    else:
        with open(rf"{path}\{date_str}.json", "r", encoding='utf-8') as f:
            res_dict = json.load(f)
        return res_dict

def load_k_json(path, date_str, symbol=None):
    if symbol:
        if not os.path.exists(rf"{path}\{date_str}\{symbol}.json"):
            return {}
        else:
            with open(rf"{path}\{date_str}\{symbol}.json", "r", encoding='utf-8') as f:
                res_dict = json.load(f)
            return res_dict
    else:
        symbol_list = [str(i)[:str(i).index(".json")] for i in get_glob_list(path_dir=rf"{path}\{date_str}\*.json")]
        dict_list = Parallel(n_jobs=16)(delayed(load_k_json)(path, date_str, symbol) for symbol in symbol_list)
        return dict(zip(symbol_list, dict_list))

def max_drawdown_plot(df,save_path):
    # 计算回撤（当前值低于历史最高点时显示回撤幅度，否则为0）
    df['peak'] = df['net_value'].cummax()
    df['is_drawdown'] = df['net_value'] < df['peak']  # 标记是否处于回撤期
    df['current_drawdown'] = np.where(
        df['is_drawdown'],
        100 * (df['peak'] - df['net_value']) / df['peak'],  # 回撤幅度
        0  # 非回撤期为0
    )

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # 绘制净值曲线（左轴）
    ax1.plot(df['date'], df['profit'], label='CumProfit', color='blue', linewidth=2)
    ax1.set_ylabel('CumProfit', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 创建右轴用于回撤
    ax2 = ax1.twinx()

    # 关键修改：只填充回撤区域
    # 找到所有回撤期的连续区间
    drawdown_periods = (df['is_drawdown'] != df['is_drawdown'].shift()).cumsum()
    for _, period_df in df[df['is_drawdown']].groupby(drawdown_periods):
        ax2.fill_between(
            period_df['date'],
            period_df['current_drawdown'],
            0,
            color='red',
            alpha=0.3,
            label='MaxDrawDown(%)'
        )

    ax2.set_ylabel('MaxDrawDown(%)', fontsize=12)
    ax2.set_ylim(0, df['current_drawdown'].max() * 1.1)  # 留出顶部空间

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2[:1], labels1 + labels2[:1], loc='upper left')
    plt.tight_layout()
    plt.savefig(rf"{save_path}\MaxDrawDown.png")
    plt.show()