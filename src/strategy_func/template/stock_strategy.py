import json

import tqdm
import dolphindb as ddb
import pandas as pd
import numpy as np
import time
import random
from src.utils import *
from src.BackTest import Backtest

def Stock_strategy(self: Backtest,
                   pos_limit: float = 0.8, # 每次的整体仓位(整体仓位*品种权重=最终头寸
                   static_loss=0.03,  # [多单]单笔交易止损比例(静态)
                   static_profit=0.06,  # [多单]单笔交易止盈比例(静态)
                   dynamic_loss= 0.05,  # [多单]单笔交易止损比例(动态)
                   dynamic_profit = 0.05,  # [多单]单笔交易止盈比例(动态)
                   ):
    """
    分钟频股票策略模板(空气指增 & 指数增强)
    """
    random.seed(self.seed)

    # 0.Init-初始化交易柜台
    self.init_counter()
    self.start_counter()
    self.time_config()

    with open(rf"{self.stock_macro_json}\{get_glob_list(path_dir=rf'{self.stock_macro_json}/*.json')[0]}","r") as f:
        self.stock_macro_dict = json.load(f)

    # 1.Iteration
    for current_date, current_str_date, current_dot_date in tqdm.tqdm(
            zip(self.date_list, self.str_date_list, self.dot_date_list),
            total=len(self.date_list),
            dynamic_ncols=True,
            desc="Pure Stock Strategy BackTesting..."):
        self.current_date = current_date
        self.current_str_date = current_str_date
        self.current_dot_date = current_dot_date

        # TODO: 后续抽象成为方法
        self.stock_k_dict = load_k_json(self.stock_counter_json, current_str_date, symbol=None)
        self.stock_signal_dict = load_k_json(self.stock_signal_json, current_str_date, symbol=None)
        self.stock_info_dict.update(load_info_json(self.stock_info_json, current_str_date))
        for symbol in self.stock_info_dict.keys():
            self.stock_info_dict[symbol]["end_date"] = "20250430"
        transformed_dict = {}
        for stock_code, data in self.stock_signal_dict.items():
            for timestamp, stock_data in data.items():
                if timestamp not in transformed_dict:
                    transformed_dict[timestamp] = {}
                transformed_dict[timestamp][stock_code] = stock_data
        self.stock_signal_dict = transformed_dict

        current_cash = self.cash * pos_limit  # 限制当前可用资金水平
        for current_timestamp in self.stock_timestamp_dict[current_date]:   # 当天需要回测的分钟时间戳
            self.current_timestamp = pd.Timestamp(current_timestamp)
            self.current_minute = int(f"{current_timestamp.hour}{str(current_timestamp.minute).zfill(2)}")
            # self.stock_counter_processing()    # 柜台判断是否能够执行
            self.stock_counter_strict_processing(open_share_threshold=0.005, close_share_threshold=0.05)  # 日内分时线成交量是以股统计的, 假设成交量不能超过这个K线的0.5%

            # 监控限价单　
            self.monitor_stock(order_sequence=False)    # 设为False更能考虑到极端情形

            if self.cash<=0:
                print(f"{self.current_timestamp},你爆仓了")

            if str(self.current_minute) not in self.stock_signal_dict:
                continue

            # 统计要买入多少标的
            k = 0
            for symbol, row in self.stock_signal_dict[str(self.current_minute)].items():
                if symbol not in self.stock_position and row['buy_signal']>0:
                    k += 1

            for symbol, row in self.stock_signal_dict[str(self.current_minute)].items():
                end_date = self.stock_info_dict[symbol]["end_date"]
                buy_signal, sell_signal, open_price, high_price, low_price, close_price = \
                    row['buy_signal'], row['sell_signal'], row['open'], row['high'], row['low'], row['close']

                # if sell_signal>0:
                #     if symbol in self.stock_position:
                #         self.order_close_stock(symbol=symbol,
                #                                    price=int(close_price),
                #                                    vol=10000000000,
                #                                    max_order_timestamp=pd.Timestamp(self.current_date) + pd.Timedelta(
                #                                        days=1),
                #                                    reason='active'
                #                                    )

                if buy_signal>0 and self.cash > (self.ori_cash * 0.5):
                    if symbol not in self.stock_position:
                        price = row['buy_signal']
                        if price <= 0:  # 除0保护
                            continue
                        vol = max(((current_cash / k) / price), 100)

                        self.order_open_stock(symbol=symbol,
                                                       price=price,
                                                       vol=vol,  # 最低下交易乘数手
                                                       static_loss=static_loss,
                                                       static_profit=static_profit,
                                                       dynamic_profit=dynamic_profit,
                                                       dynamic_loss=dynamic_loss,
                                                       min_timestamp=pd.Timestamp(current_date)+pd.Timedelta(days=1),  # 最小持仓时间
                                                       # min_timestamp=None,
                                                       max_timestamp=min(
                                                           self.date_list[min((self.date_list.index(current_date)+5),
                                                                              len(self.date_list)-1)],
                                                           pd.Timestamp(end_date) - pd.Timedelta(days=1)),  # 最长持仓时间
                                                       max_order_timestamp=self.date_list[min(
                                                           (self.date_list.index(current_date)+2),
                                                                              len(self.date_list)-1)]  # 最长挂单时间
                        )

        # 每日结算统计
        self.close_counter()    # 关闭柜台

        # 生成明日仓位
        pos_limit = 0.5 if not pos_limit else round(pos_limit, 2)
        pos_limit += self.stock_macro_dict[self.current_str_date]
        pos_limit = np.clip(pos_limit,0.2,0.8)

        # 记录
        self.profit_Dict[self.current_date]=self.profit
        self.settle_profit_Dict[self.current_date]=self.profit_settle
        self.cash_Dict[self.current_date]=self.cash
        self.pos_Dict[self.current_date]=pos_limit

        print(f"day:{self.current_date}-cash:{self.cash}-profit:{self.profit}")
        print("当前持仓长度:",len(self.stock_position),"持仓名单:",sorted(self.stock_position.keys()))

    return self

