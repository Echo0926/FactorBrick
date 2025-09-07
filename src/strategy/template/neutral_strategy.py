import tqdm
import dolphindb as ddb
import pandas as pd
import numpy as np
import time
import random
from src.utils import *
from src.BackTest import Backtest

# まほう
def Neutral_strategy(self: Backtest,
                   stock_pos_limit: float = 0.8, # 每次的整体仓位(整体仓位*品种权重=最终头寸
                     stock_static_loss=0.015,  # [多单]单笔交易止损比例(静态)
                     stock_static_profit=0.045,  # [多单]单笔交易止盈比例(静态)
                     stock_dynamic_loss=0.02,  # [多单]单笔交易止损比例(动态)
                     stock_dynamic_profit=0.06,  # [多单]单笔交易止盈比例(动态)
                     future_static_profit=0.045,  # [多单/空单]单笔交易止损比例(静态)
                     future_static_loss=0.015,  # [多单/空单]单笔交易止盈比例(静态)
                     future_dynamic_profit=0.06,  # [多单/空单]单笔交易止损比例(动态)
                     future_dynamic_loss=0.02  # [多单/空单]单笔交易止盈比例(动态)
                   ):
    """
    分钟频股票策略模板(中性策略)
    """
    random.seed(self.seed)

    # 0.Init-初始化交易柜台
    self.init_counter()
    self.start_counter()
    self.time_config()

    # 1.Iteration
    for current_date, current_str_date, current_dot_date in tqdm.tqdm(
            zip(self.date_list, self.str_date_list, self.dot_date_list),
            total=len(self.date_list),
            dynamic_ncols=True,
            desc="Simple Neutral Strategy BackTesting..."):
        self.current_date = current_date
        self.current_str_date = current_str_date
        self.current_dot_date = current_dot_date

        # TODO: 后续这块抽象成为方法
        self.stock_k_dict = load_k_json(self.stock_counter_json, current_str_date, symbol=None)
        self.stock_signal_dict = load_k_json(self.stock_signal_json, current_str_date, symbol=None)
        self.stock_info_dict.update(load_info_json(self.stock_info_json, current_str_date))
        transformed_dict = {}
        for stock_code, data in self.stock_signal_dict.items():
            for timestamp, stock_data in data.items():
                if timestamp not in transformed_dict:
                    transformed_dict[timestamp] = {}
                transformed_dict[timestamp][stock_code] = stock_data
        self.stock_signal_dict = transformed_dict
        self.future_k_dict = load_k_json(self.future_counter_json, current_str_date, symbol=None)
        current_future_info = load_info_json(self.future_info_json, current_str_date)
        self.future_info_dict.update(current_future_info)

        current_cash = self.cash * stock_pos_limit  # 限制当前可用资金水平
        for current_timestamp in self.stock_timestamp_dict[current_date]:   # 当天需要回测的分钟时间戳
            self.current_timestamp = pd.Timestamp(current_timestamp)
            self.current_minute = int(f"{current_timestamp.hour}{str(current_timestamp.minute).zfill(2)}")
            # self.stock_counter_processing()    # 柜台判断是否能够执行Order(Symbol)
            # self.future_counter_processing()    # 柜台判断是否能够执行Order(Future)
            self.stock_counter_strict_processing(open_share_threshold=0.005, close_share_threshold=0.01)
            self.future_counter_strict_processing(open_share_threshold=0.01, close_share_threshold=0.01)

            # 监控限价单　
            self.monitor_stock(order_sequence=False)    # 设为False更能考虑到极端情形
            self.monitor_future(order_type='long',order_sequence=False) # 极端情形
            self.monitor_future(order_type='short',order_sequence=True)   # 极端情形

            if self.cash<=0:
                print(f"{self.current_timestamp},你爆仓了")

            if str(self.current_minute) not in self.stock_signal_dict:
                continue

            # 统计要买入多少标的
            k = 0
            for symbol, row in self.stock_signal_dict[str(self.current_minute)].items():
                if symbol not in self.stock_position and row['buy_signal']>0:
                    k += 1

            # しょうかい しゅくだい　
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
                                                       static_profit=stock_static_profit,
                                                       static_loss=stock_static_loss,
                                                       dynamic_profit=stock_dynamic_profit,
                                                       dynamic_loss=stock_dynamic_loss,
                                                       min_timestamp=pd.Timestamp(current_date)+pd.Timedelta(days=1),  # 最小持仓时间
                                                       max_timestamp=min(
                                                           self.date_list[min((self.date_list.index(current_date)+5),
                                                                              len(self.date_list)-1)],
                                                           pd.Timestamp(end_date) - pd.Timedelta(days=1)),  # 最长持仓时间
                                                       max_order_timestamp=self.date_list[min(
                                                           (self.date_list.index(current_date)+5),
                                                                              len(self.date_list)-1)]  # 最长挂单时间
                        )
        # 收盘后看一下当前的股票持仓情况决定下多少手股指期货空单
        stock_mv = 0
        for symbol, pos_list in self.stock_position.items():
            vol = sum([i['vol'] for i in pos_list])
            stock_mv = vol * self.stock_info_dict[symbol]['close']  # 收盘价计算当前持仓市值
        future_mv = 0
        for contract, pos_list in self.short_position.items():
            vol = sum([i['vol'] for i in pos_list])
            future_mv += vol * self.future_info_dict[contract]['close']
        main_contract = list(current_future_info.keys())[0]   # 主力合约
        main_price, main_multi, main_margin, main_settle = (
            self.future_info_dict[main_contract]['close'], self.future_info_dict[main_contract]['multi'],
            self.future_info_dict[main_contract]['margin'], self.future_info_dict[main_contract]['settle'])
        main_end_date = pd.Timestamp(self.future_info_dict[main_contract]["end_date"])    # 主力合约的结束日期
        next_contract = list(current_future_info.keys())[1]   # 次主力合约
        next_price, next_multi, next_margin, next_settle = (
            self.future_info_dict[next_contract]['close'], self.future_info_dict[next_contract]['multi'],
            self.future_info_dict[next_contract]['margin'], self.future_info_dict[next_contract]['settle'])
        next_price = self.future_info_dict[next_contract]['close']
        next_end_date = pd.Timestamp(self.future_info_dict[next_contract]["end_date"])

        minimum_mv = 0.2 * main_multi * main_price   # 最小对冲的阈值市值(小于该市值一手股指期货就能覆盖,但这样做浪费了股指期货的市值)
        delta_mv =  5 * stock_mv - future_mv

        if delta_mv > minimum_mv and self.future_counter == {}:   # 说明需要补充股指期货空单,并且没有待成交的股指空单
            if self.current_timestamp + pd.Timedelta(days=5) >= main_end_date:    # 减少调仓成本,直接做空次主力合约
                vol = int(delta_mv / next_price)  # 原始计算
                # vol = (vol // next_multi) * next_multi  # 向下取整到 multi 的倍数
                price = int(next_price+1)
                self.order_open_future(order_type = "short", contract = next_contract, vol= vol, price = price,
                                       pre_settle = next_settle, margin = vol * price * next_margin,
                                       static_profit=future_static_profit,
                                       static_loss=future_static_loss,
                                       dynamic_profit=future_dynamic_profit,
                                       dynamic_loss=future_dynamic_loss,
                                       min_timestamp=None, max_timestamp=next_end_date-pd.Timedelta(days=1),
                                       min_order_timestamp=None, max_order_timestamp=pd.Timestamp(current_date)+pd.Timedelta(days=2),
                                       commission=None, reason="Neutral")
            else:       # 调仓成本较小,做空主力合约
                vol = int(delta_mv / main_price)  # 原始计算
                # vol = (vol // main_multi) * main_multi  # 向下取整到 multi 的倍数
                price = int(main_price+1)
                self.order_open_future(order_type = "short", contract = main_contract, vol= vol, price = price,
                                       pre_settle = main_settle, margin = vol * price * main_margin,
                                       static_profit=future_static_profit,
                                       static_loss=future_static_loss,
                                       dynamic_profit=future_dynamic_profit,
                                       dynamic_loss=future_dynamic_loss,
                                       min_timestamp=None, max_timestamp=main_end_date,
                                       min_order_timestamp=None, max_order_timestamp=pd.Timestamp(current_date)+pd.Timedelta(days=2),
                                       commission=None, reason="Neutral")

        # 每日结算统计
        self.calculate_future_profit(order_type = "long")
        self.calculate_future_profit(order_type = "short")
        self.close_counter()    # 关闭柜台

        # 记录
        self.profit_Dict[self.current_date]=self.profit
        self.settle_profit_Dict[self.current_date]=self.profit_settle
        self.cash_Dict[self.current_date]=self.cash

        print(f"day:{self.current_date}-cash:{self.cash}-profit:{self.profit}-profit_settle:{self.profit_settle}")
        print("当前持仓长度:",len(self.stock_position),"持仓名单:",sorted(self.stock_position.keys()))

    return self

