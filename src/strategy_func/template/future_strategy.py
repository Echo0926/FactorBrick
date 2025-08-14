import random
from src.utils import *
from src.BackTest import Backtest

def CTA_strategy(self: Backtest,
                 pos_limit: float = 0.8,    # 每次的整体仓位(整体仓位*品种权重=最终头寸)
                 static_loss: float = 0.015,  # [多单/空单]单笔交易保证金止损比例(静态)
                 static_profit: float = 0.045,  # [多单/空单]单笔交易保证金止盈比例(静态)
                 dynamic_loss: float = 0.02,   # [多单/空单]单笔交易保证金止损比例(动态)
                 dynamic_profit: float = 0.06, # [多单/空单]单笔交易保证金止盈比例(动态)
                 ):
    """
    分钟频CTA策略模板
    """
    random.seed(self.seed)  # 设置随机种子

    # 0.Init-初始化交易柜台 + 生成最终的时间
    self.init_counter()
    self.start_counter()
    self.time_config()

    # 1.Iteration
    for current_date, current_str_date, current_dot_date in tqdm.tqdm(zip(self.date_list, self.str_date_list, self.dot_date_list),
                                                                      total=len(self.date_list),
                                                                      dynamic_ncols=True,
                                                                      desc="CTA Strategy BackTesting..."):
        self.current_date = current_date
        self.current_str_date = current_str_date
        self.current_dot_date = current_dot_date

        # TODO: 后续抽象成为方法
        self.future_k_dict = load_k_json(self.future_counter_json, current_str_date, symbol=None)
        self.future_signal_dict = load_k_json(self.future_signal_json, current_str_date, symbol=None)
        self.future_info_dict.update(load_info_json(self.future_info_json ,current_str_date))

        for current_timestamp in self.future_timestamp_dict[current_date]:   # 当天需要回测的分钟时间戳
            self.current_timestamp = pd.Timestamp(current_timestamp)
            self.current_minute = int(f"{current_timestamp.hour}{str(current_timestamp.minute).zfill(2)}")

            current_cash = self.cash * pos_limit    # 限制当前可用资金水平

            # 监控限价单
            self.monitor_future(order_type='long', order_sequence=True)
            self.monitor_future(order_type='short', order_sequence=True)

            if self.future_signal_dict!={}:
                for contract, row in self.future_signal_dict.items():
                    if row == {}:
                        continue
                    if str(self.current_minute) not in row:
                        continue
                    row = row[str(self.current_minute)]
                    open_long_signal,open_short_signal,close_long_signal,close_short_signal,open_price,high_price,low_price,close_price=\
                        row['open_long'],row['open_short'],row['close_long'],row['close_short'],row['open'],row['high'],row['low'],row['close']
                    multi = self.future_info_dict[contract]["multi"]
                    pre_settle = self.future_info_dict[contract]["pre_settle"]
                    end_date = self.future_info_dict[contract]["end_date"]
                    margin = self.future_info_dict[contract]["margin"]
                    vol, price = 100, self.future_k_dict[contract][str(self.current_minute)]['close']

                    if open_long_signal>0 and self.cash > (self.ori_cash * 0.5):  # 开多
                        self.order_open_future(order_type="long",
                                               contract=contract,
                                               pre_settle=pre_settle,
                                               margin=margin * price * vol,
                                               price=price,
                                               vol=vol,  # 最低下交易乘数手
                                               static_loss=static_loss,
                                               static_profit=static_profit,
                                               dynamic_loss=dynamic_loss,
                                               dynamic_profit=dynamic_profit,
                                               min_timestamp=min(
                                                   pd.Timestamp(self.current_date) + pd.Timedelta(days=1),
                                                   pd.Timestamp(end_date) - pd.Timedelta(days=2)),  # 最小持仓时间
                                               max_timestamp=min(
                                                   pd.Timestamp(self.current_date) + pd.Timedelta(days=2),
                                                   pd.Timestamp(end_date) - pd.Timedelta(days=1)),    # 最长持仓时间
                                               max_order_timestamp= pd.Timestamp(self.current_date) + pd.Timedelta(days=2),   # 最长挂单时间
                                               )

                    if open_short_signal>0 and self.cash > (self.ori_cash * 0.5): # 开空
                        self.order_open_future(order_type="short",
                                               contract=contract,
                                               pre_settle=pre_settle,
                                               margin=margin * price * vol,
                                               price=price,
                                               vol=vol, # 最低下交易乘数手
                                               static_loss=static_loss,
                                               static_profit=static_profit,
                                               dynamic_loss=dynamic_loss,
                                               dynamic_profit=dynamic_profit,
                                               min_timestamp=min(
                                                   pd.Timestamp(self.current_date) + pd.Timedelta(days=1),
                                                   pd.Timestamp(end_date) - pd.Timedelta(days=2)),  # 最小持仓时间
                                               max_timestamp=min(
                                                   pd.Timestamp(self.current_date) + pd.Timedelta(days=2),
                                                   pd.Timestamp(end_date) - pd.Timedelta(days=1)),    # 最长持仓时间
                                               max_order_timestamp= pd.Timestamp(self.current_date) + pd.Timedelta(days=2),   # 最长挂单时间
                                               )

                    if close_long_signal>0:    # 平多
                        if contract in self.long_position:
                            self.order_close_future(order_type='long',
                                                    contract=contract,
                                                    price=int(close_price),
                                                    vol=9999,
                                                    max_order_timestamp=pd.Timestamp(self.current_date)+pd.Timedelta(days=2),
                                                    reason='active')

                    if close_short_signal>0:   # 平空
                        if contract in self.short_position:
                            self.order_close_future(order_type='short',
                                                    contract=contract,
                                                    price=int(close_price),
                                                    vol=9999,
                                                    max_order_timestamp=pd.Timestamp(self.current_date)+pd.Timedelta(days=2),
                                                    reason='active')

            # 柜台判断是否能够执行
            # self.future_counter_processing()
            self.future_counter_strict_processing(open_share_threshold=0.01, close_share_threshold=0.01)


        # 每日结算统计
        self.calculate_future_profit(order_type='long')
        self.calculate_future_profit(order_type='short')
        self.close_counter()    # 关闭柜台

        # 记录
        self.profit_Dict[self.current_date]=self.profit
        self.settle_profit_Dict[self.current_date]=self.profit_settle
        self.cash_Dict[self.current_date]=self.cash

        print(f"day:{self.current_date}-cash:{self.cash}-profit:{self.profit}")
        print("当前多单持仓长度:",len(self.long_position),"持仓名单:",sorted(self.long_position.keys()))
        print("当前空单持仓长度:",len(self.short_position),"持仓名单:",sorted(self.short_position.keys()))

    return self


