import dolphindb as ddb
import pandas as pd
import json,json5
import tqdm
import matplotlib.pyplot as plt

class SimpleStrategy:
    def __init__(self,data,start_date,end_date,ori_cash:int=500000):
        self.data = data # 行情+信号字段
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.current_date = self.start_date
        self.current_timestamp = None
        self.last_date = self.start_date
        self.long_position = {}
        self.short_position = {}
        self.OrderNum = 0   # 唯一订单号
        self.stock_record = pd.DataFrame({'state': [], 'reason': [], 'date': [], "minute": [], 'symbol': [], 'price': [], 'vol': [], 'pnl': []})
        self.future_record = pd.DataFrame({'state': [], 'reason': [], 'date': [], "minute":[], 'contract': [], 'order_type': [], 'price': [], 'vol': [], 'pnl': []})
        self.option_record = pd.DataFrame({'state': [], 'reason': [], 'date': [], "minute":[], 'option': [], 'order_type': [], 'price': [], 'vol': [], 'pnl': []})
        self.profit =0      # 平仓价-开仓价
        self.profit_settle = 0  # 平仓价-昨结算价
        self.cash = ori_cash
        self.cash_Dict = {pd.to_datetime(self.start_date): ori_cash}  # 用于记录cash的历史波动:{'date':cash}
        self.profit_Dict = {pd.to_datetime(self.start_date): 0}  # 用于记录profit的历史波动:{'date':profit}
        self.settle_profit_Dict = {pd.to_datetime(self.start_date): 0}  # 用于记录settle_profit的历史波动:{'date':settle_profit}
        self.run_future = True
        self.run_option = None
        self.run_stock = False

    def execute_future(self, order_type: str, contract: str, vol, price, pre_settle, margin,
                       min_price=None, max_price=None, max_timestamp: pd.Timestamp = None, commission=None,
                       reason: str = None):
        """
        【核心函数】期货合约开仓/加仓(默认无手续费)
        margin:每笔交易的"初始"保证金[这里是初始保证金]
        min_price:平仓最小价格(多单为止损/空单为止盈)
        max_price:平仓最大价格(多单为止盈/空单为止损)
        max_timestamp: 持仓最大时间戳
        【新增】逐日盯市制度回测 pre_settle而不是settle防止未来函数
        """
        if order_type == 'long':
            position = self.long_position
        else:
            position = self.short_position
        if contract not in position:
            position[contract] = [{'price': price,
                                   'pre_settle': pre_settle,
                                   'margin': margin,
                                   'min_price': min_price,
                                   'max_price': max_price,
                                   'max_timestamp': max_timestamp,
                                   'vol': vol,
                                   'FirstDaySettle': True}]
        else:
            position[contract].append({'price': price,
                                       'pre_settle': pre_settle,
                                       'margin': margin,
                                       'min_price': min_price,
                                       'max_price': max_price,
                                       'max_timestamp': max_timestamp,
                                       'vol': vol,
                                       'FirstDaySettle': True})
        # 赋值
        if order_type == 'long':
            self.long_position = position
        else:
            self.short_position = position
        # 记录
        self.future_record = self.future_record._append({'state': 'open',
                                                         'reason': reason,
                                                         'date': self.current_date,
                                                         "timestamp": self.current_timestamp,
                                                         'contract': contract,
                                                         'order_type': order_type,
                                                         'price': price,
                                                         'vol': vol,
                                                         'pnl': 0}, ignore_index=True)

        # 结算
        self.cash -= margin  # 减去初始保证金(该笔合约的全部保证金)

    def close_future(self, order_type, contract, vol, price, reason=None):
        """【核心函数】期货合约平仓"""
        profit = 0  # 该笔交易获得的盈利(实现盈利)
        # settle_profit = 0  # 该笔交易获得的盯市盈亏(交易价-昨结价)
        margin = 0  # 该笔交易收回的保证金
        if order_type == 'long':
            position = self.long_position.copy()
        else:
            position = self.short_position.copy()
        LS = {'long': 1, 'short': -1}[order_type]  # 避免硬编码
        if position:  # 如果目前还有持仓的话
            if contract not in position:
                print(f"合约{contract}未持仓,无法平仓")
            else:
                current_vol_list = [i['vol'] for i in position[contract]]  # 当前的合约持有情况list
                ori_price_list = [i['price'] for i in position[contract]]  # 当前合约买入价格情况list
                pre_margin_list = [i['margin'] for i in position[contract]]  # 当前合约占用的保证金情况
                pre_settle_list = [i['pre_settle'] for i in position[contract]]  # 当前合约的昨结价情况
                current_vol = sum(current_vol_list)  # 当前合约持有的数量
                max_vol = min(current_vol, vol)  # 现在需要平仓的数量
                record_vol = max_vol  # for record
                if current_vol <= 0:
                    print(f"合约{contract}未持仓,无法平仓")
                elif max_vol >= current_vol:  # 说明要全平仓
                    for i in range(0, len(current_vol_list)):
                        vol = current_vol_list[i]
                        ori_price = ori_price_list[i]
                        pre_margin = pre_margin_list[i]
                        pre_settle = pre_settle_list[i]
                        profit += (price - ori_price) * vol * LS  # 逐笔盈亏
                        margin += pre_margin
                        # settle_profit += (price - pre_settle) * vol * LS  # 盯市盈亏
                        # margin += (pre_margin + settle_profit)  # 收回的保证金
                    del position[contract]  # 直接去掉这个合约的持有 {'contract':[(price,vol),...]}
                elif max_vol < current_vol:  # 说明部分平仓
                    for i in range(0, len(current_vol_list)):
                        vol = current_vol_list[i]
                        ori_price = ori_price_list[i]
                        pre_margin = pre_margin_list[i]  # 当前合约历史订单占用的保证金
                        pre_settle = pre_settle_list[i]
                        if max_vol >= vol:  # 当前订单全部平仓
                            profit += (price - ori_price) * vol * LS
                            margin += pre_margin
                            # settle_profit += (price - pre_settle) * vol * LS
                            # margin += (pre_margin + settle_profit)  # 收回的保证金
                            del position[contract][0]  # FIFO原则
                            max_vol = max_vol - vol
                        else:  # 当前订单部分平仓
                            profit += (price - ori_price) * max_vol * LS
                            margin += pre_margin
                            # settle_profit += (price - pre_settle) * max_vol * LS
                            # margin += (pre_margin * (max_vol / vol) + settle_profit)  # 收回的保证金
                            position[contract][0]['vol'] = vol - max_vol
                            position[contract][0]['margin'] = pre_margin * (1 - max_vol / vol)  # 剩余的保证金
                            break  # 执行完毕
                # 记录
                self.future_record = self.future_record._append({'state': 'close',
                                                                 'reason': reason,
                                                                 'date': self.current_date,
                                                                 "timestamp": self.current_timestamp,
                                                                 'contract': contract,
                                                                 'order_type': order_type,
                                                                 'price': price,
                                                                 'vol': record_vol,
                                                                 'pnl': profit}, ignore_index=True)
            # 结算
            self.profit += profit  # 逐笔盈亏(平仓价-开仓价)
            self.cash += margin
            # self.profit_settle += settle_profit  # 结算盈亏(平仓价-昨结算)
            # self.cash += margin  # 保证金(pre_margin+结算盈亏)
            if order_type == 'long':
                self.long_position = position
            else:
                self.short_position = position
    def monitor_future(self):
        """[每日盘后运行]"""
        long_pos = self.long_position
        short_pos = self.short_position

    def strategy(self):

        for index, row in tqdm.tqdm(self.data.iterrows(),desc="BackTesting",total=self.data.shape[0]):
            # 数据更新
            open_long, open_short = row["open_long"], row["open_short"]
            current_date, current_timestamp = row["date"], row["timestamp"]
            minute = row["minute"]
            contract = row["contract"]
            
            # 时间更新
            self.current_date = current_date
            self.current_timestamp = current_timestamp
            
            if open_long == 1:  # 不过夜
                self.execute_future(order_type="long", contract=contract, vol=50, price=row["open"],
                                    pre_settle=row["pre_settle"], margin=50*row["open"]*10,
                                    min_price=row["open"]*0.95, max_price=row["open"]*1.15,
                                    max_timestamp=pd.Timestamp(self.current_date)+pd.Timedelta(hours=15),
                                    commission=0,reason="open_long")
            elif open_short == 1:   # 不过夜
                self.execute_future(order_type="short", contract=contract, vol=50, price=row["open"],
                                    pre_settle=row["pre_settle"], margin=50*row["open"]*10,
                                    min_price=row["open"]*0.85, max_price=row["open"]*1.05,
                                    max_timestamp=pd.Timestamp(self.current_date)+pd.Timedelta(hours=15),
                                    commission=0,reason="open_short")
            
            # 检查平仓逻辑
            if minute == 1500:  # 直接清仓
                if contract in self.long_position:
                    self.close_future(order_type="long", contract=contract, vol=999, price=row["close"], reason="close_long")
                if contract in self.short_position:
                    self.close_future(order_type="short", contract=contract, vol=999, price=row["close"], reason="close_short")

            # 盘后运行
            if self.current_date!=self.last_date:
                self.monitor_future()

            # 记录
            self.profit_Dict[self.current_timestamp] = self.profit
            self.settle_profit_Dict[self.current_timestamp] = self.profit_settle
            self.cash_Dict[self.current_timestamp] = self.cash
            self.last_date = self.current_date

    def run(self):
        """运行策略+可视化"""
        self.strategy()    # 策略运行
        plt.plot(self.cash_Dict.keys(),self.cash_Dict.values(),label='cash')
        plt.legend(frameon=False)
        plt.show()

        plt.plot(list(self.profit_Dict.keys()),list(self.profit_Dict.values()),label='profit')
        # if self.run_future or self.run_option:
        #     plt.plot(list(self.settle_profit_Dict.keys()),list(self.settle_profit_Dict.values()),label='settle_profit')
        plt.legend(frameon=False)
        plt.show()

if __name__ =="__main__":
    # Configuration
    session=ddb.session()
    session.connect("localhost",8848,"admin","123456")
    from src.strategy_func.CTASeries import get_future_signal_to_dolphindb,get_future_signal_to_dataframe, CTA_strategy
    with open(r".\config\backtest_config_pandas.json5", mode="r", encoding="UTF-8") as file:
        BackTest_config  = json5.load(file)
    with open(r".\config\returnmodel_config.json5", mode="r", encoding="UTF-8") as file:
        ReturnModel_config = json5.load(file)

    # K Generational
    K_data = session.run("""
      pt=select * from loadTable("dfs://future_cn/combination","base") where isMainContract=1;
      update pt set contract="IH";
      pt""")
    future_k_path = [BackTest_config["future_K_database"], BackTest_config["future_K_table"]]
    K_data.to_parquet(rf"{future_k_path[0]}\{future_k_path[1]}.pqt", index=False)
    print(K_data)

    # Signal Generation
    # get_future_signal_to_dolphindb(session, BackTest_config=BackTest_config, ReturnModel_config=ReturnModel_config)
    save_database, save_table = BackTest_config["future_signal_database"], BackTest_config["future_signal_table"]
    get_future_signal_to_dataframe(session, BackTest_config=BackTest_config, ReturnModel_config=ReturnModel_config)
    signal_data = pd.read_parquet(f"{save_database}/{save_table}.pqt")
    print(signal_data)

    K_data = pd.merge(K_data, signal_data[["date","minute","contract","open_long","open_short"]], on=["date","minute","contract"])
    S=SimpleStrategy(data=K_data,start_date="20240101",end_date="20250101",ori_cash=5000000)
    S.run()
    future_record = S.future_record
    print(future_record)
    future_record.to_csv(f"交易明细{pd.Timestamp.today().strftime('%Y-%m-%d')}.csv",index=None)