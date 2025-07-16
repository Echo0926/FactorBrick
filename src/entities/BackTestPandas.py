from src.entities.BackTestData import *
from src.entities.BackTestConfig import *

class BaseFunction(BackTestConfig):
    def __init__(self,session,config):
        super().__init__(session,config)
    def init_counter(self):
        """【回测前运行】期货柜台&期权柜台初始化"""
        counter_init(self)
    def start_counter(self):
        """【盘前运行】daily counter start for data receiving"""
        counter_start(self)
    def time_config(self):
        """【盘后运行】配置时间"""
        config_time(self)


class TradeBehavior(BaseFunction):
    """
    交易行为添加
    """
    def __init__(self,session,config):
        super().__init__(session,config)
    def order_open_stock(self,symbol:str, vol, price,
                         static_profit: float = None, static_loss: float = None,
                         dynamic_profit: float = None, dynamic_loss: float = None,
                         min_timestamp:pd.Timestamp=None, max_timestamp: pd.Timestamp = None,
                         min_order_timestamp: pd.Timestamp = None,
                         max_order_timestamp: pd.Timestamp = None,
                         commission:float = None, reason:str = None):
        """
        【盘中运行】股票订单发送至stock_counter,
        min_timestamp: 最短持仓时间(在该時刻之前不能平仓)
        max_timestamp: 最长持仓时间(在该時刻之前必须平仓)
        min_order_timestamp:　最早开始执行该order发送的时间(交易计划)
        max_order_timestamp: 最晚开始执行该order发送的时间

        如果不设置max_order_timestamp,
        直到回测结束每天都会尝试在min_order_date后发送该订单
        """
        if not min_timestamp:
            min_timestamp=pd.Timestamp(self.start_date)
        if not max_timestamp:
            max_timestamp=pd.Timestamp(self.end_date)
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1    # 给定订单编号(唯一值)
        order_dict = {'order_state':'open',
                       'create_date': self.current_date,
                        'create_timestamp': self.current_timestamp,
                        'min_order_timestamp': min_order_timestamp,
                        'max_order_timestamp': max_order_timestamp,
                        'min_timestamp': min_timestamp,
                        'max_timestamp': max_timestamp,
                        'symbol': symbol,
                        'vol': vol,
                        'price': price,
                        'static_profit': static_profit, # 该品种/标的所有仓位共享的全局变量,一旦更新,以最新数值为准
                        'static_loss': static_loss,
                        "dynamic_profit": dynamic_profit,
                        "dynamic_loss": dynamic_loss,
                        'commission': commission,
                        'reason': reason}
        self.stock_counter[self.orderNum]= order_dict
        self.stock_order_record.append(order_dict)

    def order_close_stock(self,symbol:str, vol, price,
                          min_order_timestamp: pd.Timestamp =None,
                          max_order_timestamp: pd.Timestamp =None,
                          reason:str =None):
        """【盘中运行】股票卖出信号发送至stock_counter,如果不设置max_order_date,每天都会在min_order_date后尝试卖出该股票"""
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1  # 给定订单编号(唯一值)
        order_dict = {'order_state': 'close',
                      'create_date': self.current_date,
                      'create_timestamp': self.current_timestamp,
                      'min_order_timestamp': min_order_timestamp,
                      'max_order_timestamp': max_order_timestamp,
                      'symbol': symbol,
                      'vol': vol,
                      'price': price,
                      'reason': reason}
        self.stock_counter[self.orderNum]= order_dict
        self.stock_order_record.append(order_dict)

    def order_open_future(self,order_type:str, contract:str, vol, price, pre_settle, margin,
                          static_profit: float = None, static_loss: float=None,
                          dynamic_profit: float = None, dynamic_loss: float = None,
                          min_timestamp:pd.Timestamp=None, max_timestamp:pd.Timestamp=None,
                          min_order_timestamp:pd.Timestamp=None, max_order_timestamp:pd.Timestamp=None,
                          commission=None, reason=None):
        """
        【盘中运行】期货订单发送至future_counter,
        min_timestamp: 最短持仓时间(在该時刻之前不能平仓)
        max_timestamp: 最长持仓时间(在该時刻之前必须平仓)
        min_order_timestamp:　最早开始执行该order发送的时间(交易计划)
        max_order_timestamp: 最晚开始执行该order发送的时间

        如果不设置max_order_timestamp,
        直到回测结束每天都会尝试在min_order_date后发送该订单
        """
        if not min_timestamp:
            min_timestamp=pd.Timestamp(self.start_date)
        if not max_timestamp:
            max_timestamp=pd.Timestamp(self.end_date)
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1    # 给定订单编号(唯一值)
        # 【Attention: pre_settle需要更新!!!】
        order_dict = {'order_state':'open',
                     'order_type':order_type,
                     'create_date':self.current_date,
                      'create_timestamp':self.current_timestamp,
                      'min_timestamp':min_timestamp,
                      'max_timestamp':max_timestamp,
                      'min_order_timestamp':min_order_timestamp,
                      'max_order_timestamp':max_order_timestamp,
                      'contract':contract,
                      'vol':vol,
                      'price':price,
                      'pre_settle':pre_settle,
                      'margin':margin,
                      'static_profit': static_profit,
                      'static_loss': static_loss,
                      "dynamic_profit": dynamic_profit,
                      "dynamic_loss": dynamic_loss,
                      'commission':commission,
                      'reason':reason}
        self.future_counter[self.orderNum]= order_dict
        self.future_order_record.append(order_dict)

    def order_close_future(self,order_type:str, contract:str, vol, price,
                           min_order_timestamp=None, max_order_timestamp=None, reason:str=None):
        """【盘中运行】期货平仓发送至future_counter,如果不设置max_order_date,每天都会在min_order_date后尝试平仓该订单"""
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1  # 给定订单编号(唯一值)
        order_dict = {'order_state':'close',
                      'order_type':order_type,
                      'create_date':self.current_date,
                      'create_timestamp':self.current_timestamp,
                      'min_order_timestamp':min_order_timestamp,
                      'max_order_timestamp':max_order_timestamp,
                      'contract':contract,
                      'vol':vol,
                      'price':price,
                      'reason':reason}
        self.future_counter[self.orderNum]= order_dict
        self.future_order_record.append(order_dict)

    def order_open_option(self,order_type:str, order_BS:str, option:str, vol, price,
                          pre_settle, strike, margin,
                          static_profit: float = None, static_loss: float=None,
                          dynamic_profit: float = None, dynamic_loss: float = None,
                          min_timestamp: pd.Timestamp = None, max_timestamp: pd.Timestamp = None,
                          min_order_timestamp:pd.Timestamp=None, max_order_timestamp:pd.Timestamp=None,
                          commission=None, reason:str=None):
        """
        【盘中运行】期权订单发送至option_counter,
        min_timestamp: 最短持仓时间(在该時刻之前不能平仓)
        max_timestamp: 最长持仓时间(在该時刻之前必须平仓)
        min_order_timestamp:　最早开始执行该order发送的时间(交易计划)
        max_order_timestamp: 最晚开始执行该order发送的时间

        如果不设置max_order_timestamp,
        直到回测结束每天都会尝试在min_order_date后发送该订单
        """
        if not min_timestamp:
            min_timestamp=pd.Timestamp(self.start_date)
        if not max_timestamp:
            max_timestamp=pd.Timestamp(self.end_date)
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        if order_BS=='buy': # 期权买方不用付保证金
            margin=0
        self.orderNum+=1    # 给定订单编号(唯一值)
        # 【Attention: pre_settle需要更新!!!】
        order_dict = {'order_state':'open',
                      'order_type':order_type,    # call/put
                      'order_BS':order_BS,        # buy/sell
                      'create_date':self.current_date,
                      "create_timestamp":self.current_timestamp,
                      'min_order_timestamp':min_order_timestamp,
                      'max_order_timestamp':max_order_timestamp,
                      'min_timestamp': min_timestamp,
                      'max_timestamp': max_timestamp,
                      'option':option,
                      'vol':vol,
                      'price':price,
                      'pre_settle':pre_settle,
                      'margin':margin,
                      'strike':strike,
                      'static_profit':static_profit,
                      'static_loss':static_loss,
                      "dynamic_profit": dynamic_profit,
                      "dynamic_loss": dynamic_loss,
                      'commission':commission,
                      'reason':reason}
        self.option_counter[self.orderNum]=order_dict
        self.option_order_record.append(order_dict)

    def order_close_option(self,order_type:str, order_BS:str, option:str, vol, price,
                           min_order_timestamp:pd.Timestamp=None, max_order_timestamp:pd.Timestamp=None,
                           reason:str=None):
        """【盘中运行】期权平仓发送至option_counter,如果不设置max_order_date,每天都会尝试平仓该订单"""
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1  # 给定订单编号(唯一值)
        order_dict = {'order_state':'close',
                      'order_type':order_type,
                      'order_BS':order_BS,
                      'create_date':self.current_date,
                      "create_timestamp":self.current_timestamp,
                      'min_order_date':min_order_timestamp,
                      'max_order_date':max_order_timestamp,
                      'option':option,
                      'vol':vol,
                      'price':price,
                      'reason':reason}
        self.option_counter[self.orderNum]=order_dict
        self.option_order_record.append(order_dict)


class CounterBehavior(TradeBehavior):
    """
    Counter Behavior:
    - execute (开仓)
    - close (平仓)
    - clear (清仓)
    - calculate_profit (核算)
    """
    def __init__(self,session,config):
        super().__init__(session, config)

    def execute_stock(self, symbol, vol, price,
                      static_profit: float = None, static_loss: float=None,
                      dynamic_profit: float = None, dynamic_loss: float = None,
                      min_timestamp:pd.Timestamp=None, max_timestamp:pd.Timestamp=None, commission=None,
                      reason:str=None):
        """
        【核心函数】股票开仓/加仓(默认无手续费)
        min_price:平仓最小价格(止损)
        max_price:平仓最大价格(止盈)
        max_timestamp:持仓最大时间戳
        """
        pos_dict = {'price':price,
                    'vol': vol,
                    'min_timestamp':min_timestamp,
                    'max_timestamp':max_timestamp,
                    'time_monitor':0}
        if symbol not in self.stock_position:
            self.stock_position[symbol]=[pos_dict]
        else:
            self.stock_position[symbol].append(pos_dict)
        if symbol not in self.stock_summary:
            self.stock_summary[symbol] = {
                "ori_price": price,   # 买入均价
                "total_vol": vol,   # 持仓总量
                'static_profit': static_profit,
                'static_loss': static_loss,
                'dynamic_profit': dynamic_profit,
                'dynamic_loss': dynamic_loss,
                "history_min": price,   # 持仓以来的历史最低价
                "history_max": price,   # 持仓以来的历史最高价
            }
        else:
            # 更新止盈止损属性
            self.stock_summary[symbol].update(
                {'static_profit': static_profit,
                'static_loss': static_loss,
                'dynamic_profit': dynamic_profit,
                'dynamic_loss': dynamic_loss})
            Dict = self.stock_summary[symbol].copy()
            # 更新买入均价 & 持仓总量
            vol0, amount0 = self.stock_summary[symbol]["total_vol"], self.stock_summary[symbol]["total_vol"] * self.stock_summary[symbol]["ori_price"]
            vol1, amount1 = vol0+vol, amount0 + price * vol
            self.stock_summary[symbol]["ori_price"] = amount1/vol1
            self.stock_summary[symbol]["total_vol"] = vol1

        # 记录
        self.stock_record=self.stock_record._append({'state':'open',
                                                     'reason':reason,
                                                     'date':self.current_date,
                                                     'minute':int(self.current_minute),
                                                     'timestamp':self.current_timestamp,
                                                     'symbol':symbol,
                                                     'price':price,
                                                     'vol':vol,
                                                     'pnl':0,
                                                    },ignore_index=True)

        # 结算
        self.cash-=vol*price  # 减去股票购买成本

    def execute_future(self, order_type:str, contract:str, vol, price, pre_settle, margin,
                       static_profit: float = None, static_loss: float = None,
                       dynamic_profit: float = None, dynamic_loss: float = None,
                       min_timestamp: pd.Timestamp=None, max_timestamp:pd.Timestamp=None,
                       commission=None, reason:str=None):
        """
        【核心函数】期货合约开仓/加仓(默认无手续费)
        margin:每笔交易的"初始"保证金[这里是初始保证金]
        min_price:平仓最小价格(多单为止损/空单为止盈)
        max_price:平仓最大价格(多单为止盈/空单为止损)
        max_timestamp: 持仓最大时间戳
        【新增】逐日盯市制度回测 pre_settle而不是settle防止未来函数
        """
        if order_type=='long':
            position=self.long_position.copy()
            summary=self.long_summary.copy()
        else:
            position=self.short_position.copy()
            summary=self.short_summary.copy()
        pos_dict = {'price':price,
                    'vol': vol,
                    'pre_settle':pre_settle,
                    'margin':margin,
                    'min_timestamp':min_timestamp,
                    'max_timestamp':max_timestamp,
                    'time_monitor': 0,
                    'hold_days': 0
                    }
        if contract not in position:
            position[contract]=[pos_dict]
        else:
            position[contract].append(pos_dict)
        if contract not in summary:
            summary[contract] = {
                "ori_price": price, # 买入均价
                "total_vol": vol,   # 持仓总量
                'static_profit': static_profit,
                'static_loss': static_loss,
                'dynamic_profit': dynamic_profit,
                'dynamic_loss': dynamic_loss,
                "history_min": price,   # 持仓以来的历史最低价
                "history_max": price,   # 持仓以来的历史最高价
            }
        else:
            # 更新止盈止损属性
            summary[contract].update({
                'static_profit': static_profit,
                'static_loss': static_loss,
                'dynamic_profit': dynamic_profit,
                'dynamic_loss': dynamic_loss,
            })
            # 更新买入均价 & 持仓总量
            vol0, amount0 = summary[contract]["total_vol"], summary[contract]["total_vol"] * summary[contract]["ori_price"]
            vol1, amount1 = vol0 + vol, amount0 + price * vol
            summary[contract]["ori_price"] = amount1 / vol1
            summary[contract]["total_vol"] = vol1

        # 赋值
        if order_type=='long':
            self.long_position=position
            self.long_summary=summary
        else:
            self.short_position=position
            self.short_summary=summary
        # 记录
        self.future_record=self.future_record._append({'state':'open',
                                                       'reason':reason,
                                                       'date':self.current_date,
                                                       'minute':int(self.current_minute),
                                                       'timestamp':self.current_timestamp,
                                                       'contract':contract,
                                                       'order_type':order_type,
                                                       'price':price,
                                                       'vol':vol,
                                                       'pnl':0},ignore_index=True)

        # 结算
        self.cash-=margin           # 减去初始保证金(该笔合约的全部保证金)

    def execute_option(self, order_type:str, order_BS:str, option:str, vol, price, strike, pre_settle,
                       margin:float=None,
                       static_profit: float = None, static_loss: float = None,
                       dynamic_profit: float = None, dynamic_loss: float = None,
                       min_timestamp:pd.Timestamp=None, max_timestamp:pd.Timestamp=None,
                       commission=None, reason:str=None):
        """【核心函数】买入看涨(order_type='call')/看跌(order_type='sell')期权"""
        if order_type=='call' and order_BS=='buy':  # 期权买方没有保证金
            position=self.buycall_position
            summary=self.buycall_summary
            margin=0
        elif order_type=='call' and order_BS=='sell':
            position=self.sellcall_position
            summary=self.sellcall_summary
        elif order_type=='put' and order_BS=='buy': # 期权买方没有保证金
            position=self.buyput_position
            summary=self.buyput_summary
            margin=0
        else:
            position=self.sellput_position
            summary=self.sellput_summary

        pos_dict = {'price':price,
                    'vol': vol,
                    'pre_settle':pre_settle,
                    'margin':margin,
                    'strike':strike,
                    'min_timestamp':min_timestamp,
                    'max_timestamp':max_timestamp,
                    'time_monitor':0,
                    'hold_days':0}

        if option not in position:
            position[option]=[pos_dict]
        else:
            position[option].append(pos_dict)

        if option not in summary:
            summary[option] = {
                "ori_price": price,  # 买入均价
                "total_vol": vol,  # 持仓总量
                'static_profit': static_profit,
                'static_loss': static_loss,
                'dynamic_profit': dynamic_profit,
                'dynamic_loss': dynamic_loss,
                "history_min": price,  # 持仓以来的历史最低价
                "history_max": price,  # 持仓以来的历史最高价
            }
        else:
            # 更新止盈止损属性
            summary[option].update({
                'static_profit': static_profit,
                'static_loss': static_loss,
                'dynamic_profit': dynamic_profit,
                'dynamic_loss': dynamic_loss,
            })
            # 更新买入均价 & 持仓总量
            vol0, amount0 = summary[option]["total_vol"], summary[option]["total_vol"] * summary[option]["ori_price"]
            vol1, amount1 = vol0 + vol, amount0 + price * vol
            summary[option]["ori_price"] = amount1 / vol1
            summary[option]["total_vol"] = vol1

        # 赋值
        if order_type=='call' and order_BS=='buy':
            self.buycall_position=position
            self.buycall_summary=summary
            self.cash-=(vol*price)  # 减去付出的权利金
        elif order_type=='call' and order_BS=='sell':
            self.sellcall_position=position
            self.sellcall_summary=summary
            self.cash+=(vol*price-margin)  # 加上得到的权利金减去保证金
        elif order_type=='put' and order_BS=='buy':
            self.buyput_position=position
            self.buyput_summary=summary
            self.cash-=(vol*price)  # 减去付出的权利金
        elif order_type=='put' and order_BS=='sell':
            self.sellput_position=position
            self.sellput_summary=summary
            self.cash+=(vol*price-margin)  # 加上得到的权利金减去保证金
        # 记录
        self.option_record=self.option_record._append({'state':order_BS,
                                                       'reason':reason,
                                                       'date':self.current_date,
                                                       "timestamp":self.current_timestamp,
                                                       'option':option,
                                                       'order_type':order_type,
                                                       'price':price,
                                                       'vol':vol,
                                                       'pnl':0},ignore_index=True)

    def close_stock(self,symbol,vol,price,reason=None):
        """【核心函数】股票平仓
        [新增]: 无法平仓time_monitor=1的仓位
        """
        profit=0    # 该笔交易获得的盈利(实现盈利)
        margin=0    # 该笔交易使用的保证金
        position=self.stock_position.copy()
        summary=self.stock_summary.copy()
        if position:    # 如果目前还有持仓的话
            if symbol not in position.keys():
                print(f"股票{symbol}未持仓,无法平仓")
            else:
                current_vol_list=[i['vol'] for i in position[symbol]]    # 当前的股票持有情况list
                ori_price_list=[i['price'] for i in position[symbol]]    # 当前股票买入价格情况list
                time_monitor_list=[i["time_monitor"] for i in position[symbol]]  # 当前股票的持仓时间状态(-2为禁止卖出)
                current_vol, state = (sum(current_vol_list[:time_monitor_list.index(-2)]), False) if -2 in time_monitor_list else (sum(current_vol_list), True)  # 当前允许平仓的最大数量
                if current_vol == 0:
                    return  # 说明不可能平仓
                max_vol=min(current_vol,vol)        # 现在需要平仓的数量
                record_vol=max_vol                  # for record
                if max_vol>=current_vol and state:  # 说明要可以平仓
                    # 先对视图进行批处理
                    del summary[symbol]
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        margin+=price*vol
                        profit+=(price-ori_price)*vol  # 逐笔盈亏
                    del position[symbol]  # 直接去掉这个股票的持有 {'symbol':[(price,vol),...]}
                else:
                    # 先对视图进行批处理
                    Dict = summary[symbol].copy()
                    vol0, amount0 = Dict["total_vol"], Dict["total_vol"] * Dict["ori_price"]
                    vol1, amount1 = Dict["total_vol"] + vol, amount0 + price * vol
                    summary[symbol]["ori_price"] = amount1 / vol1
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        if max_vol>=vol:  # 当前订单全部平仓
                            margin+=price*vol
                            profit+=(price-ori_price)*vol
                            del position[symbol][0]  # FIFO原则
                            max_vol=max_vol-vol
                        else:  # 当前订单部分平仓
                            margin+=price*max_vol
                            profit+=(price-ori_price)*max_vol
                            position[symbol][0]['vol']=vol-max_vol
                            break  # 执行完毕
                # 记录
                self.stock_record=self.stock_record._append({'state':'close',
                                                             'reason':reason,
                                                             'date':self.current_date,
                                                             'minute': int(self.current_minute),
                                                             "timestamp":self.current_timestamp,
                                                             'symbol':symbol,
                                                             'price':price,
                                                             'vol':record_vol,
                                                             'pnl':profit},ignore_index=True)
                # 结算
                self.profit+=profit                  # 逐笔盈亏(平仓价-开仓价)
                self.cash+=margin           # 获得的利润计入cash
                self.stock_position=position
                self.stock_summary=summary

    def close_future(self,order_type,contract,vol,price,reason=None):
        """【核心函数】期货合约平仓"""
        profit=0    # 该笔交易获得的盈利(实现盈利)
        settle_profit=0   # 该笔交易获得的盯市盈亏(交易价-昨结价)
        margin=0    # 该笔交易收回的保证金
        if order_type=='long':
            position=self.long_position.copy()
            summary = self.long_summary.copy()
        else:
            position=self.short_position.copy()
            summary = self.short_summary.copy()
        LS={'long':1,'short':-1}[order_type]    # 避免硬编码
        if position:    # 如果目前还有持仓的话
            if contract not in position:
                print(f"合约{contract}未持仓,无法平仓")
            else:
                current_vol_list=[i['vol'] for i in position[contract]]    # 当前的合约持有情况list
                ori_price_list=[i['price'] for i in position[contract]]    # 当前合约买入价格情况list
                pre_margin_list=[i['margin'] for i in position[contract]]  # 当前合约占用的保证金情况
                pre_settle_list=[i['pre_settle'] for i in position[contract]]   # 当前合约的昨结价情况
                hold_days_list=[i['hold_days'] for i in position[contract]] # 当前合约的持仓时间
                time_monitor_list=[i["time_monitor"] for i in position[contract]]  # 当前期货的持仓时间状态(-2为禁止卖出)
                current_vol, state = (sum(current_vol_list[:time_monitor_list.index(-2)]), False) if -2 in time_monitor_list else (sum(current_vol_list), True)  # 当前允许平仓的最大数量
                if current_vol == 0:
                    return  # 说明不可能平仓
                max_vol=min(current_vol,vol)         # 现在需要平仓的数量
                record_vol=max_vol                   # for record
                if max_vol>=current_vol and state:  # 说明可以全平仓
                    # 先对视图进行批处理
                    del summary[contract]
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        pre_margin=pre_margin_list[i]
                        pre_settle=pre_settle_list[i]
                        hold_days=hold_days_list[i]
                        profit+=(price-ori_price)*vol*LS            # 逐笔盈亏
                        if hold_days == 0:  # TODO:说明是日内平仓,添加commission或其他条件
                            settle_profit+=(price-ori_price)*vol*LS   # 此时:盯市盈亏 == 逐笔盈亏
                        else:   # 说明是隔日平仓
                            settle_profit+=(price-pre_settle)*vol*LS    # 盯市盈亏
                        margin+=(pre_margin+settle_profit)          # 收回的保证金
                    del position[contract]   # 直接去掉这个合约的持有 {'contract':[(price,vol),...]}
                else:    # 说明部分平仓
                    # 先对视图进行批处理
                    Dict = summary[contract].copy()
                    vol0, amount0 = Dict["total_vol"], Dict["total_vol"] * Dict["ori_price"]
                    vol1, amount1 = Dict["total_vol"] + vol, amount0 + price * vol
                    summary[contract]["ori_price"] = amount1 / vol1
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        pre_margin=pre_margin_list[i]   # 当前合约历史订单占用的保证金
                        pre_settle=pre_settle_list[i]
                        hold_days=hold_days_list[i]
                        if max_vol>=vol:    # 当前订单全部平仓
                            profit+=(price-ori_price)*vol*LS
                            if hold_days == 0:  # TODO:说明是日内平仓,添加commission或其他条件
                                settle_profit+=(price-ori_price)*vol*LS  # 此时:盯市盈亏 == 逐笔盈亏
                            else:  # 说明是隔日平仓
                                settle_profit+=(price-pre_settle)*vol*LS  # 盯市盈亏
                            margin+=(pre_margin+settle_profit)  # 收回的保证金
                            del position[contract][0]    # FIFO原则
                            max_vol=max_vol-vol
                        else:               # 当前订单部分平仓
                            profit+=(price-ori_price)*max_vol*LS
                            if hold_days == 0:  # TODO:说明是日内平仓,添加commission或其他条件
                                settle_profit+=(price-ori_price)*max_vol*LS  # 此时:盯市盈亏 == 逐笔盈亏
                            else:  # 说明是隔日平仓
                                settle_profit+=(price-pre_settle)*max_vol*LS  # 盯市盈亏
                            margin+=(pre_margin*(max_vol/vol)+settle_profit)    # 收回的保证金
                            position[contract][0]['vol']=vol-max_vol
                            position[contract][0]['margin']=pre_margin*(1-max_vol/vol)  # 剩余的保证金
                            break   # 执行完毕
                # 记录
                self.future_record=self.future_record._append({'state':'close',
                                                               'reason':reason,
                                                               'date': self.current_date,
                                                               'minute': int(self.current_minute),
                                                               'timestamp': self.current_timestamp,
                                                               'contract':contract,
                                                               'order_type':order_type,
                                                               'price':price,
                                                               'vol':record_vol,
                                                               'pnl':profit},ignore_index=True)
            # 结算
            self.profit+=profit                  # 逐笔盈亏(平仓价-开仓价)
            self.profit_settle+=settle_profit    # 结算盈亏(平仓价-昨结算)
            self.cash+=margin                    # 保证金(pre_margin+结算盈亏)
            if order_type=='long':
                self.long_position=position
                self.long_summary=summary
            else:
                self.short_position=position
                self.short_summary=summary

    def close_option(self,order_type,order_BS,option,vol,price,reason=None):
        """【核心函数】期权合约平仓
        #TODO: 修改期权买方的平仓逻辑
        """
        profit=0    # 该笔交易获得的盈利(实现盈利)
        settle_profit=0   # 该笔交易获得的盯市盈亏(交易价-昨结价)
        margin=0    # 该笔交易收回的保证金
        if order_BS == "buy" and order_type=='call':
            position=self.buycall_position.copy()
            summary=self.buycall_summary.copy()
        elif order_BS == "buy" and order_type=="put":
            position=self.buyput_position.copy()
            summary=self.buyput_summary.copy()
        elif order_BS == "sell" and order_type=="call":
            position=self.sellcall_position.copy()
            summary=self.sellcall_summary.copy()
        else:
            position=self.sellput_position.copy()
            summary=self.sellput_summary.copy()
        BS={'buy':1,'sell':-1}[order_BS]         # 避免硬编码
        if position:    # 如果当前还有持仓的话
            if option not in position:
                print(f"合约{option}未持仓,无法平仓")
            else:
                current_vol_list=[i['vol'] for i in position[option]]    # 当前的合约持有情况list
                ori_price_list=[i['price'] for i in position[option]]    # 当前合约买入价格情况list
                pre_margin_list=[i['margin'] for i in position[option]]  # 当前合约买入的保证金情况
                pre_settle_list=[i['pre_settle'] for i in position[option]]   # 当前合约的昨结价情况
                hold_days_list=[i['hold_days'] for i in position[option]] # 当前合约的持仓时间
                time_monitor_list=[i["time_monitor"] for i in position[option]]  # 当前期货的持仓时间状态(-2为禁止卖出)
                current_vol, state = (sum(current_vol_list[:time_monitor_list.index(-2)]), False) if -2 in time_monitor_list else (sum(current_vol_list), True)  # 当前允许平仓的最大数量
                if current_vol == 0:
                    return  # 说明不可能平仓
                max_vol=min(current_vol,vol)   # 现在需要平仓的数量
                # ??? self.cash+=max_vol*price*BS    # 期权买方(B)平仓需要卖出期权,得到cash&期权卖方(S)平仓需要买入期权,扣除cash
                record_vol=max_vol  # for record
                if max_vol>=current_vol and state:  # 说明要全平仓
                    # 先对视图进行批处理
                    del summary[option]
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        pre_margin=pre_margin_list[i]
                        pre_settle=pre_settle_list[i]
                        hold_days=hold_days_list[i]
                        profit+=(price-ori_price)*vol*BS            # 逐笔盈亏(平仓价-开仓价)
                        if hold_days == 0:  # TODO:说明是日内平仓,添加commission或其他条件
                            settle_profit+=(price-ori_price)*vol*BS   # 此时:盯市盈亏 == 逐笔盈亏
                        else:   # 说明是隔日平仓
                            settle_profit+=(price-pre_settle)*vol*BS    # 盯市盈亏
                        margin+=(pre_margin+settle_profit)          # 收回的保证金
                    del position[option]   # 直接去掉这个合约的持有 {'option':[(price,vol),...]}
                else:    # 说明部分平仓
                    # 先对视图进行批处理
                    Dict = summary[option].copy()
                    vol0, amount0 = Dict["total_vol"], Dict["total_vol"] * Dict["ori_price"]
                    vol1, amount1 = Dict["total_vol"] + vol, amount0 + price * vol
                    summary[option]["ori_price"] = amount1 / vol1
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        pre_margin=pre_margin_list[i]   # 当前合约历史订单占用的保证金
                        pre_settle=pre_settle_list[i]
                        hold_days=hold_days_list[i]
                        if max_vol>=vol:
                            profit+=(price-ori_price)*vol*BS            # 逐笔盈亏(平仓价-开仓价)
                            if hold_days == 0:  # TODO:说明是日内平仓,添加commission或其他条件
                                settle_profit+=(price-ori_price)*vol*BS  # 此时:盯市盈亏 == 逐笔盈亏
                            else:  # 说明是隔日平仓
                                settle_profit+=(price-pre_settle)*vol*BS  # 盯市盈亏
                            margin+=(pre_margin+settle_profit)          # 结算盈亏(平仓价-昨结算)
                            del position[option][0]                     # FIFO原则
                            max_vol=max_vol-vol
                        else:                 # 当前订单部分平仓
                            profit+=(price-ori_price)*max_vol*BS
                            if hold_days == 0:  # TODO:说明是日内平仓,添加commission或其他条件
                                settle_profit+=(price-ori_price)*max_vol*BS  # 此时:盯市盈亏 == 逐笔盈亏
                            else:  # 说明是隔日平仓
                                settle_profit+=(price-pre_settle)*max_vol*BS  # 盯市盈亏
                            margin+=(pre_margin*(max_vol/vol)+settle_profit)    # 收回的保证金
                            position[option][0]['vol']=vol-max_vol
                            position[option][0]['margin']=pre_margin*(1-max_vol/vol)  # 剩余的保证金
                            break   # 执行完毕
                # 记录
                self.option_record=self.option_record._append({'state':'close',
                                                               'reason':reason,
                                                               'date':self.current_date,
                                                               'minute': int(self.current_minute),
                                                               "timestamp":self.current_timestamp,
                                                               'option':option,
                                                               'order_type':order_type,
                                                               'price':price,
                                                               'vol':record_vol,
                                                               'pnl':profit},ignore_index=True)
            # 结算
            self.profit+=profit                 # 逐笔盈亏(平仓价-开仓价)
            self.profit_settle+=settle_profit   # 结算盈亏(平仓价-昨结算)
            self.cash+=margin                   # 保证金
            if order_type=='call' and order_BS=='buy':
                self.buycall_position=position
                self.buycall_summary=summary
            elif order_type=='call' and order_BS=='sell':
                self.sellcall_position=position
                self.sellcall_summary=summary
            elif order_type=='put' and order_BS=='buy':
                self.buyput_position=position
                self.buyput_summary=summary
            elif order_type=='put' and order_BS=='sell':
                self.sellput_position=position
                self.sellput_summary=summary

    def clear_option(self,order_type:str,order_BS:str,option:str,vol,reason:str="clear"):
        """【核心函数】期权到期清仓(卖方&买方通用)"""
        self.close_option(order_type=order_type,
                          order_BS=order_BS,
                          option=option,
                          vol=vol,
                          price=0,
                          reason=reason)

    def calculate_future_profit(self,order_type):
        """
        【每日盘后运行】计算未平仓合约的盯市盈亏+更新pre_settle为收盘后的settle
        【新增】settle_profit 每日盘后运行,计算浮盈浮亏(结算价-昨日结算价)并计入保证金
        【补丁】在持仓中增加了hold_days字段,仅用来计算第一天收益(结算-开仓)
        profit:逐笔平仓盈亏(平仓-开仓)+profit_settle结算盈亏(开仓-昨日结算)=平仓盈亏(平仓-昨日结算)
        order_type='long':
        order_type='short':
        """
        if order_type=='long':
            p=self.long_position.copy()
        else:
            p=self.short_position.copy()
        LS = {'long': 1, 'short': -1}[order_type]  # 【新增】为了节省代码段加了一个系数,按多头的逻辑对空头收益进行计算
        if p: # 如果有持仓的话
            POS=p.copy()
            future_info_dict = self.future_info_dict.copy()
            for contract,List in p.items():   # 获取当前结算价(waiting)
                if contract not in future_info_dict:
                    continue
                info_dict =  future_info_dict[contract]
                L=[]
                for Dict in List:
                    pre_settle,settle_price=info_dict['pre_settle'],info_dict['settle'] # 未平仓合约昨日结算价&当日结算价
                    vol=Dict['vol']
                    if Dict["hold_days"] >=1: # 说明不是第一天持仓
                        settle_profit=(settle_price-pre_settle)*vol*LS          # 第K天盯市盈亏 (settle-pre_settle)
                    else:       # 说明是第一天持仓
                        settle_profit=(settle_price-Dict['price'])*vol*LS       # 第1天盯市盈亏 (settle-price)
                    self.profit_settle+=settle_profit
                    Dict["hold_days"] += 1  # 说明是第一天持仓
                    Dict['margin']+=settle_profit
                    Dict['pre_settle']=settle_price # 更新pre_settle为收盘后的settle
                    L.append(Dict)
                POS[contract]=L

            """更新self.long_position/self.short_position"""
            if order_type=='long':
                self.long_position=POS
            else:
                self.short_position=POS
        else:
            pass

    def calculate_option_profit(self,order_type,order_BS):
        """【每日盘后运行】计算期权逐日盈亏&盯市盈亏
        【补丁】在持仓中增加了hold_days,仅用来计算第一天收益(结算-开仓)
        """
        if order_type=='call' and order_BS=='buy':
            p=self.buycall_position.copy()
        elif order_type=='call' and order_BS=='sell':
            p=self.sellcall_position.copy()
        elif order_type=='put' and order_BS=='buy':
            p=self.buyput_position.copy()
        else:
            p=self.sellput_position.copy()
        if p:   # 如果有持仓的话
            POS=p.copy()
            option_info_dict = self.option_info_dict.copy()
            BS={'buy':1,'sell':-1}[order_BS]
            for option,List in p.items():   # 获取当前结算价(waiting)
                if option not in option_info_dict:
                    continue
                info_dict = option_info_dict[option]
                L=[]
                for Dict in List:
                    pre_settle,settle_price=info_dict['pre_settle'],info_dict['settle'] # 未平仓合约昨日结算价&当日结算价
                    vol=Dict['vol']
                    if Dict["hold_days"] >=1: # 说明不是第一天持仓
                        settle_profit=(settle_price-pre_settle)*vol*BS          # 第K天盯市盈亏 (settle-pre_settle)
                    else:       # 说明是第一天持仓
                        settle_profit=(settle_price-Dict['price'])*vol*BS       # 第1天盯市盈亏 (settle-price)
                    self.profit_settle+=settle_profit
                    Dict["hold_days"] += 1  # 说明是第一天持仓
                    Dict['margin']+=settle_profit
                    Dict['pre_settle']=settle_price # 更新pre_settle为收盘后的settle
                    L.append(Dict)
                POS[option]=L

            if order_type=='call' and order_BS=='buy':
                self.buycall_position=POS
            elif order_type=='call' and order_BS=='sell':
                self.sellcall_position=POS
            elif order_type=='put' and order_BS=='buy':
                self.buyput_position=POS
            else:
                self.sellput_position=POS
        else:
            pass

    def monitor_stock(self,order_sequence):
        """
        【柜台处理订单后运行,可重复运行】每日盘中运行,负责监控当前持仓是否满足限制平仓要求
        order_sequence=True 假设max_price先判断
        order_sequence=False 假设min_price先判断
        """
        pos=self.stock_position.copy()
        summary=self.stock_summary.copy()
        if len(pos) == 0:
            return

        date, minute = self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式
        stock_info_dict = self.stock_info_dict.copy() # 今天的标的基本信息
        stock_k_dict = self.stock_k_dict.copy()       # 今天的K线信息

        for symbol,List in pos.items():
            info_dict = stock_info_dict[symbol]
            k_dict = stock_k_dict[symbol]
            if minute not in k_dict:
                continue
            k_dict = k_dict[minute]
            SummaryDict = summary[symbol]

            # Step0. 基本信息
            end_date,daily_max_price,daily_min_price=info_dict['end_date'],info_dict['high'],info_dict['low']  # 基本信息
            high_price, low_price, close_price = k_dict['high'], k_dict['low'], k_dict['close']  # K线信息
            ori_price,static_profit,static_loss,dynamic_profit,dynamic_loss,history_high,history_low = (
                SummaryDict["ori_price"],SummaryDict['static_profit'],SummaryDict['static_loss'],SummaryDict['dynamic_profit'],
                SummaryDict['dynamic_loss'],SummaryDict['history_max'],SummaryDict['history_min'])
            static_high = ori_price * (1 + static_profit) if static_profit else None
            static_low = ori_price * (1 - static_loss) if static_loss else None
            dynamic_high = history_low * (1 + dynamic_profit) if dynamic_profit else None
            dynamic_low = history_high * (1 - dynamic_loss) if dynamic_loss else None
            SummaryDict['history_min'] = min(SummaryDict['history_min'], low_price)   # 更新视图中的history_min & history_max
            SummaryDict['history_max'] = max(SummaryDict['history_max'], high_price)
            self.stock_summary[symbol] = SummaryDict

            # Step1. 先判断每一个仓位有没有到达最长持仓时间
            i = 0
            while i < len(List):
                Dict = List[i]
                if symbol not in self.stock_position:                # 保护
                    i+=1
                    continue
                if Dict['time_monitor'] < 0:  # 该日禁止平仓(如股票的T+1制度)[-2]/没到最长持仓时间
                    i+=1
                    continue
                price,vol,first_timestamp,last_timestamp=(Dict['price'],Dict['vol'],Dict['min_timestamp'],Dict['max_timestamp'])    # 持仓信息
                time_monitor = Dict['time_monitor']
                if Dict['time_monitor'] == 0:  # 还没判断过
                    if pd.Timestamp(first_timestamp) >= pd.Timestamp(date):
                        time_monitor = -2   # 禁止平仓(如股票的T+1制度)
                    elif pd.Timestamp(end_date) > pd.Timestamp(date) and pd.Timestamp(first_timestamp) < pd.Timestamp(date) < pd.Timestamp(last_timestamp):   # 这里date省略了分钟,不能带等号
                        time_monitor = -1
                    else:
                        time_monitor = 1
                    self.stock_position[symbol][i]['time_monitor'] = time_monitor

                i+=1    # 自動index

                # Step1. 限时单
                if time_monitor == 1:
                    if self.current_timestamp>=pd.Timestamp(end_date): # 最长持仓时间的股票持仓
                        self.close_stock(symbol=symbol, price=close_price, vol=vol, reason='end_date')  # end_date都是所有仓位共享的,这样处理不会有问题
                        continue
                    if self.current_timestamp>=pd.Timestamp(last_timestamp): # 最长持仓时间的期货持仓
                        self.close_stock(symbol=symbol, price=close_price, vol=vol, reason='max_timestamp') # 由于这里一定是按照FIFO，且end_timestamp依次递增,所以不会有问题

            # Step2. 再判断是否要止盈止损
            # if Dict['static_monitor'] == 0:  # 原始逻辑
            if True:
                if not static_high and not static_low:
                    static_monitor = -1
                elif not static_high and daily_min_price < static_low:  # 不设最大价格+当日最低价格<目标最低价(静态)
                    static_monitor = -1
                elif not static_low and daily_max_price > static_high:  # 不设最小价格+当日最高价格>目标最高价(静态)
                    static_monitor = -1
                elif daily_min_price < static_low and daily_max_price > static_high:
                    static_monitor = -1
                else:
                    static_monitor = 1
                if symbol in self.stock_summary:
                    self.stock_summary[symbol]['static_monitor'] = static_monitor

            # if Dict['dynamic_monitor'] == 0:  # 原始逻辑
            if True:
                if not dynamic_high and not dynamic_low:
                    dynamic_monitor = -1
                elif not dynamic_high and daily_min_price < dynamic_low:  # 不设最大价格+当日最低价格<目标最低价(动态)
                    dynamic_monitor = -1
                elif not dynamic_low and daily_max_price > dynamic_high:  # 不设最小价格+当日最高价格>目标最高价(动态)
                    dynamic_monitor = -1
                elif daily_min_price < dynamic_low and daily_max_price > dynamic_high:
                    dynamic_monitor = -1
                else:
                    dynamic_monitor = 1
                if symbol in self.stock_summary:
                    self.stock_summary[symbol]['dynamic_monitor'] = dynamic_monitor

            # Step2. 限价单(静态)
            if static_monitor == 1:
                if order_sequence:  # 【模拟撮合】最高价先被触发
                    if static_high:
                        if high_price >= static_high:
                            self.close_stock(symbol=symbol, price=static_high, vol=SummaryDict["total_vol"], reason='static_high')
                            continue
                    if static_low:
                        if low_price <= static_low:
                            self.close_stock(symbol=symbol, price=static_low, vol=SummaryDict["total_vol"], reason='static_low')
                            continue
                elif not order_sequence:  # 【模拟撮合】最低价先被触发
                    if static_low:
                        if low_price <= static_low:
                            self.close_stock(symbol=symbol, price=static_low, vol=SummaryDict["total_vol"], reason='static_low')
                            continue
                    if static_high:
                        if high_price >= static_high:
                            self.close_stock(symbol=symbol, price=static_high, vol=SummaryDict["total_vol"], reason='static_high')
                            continue

            # Step2. 限价单(动态)
            if dynamic_monitor == 1:
                if order_sequence:  # 【模拟撮合】最高价先被触发
                    if dynamic_high:
                        if high_price >= dynamic_high:
                            self.close_stock(symbol=symbol, price=dynamic_high, vol=SummaryDict["total_vol"], reason='dynamic_high') # 尝试平仓所有仓位,但是time_monitor=-2的仓位被排除在外
                            continue
                    if dynamic_low:
                        if low_price <= dynamic_low:
                            self.close_stock(symbol=symbol, price=dynamic_low, vol=SummaryDict["total_vol"], reason='dynamic_low') # 尝试平仓所有仓位,但是time_monitor=-2的仓位被排除在外
                            continue
                elif not order_sequence:  # 【模拟撮合】最低价先被触发
                    if dynamic_low:
                        if low_price <= dynamic_low:
                            self.close_stock(symbol=symbol, price=dynamic_low, vol=SummaryDict["total_vol"], reason='dynamic_low')
                            continue
                    if dynamic_high:
                        if high_price >= dynamic_high:
                            self.close_stock(symbol=symbol, price=dynamic_high, vol=SummaryDict["total_vol"], reason='dynamic_high')
                            continue

    def monitor_future(self,order_type,order_sequence):
        """
        【柜台处理订单后运行,可重复运行】每日盘中运行,负责监控当前持仓是否满足限制平仓要求
        order_sequence=True 假设max_price先判断
        order_sequence=False 假设min_price先判断
        """
        if order_type=='long':
            pos=self.long_position.copy()
            summary=self.long_summary.copy()
        else:
            pos=self.short_position.copy()
            summary=self.short_summary.copy()
        if len(pos) == 0:
            return

        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式
        future_info_dict = self.future_info_dict.copy()  # 今天的标的基本信息
        future_k_dict = self.future_k_dict.copy()  # 今天的K线信息

        for contract,List in pos.items():
            info_dict = future_info_dict[contract]
            k_dict = future_k_dict[contract]
            if minute not in k_dict:
                continue
            k_dict = k_dict[minute]
            SummaryDict = summary[contract]

            # Step0. 基本信息
            end_date,daily_max_price,daily_min_price=info_dict['end_date'],info_dict['high'],info_dict['low']
            high_price,low_price,close_price=k_dict['high'],k_dict['low'],k_dict['close']   # K线信息
            ori_price,static_profit,static_loss,dynamic_profit,dynamic_loss,history_high,history_low=(
                SummaryDict['ori_price'],SummaryDict['static_profit'],SummaryDict['static_loss'],
                SummaryDict['dynamic_profit'],SummaryDict['dynamic_loss'],SummaryDict['history_max'],SummaryDict['history_min'])    # 持仓信息
            if order_type == "long":  # 多单止盈止损的价格
                static_high = ori_price * (1 + static_profit) if static_profit else None
                static_low = ori_price * (1 - static_loss) if static_loss else None
                dynamic_high = history_low * (1 + dynamic_profit) if dynamic_profit else None
                dynamic_low = history_high * (1 - dynamic_loss) if dynamic_loss else None
            else:  # 空单止盈止损的价格
                static_high = ori_price * (1 + static_loss) if static_loss else None
                static_low = ori_price * (1 - static_profit) if static_profit else None
                dynamic_high = history_low * (1 + dynamic_loss) if dynamic_loss else None
                dynamic_low = history_high * (1 - dynamic_profit) if dynamic_profit else None
            SummaryDict['history_min'] = min(SummaryDict['history_min'], low_price)
            SummaryDict['history_max'] = max(SummaryDict['history_max'], high_price)
            if order_type == "long":
                self.long_summary[contract] = SummaryDict
            else:
                self.short_summary[contract] = SummaryDict

            # 最小持仓单元
            i = 0
            while i < len(List):
                Dict = List[i]
                if order_type == "long" and contract not in self.long_position:
                    i+=1
                    continue
                if order_type == "short" and contract not in self.short_position:
                    i+=1
                    continue
                if Dict['time_monitor'] < 0:  # 该日禁止平仓(如股票的T+1制度)[-2]/没到最长持仓时间
                    i+=1
                    continue
                price,vol,first_timestamp,last_timestamp=(Dict['price'],Dict['vol'],Dict['min_timestamp'],Dict['max_timestamp'])    # 持仓信息
                time_monitor = Dict['time_monitor']
                if Dict['time_monitor'] == 0:  # 还没判断过
                    if pd.Timestamp(first_timestamp) >= pd.Timestamp(date):
                        time_monitor = -2   # 禁止平仓(如股票的T+1制度)
                    elif pd.Timestamp(end_date) > pd.Timestamp(date) and pd.Timestamp(first_timestamp) < pd.Timestamp(date) < pd.Timestamp(last_timestamp):   # 这里date省略了分钟,不能带等号
                        time_monitor = -1
                    else:
                        time_monitor = 1
                    if order_type == 'long':
                        self.long_position[contract][i]['time_monitor'] = time_monitor
                    else:
                        self.short_position[contract][i]['time_monitor'] = time_monitor

                i+=1    # 自動index

                # Step1. 限时单
                if time_monitor == 1:
                    if self.current_timestamp>=pd.Timestamp(end_date): #TODO:这里可以加上移仓换月的逻辑
                        self.close_future(order_type=order_type,contract=contract,price=close_price,vol=vol,reason='end_date')
                        continue
                    if self.current_timestamp>=pd.Timestamp(last_timestamp): # 最后交易日的期货持仓
                        self.close_future(order_type=order_type,contract=contract,price=close_price,vol=vol,reason='max_timestamp')

            # Step2. 再判断是否要止盈止损
            # if Dict['static_monitor'] == 0:  # 原始逻辑
            if True:
                if not static_high and not static_low:
                    static_monitor = -1
                elif not static_high and daily_min_price < static_low:    # 不设最大价格+当日最低价格<目标最低价(静态)
                    static_monitor = -1
                elif not static_low and daily_max_price > static_high:   # 不设最小价格+当日最高价格>目标最高价(静态)
                    static_monitor = -1
                elif daily_min_price < static_low and daily_max_price > static_high:
                    static_monitor = -1
                else:
                    static_monitor = 1

                if order_type == 'long' and contract in self.long_summary:
                    self.long_summary[contract]['static_monitor'] = static_monitor
                elif order_type == "short" and contract in self.short_summary:
                    self.short_summary[contract]['static_monitor'] = static_monitor

            # if Dict['dynamic_monitor'] == 0:  # 还没判断过
            if True:
                if not dynamic_high and not dynamic_low:
                    dynamic_monitor = -1
                elif not dynamic_high and daily_min_price < dynamic_low:    # 不设最大价格+当日最低价格<目标最低价(动态)
                    dynamic_monitor = -1
                elif not dynamic_low and daily_max_price > dynamic_high:   # 不设最小价格+当日最高价格>目标最高价(动态)
                    dynamic_monitor = -1
                elif daily_min_price < dynamic_low and daily_max_price > dynamic_high:
                    dynamic_monitor = -1
                else:
                    dynamic_monitor = 1

                if order_type == 'long' and contract in self.long_summary:
                    self.long_summary[contract]['dynamic_monitor'] = dynamic_monitor
                elif order_type == "short" and contract in self.short_summary:
                    self.short_summary[contract]['dynamic_monitor'] = dynamic_monitor

            # Step 2.限价单(静态)
            if static_monitor == 1:
                if order_sequence:  # 【模拟撮合】最高价先被触发
                    if static_high:
                        if high_price>=static_high:
                            self.close_future(order_type=order_type,contract=contract,price=static_high,vol=SummaryDict["total_vol"],reason='static_high')
                            continue
                    if static_low:
                        if low_price<=static_low:
                            self.close_future(order_type=order_type,contract=contract,price=static_low,vol=SummaryDict["total_vol"],reason='static_low')
                            continue
                elif not order_sequence:  # 【模拟撮合】最低价先被触发
                    if static_low:
                        if low_price<=static_low:
                            self.close_future(order_type=order_type,contract=contract,price=static_low,vol=SummaryDict["total_vol"],reason='static_low')
                            continue
                    if static_high:
                        if high_price>=static_high:
                            self.close_future(order_type=order_type,contract=contract,price=static_high,vol=SummaryDict["total_vol"],reason='static_high')
                            continue

            # Step2.限价单(动态)
            if dynamic_monitor == 1:
                if order_sequence:  # 【模拟撮合】最高价先被触发
                    if dynamic_high:
                        if high_price>=dynamic_high:
                            self.close_future(order_type=order_type,contract=contract,price=dynamic_high,vol=SummaryDict["total_vol"],reason='dynamic_high')
                            continue
                    if dynamic_low:
                        if low_price<=dynamic_low:
                            self.close_future(order_type=order_type,contract=contract,price=dynamic_low,vol=SummaryDict["total_vol"],reason='dynamic_low')
                            continue
                elif not order_sequence:  # 【模拟撮合】最低价先被触发
                    if dynamic_low:
                        if low_price<=dynamic_low:
                            self.close_future(order_type=order_type,contract=contract,price=dynamic_low,vol=SummaryDict["total_vol"],reason='dynamic_low')
                            continue
                    if dynamic_high:
                        if high_price>=dynamic_high:
                            self.close_future(order_type=order_type,contract=contract,price=dynamic_high,vol=SummaryDict["total_vol"],reason='dynamic_high')
                            continue

    def monitor_option(self,order_type,order_BS,order_sequence):
        """
        【柜台处理订单后运行,可重复运行】每日盘中运行,负责监控当前持仓是否满足限制平仓要求
        order_sequence=True 假设max_price先判断
        order_sequence=False 假设min_price先判断
        【新增】买方/卖方到期日未平仓虚值期权自动清算
        """
        if order_type=='call' and order_BS=='buy':
            pos=self.buycall_position.copy()
            summary=self.buycall_summary.copy()
        elif order_type=='call' and order_BS=='sell':
            pos=self.sellcall_position.copy()
            summary=self.sellcall_summary.copy()
        elif order_type=='put' and order_BS=='buy':
            pos=self.buyput_position.copy()
            summary=self.buyput_summary.copy()
        else:
            pos=self.sellput_position.copy()
            summary=self.sellput_summary.copy()
        if len(pos) == 0:
            return

        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式
        option_info_dict = self.option_info_dict.copy()  # 今天的标的基本信息
        option_k_dict = self.option_k_dict.copy()  # 今天的K线信息

        for option,List in pos.items():
            info_dict = option_info_dict[option]
            k_dict = option_k_dict[option]
            if minute not in k_dict:
                continue
            k_dict = k_dict[minute]
            SummaryDict = summary[option]

            # Step0. 基本信息
            end_date,daily_max_price,daily_min_price=info_dict['end_date'],info_dict['high'],info_dict['low']  # 基本信息
            high_price,low_price,close_price,level=k_dict['high'],k_dict['low'],k_dict['close'],k_dict['level']   # K线信息
            ori_price, static_profit, static_loss, dynamic_profit, dynamic_loss, history_high, history_low = (
                SummaryDict['ori_price'], SummaryDict['static_profit'], SummaryDict['static_loss'],
                SummaryDict['dynamic_profit'], SummaryDict['dynamic_loss'], SummaryDict['history_max'],
                SummaryDict['history_min'])  # 持仓信息
            if (order_type, order_BS) in [("buy","call"), ("sell","put")]:  # 买入看涨/卖出看跌期权止盈止损的价格
                static_high = ori_price * (1 + static_profit) if static_profit else None
                static_low = ori_price * (1 - static_loss) if static_loss else None
                dynamic_high = history_low * (1 + dynamic_profit) if dynamic_profit else None
                dynamic_low = history_high * (1 - dynamic_loss) if dynamic_loss else None
            else:  # 买入看跌/卖出看涨期权止盈止损的价格
                static_high = ori_price * (1 + static_loss) if static_loss else None
                static_low = ori_price * (1 - static_profit) if static_profit else None
                dynamic_high = history_low * (1 + dynamic_loss) if dynamic_loss else None
                dynamic_low = history_high * (1 - dynamic_profit) if dynamic_profit else None
            SummaryDict['history_min'] = min(SummaryDict['history_min'], low_price)
            SummaryDict['history_max'] = max(SummaryDict['history_max'], high_price)
            if order_type == 'call' and order_BS == 'buy':
                self.buycall_summary[option]=SummaryDict
            elif order_type == 'call' and order_BS == 'sell':
                self.sellcall_summary[option]=SummaryDict
            elif order_type == 'put' and order_BS == 'buy':
                self.buyput_summary[option]=SummaryDict
            else:
                self.sellput_summary[option]=SummaryDict

            # 最小持仓单元
            i = 0
            while i < len(List):
                Dict = List[i]
                if (order_type,order_BS) == ("buy","call") and option not in self.buycall_position:
                    i+=1
                    continue
                if (order_type,order_BS) == ("buy","put") and option not in self.buyput_position:
                    i+=1
                    continue
                if (order_type,order_BS) == ("sell","call") and option not in self.sellcall_position:
                    i+=1
                    continue
                if (order_type,order_BS) == ("sell","put") and option not in self.sellput_position:
                    i+=1
                    continue
                if Dict['time_monitor'] < 0:  # 该日禁止平仓(如股票的T+1制度)[-2]/没到最长持仓时间
                    i+=1
                    continue
                price,vol,first_timestamp,last_timestamp=(Dict['price'],Dict['vol'],Dict['min_timestamp'],Dict['max_timestamp'])    # 持仓信息
                time_monitor = Dict['time_monitor']
                if Dict['time_monitor'] == 0:  # 还没判断过
                    if pd.Timestamp(first_timestamp) >= pd.Timestamp(date):
                        time_monitor = -2   # 禁止平仓(如股票的T+1制度)
                    elif pd.Timestamp(end_date) > pd.Timestamp(date) and pd.Timestamp(first_timestamp) < pd.Timestamp(date) < pd.Timestamp(last_timestamp):   # 这里date省略了分钟,不能带等号
                        time_monitor = -1
                    else:
                        time_monitor = 1

                    if order_type == 'call' and order_BS == 'buy':
                        self.buycall_position[option][i]['time_monitor'] = time_monitor
                    elif order_type == 'call' and order_BS == 'sell':
                        self.sellcall_position[option][i]['time_monitor'] = time_monitor
                    elif order_type == 'put' and order_BS == 'buy':
                        self.buyput_position[option][i]['time_monitor'] = time_monitor
                    else:
                        self.sellput_position[option][i]['time_monitor'] = time_monitor

                i+=1    # 自動index

                # Step1. 限时单
                if time_monitor == 1:
                    # 期权清仓 (clear_option)
                    if self.current_date>=end_date and level<0: # 注:一定是到期日还是虚值期权的才可以
                        self.clear_option(order_type=order_type,order_BS=order_BS,option=option,vol=vol,reason='clear')
                        continue

                    # 再处理未到期权到期日但到指令到期日的期权(实值期权)
                    if self.current_timestamp>=pd.Timestamp(last_timestamp):
                        self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=close_price,vol=vol,reason='max_timestamp')

            # Step2. 再判断是否要止盈止损
            # if Dict['static_monitor'] == 0:  # 原始逻辑
            if True:
                if not static_high and not static_low:
                    static_monitor = -1
                elif not static_high and daily_min_price < static_low:  # 不设最大价格+当日最低价格<目标最低价(静态)
                    static_monitor = -1
                elif not static_low and daily_max_price > static_high:  # 不设最小价格+当日最高价格>目标最高价(静态)
                    static_monitor = -1
                elif daily_min_price < static_low and daily_max_price > static_high:
                    static_monitor = -1
                else:
                    static_monitor = 1

                if order_type == 'call' and order_BS == 'buy':
                    self.buycall_summary[option]['static_monitor'] = static_monitor
                elif order_type == 'call' and order_BS == 'sell':
                    self.sellcall_summary[option]['static_monitor'] = static_monitor
                elif order_type == 'put' and order_BS == 'buy':
                    self.buyput_summary[option]['static_monitor'] = static_monitor
                else:
                    self.sellput_summary[option]['static_monitor'] = static_monitor

            # if Dict['dynamic_monitor'] == 0:  # 还没判断过
            if True:
                if not dynamic_high and not dynamic_low:
                    dynamic_monitor = -1
                elif not dynamic_high and daily_min_price < dynamic_low:  # 不设最大价格+当日最低价格<目标最低价(动态)
                    dynamic_monitor = -1
                elif not dynamic_low and daily_max_price > dynamic_high:  # 不设最小价格+当日最高价格>目标最高价(动态)
                    dynamic_monitor = -1
                elif daily_min_price < dynamic_low and daily_max_price > dynamic_high:
                    dynamic_monitor = -1
                else:
                    dynamic_monitor = 1

                if order_type == 'call' and order_BS == 'buy':
                    self.buycall_summary[option]['dynamic_monitor'] = dynamic_monitor
                elif order_type == 'call' and order_BS == 'sell':
                    self.sellcall_summary[option]['dynamic_monitor'] = dynamic_monitor
                elif order_type == 'put' and order_BS == 'buy':
                    self.buyput_summary[option]['dynamic_monitor'] = dynamic_monitor
                else:
                    self.sellput_summary[option]['dynamic_monitor'] = dynamic_monitor


            # Step2. 限价单(静态)
            if static_monitor == 1:
                if order_sequence:  # 【模拟撮合】最高价先被触发
                    if static_high:
                        if high_price>=static_high:
                           self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=static_high,vol=SummaryDict["total_vol"],reason='static_high')
                           continue
                    if static_low:
                        if low_price<=static_low:
                           self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=static_low,vol=SummaryDict["total_vol"],reason='static_low')
                           continue
                elif not order_sequence:  # 【模拟撮合】最低价先被触发
                    if static_low:
                        if low_price<=static_low:
                            self.close_option(order_type=order_type, order_BS=order_BS, option=option, price=static_low,vol=SummaryDict["total_vol"], reason='static_low')
                            continue
                    if static_high:
                        if high_price>=static_high:
                            self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=static_high,vol=SummaryDict["total_vol"],reason='static_high')
                            continue

            # Step2. 限价单(动态)
            if dynamic_monitor == 1:
                if order_sequence:  # 【模拟撮合】最高价先被触发
                    if dynamic_high:
                        if high_price>=dynamic_high:
                            self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=dynamic_high,vol=SummaryDict["total_vol"],reason='dynamic_high')
                            continue
                    if dynamic_low:
                        if low_price<=dynamic_low:
                            self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=dynamic_low,vol=SummaryDict["total_vol"],reason='dynamic_low')
                            continue
                elif not order_sequence:  # 【模拟撮合】最低价先被触发
                    if dynamic_low:
                        if low_price<=dynamic_low:
                            self.close_option(order_type=order_type, order_BS=order_BS, option=option, price=dynamic_low,vol=SummaryDict["total_vol"], reason='dynamic_low')
                            continue
                    if dynamic_high:
                        if high_price>=dynamic_high:
                            self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=dynamic_high,vol=SummaryDict["total_vol"],reason='dynamic_high')
                            continue


class Counter(CounterBehavior):
    """
    【可修改, 特别是撮合逻辑需要自定义】
    继承所有交易行为CounterBehavior+TradeBehavior的交易柜台类
    """
    def __init__(self,session,config):
        super().__init__(session,config)

    def stock_counter_processing(self):
        """一般柜台撮合函数：
        【开仓/平仓order处理后运行,可重复运行】柜台判断open/close是否能够执行,若能则执行,并在柜台删除该订单
        【后续还需要添加volume判断条件,并添加部分成交+剩余继续挂单的情形】
        【同时,由于开仓设置时间是合理的,平仓如果时间过了平不了那大概率真的平不了,所以需要考虑流动性的问题进一步地优化代码】
        """
        stock_counter=self.stock_counter.copy()
        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式

        for i,orderDict in stock_counter.items():   # 订单编号,订单详情
            order_state,symbol,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['symbol'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_timestamp:   # 说明这个订单时间太长了,搞不了
                del self.stock_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}-Symbol{symbol}:Price{price}&Vol{vol} failed[Out of Timestamp]")

            elif self.current_timestamp>=min_order_timestamp:
                low,high=None,None
                if symbol in self.stock_k_dict:
                    if minute in self.stock_k_dict[symbol]:
                        low,high=self.stock_k_dict[symbol][minute]['low'],self.stock_k_dict[symbol][minute]['high']

                if low and high:    # 说明这根K线上有该股票的数据
                    if low<=price<=high:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            self.execute_stock(symbol=symbol,vol=vol,price=price,
                                               static_profit=orderDict['static_profit'],
                                               static_loss=orderDict['static_loss'],
                                               dynamic_profit=orderDict['dynamic_profit'],
                                               dynamic_loss=orderDict['dynamic_loss'],
                                               min_timestamp=orderDict['min_timestamp'],
                                               max_timestamp=orderDict['max_timestamp'],
                                               commission=orderDict['commission'],
                                               reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            self.close_stock(symbol=symbol,vol=vol,price=price,reason=orderDict['reason'])
                        del self.stock_counter[i]  # 删除柜台的订单
                    else:       # 说明不能在这根K线上成交
                        pass
                else:   # 说明当前K线没有该股票的数据
                    pass

    def stock_counter_strict_processing(self, open_share_threshold: float, close_share_threshold: float):
        """
        更严格的柜台撮合函数：
        当前开仓最多只能成交这根K线成交量的open_share_threshold倍
        当前平仓最多只能成交这根K线成交量的close_share_threshold倍
        """
        stock_counter=self.stock_counter.copy()
        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式

        for i,orderDict in stock_counter.items():   # 订单编号,订单详情
            order_state,symbol,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['symbol'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_timestamp:   # 说明这个订单时间太长了,搞不了
                del self.stock_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}-Symbol{symbol}:Price{price}&Vol{vol} failed[Out of Timestamp]")

            elif self.current_timestamp>=min_order_timestamp:
                low,high,close,volume=None,None,None,None
                if symbol in self.stock_k_dict:
                    if minute in self.stock_k_dict[symbol]:
                        low,high,close,volume=(self.stock_k_dict[symbol][minute]['low'],self.stock_k_dict[symbol][minute]['high'],
                                               self.stock_k_dict[symbol][minute]['close'],self.stock_k_dict[symbol][minute]['volume'])

                if low and high and close and volume:    # 说明这根K线上有该股票的数据
                    if "partialorder" in orderDict:
                        price = close       # 说明是部分成交的订单,第一次成交按照挂单价成交,后续假设都按照close成交
                    if low<=price<=high:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            if vol <= int(volume * open_share_threshold):  # 说明该订单可以全部成交
                                del self.stock_counter[i]  # 删除柜台的订单
                            else:
                                vol = int(volume * open_share_threshold)  # 只能成交这一份额
                                self.stock_counter[i]['vol'] -= vol
                                self.stock_counter[i]['partialorder'] = True  # 标记为部分成交的订单
                            self.execute_stock(symbol=symbol,vol=vol,price=price,
                                                static_profit=orderDict['static_profit'],
                                                static_loss=orderDict['static_loss'],
                                                dynamic_profit=orderDict['dynamic_profit'],
                                                dynamic_loss=orderDict['dynamic_loss'],
                                                min_timestamp=orderDict['min_timestamp'],
                                                max_timestamp=orderDict['max_timestamp'],
                                                commission=orderDict['commission'],
                                                reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            if vol <= int(volume * close_share_threshold):  # 说明该订单可以全部成交
                                del self.stock_counter[i]  # 删除柜台的订单
                            else:
                                vol = int(volume * close_share_threshold)  # 只能成交这一份额
                                self.stock_counter[i]['vol'] -= vol
                                self.stock_counter[i]['partialorder'] = True  # 标记为部分成交的订单
                            self.close_stock(symbol=symbol,vol=vol,price=price,reason=orderDict['reason'])
                    else:       # 说明不能在这根K线上成交
                        pass
                else:   # 说明当前K线没有该股票的数据
                    pass

    def future_counter_processing(self):
        """【开仓/平仓order处理后运行,可重复运行】柜台判断open/close是否能够执行,若能则执行,并在柜台删除该订单
        【后续还需要添加volume判断条件,并添加部分成交+剩余继续挂单的情形】
        【同时,由于开仓设置时间是合理的,平仓如果时间过了平不了那大概率真的平不了,所以需要考虑流动性的问题进一步地优化代码】
        """
        future_counter=self.future_counter.copy()
        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式
        for i,orderDict in future_counter.items(): # 订单编号,订单详情
            order_state,order_type,contract,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['order_type'],orderDict['contract'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_date:   # 说明这个订单时间太长了,搞不了
                del self.future_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}{order_type}-Contract{contract}:Price{price}&Vol{vol} failed[Out of Timestamp]")

            elif self.current_timestamp>=min_order_timestamp:
                low,high=None,None
                if contract in self.future_k_dict:
                    if minute in self.future_k_dict[contract]:
                        low,high=self.future_k_dict[contract][minute]['low'], self.future_k_dict[contract][minute]['high']
                if low and high:
                    if low<=price<=high:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            pre_settle=self.future_info_dict[contract]['pre_settle']
                            self.execute_future(order_type=order_type,contract=contract,vol=vol,price=price,
                                                pre_settle=pre_settle,margin=orderDict['margin'],
                                                static_profit=orderDict['static_profit'],
                                                static_loss=orderDict['static_loss'],
                                                dynamic_profit=orderDict['dynamic_profit'],
                                                dynamic_loss=orderDict['dynamic_loss'],
                                                min_timestamp=orderDict['min_timestamp'],
                                                max_timestamp=orderDict['max_timestamp'],
                                                commission=orderDict['commission'],reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            self.close_future(order_type=order_type,contract=contract,vol=vol,price=price,reason=orderDict['reason'])
                        del self.future_counter[i]  # 删除柜台的订单
                    else:       # 说明不能在这根K线上成交
                        pass
                else:   # 说明这根K线没有该合约的数据
                    pass

    def future_counter_strict_processing(self, open_share_threshold: float, close_share_threshold: float):
        """
        更严格的柜台撮合函数：
        当前开仓最多只能成交这根K线成交量的open_share_threshold倍
        当前平仓最多只能成交这根K线成交量的close_share_threshold倍
        """
        future_counter=self.future_counter.copy()
        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式

        for i,orderDict in future_counter.items(): # 订单编号,订单详情
            order_state,order_type,contract,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['order_type'],orderDict['contract'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_date:   # 说明这个订单时间太长了,搞不了
                del self.future_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}{order_type}-Contract{contract}:Price{price}&Vol{vol} failed[Out of Timestamp]")

            elif self.current_timestamp>=min_order_timestamp:
                low,high,close,volume=None,None,None,None
                if contract in self.future_k_dict:
                    if minute in self.future_k_dict[contract]:
                        low,high,close,volume=self.future_k_dict[contract][minute]['low'], self.future_k_dict[contract][minute]['high'],self.future_k_dict[contract][minute]['close'],self.future_k_dict[contract][minute]['volume']

                if low and high and close and volume:   # 说明这根K线上有该合约的数据
                    if "partialorder" in orderDict:
                        price = close                   # 说明是部分成交的订单,第一次成交按照挂单价成交,后续假设都按照close成交
                    if low<=price<=high:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            if vol <= int(volume * open_share_threshold):  # 说明该订单可以全部成交
                                del self.future_counter[i]  # 删除柜台的订单
                            else:
                                vol = int(volume * open_share_threshold)  # 只能成交这一份额
                                self.future_counter[i]['vol'] -= vol
                                self.future_counter[i]['partialorder'] = True  # 标记为部分成交的订单
                            self.execute_future(order_type=order_type,contract=contract,vol=vol,price=price,
                                                pre_settle=self.future_info_dict[contract]['pre_settle'],
                                                margin=orderDict['margin'],
                                                static_profit=orderDict['static_profit'],
                                                static_loss=orderDict['static_loss'],
                                                dynamic_profit=orderDict['dynamic_profit'],
                                                dynamic_loss=orderDict['dynamic_loss'],
                                                min_timestamp=orderDict['min_timestamp'],
                                                max_timestamp=orderDict['max_timestamp'],
                                                commission=orderDict['commission'],reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            if vol <= int(volume * close_share_threshold):  # 说明该订单可以全部成交
                                del self.future_counter[i]  # 删除柜台的订单
                            else:
                                vol = int(volume * close_share_threshold)  # 只能成交这一份额
                                self.future_counter[i]['vol'] -= vol
                                self.future_counter[i]['partialorder'] = True  # 标记为部分成交的订单
                            self.close_future(order_type=order_type,contract=contract,vol=vol,price=price,reason=orderDict['reason'])
                        else:       # 说明不能在这根K线上成交
                            pass
                    else:   # 说明这根K线没有该合约的数据
                        pass

    def option_counter_processing(self):
        """【开仓/平仓order处理后运行,可重复运行】柜台判断open/close是否能够执行,若能则执行,并在柜台删除该订单
        【后续还需要添加volume判断条件,并添加部分成交+剩余继续挂单的情形】
        【同时,由于开仓设置时间是合理的,平仓如果时间过了平不了那大概率真的平不了,所以需要考虑流动性的问题进一步地优化代码】
        """
        option_counter=self.option_counter.copy()
        date, minute =self.current_dot_date, str(self.current_minute)   # 注意: json里的键是string格式
        for i,orderDict in option_counter.items():  # 订单编号,订单详情
            order_state,order_type,order_BS,option,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['order_type'],orderDict['order_BS'],orderDict['option'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_timestamp:  # 说明这个订单时间太长了,搞不了
                del self.option_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}{order_BS}{order_type}-Option{option}:Price{price}&Vol{vol} failed[Out of Timestamp]")
            elif self.current_timestamp>=min_order_timestamp:   # 時計
                low,high=None,None
                if option in self.option_k_dict:
                    if minute in self.option_k_dict[option]:
                        low, high = self.option_k_dict[option][minute]
                if low and high:
                    if low <= price <= high:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            self.execute_option(order_type=order_type,order_BS=order_BS,option=option,vol=vol,price=price,strike=orderDict['strike'],
                                                pre_settle=orderDict['pre_settle'],margin=orderDict['margin'],
                                                static_profit=orderDict['static_profit'],
                                                static_loss=orderDict['static_loss'],
                                                dynamic_profit=orderDict['dynamic_profit'],
                                                dynamic_loss=orderDict['dynamic_loss'],
                                                min_timestamp=orderDict['min_timestamp'],max_timestamp=orderDict['max_timestamp'],
                                                commission=orderDict['commission'],reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            self.close_option(order_type=order_type,order_BS=order_BS,option=option,vol=vol,price=price,reason=orderDict['reason'])
                        del self.option_counter[i]  # 删除柜台的订单
                    else:  # 说明当日不能成交
                        pass
                else:  # 说明当日没有该合约的数据
                    pass

    def close_counter(self):
        """【每日盘后运行】
        1. 更新counter中未完成订单的pre_settle为当日settle
        2. 更新position中的monitor_price与monitor_time重置为0 + 删除持仓为空的标的 + 更新history_min与history_max (dynamic_monitor=-1的仓位)
        """
        if self.run_stock:
            pos=self.stock_position.copy()
            if len(pos)>0:
                for symbol, List in pos.items():
                    if not List:
                        del self.stock_position[symbol]
                        continue
                    L = []
                    for Dict in List:
                        # if Dict['dynamic_monitor'] == 1:  # 更新history_min/history_max
                        #     Dict['history_min'] = min(Dict['history_min'], self.stock_info_dict[symbol]['low'])
                        #     Dict['history_max'] = max(Dict['history_max'], self.stock_info_dict[symbol]['high'])
                        Dict["static_monitor"] = 0
                        Dict["dynamic_monitor"] = 0
                        Dict["time_monitor"] = 0
                        L.append(Dict)
                    self.stock_position[symbol] = L

                # 新增:强制清仓end_date>当前日期的标的
                for symbol, List in pos.items():
                    end_date = pd.Timestamp(self.stock_info_dict[symbol]['end_date'])
                    if end_date < self.current_timestamp:  # 已经过期了
                        close = self.stock_info_dict[symbol]['close']
                        for Dict in List:
                            self.close_stock(symbol=symbol, vol=Dict['vol'], price=close, reason='force_close')

        if self.run_future:
            # 更新pre_settle为当日settle
            Dict=self.future_counter.copy()
            future_info_dict = self.future_info_dict
            if len(Dict)>0: # 说明有积压的订单
                for orderNum,order in Dict.items():
                    self.future_counter[orderNum]['pre_settle']=future_info_dict[order['contract']]['settle']

            pos=self.long_position.copy()
            if len(pos)>0:
                for symbol, List in pos.items():
                    if not List:
                        del self.long_position[symbol]
                        continue
                    L = []
                    for Dict in List:
                        # if Dict['dynamic_monitor'] == 1:  # 更新history_min/history_max
                        #     Dict['history_min'] = min(Dict['history_min'], self.future_info_dict[symbol]['low'])
                        #     Dict['history_max'] = max(Dict['history_max'], self.future_info_dict[symbol]['high'])
                        Dict["static_monitor"] = 0
                        Dict["dynamic_monitor"] = 0
                        Dict["time_monitor"] = 0
                        L.append(Dict)
                    self.long_position[symbol] = L

            # 新增:强制清仓end_date>当前日期的标的
            if len(pos)>0:
                for symbol, List in pos.items():
                    end_date = pd.Timestamp(self.future_info_dict[symbol]['end_date'])
                    if end_date < self.current_timestamp:   # 已经过期了
                        close = self.future_info_dict[symbol]['close']
                        for Dict in List:
                            self.close_future(order_type='long',contract=symbol,vol=Dict['vol'],price=close,reason='force_close')

            pos=self.short_position.copy()
            if len(pos)>0:
                for symbol, List in pos.items():
                    if not List:
                        del self.short_position[symbol]
                        continue
                    L=[]
                    for Dict in List:
                        # if Dict['dynamic_monitor'] == 1:  # 更新history_min/history_max
                        #     Dict['history_min'] = min(Dict['history_min'], self.future_info_dict[symbol]['low'])
                        #     Dict['history_max'] = max(Dict['history_max'], self.future_info_dict[symbol]['high'])
                        Dict["static_monitor"] = 0
                        Dict["dynamic_monitor"] = 0
                        Dict["time_monitor"] = 0
                        L.append(Dict)
                    self.short_position[symbol] = L

                # 新增:强制清仓end_date>当前日期的标的
                for symbol, List in pos.items():
                    end_date = pd.Timestamp(self.future_info_dict[symbol]['end_date'])
                    if end_date < self.current_timestamp:  # 已经过期了
                        close = self.future_info_dict[symbol]['close']
                        for Dict in List:
                            self.close_future(order_type='short', contract=symbol, vol=Dict['vol'], price=close,
                                              reason='force_close')

        if self.run_option:
            Dict=self.option_counter
            if len(Dict)>0: # 说明有积压的订单
                for orderNum,order in Dict.items():
                    self.option_counter[orderNum]['pre_settle']=self.option_info_dict[order['option']]['settle'] # 更新pre_settle

            pos = self.buycall_position.copy()
            if len(pos) > 0:
                for symbol, List in pos.items():
                    if not List:
                        del self.buycall_position[symbol]
                        continue
                    L=[]
                    for Dict in List:
                        # if Dict['dynamic_monitor']==1:  # 更新history_min/history_max
                        #     Dict['history_min']=min(Dict['history_min'],self.option_info_dict[symbol]['low'])
                        #     Dict['history_max']=max(Dict['history_max'],self.option_info_dict[symbol]['high'])
                        Dict["static_monitor"]=0
                        Dict["dynamic_monitor"]=0
                        Dict["time_monitor"]=0
                        L.append(Dict)
                    self.buycall_position[symbol] = L

            pos = self.buyput_position.copy()
            if len(pos) > 0:
                for symbol, List in pos.items():
                    if not List:
                        del self.buyput_position[symbol]
                        continue
                    L=[]
                    for Dict in List:
                        # if Dict['dynamic_monitor']==1:  # 更新history_min/history_max
                        #     Dict['history_min']=min(Dict['history_min'],self.option_info_dict[symbol]['low'])
                        #     Dict['history_max']=max(Dict['history_max'],self.option_info_dict[symbol]['high'])
                        Dict["static_monitor"]=0
                        Dict["dynamic_monitor"]=0
                        Dict["time_monitor"]=0
                        L.append(Dict)
                    self.buyput_position[symbol] = L

            pos = self.sellcall_position.copy()
            if len(pos) > 0:
                for symbol, List in pos.items():
                    if not List:
                        del self.sellcall_position[symbol]
                        continue
                    L=[]
                    for Dict in List:
                        # if Dict['dynamic_monitor']==1:  # 更新history_min/history_max
                        #     Dict['history_min']=min(Dict['history_min'],self.option_info_dict[symbol]['low'])
                        #     Dict['history_max']=max(Dict['history_max'],self.option_info_dict[symbol]['high'])
                        Dict["static_monitor"]=0
                        Dict["dynamic_monitor"]=0
                        Dict["time_monitor"]=0
                        L.append(Dict)
                    self.sellcall_position[symbol] = L

            pos = self.sellput_position.copy()
            if len(pos) > 0:
                for symbol, List in pos.items():
                    if not List:
                        del self.sellput_position[symbol]
                        continue
                    L=[]
                    for Dict in List:
                        # if Dict['dynamic_monitor']==1:  # 更新history_min/history_max
                        #     Dict['history_min']=min(Dict['history_min'],self.option_info_dict[symbol]['low'])
                        #     Dict['history_max']=max(Dict['history_max'],self.option_info_dict[symbol]['high'])
                        Dict["static_monitor"]=0
                        Dict["dynamic_monitor"]=0
                        Dict["time_monitor"]=0
                        L.append(Dict)
                    self.sellput_position[symbol] = L


