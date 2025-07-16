# base_backtest.py
import pandas as pd
import dolphindb as ddb
from numba import jit
from src.entities.BackTestData import *
from src.entities.BackTestConfig import *

class BaseFunction(BackTestConfig):
    def __init__(self,session,config):
        super().__init__(session,config)    # 继承所有属性

    def init_counter(self):
        """【回测前运行】期货柜台&期权柜台初始化"""
        counter_init(self)

    def start_counter(self):
        """【盘前运行】daily counter start for data receiving"""
        counter_start(self)

    def close_counter(self):
        """【每日盘后运行】更新counter中未完成订单的pre_settle为当日settle"""
        if self.run_future:
            Dict=self.future_counter
            if len(Dict)>0: # 说明有积压的订单
                for orderNum,order in Dict.items():
                    df=self.session.run(f"""select settle from loadTable('{self.future_counter_database}','{self.future_counter_table}') where contract="{order['contract']}" and date==date({self.current_dot_date})""")
                    if not df.empty:
                        self.future_counter[orderNum]['pre_settle']=df.loc[0]['settle']
                    else:   # 说明当天future_settle数据缺失
                        pass

        if self.run_option:
            Dict=self.option_counter
            if len(Dict)>0: # 说明有积压的订单
                for orderNum,order in Dict.items():
                    df=self.session.run(f"""select settle from loadTable('{self.option_counter_database}','{self.option_counter_table}') where option="{order['option']}" and date==date({self.current_dot_date})""")
                    if not df.empty:
                        self.option_counter[orderNum]['pre_settle']=df.loc[0]['settle']
                    else:   # 说明当天option_settle数据缺失
                        pass


class TradeBehavior(BaseFunction):
    """
    向基础的回测框架添加交易行为的方法
    Order()
    """
    def __init__(self,session,config):
        super().__init__(session,config)    # 继承所有属性

    def order_open_stock(self,symbol:str, vol, price, min_price = None, max_price = None,
                         max_timestamp: pd.Timestamp = None,
                         min_order_timestamp: pd.Timestamp = None,
                         max_order_timestamp: pd.Timestamp = None,
                         commission:float = None, reason:str = None):
        """
        【盘中运行】股票订单发送至stock_counter,
        如果不设置max_order_date,
        每天都会尝试在min_order_date后发送该订单
        """
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1    # 给定订单编号(唯一值)
        self.stock_counter[self.orderNum]={'order_state':'open',
                                            'create_date': self.current_date,
                                            'create_timestamp': self.current_timestamp,
                                            'min_order_timestamp': min_order_timestamp,
                                            'max_order_timestamp': max_order_timestamp,
                                            'max_timestamp': max_timestamp,
                                            'symbol': symbol,
                                            'vol': vol,
                                            'price': price,
                                            'min_price': min_price,
                                            'max_price': max_price,
                                            'commission': commission,
                                            'reason': reason}

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
        self.stock_counter[self.orderNum]={'order_state': 'close',
                                            'create_date': self.current_date,
                                            'create_timestamp': self.current_timestamp,
                                            'min_order_timestamp': min_order_timestamp,
                                            'max_order_timestamp': max_order_timestamp,
                                            'symbol': symbol,
                                            'vol': vol,
                                            'price': price,
                                            'reason': reason}

    def order_open_future(self,order_type:str, contract:str, vol, price, pre_settle, margin,
                          min_price=None, max_price=None,
                          max_timestamp:pd.Timestamp=None, min_order_timestamp:pd.Timestamp=None, max_order_timestamp:pd.Timestamp=None,
                          commission=None, reason=None):
        """【盘中运行】期货订单发送至future_counter,如果不设置max_order_date,每天都会尝试在min_order_date后发送该订单"""
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1    # 给定订单编号(唯一值)
        # 【Attention: pre_settle需要更新!!!】
        self.future_counter[self.orderNum]={'order_state':'open',
                                            'order_type':order_type,
                                            'create_date':self.current_date,
                                            'create_timestamp':self.current_timestamp,
                                            'max_timestamp':max_timestamp,
                                            'min_order_timestamp':min_order_timestamp,
                                            'max_order_timestamp':max_order_timestamp,
                                            'contract':contract,
                                            'vol':vol,
                                            'price':price,
                                            'pre_settle':pre_settle,
                                            'margin':margin,
                                            'min_price':min_price,
                                            'max_price':max_price,
                                            'commission':commission,
                                            'reason':reason}

    def order_close_future(self,order_type:str, contract:str, vol, price,
                           min_order_timestamp=None, max_order_timestamp=None, reason:str=None):
        """【盘中运行】期货平仓发送至future_counter,如果不设置max_order_date,每天都会在min_order_date后尝试平仓该订单"""
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1  # 给定订单编号(唯一值)
        self.future_counter[self.orderNum]={'order_state':'close',
                                            'order_type':order_type,
                                            'create_date':self.current_date,
                                            'create_timestamp':self.current_timestamp,
                                            'min_order_timestamp':min_order_timestamp,
                                            'max_order_timestamp':max_order_timestamp,
                                            'contract':contract,
                                            'vol':vol,
                                            'price':price,
                                            'reason':reason}

    def order_open_option(self,order_type:str, order_BS:str, option:str, vol, price,
                          pre_settle, strike, margin, min_price=None, max_price=None, max_timestamp=None,
                          min_order_timestamp=None, max_order_timestamp=None, commission=None, reason:str=None):
        """
        【盘中运行】期权买入订单发送至option_counter,如果不设置max_order_date,每天都会尝试发送该订单
        """
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        if order_BS=='buy': # 期权买方不用付保证金
            margin=0
        self.orderNum+=1    # 给定订单编号(唯一值)
        # 【Attention: pre_settle需要更新!!!】
        self.option_counter[self.orderNum]={'order_state':'open',
                                            'order_type':order_type,    # call/put
                                            'order_BS':order_BS,        # buy/sell
                                            'create_date':self.current_date,
                                            "create_timestamp":self.current_timestamp,
                                            'min_order_timestamp':min_order_timestamp,
                                            'max_order_timestamp':max_order_timestamp,
                                            'max_timestamp': max_timestamp,
                                            'option':option,
                                            'vol':vol,
                                            'price':price,
                                            'pre_settle':pre_settle,
                                            'margin':margin,
                                            'strike':strike,
                                            'min_price':min_price,
                                            'max_price':max_price,
                                            'commission':commission,
                                            'reason':reason}

    def order_close_option(self,order_type:str, order_BS:str, option:str, vol, price,
                           min_order_timestamp:pd.Timestamp=None, max_order_timestamp:pd.Timestamp=None,
                           reason:str=None):
        """【盘中运行】期权平仓发送至option_counter,如果不设置max_order_date,每天都会尝试平仓该订单"""
        if not min_order_timestamp:
            min_order_timestamp=pd.Timestamp(self.start_date)
        if not max_order_timestamp:
            max_order_timestamp=pd.Timestamp(self.end_date)
        self.orderNum+=1  # 给定订单编号(唯一值)
        self.option_counter[self.orderNum]={'order_state':'close',
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


class CounterBehavior(TradeBehavior):
    """
    Counter Behavior:
    - execute (开仓)
    - close (平仓)
    - clear (清仓)
    - calculate_profit (核算)
    """
    def __init__(self,session,config):  # 继承所有属性
        super().__init__(session, config)

    def execute_stock(self, symbol, vol, price, min_price=None, max_price=None,
                      max_timestamp:pd.Timestamp=None, commission=None,
                      reason:str=None):
        """
        【核心函数】股票开仓/加仓(默认无手续费)
        min_price:平仓最小价格(止损)
        max_price:平仓最大价格(止盈)
        max_timestamp:持仓最大时间戳
        """
        position=self.stock_position    # 当前股票持仓
        if symbol not in position:
            position[symbol]=[{'price':price,
                               'min_price':min_price,
                               'max_price':max_price,
                               'max_timestamp':max_timestamp,
                               'vol':vol}]
        else:
            position[symbol].append({'price':price,
                                     'min_price':min_price,
                                     'max_price':max_price,
                                     'max_timestamp':max_timestamp,
                                     'vol':vol})
        self.stock_position=position

        # 记录
        self.future_record=self.future_record._append({'state':'open',
                                                       'reason':reason,
                                                       'date':self.current_date,
                                                       'timestamp':self.current_timestamp,
                                                       'symbol':symbol,
                                                       'price':price,
                                                       'vol':vol,
                                                       'pnl':0},ignore_index=True)

        # 结算
        self.cash-=vol*price  # 减去股票购买成本

    def execute_future(self, order_type:str, contract:str, vol, price, pre_settle, margin,
                       min_price=None, max_price=None, max_timestamp:pd.Timestamp=None, commission=None, reason:str=None):
        """
        【核心函数】期货合约开仓/加仓(默认无手续费)
        margin:每笔交易的"初始"保证金[这里是初始保证金]
        min_price:平仓最小价格(多单为止损/空单为止盈)
        max_price:平仓最大价格(多单为止盈/空单为止损)
        max_timestamp: 持仓最大时间戳
        【新增】逐日盯市制度回测 pre_settle而不是settle防止未来函数
        """
        if order_type=='long':
            position=self.long_position
        else:
            position=self.short_position
        if contract not in position:
            position[contract]=[{'price':price,
                                 'pre_settle':pre_settle,
                                 'margin':margin,
                                 'min_price':min_price,
                                 'max_price':max_price,
                                 'max_timestamp':max_timestamp,
                                 'vol':vol,
                                 'FirstDaySettle':None}]
        else:
            position[contract].append({'price':price,
                                       'pre_settle':pre_settle,
                                       'margin':margin,
                                       'min_price':min_price,
                                       'max_price':max_price,
                                       'max_timestamp': max_timestamp,
                                       'vol':vol,
                                       'FirstDaySettle':None})
        # 赋值
        if order_type=='long':
            self.long_position=position
        else:
            self.short_position=position
        # 记录
        self.future_record=self.future_record._append({'state':'open',
                                                       'reason':reason,
                                                       'date':self.current_date,
                                                       "timestamp":self.current_timestamp,
                                                       'contract':contract,
                                                       'order_type':order_type,
                                                       'price':price,
                                                       'vol':vol,
                                                       'pnl':0},ignore_index=True)

        # 结算
        self.cash-=margin           # 减去初始保证金(该笔合约的全部保证金)

    def execute_option(self, order_type:str, order_BS:str, option:str, vol, price, strike, pre_settle,
                       margin:float=None, min_price=None, max_price=None, max_timestamp:pd.Timestamp=None, commission=None, reason:str=None):
        """【核心函数】买入看涨(order_type='call')/看跌(order_type='sell')期权"""
        if order_type=='call' and order_BS=='buy':  # 期权买方没有保证金
            position=self.buycall_position
            margin=0
        elif order_type=='call' and order_BS=='sell':
            position=self.sellcall_position
        elif order_type=='put' and order_BS=='buy': # 期权买方没有保证金
            position=self.buyput_position
            margin=0
        else:
            position=self.sellput_position

        if option not in position:
            position[option]=[{'price':price,
                               'pre_settle':pre_settle,
                               'margin':margin,
                               'strike':strike,
                               'min_price':min_price,
                               'max_price':max_price,
                               'max_timestamp':max_timestamp,
                               'vol':vol,
                               'FirstDaySettle':None}]
        else:
            position[option].append([{'price':price,
                                      'pre_settle':pre_settle,
                                      'margin':margin,
                                      'strike':strike,
                                      'min_price':min_price,
                                      'max_price':max_price,
                                      'max_timestamp':max_timestamp,
                                      'vol':vol,
                                      'FirstDaySettle':None}])
        # 赋值
        if order_type=='call' and order_BS=='buy':
            self.buycall_position=position
            self.cash-=(vol*price)  # 减去付出的权利金
        elif order_type=='call' and order_BS=='sell':
            self.sellcall_position=position
            self.cash+=(vol*price-margin)  # 加上得到的权利金减去保证金
        elif order_type=='put' and order_BS=='buy':
            self.buyput_position=position
            self.cash-=(vol*price)  # 减去付出的权利金
        elif order_type=='put' and order_BS=='sell':
            self.sellput_position=position
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
        """【核心函数】股票平仓"""
        profit=0    # 该笔交易获得的盈利(实现盈利)
        position=self.stock_position.copy()
        if position:    # 如果目前还有持仓的话
            if symbol not in position.keys():
                print(f"股票{symbol}未持仓,无法平仓")
            else:
                current_vol_list=[i['vol'] for i in position[symbol]]    # 当前的股票持有情况list
                ori_price_list=[i['price'] for i in position[symbol]]    # 当前股票买入价格情况list
                current_vol=sum(current_vol_list)   # 当前持股数量
                max_vol=min(current_vol,vol)        # 现在需要平仓的数量
                record_vol=max_vol                  # for record
                if max_vol>=current_vol:  # 说明要全平仓
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        profit+=(price-ori_price)*vol  # 逐笔盈亏
                    del position[symbol]  # 直接去掉这个股票的持有 {'symbol':[(price,vol),...]}
                elif max_vol<current_vol:  # 说明部分平仓
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        if max_vol>=vol:  # 当前订单全部平仓
                            profit+=(price-ori_price)*vol
                            del position[symbol][0]  # FIFO原则
                            max_vol=max_vol-vol
                        else:  # 当前订单部分平仓
                            profit+=(price-ori_price)*max_vol
                            position[symbol][0]['vol']=vol-max_vol
                            break  # 执行完毕
                # 记录
                self.stock_record=self.stock_record._append({'state':'close',
                                                             'reason':reason,
                                                             'date':self.current_date,
                                                             "timestamp":self.current_timestamp,
                                                             'symbol':symbol,
                                                             'price':price,
                                                             'vol':record_vol,
                                                             'pnl':profit},ignore_index=True)
                # 结算
                self.profit+=profit                  # 逐笔盈亏(平仓价-开仓价)
                self.cash+=profit                    # 获得的利润计入cash
                self.stock_position=position

    def close_future(self,order_type,contract,vol,price,reason=None):
        """【核心函数】期货合约平仓"""
        profit=0    # 该笔交易获得的盈利(实现盈利)
        settle_profit=0   # 该笔交易获得的盯市盈亏(交易价-昨结价)
        margin=0    # 该笔交易收回的保证金
        if order_type=='long':
            position=self.long_position.copy()
        else:
            position=self.short_position.copy()
        LS={'long':1,'short':-1}[order_type]    # 避免硬编码
        if position:    # 如果目前还有持仓的话
            if contract not in position:
                print(f"合约{contract}未持仓,无法平仓")
            else:
                current_vol_list=[i['vol'] for i in position[contract]]    # 当前的合约持有情况list
                ori_price_list=[i['price'] for i in position[contract]]    # 当前合约买入价格情况list
                pre_margin_list=[i['margin'] for i in position[contract]]  # 当前合约占用的保证金情况
                pre_settle_list=[i['pre_settle'] for i in position[contract]]   # 当前合约的昨结价情况
                current_vol=sum(current_vol_list)    # 当前合约持有的数量
                max_vol=min(current_vol,vol)         # 现在需要平仓的数量
                record_vol=max_vol                   # for record
                if current_vol<=0:
                    print(f"合约{contract}未持仓,无法平仓")
                elif max_vol>=current_vol:  # 说明要全平仓
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        pre_margin=pre_margin_list[i]
                        pre_settle=pre_settle_list[i]
                        profit+=(price-ori_price)*vol*LS            # 逐笔盈亏
                        settle_profit+=(price-pre_settle)*vol*LS    # 盯市盈亏
                        margin+=(pre_margin+settle_profit)          # 收回的保证金
                    del position[contract]   # 直接去掉这个合约的持有 {'contract':[(price,vol),...]}
                elif max_vol<current_vol:    # 说明部分平仓
                    for i in range(0,len(current_vol_list)):
                        vol=current_vol_list[i]
                        ori_price=ori_price_list[i]
                        pre_margin=pre_margin_list[i]   # 当前合约历史订单占用的保证金
                        pre_settle=pre_settle_list[i]
                        if max_vol>=vol:    # 当前订单全部平仓
                            profit+=(price-ori_price)*vol*LS
                            settle_profit+=(price-pre_settle)*vol*LS
                            margin+=(pre_margin+settle_profit)  # 收回的保证金
                            del position[contract][0]    # FIFO原则
                            max_vol=max_vol-vol
                        else:               # 当前订单部分平仓
                            profit+=(price-ori_price)*max_vol*LS
                            settle_profit+=(price-pre_settle)*max_vol*LS
                            margin+=(pre_margin*(max_vol/vol)+settle_profit)    # 收回的保证金
                            position[contract][0]['vol']=vol-max_vol
                            position[contract][0]['margin']=pre_margin*(1-max_vol/vol)  # 剩余的保证金
                            break   # 执行完毕
                # 记录
                self.future_record=self.future_record._append({'state':'close',
                                                               'reason':reason,
                                                               'date':self.current_date,
                                                               "timestamp":self.current_timestamp,
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
            else:
                self.short_position=position

    def close_option(self,order_type,order_BS,option,vol,price,reason=None):
        """【核心函数】期权合约平仓
        【需要进行修改】加入期权买方的平仓逻辑
        """
        profit=0    # 该笔交易获得的盈利(实现盈利)
        settle_profit=0   # 该笔交易获得的盯市盈亏(交易价-昨结价)
        margin=0    # 该笔交易收回的保证金
        if order_type=='call':
            position=self.buycall_position.copy()
        else:
            position=self.buyput_position.copy()
        BS={'buy':1,'sell':-1}[order_BS]         # 【新增】为了节省代码段加了一个系数,按买入期权的逻辑对卖出期权收益进行计算
        if position:    # 如果当前还有持仓的话
            current_vol_list=[i['vol'] for i in position[option]]    # 当前的合约持有情况list
            ori_price_list=[i['price'] for i in position[option]]    # 当前合约买入价格情况list
            pre_margin_list=[i['margin'] for i in position[option]]  # 当前合约买入的保证金情况
            pre_settle_list=[i['pre_settle'] for i in position[option]]   # 当前合约的昨结价情况
            current_vol=sum(current_vol_list)    # 当前合约持有的数量
            max_vol=min(current_vol,vol)   # 现在需要平仓的数量
            # ??? self.cash+=max_vol*price*BS    # 期权买方(B)平仓需要卖出期权,得到cash&期权卖方(S)平仓需要买入期权,扣除cash
            record_vol=max_vol  # for record
            if current_vol<=0:
                print(f"合约{option}未持仓,无法平仓")
            elif max_vol>=current_vol:  # 说明要全平仓
                for i in range(0,len(current_vol_list)):
                    vol=current_vol_list[i]
                    ori_price=ori_price_list[i]
                    pre_margin=pre_margin_list[i]
                    pre_settle=pre_settle_list[i]
                    profit+=(price-ori_price)*vol*BS            # 逐笔盈亏(平仓价-开仓价)
                    settle_profit+=(price-pre_settle)*vol*BS    # 结算盈亏(平仓价-昨结算)
                    margin+=(pre_margin+settle_profit)
                del position[option]   # 直接去掉这个合约的持有 {'option':[(price,vol),...]}
            elif max_vol<current_vol:    # 说明部分平仓
                for i in range(0,len(current_vol_list)):
                    vol=current_vol_list[i]
                    ori_price=ori_price_list[i]
                    pre_margin=pre_margin_list[i]
                    pre_settle=pre_settle_list[i]
                    if max_vol>=vol:
                        profit+=(price-ori_price)*vol*BS            # 逐笔盈亏(平仓价-开仓价)
                        settle_profit+=(price-pre_settle)*vol*BS    # 结算盈亏(平仓价-昨结算)
                        margin+=(pre_margin+settle_profit)          # 结算盈亏(平仓价-昨结算)
                        del position[option][0]                     # FIFO原则
                        max_vol=max_vol-vol
                    else:
                        profit+=(price-ori_price)*max_vol*BS
                        settle_profit+=(price-pre_settle)*max_vol*BS
                        margin+=(pre_margin+settle_profit)
                        position[option][0]['vol']=vol-max_vol
                        position[option][0]['margin']=pre_margin*(1-max_vol/vol)  # 剩余的保证金
                        break   # 执行完毕
            # 记录
            self.option_record=self.option_record._append({'state':'close',
                                                           'reason':reason,
                                                           'date':self.current_date,
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
            elif order_type=='call' and order_BS=='sell':
                self.sellcall_position=position
            elif order_type=='put' and order_BS=='buy':
                self.buyput_position=position
            elif order_type=='put' and order_BS=='sell':
                self.sellput_position=position

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
        【补丁】在持仓中增加了FirstDaySettle,仅用来计算第一天收益(结算-开仓)
        profit:逐笔平仓盈亏(平仓-开仓)+profit_settle结算盈亏(开仓-昨日结算)=平仓盈亏(平仓-昨日结算)
        order_type='long':
        order_type='short':
        """
        if order_type=='long':
            pos=self.long_position
        else:
            pos=self.short_position
        if pos: # 如果有持仓的话
            POS=pos.copy()
            LS={'long':1,'short':-1}[order_type]  # 【新增】为了节省代码段加了一个系数,按多头的逻辑对空头收益进行计算
            for contract,List in pos.items():   # 获取当前结算价(waiting)
                df=self.session.run(f"""select pre_settle,settle from loadTable('{self.future_counter_database}','{self.future_counter_table}') where contract=='{contract}' and date==date({self.current_dot_date})""")
                L=[]
                for Dict in List:
                    if not df.empty: # 计算未平仓合约的盯市盈亏
                        slice_df=df.loc[0]
                        pre_settle,settle_price=slice_df['pre_settle'],slice_df['settle'] # 未平仓合约昨日结算价&当日结算价
                        vol=Dict['vol']
                        if 'FirstDaySettle' not in Dict.keys(): # 说明已经不是第一天持仓了
                            settle_profit=(settle_price-pre_settle)*vol*LS
                        else:       # 说明是第一天持仓
                            settle_profit=(settle_price-Dict['price'])*vol*LS
                            del Dict['FirstDaySettle']
                        self.profit_settle+=settle_profit
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
        【补丁】在持仓中增加了FirstDaySettle,仅用来计算第一天收益(结算-开仓)
        """
        if order_type=='call' and order_BS=='buy':
            pos=self.buycall_position
        elif order_type=='call' and order_BS=='sell':
            pos=self.sellcall_position
        elif order_type=='put' and order_BS=='buy':
            pos=self.buyput_position
        else:
            pos=self.sellput_position
        if pos:
            POS=pos.copy()
            BS={'buy':1,'sell':-1}[order_BS]
            for option,List in pos.items():   # 获取当前结算价(waiting)
                df=self.session.run(f"""select pre_settle,settle from loadTable('{self.option_counter_database}','{self.option_counter_table}') where option=='{option}' and date==date({self.current_dot_date})""")
                L=[]
                for Dict in List:
                    if not df.empty: # 计算未平仓合约的盯市盈亏
                        slice_df=df.loc[0]
                        pre_settle,settle_price=slice_df['pre_settle'],slice_df['settle'] # 未平仓合约昨日结算价&当日结算价
                        vol=Dict['vol']
                        if 'FirstDaySettle' not in Dict.keys():  # 说明已经不是第一天持仓了
                            settle_profit=(settle_price-pre_settle)*vol*BS
                        else:  # 说明是第一天持仓
                            settle_profit=(settle_price-Dict['price'])*vol*BS
                            del Dict['FirstDaySettle']
                        self.profit_settle+=settle_profit
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
        pos=self.stock_position
        date, minute =self.current_dot_date, self.current_minute
        for symbol,List in pos.items():
            df=self.session.run(f"""select high,low,close from loadTable('{self.stock_counter_database}','{self.stock_counter_table}') where symbol=="{symbol}" and date==date({date}) and minute ={minute}""")
            for Dict in List:
                high_limit,low_limit,last_timestamp,vol=Dict['max_price'],Dict['min_price'],Dict['max_timestamp'],Dict['vol']
                if not df.empty:
                    # 【盘中】先处理限价单
                    slice_df=df.loc[0]
                    high_price,low_price,close_price=slice_df['high'],slice_df['low'],slice_df['close']
                    state=False
                    if order_sequence:  # 【模拟撮合】最高价先被触发
                        if high_limit:
                            if high_price>=high_limit:
                                self.close_stock(symbol=symbol, price=high_limit, vol=vol, reason='high_limit')
                                state=True
                        elif low_limit:
                            if low_price<=low_limit:
                                self.close_stock(symbol=symbol, price=low_limit, vol=vol, reason='low_limit')
                                state=True
                    elif not order_sequence:  # 【模拟撮合】最低价先被触发
                        if low_limit:
                            if low_price<=low_limit:
                                self.close_stock(symbol=symbol, price=low_limit, vol=vol, reason='low_limit')
                                state=True
                        elif high_limit:
                            if high_price>=high_limit:
                                self.close_stock(symbol=symbol, price=high_limit, vol=vol, reason='high_limit')
                                state=True
                    # 【收盘】处理到最大持仓时间的期货持仓
                    if self.current_timestamp>=pd.Timestamp(last_timestamp) and not state: # 最长持仓时间的股票持仓
                        self.close_stock(symbol=symbol, price=close_price, vol=vol, reason='max_timestamp')
                else:
                    self.close_stock(symbol=symbol, price=0.5*(high_limit+low_limit), vol=vol, reason='data missing')
                    print(f"{symbol}-{self.current_timestamp}'s data is missed, couldn't close this stock")

    def monitor_future(self,order_type,order_sequence):
        """
        【柜台处理订单后运行,可重复运行】每日盘中运行,负责监控当前持仓是否满足限制平仓要求
        order_sequence=True 假设max_price先判断
        order_sequence=False 假设min_price先判断
        """
        if order_type=='long':
            pos=self.long_position
        else:
            pos=self.short_position
        date, minute =self.current_dot_date, self.current_minute
        for contract,List in pos.items():
            df=self.session.run(f"""select high,low,close,end_date from loadTable('{self.future_counter_database}','{self.future_counter_table}') where contract=="{contract}" and date==date({date}) and minute=={minute}""")
            for Dict in List:
                high_limit,low_limit,last_date,vol=Dict['max_price'],Dict['min_price'],Dict['max_timestamp'],Dict['vol']
                if not df.empty:
                    # 【盘中】先处理限价单
                    slice_df=df.loc[0]
                    high_price,low_price,close_price,end_date=slice_df['high'],slice_df['low'],slice_df['close'],slice_df['end_date']
                    state=False
                    if order_sequence:  # 【模拟撮合】最高价先被触发
                        if high_limit:
                            if high_price>=high_limit:
                                self.close_future(order_type=order_type,contract=contract,price=high_limit,vol=vol,reason='high_limit')
                                state=True
                        elif low_limit:
                            if low_price<=low_limit:
                                self.close_future(order_type=order_type,contract=contract,price=low_limit,vol=vol,reason='low_limit')
                                state=True
                    elif not order_sequence:  # 【模拟撮合】最低价先被触发
                        if low_limit:
                            if low_price<=low_limit:
                                self.close_future(order_type=order_type,contract=contract,price=low_limit,vol=vol,reason='low_limit')
                                state=True
                        elif high_limit:
                            if high_price>=high_limit:
                                self.close_future(order_type=order_type,contract=contract,price=high_limit,vol=vol,reason='high_limit')
                                state=True
                    # 【收盘】先处理到最后交易日的期货持仓
                    if self.current_timestamp>=pd.Timestamp(end_date) and not state:
                        """这里可以加上移仓换月的逻辑"""
                        self.close_future(order_type=order_type,contract=contract,price=close_price,vol=vol,reason='end_date')
                        state=True
                    # 【收盘】再处理到最大持仓时间的期货持仓
                    if self.current_timestamp>=pd.Timestamp(last_date) and not state: # 最长持仓时间的期货持仓
                        self.close_future(order_type=order_type,contract=contract,price=close_price,vol=vol,reason='max_timestamp')
                else:
                    self.close_future(order_type=order_type,contract=contract,price=0.5*(high_limit+low_limit),vol=vol,reason='data_missing')
                    print(f"{contract}-{self.current_timestamp}'s data is missed, couldn't close this contract")

    def monitor_option(self,order_type,order_BS,order_sequence):
        """
        【柜台处理订单后运行,可重复运行】每日盘中运行,负责监控当前持仓是否满足限制平仓要求
        order_sequence=True 假设max_price先判断
        order_sequence=False 假设min_price先判断
        【新增】买方/卖方到期日未平仓虚值期权自动清算
        """
        if order_type=='call' and order_BS=='buy':
            pos=self.buycall_position.copy()
        elif order_type=='call' and order_BS=='sell':
            pos=self.sellcall_position.copy()
        elif order_type=='put' and order_BS=='buy':
            pos=self.buyput_position.copy()
        else:
            pos=self.sellput_position.copy()

        date, minute =self.current_dot_date, self.current_minute
        for option,List in pos.items():
            df=self.session.run(f"""select high,low,close,end_date,level from loadTable('{self.option_counter_database}','{self.option_counter_table}') where option=='{option}' and date=date({date}) and minute={minute}""")
            for Dict in List:
                high_limit,low_limit,max_timestamp,vol=Dict['max_price'],Dict['min_price'],Dict['max_timestamp'],Dict['vol']
                if not df.empty:
                    # 说明该日可以交易
                    # 【盘中】先处理限价单
                    slice_df=df.loc[0]
                    high_price,low_price,close_price,end_date,level=slice_df['high'],slice_df['low'],slice_df['close'],slice_df['end_date'],slice_df['level']   # end_date&level用于判断末日期权是否平仓还是等待清算(可能用到未来函数,需要以后进一步确认)
                    state=False
                    if order_sequence:  # 【模拟撮合】最高价先被触发
                        if high_limit:
                            if high_price>=high_limit:
                                self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=high_limit,vol=vol,reason='high_limit')
                                state=True
                        elif low_limit:
                            if low_price<=low_limit:
                                self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=low_limit,vol=vol,reason='low_limit')
                                state=True
                    elif not order_sequence:  # 【模拟撮合】最低价先被触发
                        if low_limit:
                            if low_price<=low_limit:
                                self.close_option(order_type=order_type, order_BS=order_BS, option=option, price=low_limit,vol=vol, reason='low_limit')
                                state=True
                        elif high_limit:
                            if high_price>=high_limit:
                                self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=high_limit,vol=vol,reason='high_limit')
                                state=True
                    # 【收盘】先处理到期权到期日的期权(虚值期权)
                    # 期权特殊处理(clear_option)
                    if self.current_date==end_date and not state and level<0: # 注:一定是到期日还是虚值期权的才可以
                        self.clear_option(order_type=order_type,order_BS=order_BS,option=option,vol=vol,reason='clear')
                        state=1
                    # 【收盘】再处理未到期权到期日但到指令到期日的期权(实值期权)
                    if self.current_timestamp>=pd.Timestamp(max_timestamp) and not state:
                        self.close_option(order_type=order_type,order_BS=order_BS,option=option,price=close_price,vol=vol,reason='max_timestamp')
                else:
                    print(f"{option}-{self.current_timestamp}'s data is missed, couldn't close this option")


class Counter(CounterBehavior):
    """
    交易柜台实例
    """
    def __init__(self,session,config):
        super().__init__(session,config)

    def stock_counter_processing(self):
        """【开仓/平仓order处理后运行,可重复运行】柜台判断open/close是否能够执行,若能则执行,并在柜台删除该订单
        【后续还需要添加volume判断条件,并添加部分成交+剩余继续挂单的情形】
        【同时,由于开仓设置时间是合理的,平仓如果时间过了平不了那大概率真的平不了,所以需要考虑流动性的问题进一步地优化代码】
        """
        stock_counter=self.stock_counter.copy()
        date,minute=self.current_dot_date,self.current_minute
        for i,orderDict in stock_counter.items():   # 订单编号,订单详情
            order_state,symbol,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['symbol'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_timestamp:   # 说明这个订单时间太长了,搞不了
                del self.stock_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}-Symbol{symbol}:Price{price}&Vol{vol} failed[Out of Timestamp]")

            elif self.current_timestamp>=min_order_timestamp:
                df=self.session.run(f"""select low,high from loadTable("{self.stock_counter_database}","{self.stock_counter_table}") where symbol="{symbol}" and date=date({date}) and minute={minute}""")
                if not df.empty:    # 说明这根K线上有该股票的数据
                    if df.loc[0]['low']<=price<=df.loc[0]['high']:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            self.execute_stock(symbol=symbol,vol=vol,price=price,
                                               min_price=orderDict['min_price'],max_price=orderDict['max_price'],
                                               max_timestamp=orderDict['max_timestamp'],commission=orderDict['commission'],
                                               reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            self.close_stock(symbol=symbol,vol=vol,price=price,reason=orderDict['reason'])
                        del self.stock_counter[i]  # 删除柜台的订单
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
        date,minute=self.current_dot_date,self.current_minute
        for i,orderDict in future_counter.items(): # 订单编号,订单详情
            order_state,order_type,contract,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['order_type'],orderDict['contract'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_date:   # 说明这个订单时间太长了,搞不了
                del self.future_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}{order_type}-Contract{contract}:Price{price}&Vol{vol} failed[Out of Timestamp]")
            elif self.current_date>=min_order_timestamp:
                df=self.session.run(f"""select low,high,pre_settle from loadTable('{self.future_counter_database}','{self.future_counter_table}') where contract='{contract}' and date=date({date}) and minute={minute}""")
                if not df.empty:    # 说明这根K线上有该合约的数据
                    if df.loc[0]['low']<=price<=df.loc[0]['high']:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            pre_settle=df.loc[0]['pre_settle']
                            self.execute_future(order_type=order_type,contract=contract,vol=vol,price=price,
                                                pre_settle=pre_settle,margin=orderDict['margin'],
                                                min_price=orderDict['min_price'],max_price=orderDict['max_price'],max_timestamp=orderDict['max_timestamp'],
                                                commission=orderDict['commission'],reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            self.close_future(order_type=order_type,contract=contract,vol=vol,price=price,reason=orderDict['reason'])
                        del self.future_counter[i]  # 删除柜台的订单
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
        for i,orderDict in option_counter.items():  # 订单编号,订单详情
            order_state,order_type,order_BS,option,price,vol,min_order_timestamp,max_order_timestamp=orderDict['order_state'],orderDict['order_type'],orderDict['order_BS'],orderDict['option'],orderDict['price'],orderDict['vol'],orderDict['min_order_timestamp'],orderDict['max_order_timestamp']
            if max_order_timestamp<=self.current_timestamp:  # 说明这个订单时间太长了,搞不了
                del self.option_counter[i]
                print(f"OrderNum{i}:Behavior{order_state}{order_BS}{order_type}-Option{option}:Price{price}&Vol{vol} failed[Out of Date]")
            elif self.current_timestamp>=min_order_timestamp:
                df=self.session.run(f"""select low,high,pre_settle from loadTable('{self.option_counter_database}','{self.option_counter_table}') where option='{option}' and date==date({self.current_dot_date}) and low<={price} and high>={price}""")
                if not df.empty:    # 说明K线上有该合约的数据
                    if df.loc[0]['low']<=price<=df.loc[0]['high']:  # 说明可以成交
                        if order_state=='open': # 开仓命令
                            self.execute_option(order_type=order_type,order_BS=order_BS,option=option,vol=vol,price=price,strike=orderDict['strike'],
                                                pre_settle=orderDict['pre_settle'],margin=orderDict['margin'],
                                                min_price=orderDict['min_price'],max_price=orderDict['max_price'],max_timestamp=orderDict['max_timestamp'],
                                                commission=orderDict['commission'],reason=orderDict['reason'])
                        elif order_state=='close':  # 平仓命令
                            self.close_option(order_type=order_type,order_BS=order_BS,option=option,vol=vol,price=price,reason=orderDict['reason'])
                        del self.option_counter[i]  # 删除柜台的订单
                    else:  # 说明当日不能成交
                        pass
                else:  # 说明当日没有该合约的数据
                    pass

