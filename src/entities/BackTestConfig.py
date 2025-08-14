import pandas as pd
import json,json5

"""
所有数据类型的回测共有参数
"""

class BackTestConfig:
    def __init__(self, session, config):
        self.session = session
        self.config = config

        """基本信息"""
        self.strategy_name = config["strategy_name"]  # 策略名称(默认为strategy)
        self.data_type = config["data_type"]    # dolphindb or dataframe

        """回测模块"""
        # 0.时间类
        self.date_list = []         # 程序确定下来的回测周期
        self.dot_date_list = []
        self.str_date_list = []
        self.current_date = pd.Timestamp(config["start_date"])  # 循环时候的当前交易日
        self.current_minute = 915  # 循环时候的当前分钟
        self.current_timestamp = pd.Timestamp(config["start_date"]) # 当前timestamp时间戳
        self.current_str_date = self.current_date.strftime('%Y%m%d')
        self.current_dot_date = self.current_date.strftime('%Y.%m.%d')
        self.start_date = config["start_date"].replace("-", "").replace(".", "")  # 策略回测开始日期
        self.end_date = config["end_date"].replace("-", "").replace(".", "")  # 策略回测结束日期
        self.start_dot_date = pd.Timestamp(self.start_date).strftime('%Y.%m.%d')
        self.end_dot_date = pd.Timestamp(self.end_date).strftime('%Y.%m.%d')

        # 0.柜台类(中间变量,运行结束会删除这两张表)
        self.orderNum = 0  # 订单编号(唯一值)
        self.run_stock = config["run_stock"]  # 策略中是否包含股票
        self.run_future = config["run_future"]  # 策略中是否包含期货
        self.run_option = config["run_option"]  # 策略中是否包含期权
        self.stock_counter_json = config["stock_counter_json"]
        self.future_counter_json = config["future_counter_json"]
        self.option_counter_json = config["option_counter_json"]

        # 1.股票类
        self.stock_k_dict= {}
        self.stock_macro_dict={}
        self.stock_info_dict = {}
        self.stock_signal_dict = {}
        self.stock_timestamp_dict = {}  # 每天的交易分钟dict(笛卡尔积) {date:[timestamp1,...timestampN]}
        self.stock_date_list = []         # 股票的回测周期
        self.stock_dot_date_list = []
        self.stock_str_date_list = []
        self.stock_K_database = config["stock_K_database"]
        self.stock_K_table = config["stock_K_table"]
        self.stock_signal_database = config["stock_signal_database"]
        self.stock_signal_table = config["stock_signal_table"]
        self.stock_macro_json = config["stock_macro_json"]
        self.stock_info_json = config["stock_info_json"]
        self.stock_signal_json = config["stock_signal_json"]

        # 2.期货类
        self.future_k_dict = {}
        self.future_macro_dict = {}
        self.future_info_dict = {}
        self.future_signal_dict = {}
        self.future_timestamp_dict = {}  # 每天的交易分钟dict(笛卡尔积) {date:[timestamp1,...timestampN]}
        self.future_date_list = []         # 期货的回测周期
        self.future_dot_date_list = []
        self.future_str_date_list = []
        self.future_K_database = config["future_K_database"]  # product contract date open close settle...+fundamental
        self.future_K_table = config["future_K_table"]
        self.future_signal_database = config["future_signal_database"]  # product contract date long_signal short_signal;
        self.future_signal_table = config["future_signal_table"]
        self.future_macro_json = config["future_macro_json"]
        self.future_info_json = config["future_info_json"]
        self.future_signal_json = config["future_signal_json"]

        # 3.期权类
        self.option_k_dict = {}
        self.option_macro_dict = {}
        self.option_info_dict = {}
        self.option_signal_dict = {}
        self.option_timestamp_dict = {}  # 每天的交易分钟dict(笛卡尔积) {date:[timestamp1,...timestampN]}
        self.option_date_list = []         # 期货的回测周期
        self.option_dot_date_list = []
        self.option_str_date_list = []
        self.option_K_database = config["option_K_database"]  # product contract date option open close settle...+fundamental
        self.option_K_table = config["option_K_table"]
        self.option_signal_database = config["option_signal_database"]  # product contract date option buycall_signal sellcall_signal buyput_signal sellput_signal
        self.option_signal_table = config["option_signal_table"]
        self.option_macro_json = config["option_macro_json"]
        self.option_info_json = config["option_info_json"]
        self.option_signal_json = config["option_signal_json"]

        """柜台模块"""
        # 0.柜台类
        self.stock_counter = {}  # 股票柜台
        self.future_counter = {}  # 期货柜台
        self.option_counter = {}  # 期权柜台

        # 1.持仓类
        self.stock_record = pd.DataFrame({'state': [],
                                          'reason': [],
                                          'date': [],
                                          "minute": [],
                                          'symbol': [],
                                          'price': [],
                                          'vol': [],
                                          'pnl': []})
        self.future_record = pd.DataFrame({'state': [], 'reason': [], 'date': [], "minute":[], 'contract': [], 'order_type': [], 'price': [], 'vol': [], 'pnl': []})
        self.option_record = pd.DataFrame({'state': [], 'reason': [], 'date': [], "minute":[], 'option': [], 'order_type': [], 'price': [], 'vol': [], 'pnl': []})  # order_type:['BC','SC','BP','SP']
        self.stock_position = {}  # 当前股票持仓情况 format:见开头注释
        self.stock_summary = {}   # [新增：用于优化止盈止损]
        self.long_position = {}  # 当前多单期货持仓情况 format:见开头注释
        self.long_summary = {}
        self.short_position = {}  # 当前空单期货持仓情况
        self.short_summary = {}
        self.buycall_position = {}  # 当前买入看涨期权持仓情况  format:见开头注释
        self.buycall_summary = {}
        self.buyput_position = {}  # 当前买入看跌期权持仓情况
        self.buyput_summary = {}
        self.sellcall_position = {}  # 当前卖出看涨期权持仓情况  format:见开头注释
        self.sellcall_summary = {}
        self.sellput_position = {}  # 当前卖出看跌期权持仓情况
        self.sellput_summary = {}

        """评价模块"""
        # 1.利润类# TODO: 之后需要对不同资产(option/future)的收益进行统计
        self.cash = config["cash"]  # format:1000000 初始资金
        self.ori_cash = self.cash  # 初始资金(const，用于计算收益率)
        self.profit = 0  # format:0 逐笔盈亏(卖出价-买入价)   # 只对已经平仓的合约进行计算
        self.profit_settle = 0  # format:0 盯市盈亏(结算价/卖出价-昨结算价)  # 先对当日平仓的合约进行计算,之后对未平仓的合约进行计算
        self.cash_Dict = {pd.to_datetime(self.start_date): self.ori_cash}  # 用于记录cash的历史波动:{'date':cash}
        self.profit_Dict = {pd.to_datetime(self.start_date): 0}  # 用于记录profit的历史波动:{'date':profit}
        self.settle_profit_Dict = {pd.to_datetime(self.start_date): 0}  # 用于记录settle_profit的历史波动:{'date':settle_profit}
        self.pos_Dict = {pd.to_datetime(self.start_date): 0}    # 用于记录整体仓位水平
        self.stock_order_record = []
        self.future_order_record = []
        self.option_order_record = []