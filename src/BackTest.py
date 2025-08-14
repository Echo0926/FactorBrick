import json5
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
import matplotlib.pyplot as plt
from src.utils import *
from src.entities.BackTestPandas import Counter

pd.set_option("display.max_columns",None)
plt.rcParams['font.sans-serif']=['KaiTi'] # 显示中文
plt.rcParams['axes.unicode_minus']=False # 显示负号

class Backtest(Counter):
    """
    股票+期货+期权回测框架
    """
    def __init__(self,strategy,session,config):
        """
        初始化策略参数
        """
        super().__init__(session,config)
        self.strategy=strategy
        self.seed=config["seed"]

    def run(self):
        """运行策略+可视化"""
        self.strategy(self=self)    # 策略运行

        fig, ax1 = plt.subplots()
        ax1.plot(self.profit_Dict.keys(), self.profit_Dict.values(), 'b-', label='profit')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Profit', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()         # 创建右轴
        ax2.plot(self.pos_Dict.keys(), self.pos_Dict.values(), 'r-', label="position")
        ax2.set_ylabel('Position', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False)
        plt.savefig(r"日度盈亏.png")
        plt.show()

        if self.run_future or self.run_option:
            plt.plot(self.settle_profit_Dict.keys(), self.settle_profit_Dict.values(), label='settle_profit')
            plt.savefig(r"盯市盈亏.png")
            plt.show()

        plt.plot(self.cash_Dict.keys(),self.cash_Dict.values(),label='cash')
        plt.legend(frameon=False)
        plt.savefig(r"总资金.png")
        plt.show()

        # Recording
        stock_record, future_record, option_record = self.stock_record, self.future_record, self.option_record
        stock_record.to_csv(f"Stock交易明细{pd.Timestamp.today().strftime('%Y-%m-%d')}(Stock).csv", index=None)
        future_record.to_csv(f"Stock交易明细{pd.Timestamp.today().strftime('%Y-%m-%d')}(Future).csv", index=None)


if __name__=="__main__":
    # Configuration
    session=ddb.session()
    session.connect("localhost",8848,"admin","123456")
    # from src.strategy_func.CTASeries import get_future_signal_to_dataframe, CTA_strategy
    from src.strategy_func.StockPanel import get_stock_signal_to_dataframe, Stock_strategy
    # from src.strategy_func.Neutral import get_stock_signal_to_dataframe, Neutral_strategy
    with open(r".\config\backtest_config.json5", mode="r", encoding="UTF-8") as file:
        BackTest_config = json5.load(file)
    with open(r".\config\returnmodel_config.json5", mode="r", encoding="UTF-8") as file:
        ReturnModel_config = json5.load(file)

    # # # K Data Generation
    # K_data=session.run("""
    # pt=select symbol,date,1500 as minute,open,high,low,close,volume from loadTable("dfs://stock_cn/value","market_hfq") where date>=2020.01.01;
    # update pt set timestamp = datetime(string(date)+"T15:00:00");
    # pt""")
    # stock_K_path = [BackTest_config["stock_K_database"], BackTest_config["stock_K_table"]]
    # K_data.to_parquet(rf"{stock_K_path[0]}\{stock_K_path[1]}.pqt", index=False)

    # Info Generation
    # Info_data = session.run("""
    # pt = select * from loadTable("dfs://stock_cn/info","info") where date>=2020.01.01;
    # update pt set end_date = 2030.01.01;
    # pt
    # """)
    # stock_info_path = BackTest_config["stock_info_json"]
    # write_info_json(data=Info_data, date_col="date", symbol_col="symbol", save_path=stock_info_path,
    #              index_col="symbol")

    # # Macro Position Generation
    # macro_data = session.run("""
    # pt = select * from loadTable("dfs://index_cn/stock_index","stock_index") where symbol = "000985";
    # update pt set bias = nullFill((prev(close)-prev(ma(close,5,1)))/prev(ma(close,5,1)),0);
    # select date,bias from pt
    # """)
    # macro_data["date"]=macro_data["date"].apply(lambda x: pd.Timestamp(x).strftime("%Y%m%d"))
    # write_json(data=dict(zip(macro_data["date"],macro_data["bias"])),
    #            output_path="E:\\factorbrick\\data\\backtest\\stock_cn\\macro_json/",
    #            file_name="macro.json")

    # K_data=session.run("""
    # pt=select * from loadTable("dfs://future_cn/combination","base") where isMainContract=1;
    # pt""")
    # future_k_path = [BackTest_config["future_K_database"], BackTest_config["future_K_table"]]
    # K_data.to_parquet(rf"{future_k_path[0]}\{future_k_path[1]}.pqt", index=False)

    # # Info Generation
    # Info_data = session.run("""
    # pt = select * from loadTable("dfs://future_cn/info","info");
    # pt
    # """)
    # future_info_path = BackTest_config["future_info_json"]
    # write_info_json(data=Info_data, date_col="date", symbol_col="contract", save_path=future_info_path,
    #              index_col="contract")

    # Signal Generation
    get_stock_signal_to_dataframe(session, BackTest_config=BackTest_config, ReturnModel_config=ReturnModel_config)
    # # get_future_signal_to_dataframe(session, BackTest_config=BackTest_config, ReturnModel_config=ReturnModel_config)
    #
    # BackTesting
    S=Backtest(session=session,strategy=Stock_strategy,config=BackTest_config)
    S.run()

    print(S.stock_position)
    print(S.long_position)
    print(S.short_position)

    df = pd.DataFrame({"date":S.profit_Dict.keys(),"profit":S.profit_Dict.values()})
    df["net_value"] = 1+(df["profit"]/S.ori_cash)
    max_drawdown_plot(df=df,save_path=r"E:\factorbrick\reports")



