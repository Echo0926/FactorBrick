import pandas as pd
import numpy as np
import dolphindb as ddb
from src.utils import *

# しゅくだい宿題
def get_stock_k_data(session, pool, save_database:str, save_table:str):
    # 默认删除原来的数据库
    if session.existsTable(dbUrl=save_database,tableName=save_table):
        session.dropTable(dbPath=save_database,tableName=save_table)
    if not session.existsTable(dbUrl=save_database,tableName=save_table):
        columns_name=["date","minute","timestamp","symbol",
                      "open","high","low","close","volume"]
        columns_type=["DATE","INT","DATETIME","SYMBOL",
                      "DOUBLE","DOUBLE","DOUBLE","DOUBLE","DOUBLE"]
        session.run(f"""
            db=database("{save_database}",RANGE,2000.01M+(0..30)*30,engine="OLAP");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns=["date"])
    """)
    appender = ddb.PartitionedTableAppender(dbPath=save_database,
                                            tableName=save_table,
                                            partitionColName="date",
                                            dbConnectionPool=pool)

    df = session.run("""
        select date,1500 as minute,symbol,open,high,low,close,volume from loadTable("dfs://stock_cn/value","market_hfq");        
    """)

    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    df["date"] = df["date"].apply(str).apply(pd.Timestamp)
    df["minute"] = df["minute"].apply(float).apply(int)
    df['time'] = df['minute'].astype(str).str.zfill(4)  # 补零到4位（如 930 → '0930'）
    df['time'] = df['time'].str[:2] + ':' + df['time'].str[2:] + ':00'  # 格式化为 '09:30:00'
    df['timestamp'] = pd.to_datetime(df['date'].dt.date.astype(str) + ' ' + df['time'])
    df = df[["date","minute","timestamp","symbol","open","high","low","close","volume"]]
    # 批量添加数据
    chunk_size = 100000  # 每个 chunk 的大小
    for start in tqdm.tqdm(range(0, len(df), chunk_size), desc="Adding Stock K Data"):
        end = start + chunk_size
        chunk = df[start:end]
        appender.append(chunk)

def get_future_k_data(session, save_database:str, save_table:str):
    # 默认删除原来的数据库
    if session.existsTable(dbUrl=save_database,tableName=save_table):
        session.dropTable(dbPath=save_database,tableName=save_table)
    if not session.existsTable(dbUrl=save_database,tableName=save_table):
        columns_name=["date","minute","timestamp","contract",
                          "pre_settle","open","high","low","close","settle","volume",
                          "multi","margin","start_date","end_date","isMainContract"]
        columns_type=["DATE","INT","DATETIME","SYMBOL",
                          "DOUBLE","DOUBLE","DOUBLE","DOUBLE","DOUBLE","DOUBLE","DOUBLE",
                          "DOUBLE","DOUBLE","DATE","DATE","DOUBLE"]
        session.run(f"""
            db=database("{save_database}",RANGE,2000.01M+(0..30)*30,engine="OLAP");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns=["date"])
        """)

    session.run(f"""
        // 初始行情数据
        pt = select product,contract,date,minute,open,high,low,close,settle,volume from loadTable("dfs://future_cn/value","market");
        
        // Filter
        pt=select * from pt where product in ["IH"];
        
        // 添加timestamp字段
        update pt set dot_date = string(date);
        update pt set fill_minute = string(minute).lpad(4, "0");
        update pt set dot_minute = left(fill_minute,2)+":"+right(fill_minute,2)+":00";
        update pt set timestamp =datetime(string(dot_date+" "+dot_minute));
        
        // 添加pre_settle字段
        settle_pt = select firstNot(settle) as settle from pt group by date,contract;
        update settle_pt set pre_settle = prev(settle) context by contract;
        dropColumns!(settle_pt, `settle);
        pt = select * from pt left join settle_pt on pt.date=settle_pt.date and pt.contract=settle_pt.contract;
        
        // 添加start_date/end_date字段(合约开始/结束时间)
        date_pt =select firstNot(date) as start_date, lastNot(date) as end_date from pt group by contract;
        pt = select * from pt left join date_pt on date_pt.contract= pt.contract;
        
        // 添加交易乘数字段
        update pt set multi = 300.0; // 一手IH期货交易乘数为300
        
        // 添加margin字段(保证金率)
        update pt set margin = 0.1;
                
        // 主力合约判断(成交量最大的合约, 当一个合约成为主力合约之后只允许比他大的合约成为主力合约)
        df = select sum(volume) as daily_volume from pt group by product,contract,date order by date; 
        update df set time = int(right(contract,4)) // 提取合约时间
        update df set rank = rank(daily_volume, false) context by date;
        // 主力合约信息(假定当该合约成为主力合约, 只允许比晚的合约再次成为主力合约)
        info = select date, contract, time from df where rank = 0
        update info set contract = NULL where time<prev(time);  // 检测如果time<prev(time), 就把contract填充为NULL
        update info set contract = contract.ffill();  // 向后填充
        update info set isMainContract = 1.0; // 主力合约标识
        dropColumns!(info,`time) 
        
        // 添加isMainContract字段
        pt = select * from pt left join info on pt.date =info.date and pt.contract= info.contract;
        update pt set isMainContract = nullFill(isMainContract, 0.0); 
        
        // 最终结果
        pt = select date,minute,timestamp,contract, nullFill(pre_settle,open) as pre_settle,open,high,low,close,settle,volume,multi,margin,start_date,end_date,isMainContract from pt;
        loadTable("{save_database}","{save_table}").append!(pt);
        undef(`pt); // 释放内存
    """)

def get_stock_info(session, save_database:str, save_table:str):
    # 默认删除原来的数据库
    if session.existsTable(dbUrl=save_database,tableName=save_table):
        session.dropTable(dbPath=save_database,tableName=save_table)
    if not session.existsTable(dbUrl=save_database,tableName=save_table):
        columns_name=["date","symbol","open","high","low","close","start_date","end_date"]
        columns_type=["DATE","SYMBOL","DOUBLE","DOUBLE","DOUBLE","DOUBLE","DATE","DATE"]
        session.run(f"""
            db=database("{save_database}",RANGE,2000.01M+(0..30)*30,engine="OLAP");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns=["date"])
        """)

    session.run(f"""
        df = select * from loadTable("dfs://stock_cn/value","market_hfq");
        period = select firstNot(date) as start_date, lastNot(date) as end_date from df group by symbol;
        df = select first(open) as open, max(high) as high, min(low) as low, last(close) as close from df group by symbol,date order by date;
        // 合并df
        df = select * from df left join period on df.symbol=period.symbol;
        undef(`period);
        df = select date,symbol,open,high,low,close,start_date,end_date from df;
        loadTable("{save_database}","{save_table}").append!(df);
    """)

def get_future_info(session, save_database:str, save_table:str):
    # 默认删除原来的数据库
    if session.existsTable(dbUrl=save_database,tableName=save_table):
        session.dropTable(dbPath=save_database,tableName=save_table)
    if not session.existsTable(dbUrl=save_database,tableName=save_table):
        columns_name=["date","contract","pre_settle","open","high","low","close","settle",
                    "multi","margin","start_date","end_date","isMainContract"]
        columns_type=["DATE","SYMBOL","DOUBLE","DOUBLE","DOUBLE","DOUBLE",
                      "DOUBLE","DOUBLE","DOUBLE","DOUBLE","DATE","DATE","DOUBLE"]
        session.run(f"""
            db=database("{save_database}",RANGE,2000.01M+(0..30)*30,engine="OLAP");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{save_table}",partitionColumns=["date"])
        """)

    session.run(f"""
        df = select * from loadTable("dfs://future_cn/backtest","base");
        df = select first(open) as open, first(pre_settle) as pre_settle, max(high) as high, min(low) as low, last(close) as close, firstNot(settle) as settle, firstNot(multi) as multi, firstNot(margin) as margin, firstNot(start_date) as start_date, firstNot(end_date) as end_date, firstNot(isMainContract) as isMainContract from df group by contract, date order by date;
        df = select date,contract,pre_settle,open,high,low,close,settle,multi,margin,start_date,end_date,isMainContract from df;
        loadTable("{save_database}","{save_table}").append!(df);
    """)


if __name__ == "__main__":
    # Configuration
    session=ddb.session()
    session.connect("localhost",8848,"admin","123456")
    pool=ddb.DBConnectionPool("localhost",8848,10,"admin","123456")

    # get_stock_k_data(session, pool, save_database="dfs://stock_cn/backtest", save_table="base")
    get_stock_info(session, save_database="dfs://stock_cn/info", save_table="info")

    # get_future_k_data(session, save_database="dfs://future_cn/backtest", save_table="base")
    # get_future_info(session, save_database="dfs://future_cn/info", save_table="info")
