import json,json5
import dolphindb as ddb
import pandas as pd
import numpy as np
import tqdm

if __name__ == "__main__":
    session=ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")
    # session.run(
    # """
    #     if (existsTable("dfs://component","component_cn")){
    #      dropTable(database("dfs://component"),"component_cn")
    #     }
    #     columns_name=["index","date","component"]
    #     columns_type=[SYMBOL,DATE,SYMBOL]
    #     db=database("dfs://component",RANGE,2000.01M+(0..30)*12,engine="TSDB");
    #     schemaTb=table(1:0,columns_name,columns_type);
    #     t=db.createPartitionedTable(table=schemaTb,tableName="component_cn",partitionColumns=`date,sortColumns=`index`component`date,keepDuplicates=LAST)
    # """
    # )
    with open("index50_symbol.json","r") as f:
        comp = json.load(f)
    appender = ddb.PartitionedTableAppender(dbPath="dfs://component",
                                            tableName="component_cn",
                                            partitionColName="date",
                                            dbConnectionPool=pool)  # 写入数据的appender
    for date,component_list in tqdm.tqdm(comp.items()):
        appender.append(pd.DataFrame({"index":["000016"]*len(component_list),
                                      "date":[pd.Timestamp(date)]*len(component_list),
                                      "component":component_list}))


