import json5
import pandas as pd
import dolphindb as ddb
import numpy as np
from functools import lru_cache
import streamlit as st
from ReturnModel_light import SingleFactorBackTest


class ReturnModel_Plot(SingleFactorBackTest):
    def __init__(self, session, pool, config):
        # 基本信息
        super().__init__(session, pool, config)

    @lru_cache(128)
    def Summary_plot(self: SingleFactorBackTest):
        """
        所有因子横向比较可视化
        including: avg(IC), avg(RankIC), ICIR
        """
        benchmark = st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        r_interval = st.selectbox(
            label="请输入未来收益率区间长度",
            options=(i for i in self.returnIntervals),
            index=0,
            format_func=str,
            help='即ReturnModel中的returnIntervals'
        )
        st.title("_Total Factor Performance Comparison_")
        summary_table = self.summary_table
        Dict = self.session.run(f"""
        pt = select ReturnInterval,period,factor,indicator,value from loadTable("{self.result_database}","{summary_table}") 
            where Benchmark=="{benchmark}" and 
            ReturnInterval == int({r_interval}) and 
            indicator in ["IC","RankIC"]
        template_pt = select startDate as date,period from loadTable("{self.combine_database}","{self.template_table}")
        // 添加时间
        pt = lj(template_pt,pt,`period);
        update pt set yearInt = year(date);
        update pt set yearStr = "Year"+string(yearInt)
        year_list = sort(exec distinct(yearInt) from pt)
        undef(`template_pt);

        // avg(IC)
        TotalIC_pt = select avg(value) as Total from pt where indicator == "IC" group by factor
        sortBy!(TotalIC_pt,`factor)
        YearIC_pt = select avg(value) as value from pt where indicator == "IC" pivot by factor, yearStr
        YearIC_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearIC_pt).eval()
        TotalIC_pt = lj(TotalIC_pt, YearIC_pt, `factor)

        // avg(RankIC)
        TotalRankIC_pt = select avg(value) as Total from pt where indicator == "RankIC" group by factor 
        sortBy!(TotalRankIC_pt,`factor)
        YearRankIC_pt = select avg(value) as value from pt where indicator == "RankIC" pivot by factor, yearStr
        YearRankIC_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearRankIC_pt).eval()
        TotalRankIC_pt = lj(TotalRankIC_pt, YearRankIC_pt, `factor)

        // avg(IC)\std(IC)
        TotalICIR_pt = select avg(value)\std(value) as Total from pt where indicator == "IC" group by factor
        sortBy!(TotalICIR_pt,`factor)
        YearICIR_pt = select avg(value)\std(value) as value from pt where indicator == "IC" pivot by factor, yearStr
        YearICIR_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearICIR_pt).eval()
        TotalICIR_pt = lj(TotalICIR_pt, YearICIR_pt, `factor)

        // avg(RankIC)\std(RankIC)
        TotalRankICIR_pt = select avg(value)\std(value) as Total from pt where indicator == "RankIC" group by factor
        sortBy!(TotalRankICIR_pt,`factor)
        YearRankICIR_pt = select avg(value)\std(value) as value from pt where indicator == "RankIC" pivot by factor, yearStr
        YearRankICIR_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearRankICIR_pt).eval()
        TotalRankICIR_pt = lj(TotalRankICIR_pt, YearRankICIR_pt, `factor)

        // 返回结果
        res_dict = dict(["TotalIC","TotalRankIC","TotalICIR","TotalRankICIR"], [TotalIC_pt,TotalRankIC_pt,TotalICIR_pt,TotalRankICIR_pt])
        res_dict
        """)
        TotalIC_df = Dict["TotalIC"]
        TotalRankIC_df = Dict["TotalRankIC"]
        TotalICIR_df = Dict["TotalICIR"]
        TotalRankICIR_df = Dict["TotalRankICIR"]
        st.subheader("All Factors' avg(IC)", divider=True)
        st.dataframe(data=TotalIC_df)
        st.subheader("All Factors' avg(RankIC)", divider=True)
        st.dataframe(data=TotalRankIC_df)
        st.subheader("All Factors' ICIR", divider=True)
        st.dataframe(data=TotalICIR_df)
        st.subheader("All Factors' RankICIR", divider=True)
        st.dataframe(data=TotalRankICIR_df)

    @lru_cache(128)
    def FactorR_plot(self: SingleFactorBackTest):
        """单因子收益率可视化
        including: R/IC/RankIC/Tstats/Reg_stats(R_square/Adj_square/Obs)
        """
        benchmark = st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        factor = st.selectbox(
            label="请选择因子",
            options=self.factor_list,
            index=0,
            format_func=str,
            help="选择当前因子进行因子分层收益展示"
        )
        st.title("_Single Factor BackTest Analysis_")
        tabReg, tabIC, tabQuantile, tabStats = st.tabs(["回归法", "IC法", "分层回测", "其他指标"])
        Dict = self.session.run(rf"""
        pt=select * from loadTable("{self.result_database}","{self.summary_table}") 
            where Benchmark="{benchmark}" and factor ilike "%{factor}" 
            and (indicator ilike "%_OLS" or indicator in ["IC","RankIC"]);
        quantile_pt=select * from loadTable("{self.result_database}","{self.quantile_table}") 
                    where factor == "{factor}"

        // 因子收益率&累计因子收益率
        R=select value from pt where indicator ="R_OLS" pivot by TradeTime,indicator;
        R_cumsum=R.copy();
        L=R_cumsum["TradeTime"];
        dropColumns!(R_cumsum,`TradeTime);
        R_cumsum=cumsum(R_cumsum);
        R_cumsum=select L as TradeTime,* from R_cumsum;

        // Reg_stat
        Obs=select value from pt where indicator ilike "Obs_%" pivot by TradeTime,indicator;
        Std_Error=select value from pt where indicator ilike "Std_Error_%" pivot by TradeTime,indicator;  // 残差标准差
        R_square=select value from pt where indicator ilike "R_square_%" pivot by TradeTime,indicator;
        Adj_square=select value from pt where indicator ilike "Adj_square_%" pivot by TradeTime,indicator;

        // Tstat
        t_stat=select value from pt where indicator ilike "tstat_%" pivot by TradeTime,indicator;

        // IC & 累计IC
        IC=select value from pt where indicator="IC" pivot by TradeTime,indicator;
        IC_cumsum=IC.copy();
        L=IC_cumsum["TradeTime"];
        dropColumns!(IC_cumsum,`TradeTime);
        IC_cumsum=cumsum(IC_cumsum);
        IC_cumsum=select L as TradeTime,* from IC_cumsum;

        // RankIC & 累计RankIC
        RankIC=select value from pt where indicator="RankIC" pivot by TradeTime,indicator;
        RankIC_cumsum=RankIC.copy();
        L=RankIC_cumsum["TradeTime"];
        dropColumns!(RankIC_cumsum,`TradeTime);
        RankIC_cumsum=cumsum(RankIC_cumsum);
        RankIC_cumsum=select L as TradeTime,* from RankIC_cumsum;

        // Yearly avg(IC)&IR
        data=unpivot(IC,keyColNames="TradeTime",valueColNames=columnNames(IC)[1:])
        rename!(data,`TradeTime`factor`factor_IC);
        avg_IC=select avg(factor_IC) from data pivot by year(TradeTime) as year,factor;
        IR=select avg(factor_IC)/std(factor_IC) from data pivot by year(TradeTime) as year,factor;

        // Yearly avg(RankIC)&RankIR
        data=unpivot(RankIC,keyColNames="TradeTime",valueColNames=columnNames(RankIC)[1:])
        rename!(data,`TradeTime`factor`factor_RankIC);
        avg_RankIC=select avg(factor_RankIC) from data pivot by year(TradeTime) as year,factor;
        RankIR=select avg(factor_RankIC)/std(factor_RankIC) from data pivot by year(TradeTime) as year,factor;

        // 返回为字典格式
        Dict=dict(["R_square","Adj_square","Obs","Std_Error","R","R_cumsum","t_stat",
                  "IC","IC_cumsum","RankIC","RankIC_cumsum","avg_IC","IR","avg_RankIC","RankIR"],
                  [R_square,Adj_square,Obs,Std_Error,R,R_cumsum,t_stat,IC,IC_cumsum,RankIC,RankIC_cumsum,avg_IC,IR,avg_RankIC,RankIR]);

        // Quantile Return & Quantile Cumsum Return
        returnIntervals = {self.returnIntervals}
        for (r_interval in returnIntervals){{  // 这里只统计累计值(cumsum)
            df = sql(select=[sqlCol(`TradeTime)].append!(sqlCol("QuantileReturn"+string(1..{self.Quantile}))),
                    from=quantile_pt, where=<ReturnInterval == r_interval>).eval()
            ts_list = df[`TradeTime];
            dropColumns!(df,`TradeTime);
            df = cumsum(df) + 1
            Dict["Return"+string(r_interval)] = select ts_list as `TradeTime, * from df
        }}
        undef(`pt); // 清除缓存

        Dict
        """)
        R_square = Dict["R_square"]
        Adj_square = Dict["Adj_square"]
        Obs = Dict["Obs"]
        Std_Error = Dict["Std_Error"]
        R = Dict["R"]
        R_cumsum = Dict["R_cumsum"]
        t_stat = Dict["t_stat"]
        # IC=Dict["IC"]
        IC_cumsum = Dict["IC_cumsum"]
        # RankIC=Dict["RankIC"]
        RankIC_cumsum = Dict["RankIC_cumsum"]
        avg_IC = Dict["avg_IC"]
        IR = Dict["IR"]
        avg_RankIC = Dict["avg_RankIC"]
        RankIR = Dict["RankIR"]
        with tabReg:
            st.subheader("Single Factor Return", divider=True)
            st.line_chart(data=R, x="date", y=None)
            st.subheader("Single Factor Return(cumsum)", divider=True)
            st.line_chart(data=R_cumsum, x="date", y=None)
            st.subheader("Factor Tstat", divider=True)
            st.bar_chart(data=t_stat, x="date", y=None, stack=False)
            st.write("T值绝对值大于等于2的比例")
            t_stat = t_stat.set_index("date")
            t_stat = (t_stat.abs() >= 2).mean()  # .mean()计算|T|≥2的比例
            st.dataframe(data=t_stat)
        with tabIC:
            # st.subheader("Factor IC",divider=True)
            # st.bar_chart(data=IC,x="TradeTime",y=None,stack=False)
            # st.subheader("Factor RankIC",divider=True)
            # st.bar_chart(data=RankIC,x="TradeTime",y=None,stack=False)
            st.subheader("Factor IC(cumsum)", divider=True)
            st.line_chart(data=IC_cumsum, x="TradeTime", y=None)
            st.subheader("Factor RankIC(cumsum)", divider=True)
            st.line_chart(data=RankIC_cumsum, x="TradeTime", y=None)
            st.subheader("Factor avg(IC)", divider=True)
            st.bar_chart(data=avg_IC, x="year", y=None, stack=False)
            st.dataframe(data=avg_IC)
            st.write("Total avg(IC):")
            st.dataframe(data=avg_IC.set_index("year").mean())
            st.subheader("Factor IR", divider=True)
            st.bar_chart(data=IR, x="year", y=None, stack=False)
            st.dataframe(data=IR)
            st.subheader("Factor avg(RankIC)", divider=True)
            st.bar_chart(data=avg_RankIC, x="year", y=None, stack=False)
            st.dataframe(data=avg_RankIC)
            st.write("Total avg(RankIC):")
            st.dataframe(data=avg_RankIC.set_index("year").mean())
            st.subheader("Factor RankIR", divider=True)
            st.bar_chart(data=RankIR, x="year", y=None, stack=False)
            st.dataframe(data=RankIR)
        with tabQuantile:
            for r_interval in self.returnIntervals:
                st.subheader(f"Single Factor Quantile Return(ReturnInterval={r_interval})", divider=True)
                st.line_chart(data=Dict["Return" + str(r_interval)], x="TradeTime", y=None)
        with tabReg:
            st.subheader("R square", divider=True)
            st.bar_chart(data=R_square, x="TradeTime", y=None, stack=False)
            st.subheader("Adj R suqare", divider=True)
            st.bar_chart(data=Adj_square, x="TradeTime", y=None, stack=False)
            st.subheader("Std Error(残差标准差)", divider=True)
            st.bar_chart(data=Std_Error, x="TradeTime", y=None, stack=False)
            st.subheader("Num of Obs", divider=True)
            st.line_chart(data=Obs, x="TradeTime", y=None)

        return Dict  # 返回绘图用的数据Dictionary

if __name__ == "__main__":
    with open(r"D:\DolphinDB\Project\FactorBrick\src\config\returnmodel_config_light.json5", mode="r",
              encoding="UTF-8") as file:
        cfg = json5.load(file)
    session = ddb.session()
    session.connect("172.16.0.184", 8001, "maxim", "dyJmoc-tiznem-1figgu")
    pool = ddb.DBConnectionPool("172.16.0.184", 8001, 10, "maxim", "dyJmoc-tiznem-1figgu")
    P = ReturnModel_Plot(session=session, pool=pool, config=cfg)
    P.Summary_plot()
    # P.FactorR_plot()  # 绘制单因子模型回测结果

