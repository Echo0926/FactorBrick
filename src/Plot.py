import json5
import pandas as pd
import dolphindb as ddb
import numpy as np
import streamlit as st
from ReturnModel_mr import ReturnModel_Backtest_mr

class ReturnModel_Plot(ReturnModel_Backtest_mr):
    def __init__(self, session, pool, config, asset_return_Algo="nullFill((open-prev(open))/prev(open),0)",
                 bench_return_Algo="nullFill((benchmark_open-prev(benchmark_open)))"):
        # 基本信息
        super().__init__(session, pool, config)
        self.Asset_return_Algo=asset_return_Algo    # 计算Daily资产收益率
        self.Bench_return_Algo=bench_return_Algo    # 计算Daily基准收益率

    def Summary_plot(self: ReturnModel_Backtest_mr):
        """
        所有因子横向比较可视化
        including: avg(IC), avg(RankIC), ICIR
        """
        benchmark=st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        r_interval=st.selectbox(
            label="请输入未来收益率区间长度",
            options=(i for i in self.returnIntervals),
            index=0,
            format_func=str,
            help='即ReturnModel中的returnIntervals'
        )
        st.title("_Total Factor Performance Comparison_")
        if not self.SingleFactor_estimation and self.MultiFactor_estimation:
            summary_table = self.Multisummary_table
        else:
            summary_table = self.summary_table
        Dict=self.session.run(f"""
        pt = select ReturnInterval,period,class,indicator,value from loadTable("{self.result_database}","{summary_table}") 
            where Benchmark=="{benchmark}" and 
            ReturnInterval == int({r_interval}) and 
            class in ["IC","RankIC"]
        template_pt = select start_date as date,period from loadTable("{self.combine_database}","{self.template_table}")
        // 添加时间
        pt = lj(template_pt,pt,`period);
        update pt set yearInt = year(date);
        update pt set yearStr = "Year"+string(yearInt)
        year_list = sort(exec distinct(yearInt) from pt)
        undef(`template_pt);
        
        // avg(IC)
        TotalIC_pt = select avg(value) as Total from pt where class == "IC" group by indicator as factor
        sortBy!(TotalIC_pt,`factor)
        YearIC_pt = select avg(value) as value from pt where class == "IC" pivot by indicator as factor, yearStr
        YearIC_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearIC_pt).eval()
        TotalIC_pt = lj(TotalIC_pt, YearIC_pt, `factor)
        
        // avg(RankIC)
        TotalRankIC_pt = select avg(value) as Total from pt where class == "RankIC" group by indicator as factor 
        sortBy!(TotalRankIC_pt,`factor)
        YearRankIC_pt = select avg(value) as value from pt where class == "RankIC" pivot by indicator as factor, yearStr
        YearRankIC_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearRankIC_pt).eval()
        TotalRankIC_pt = lj(TotalRankIC_pt, YearRankIC_pt, `factor)
        
        // avg(IC)\std(IC)
        TotalICIR_pt = select avg(value)\std(value) as Total from pt where class == "IC" group by indicator as factor
        sortBy!(TotalICIR_pt,`factor)
        YearICIR_pt = select avg(value)\std(value) as value from pt where class == "IC" pivot by indicator as factor, yearStr
        YearICIR_pt = sql(select=[sqlCol(`factor)].append!(sqlCol("Year"+string(year_list))), from=YearICIR_pt).eval()
        TotalICIR_pt = lj(TotalICIR_pt, YearICIR_pt, `factor)
        
        // avg(RankIC)\std(RankIC)
        TotalRankICIR_pt = select avg(value)\std(value) as Total from pt where class == "RankIC" group by indicator as factor
        sortBy!(TotalRankICIR_pt,`factor)
        YearRankICIR_pt = select avg(value)\std(value) as value from pt where class == "RankIC" pivot by indicator as factor, yearStr
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

    def FactorR_plot(self: ReturnModel_Backtest_mr):
        """单因子收益率可视化
        including: R/IC/RankIC/Tstats/Reg_stats(R_square/Adj_square/Obs)
        """
        benchmark=st.selectbox(
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
        method=st.selectbox(
            label="请输入估计方法",
            options=("OLS","Ridge","Lasso","ElasticNet"),
            index=0,
            format_func=str,
            help="即因子回归时候的统计估计方法method"
        )
        st.title("_Single Factor BackTest Analysis_")
        tabR,tabIC,tabT,tabQuantile,tabReg=st.tabs(["因子收益率R","因子IC值","因子T值","因子分组收益率","回归统计结果"])
        Dict=self.session.run(rf"""
        pt=select * from loadTable("{self.result_database}","{self.summary_table}") where Benchmark="{benchmark}";
        template_pt=select start_date,end_date,period from loadTable("{self.combine_database}","{self.template_table}");
        pt=lj(template_pt,pt,`period)
        pt=select * from pt where start_date>=date({self.start_dot_date}) and end_date<=date({self.end_dot_date});
        pt=select * from pt where class ilike "%_{method}" or class in ["IC","RankIC"];        
        
        quantile_pt=select * from loadTable("{self.result_database}","{self.quantile_table}") where Benchmark="{benchmark}"
        undef(`template_pt);
        
        // 因子收益率&累计因子收益率
        R=select value from pt where class ="R_"+"{method}" pivot by start_date as date,indicator;
        R_cumsum=R.copy();
        L=R_cumsum["date"];
        dropColumns!(R_cumsum,`date);
        R_cumsum=cumsum(R_cumsum);
        R_cumsum=select L as date,* from R_cumsum;
        
        // Reg_stat
        Obs=select value from pt where class ilike "Obs_%" pivot by start_date as date,indicator;
        Std_Error=select value from pt where class ilike "Std_Error_%" pivot by start_date as date,indicator;  // 残差标准差
        R_square=select value from pt where class ilike "R_square_%" pivot by start_date as date,indicator;
        Adj_square=select value from pt where class ilike "Adj_square_%" pivot by start_date as date,indicator;
        
        // Tstat
        t_stat=select value from pt where class ilike "tstat_%" pivot by start_date as date,indicator;
        
        // IC & 累计IC
        IC=select value from pt where class="IC" pivot by start_date as date,indicator;
        IC_cumsum=IC.copy();
        L=IC_cumsum["date"];
        dropColumns!(IC_cumsum,`date);
        IC_cumsum=cumsum(IC_cumsum);
        IC_cumsum=select L as date,* from IC_cumsum;
        
        // RankIC & 累计RankIC
        RankIC=select value from pt where class="RankIC" pivot by start_date as date,indicator;
        RankIC_cumsum=RankIC.copy();
        L=RankIC_cumsum["date"];
        dropColumns!(RankIC_cumsum,`date);
        RankIC_cumsum=cumsum(RankIC_cumsum);
        RankIC_cumsum=select L as date,* from RankIC_cumsum;
        
        // Yearly avg(IC)&IR
        data=unpivot(IC,keyColNames="date",valueColNames=columnNames(IC)[1:])
        rename!(data,`date`factor`factor_IC);
        avg_IC=select avg(factor_IC) from data pivot by year(date) as year,factor;
        IR=select avg(factor_IC)/std(factor_IC) from data pivot by year(date) as year,factor;
        
        // Yearly avg(RankIC)&RankIR
        data=unpivot(RankIC,keyColNames="date",valueColNames=columnNames(RankIC)[1:])
        rename!(data,`date`factor`factor_RankIC);
        avg_RankIC=select avg(factor_RankIC) from data pivot by year(date) as year,factor;
        RankIR=select avg(factor_RankIC)/std(factor_RankIC) from data pivot by year(date) as year,factor;
        
        undef(`pt); // 清除缓存
        
        // 返回为字典格式
        Dict=dict(["R_square","Adj_square","Obs","Std_Error","R","R_cumsum","t_stat","Quantile_Return","Quantile_Return_cumprod","IC","IC_cumsum","RankIC","RankIC_cumsum","avg_IC","IR","avg_RankIC","RankIR"],
        [R_square,Adj_square,Obs,Std_Error,R,R_cumsum,t_stat,Quantile_Return,Quantile_Return_cumprod,IC,IC_cumsum,RankIC,RankIC_cumsum,avg_IC,IR,avg_RankIC,RankIR]);
        Dict
        """)
        R_square=Dict["R_square"]
        Adj_square=Dict["Adj_square"]
        Obs=Dict["Obs"]
        Std_Error=Dict["Std_Error"]
        R=Dict["R"]
        R_cumsum=Dict["R_cumsum"]
        t_stat=Dict["t_stat"]
        Quantile_Return=Dict["Quantile_Return"]
        Quantile_Return_cumprod=Dict["Quantile_Return_cumprod"]
        IC=Dict["IC"]
        IC_cumsum=Dict["IC_cumsum"]
        RankIC=Dict["RankIC"]
        RankIC_cumsum=Dict["RankIC_cumsum"]
        avg_IC=Dict["avg_IC"]
        IR=Dict["IR"]
        avg_RankIC=Dict["avg_RankIC"]
        RankIR=Dict["RankIR"]
        with tabR:
            st.subheader("Single Factor Return",divider=True)
            st.line_chart(data=R,x="date",y=None)
            st.subheader("Single Factor Return(cumsum)",divider=True)
            st.line_chart(data=R_cumsum,x="date",y=None)
        with tabIC:
            # st.subheader("Factor IC",divider=True)
            # st.bar_chart(data=IC,x="date",y=None,stack=False)
            # st.subheader("Factor RankIC",divider=True)
            # st.bar_chart(data=RankIC,x="date",y=None,stack=False)
            st.subheader("Factor IC(cumsum)",divider=True)
            st.line_chart(data=IC_cumsum,x="date",y=None)
            st.subheader("Factor RankIC(cumsum)",divider=True)
            st.line_chart(data=RankIC_cumsum,x="date",y=None)
            st.subheader("Factor avg(IC)",divider=True)
            st.bar_chart(data=avg_IC,x="year",y=None,stack=False)
            st.dataframe(data=avg_IC)
            st.write("Total avg(IC):")
            st.dataframe(data=avg_IC.set_index("year").mean())
            st.subheader("Factor IR",divider=True)
            st.bar_chart(data=IR,x="year",y=None,stack=False)
            st.dataframe(data=IR)
            st.subheader("Factor avg(RankIC)",divider=True)
            st.bar_chart(data=avg_RankIC,x="year",y=None,stack=False)
            st.dataframe(data=avg_RankIC)
            st.write("Total avg(RankIC):")
            st.dataframe(data=avg_RankIC.set_index("year").mean())
            st.subheader("Factor RankIR",divider=True)
            st.bar_chart(data=RankIR,x="year",y=None,stack=False)
            st.dataframe(data=RankIR)
        with tabT:
            st.subheader("Factor Tstat",divider=True)
            st.bar_chart(data=t_stat,x="date",y=None,stack=False)
            st.write("T值绝对值大于等于2的比例")
            t_stat=t_stat.set_index("date")
            t_stat=(t_stat.abs()>=2).mean()  # .mean()计算|T|≥2的比例
            st.dataframe(data=t_stat)
        with tabQuantile:
            st.subheader("Single Factor Quantile Return", divider=True)
            st.line_chart(data=Quantile_Return, x="date", y=None)
            st.subheader("Single Factor Quantile Return(cumprod)", divider=True)
            st.line_chart(data=Quantile_Return_cumprod, x="date", y=None)
        with tabReg:
            st.subheader("R square",divider=True)
            st.bar_chart(data=R_square,x="date",y=None,stack=False)
            st.subheader("Adj R suqare",divider=True)
            st.bar_chart(data=Adj_square,x="date",y=None,stack=False)
            st.subheader("Std Error(残差标准差)",divider=True)
            st.bar_chart(data=Std_Error,x="date",y=None,stack=False)
            st.subheader("Num of Obs",divider=True)
            st.line_chart(data=Obs,x="date",y=None)

        return Dict # 返回绘图用的数据Dictionary

    def MultiFactorR_plot(self: ReturnModel_Backtest_mr):
        """多因子收益率可视化
        including: R/Tstats/Reg_stats(R_square/Adj_square/Obs)
        """
        benchmark=st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        method=st.selectbox(
            label="请输入估计方法",
            options=("OLS","Ridge","Lasso","ElasticNet"),
            index=0,
            format_func=str,
            help="即因子回归时候的统计估计方法method"
        )
        st.title("_Multi Factor BackTest Analysis_")
        tabR,tabIC,tabT,tabReg=st.tabs(["因子收益率R","因子IC值","因子T值","回归统计结果"])

        Dict=self.session.run(rf"""
        pt=select * from loadTable("{self.result_database}","{self.Multisummary_table}") where Benchmark="{benchmark}";
        template_pt=select start_date,end_date,period from loadTable("{self.combine_database}","{self.template_table}");
        pt=select * from pt left join template_pt on template_pt.period=pt.period;
        undef(`template_pt);
        pt=select * from pt where start_date>=date({self.start_dot_date}) and end_date<=date({self.end_dot_date});
        pt=select * from pt where class ilike "%_{method}" or class in ["IC","RankIC"];        
        // 因子收益率&累计因子收益率
        R=select value from pt where class ="R_"+"{method}" pivot by start_date as date,indicator;
        R_cumsum=R.copy();
        L=R_cumsum["date"];
        dropColumns!(R_cumsum,`date);
        R_cumsum=cumsum(R_cumsum);
        R_cumsum=select L as date,* from R_cumsum;
        
        // Reg_stat
        Obs=select value from pt where class ilike "Obs_%" pivot by start_date as date,indicator;
        Std_Error=select value from pt where class ilike "Std_Error_%" pivot by start_date as date,indicator;  // 残差标准差
        R_square=select value from pt where class ilike "R_square_%" pivot by start_date as date,indicator;
        Adj_square=select value from pt where class ilike "Adj_square_%" pivot by start_date as date,indicator;
        
        // Tstat
        t_stat=select value from pt where class ilike "tstat_%" pivot by start_date as date,indicator;
        
        // IC & 累计IC
        IC=select value from pt where class="IC" pivot by start_date as date,indicator;
        IC_cumsum=IC.copy();
        L=IC_cumsum["date"];
        dropColumns!(IC_cumsum,`date);
        IC_cumsum=cumsum(IC_cumsum);
        IC_cumsum=select L as date,* from IC_cumsum;
        
        // RankIC & 累计RankIC
        RankIC=select value from pt where class="RankIC" pivot by start_date as date,indicator;
        RankIC_cumsum=RankIC.copy();
        L=RankIC_cumsum["date"];
        dropColumns!(RankIC_cumsum,`date);
        RankIC_cumsum=cumsum(RankIC_cumsum);
        RankIC_cumsum=select L as date,* from RankIC_cumsum;
        
        // Yearly avg(IC)&IR
        data=unpivot(IC,keyColNames="date",valueColNames=columnNames(IC)[1:])
        rename!(data,`date`factor`factor_IC);
        avg_IC=select avg(factor_IC) from data pivot by year(date) as year,factor;
        IR=select avg(factor_IC)/std(factor_IC) from data pivot by year(date) as year,factor;
        
        // Yearly avg(RankIC)&RankIR
        data=unpivot(RankIC,keyColNames="date",valueColNames=columnNames(RankIC)[1:])
        rename!(data,`date`factor`factor_RankIC);
        avg_RankIC=select avg(factor_RankIC) from data pivot by year(date) as year,factor;
        RankIR=select avg(factor_RankIC)/std(factor_RankIC) from data pivot by year(date) as year,factor;
        
        undef(`pt); // 清除缓存
        
        // 返回为字典格式
        Dict=dict(["R_square","Adj_square","Obs","Std_Error","R","R_cumsum","t_stat","IC","IC_cumsum","RankIC","RankIC_cumsum","avg_IC","IR","avg_RankIC","RankIR"],
        [R_square,Adj_square,Obs,Std_Error,R,R_cumsum,t_stat,IC,IC_cumsum,RankIC,RankIC_cumsum,avg_IC,IR,avg_RankIC,RankIR]);
        Dict
        """)
        R_square=Dict["R_square"]
        Adj_square=Dict["Adj_square"]
        Obs=Dict["Obs"]
        Std_Error=Dict["Std_Error"]
        R=Dict["R"]
        R_cumsum=Dict["R_cumsum"]
        t_stat=Dict["t_stat"]
        IC=Dict["IC"]
        IC_cumsum=Dict["IC_cumsum"]
        RankIC=Dict["RankIC"]
        RankIC_cumsum=Dict["RankIC_cumsum"]
        avg_IC=Dict["avg_IC"]
        IR=Dict["IR"]
        avg_RankIC=Dict["avg_RankIC"]
        RankIR=Dict["RankIR"]

        with tabR:
            st.subheader("Multi Factor Return",divider=True)
            st.line_chart(data=R,x="date",y=None)
            st.subheader("Multi Factor Return(cumsum)",divider=True)
            st.line_chart(data=R_cumsum,x="date",y=None)
        with tabIC:
            # st.subheader("Factor IC",divider=True)
            # st.bar_chart(data=IC,x="date",y=None,stack=False)
            # st.subheader("Factor RankIC",divider=True)
            # st.bar_chart(data=RankIC,x="date",y=None,stack=False)
            st.subheader("Factor IC(cumsum)",divider=True)
            st.line_chart(data=IC_cumsum,x="date",y=None)
            st.subheader("Factor RankIC(cumsum)",divider=True)
            st.line_chart(data=RankIC_cumsum,x="date",y=None)
            st.subheader("Factor avg(IC)",divider=True)
            st.bar_chart(data=avg_IC,x="year",y=None,stack=False)
            st.dataframe(data=avg_IC)
            st.write("Total avg(IC):")
            st.dataframe(data=avg_IC.set_index("year").mean())
            st.subheader("Factor IR",divider=True)
            st.bar_chart(data=IR,x="year",y=None,stack=False)
            st.dataframe(data=IR)
            st.subheader("Factor avg(RankIC)",divider=True)
            st.bar_chart(data=avg_RankIC,x="year",y=None,stack=False)
            st.dataframe(data=avg_RankIC)
            st.write("Total avg(RankIC):")
            st.dataframe(data=avg_RankIC.set_index("year").mean())
            st.subheader("Factor RankIR",divider=True)
            st.bar_chart(data=RankIR,x="year",y=None,stack=False)
            st.dataframe(data=RankIR)
        with tabT:
            st.subheader("Factor Tstat",divider=True)
            st.bar_chart(data=t_stat,x="date",y=None,stack=False)
            st.write("T值绝对值大于等于2的比例")
            t_stat=t_stat.set_index("date")
            t_stat=(t_stat.abs()>=2).mean()  # .mean()计算|T|≥2的比例
            st.dataframe(data=t_stat)
        with tabReg:
            st.subheader("R square",divider=True)
            st.bar_chart(data=R_square,x="date",y=None,stack=False)
            st.subheader("Adj R suqare",divider=True)
            st.bar_chart(data=Adj_square,x="date",y=None,stack=False)
            st.subheader("Std Error(残差标准差)",divider=True)
            st.bar_chart(data=Std_Error,x="date",y=None,stack=False)
            st.subheader("Num of Obs",divider=True)
            st.line_chart(data=Obs,x="date",y=None)

        return Dict # 返回绘图用的数据Dictionary

    def Return_Fitted_plot(self: ReturnModel_Backtest_mr):
        """MAE MSE可视化"""
        benchmark=st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        method=st.selectbox(
            label="请输入估计方法",
            options=("OLS","Ridge","Lasso","ElasticNet"),
            index=0,
            format_func=str,
            help="即因子回归时候的统计估计方法method"
        )
        st.title("_Fitting Effect Analysis_")
        if self.Model_list:
            tab_SingleF,tab_MultiF,tab_ModelF=st.tabs(["单因子预测效果","多因子预测效果","自定义模型预测效果"])
        else:
            tab_SingleF,tab_MultiF=st.tabs(["单因子预测效果","多因子预测效果"])

        # 单因子模型预测效果
        if self.SingleFactor_estimation:
            Single_Dict=self.session.run(rf"""
            pt=select * from loadTable("{self.result_database}","{self.individualR_table}") where Benchmark="{benchmark}" and method="{method}";
            template_pt=select start_date,end_date,period from loadTable("{self.combine_database}","{self.template_table}");
            pt=select * from pt left join template_pt on template_pt.period=pt.period;
            undef(`template_pt);
            pt=select * from pt where start_date>=date({self.start_dot_date}) and end_date<=date({self.end_dot_date});
            
            // 单因子模型IC值(corr(return_pred,real_return)))
            Single_IC=select {",".join([f"corr({factor}_return_pred, real_return) as {factor}_IC" for factor in self.factor_list])} from pt group by start_date as date;
            sortBy!(Single_IC,`date,true);
            
            // 单因子模型累计IC值
            Single_CumIC=select date,{",".join([f"cumsum({factor}_IC) as {factor}_cumIC" for factor in self.factor_list])} from Single_IC;
            sortBy!(Single_CumIC,`date,true); 
            
            // 单因子模型MAE
            Single_MAE=select {",".join([f"avg(abs({factor}_return_pred-real_return)) as {factor}_MAE" for factor in self.factor_list])} from pt group by start_date as date;
            sortBy!(Single_MAE,`date,true);
            
            // 单因子模型MSE
            Single_MSE=select {",".join([f"avg(square({factor}_return_pred-real_return)) as {factor}_MSE" for factor in self.factor_list])} from pt group by start_date as date;
            sortBy!(Single_MSE,`date,true);
        
            // 单因子模型MRE
            Single_MRE=select {",".join([f"avg(abs(({factor}_return_pred-real_return)/real_return)) as {factor}_MRE" for factor in self.factor_list])} from pt group by start_date as date;
            sortBy!(Single_MRE,`date,true);
            undef(`pt); //释放内存
            Dict=dict(["IC","CumIC","MAE","MSE","MRE"],[Single_IC,Single_CumIC,Single_MAE,Single_MSE,Single_MRE]);
            Dict
            """)
            Single_IC=Single_Dict["IC"]
            Single_CumIC=Single_Dict["CumIC"]
            Single_MAE=Single_Dict["MAE"]
            Single_MSE=Single_Dict["MSE"]
            Single_MRE=Single_Dict["MRE"]

        # 多因子模型预测效果
        if self.MultiFactor_estimation:
            Multi_Dict=self.session.run(rf"""
            pt=select * from loadTable("{self.result_database}","{self.MultiIndividualR_table}") where Benchmark="{benchmark}";
            template_pt=select start_date,end_date,period from loadTable("{self.combine_database}","{self.template_table}");
            pt=select * from pt left join template_pt on template_pt.period=pt.period;
            undef(`template_pt);
            pt=select * from pt where start_date>=date({self.start_dot_date}) and end_date<=date({self.end_dot_date});
            
            // 多因子模型IC值
            Multi_IC=select {",".join([f"corr(return_pred_{model},real_return) as {model}_IC" for model in [method]])} from pt group by start_date as date;
            sortBy!(Multi_IC,`date,true);
            
            // 多因子模型累计IC值
            Multi_CumIC=select date,{",".join([f"cumsum({model}_IC) as {model}_cumIC" for model in [method]])} from Multi_IC;
            sortBy!(Multi_CumIC,`date,true); 
            
            // 多因子模型MAE
            Multi_MAE=select {",".join([f"avg(abs(return_pred_{model}-real_return)) as {model}_MAE" for model in [method]])} from pt group by start_date as date;
            sortBy!(Multi_MAE,`date,true);
            
            // 多因子模型MSE
            Multi_MSE=select {",".join([f"avg(square(return_pred_{model}-real_return)) as {model}_MSE" for model in [method]])} from pt group by start_date as date;
            sortBy!(Multi_MSE,`date,true);
            
            // 多因子模型MRE
            Multi_MRE=select {",".join([f"avg(abs((return_pred_{model}-real_return)/real_return)) as {model}_MRE" for model in [method]])} from pt group by start_date as date;
            sortBy!(Multi_MRE,`date,true);
        
            undef(`pt); //释放内存
            Dict=dict(["IC","CumIC","MAE","MSE","MRE"],[Multi_IC,Multi_CumIC,Multi_MAE,Multi_MSE,Multi_MRE]);
            Dict
            """)
            Multi_IC=Multi_Dict["IC"]
            Multi_CumIC=Multi_Dict["CumIC"]
            Multi_MAE=Multi_Dict["MAE"]
            Multi_MSE=Multi_Dict["MSE"]
            Multi_MRE=Multi_Dict["MRE"]

        # 自定义模型收益率预测效果
        if self.Model_list: # 说明应用了自定义ML/DL模型
            Model_Dict=self.session.run(rf"""
            pt=select * from loadTable("{self.model_database}","{self.ModelIndividualR_table}") where Benchmark="{benchmark}";
            pt=select * from pt where date between date({self.start_dot_date}) and date({self.end_dot_date});
            // Sort
            sortBy!(pt,`period`date`minute,[1,1,1]);
            
            // FirstNot GroupBy(选每一天第一分钟的样本)
            // pt=select firstNot(real_return) as real_return, {",".join([f"firstNot({model}_return_pred) as {model}_return_pred" for model in self.Model_list])} from pt group by period,date,symbol;
            // dropColumns!(pt,`date);
                        
            template_pt=select symbol,date,period from loadTable("{self.combine_database}","{self.template_daily_table}");
            pt=select * from pt left join template_pt on template_pt.period=pt.period and template_pt.symbol=pt.symbol;
            undef(`template_pt);
            
            // 自定义模型的IC
            Model_IC=select {",".join([f"corr({model}_return_pred,real_return) as {model}_IC" for model in self.Model_list])} from pt group by date;
            sortBy!(Model_IC,`date,true);
            
            // 自定义模型的CumIC
            Model_CumIC=select date,{",".join([f"cumsum({model}_IC) as {model}_cumIC" for model in self.Model_list])} from Model_IC;
            sortBy!(Model_CumIC,`date,true);
            
            // 自定义模型的MAE
            Model_MAE=select {",".join([f"avg(abs({model}_return_pred-real_return)) as {model}_MAE" for model in self.Model_list])} from pt group by date;
            sortBy!(Model_MAE,`date,true);
                       
            // 自定义模型的MSE
            Model_MSE=select {",".join([f"avg(square({model}_return_pred-real_return)) as {model}_MSE" for model in self.Model_list])} from pt group by date;
            sortBy!(Model_MSE,`date,true);
            
            // 自定义模型的MRE
            Model_MRE=select {",".join([f"avg(abs(({model}_return_pred-real_return)/real_return)) as {model}_MRE" for model in self.Model_list])} from pt group by date;
            sortBy!(Model_MRE,`date,true);
            
            undef(`pt); // 清除缓存
            Dict=dict(["IC","CumIC","MAE","MSE","MRE"],[Model_IC,Model_CumIC,Model_MAE,Model_MSE,Model_MRE]);
            Dict
            """)

            Model_IC=Model_Dict["IC"]
            Model_CumIC=Model_Dict["CumIC"]
            Model_MAE=Model_Dict["MAE"]
            Model_MSE=Model_Dict["MSE"]
            Model_MRE=Model_Dict["MRE"]

        if self.SingleFactor_estimation:
            with tab_SingleF:
                st.title("单因子模型收益率预测效果")
                st.subheader("单因子模型 IC",divider=True)
                st.bar_chart(data=Single_IC,x="date",y=None,stack=False)
                st.subheader("单因子模型 Cumulative IC",divider=True)
                st.line_chart(data=Single_CumIC,x="date",y=None)
                st.subheader("单因子模型 MAE",divider=True)
                st.bar_chart(data=Single_MAE,x="date",y=None,stack=False)
                st.subheader("单因子模型 MSE",divider=True)
                st.bar_chart(data=Single_MSE,x="date",y=None,stack=False)
                st.subheader("单因子模型 MRE",divider=True)
                st.bar_chart(data=Single_MRE,x="date",y=None,stack=False)
        if self.MultiFactor_estimation:
            with tab_MultiF:
                st.title("多因子模型收益率预测效果")
                st.subheader("多因子模型 IC",divider=True)
                st.bar_chart(data=Multi_IC,x="date",y=None,stack=False)
                st.subheader("多因子模型 Cumulative IC",divider=True)
                st.line_chart(data=Multi_CumIC,x="date",y=None)
                st.subheader("多因子模型 MAE",divider=True)
                st.bar_chart(data=Multi_MAE,x="date",y=None,stack=False)
                st.subheader("多因子模型 MSE",divider=True)
                st.bar_chart(data=Multi_MSE,x="date",y=None,stack=False)
                st.subheader("多因子模型 MRE",divider=True)
                st.bar_chart(data=Multi_MRE,x="date",y=None,stack=False)
        if self.Model_list:
            with tab_ModelF:
                st.title("自定义模型收益率预测效果")
                st.subheader("自定义模型 IC", divider=True)
                st.bar_chart(data=Model_IC,x="date",y=None,stack=False)
                st.subheader("自定义模型 Cumulative IC", divider=True)
                st.line_chart(data=Model_CumIC,x="date",y=None)
                st.dataframe(data=Model_IC.drop(columns=["date"]).mean(axis=0))
                st.subheader("自定义模型 MAE",divider=True)
                st.bar_chart(data=Model_MAE,x="date",y=None,stack=False)
                st.subheader("自定义模型 MSE",divider=True)
                st.bar_chart(data=Model_MSE,x="date",y=None,stack=False)
                st.subheader("自定义模型 MRE",divider=True)
                st.bar_chart(data=Model_MRE,x="date",y=None,stack=False)

        res_dict = {}
        if self.SingleFactor_estimation:
            res_dict["Single_Dict"] = Single_Dict
        if self.MultiFactor_estimation:
            res_dict["Multi_Dict"] = Multi_Dict
        if self.Model_list:
            res_dict["Model_Dict"] = Model_Dict
        return res_dict

    def GroupReturn_Plot(self: ReturnModel_Backtest_mr):
        """分组收益率可视化"""
        benchmark=st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        # 创建选项卡
        tabs=st.tabs(self.Group_list)
        for tab,group in zip(tabs,self.Group_list):
            with tab:
                Dict=self.session.run(rf"""
                    // 资产选择分组收益率(等权投资)可视化
                    // 模板数据库
                    template_pt=select start_date,end_date,period from loadTable("{self.combine_database}","{self.template_table}");
                    
                    // 行情数据库
                    R_pt=select firstNot(open) as open,lastNot(close) as close,firstNot({benchmark}_open) as benchmark_open,lastNot({benchmark}_close) as benchmark_close,firstNot({self.label_pred}) as {self.label_pred} from loadTable("{self.combine_database}","{self.combine_table}") group by symbol,date;
                    sortBy!(R_pt,[`symbol,`date],[1,1]);
                    update R_pt set DailyR={self.Asset_return_Algo};
                    update R_pt set BenchmarkR={self.Bench_return_Algo};
                    
                    // 资产选择Asset_slice数据库
                    asset_pt=select symbol,period,{group} as group from loadTable("{self.result_database}","{self.asset_slice_table}") where Benchmark="{benchmark}";
                    asset_pt=select * from asset_pt left join template_pt on template_pt.period=asset_pt.period;
                    undef(`template_pt);
                    
                    // Combination
                    asset_pt0=select symbol,start_date as date,group from asset_pt;
                    R_pt=select * from R_pt left join asset_pt0 on asset_pt0.symbol=R_pt.symbol and asset_pt0.date=R_pt.date;
                    undef(`asset_pt0);
                    asset_pt1=select symbol,end_date as date,group as group1 from asset_pt;
                    R_pt=select * from R_pt left join asset_pt1 on asset_pt1.symbol=R_pt.symbol and asset_pt1.date=R_pt.date;
                    undef(`asset_pt1`asset_pt);
                    update R_pt set group=group1 where isNull(group);
                    update R_pt set group=group.ffill() context by symbol;
                    dropColumns!(R_pt,`group1)
            
                    // 统计所有group_list
                    total_group_list=sort(exec distinct(group) as group from R_pt,true);
                    total_select_list=["date"];  // 最终选择的组名 ("-1","-2",...)
                    result_group_list=[];       // 最终放到Dict中的组名称 ("Div1","Div2"...)
                    DailyR_group_name=["date"];
                    BenchmarkR_group_name=["date"];
                    DailyR_cumprod_name=["date"];
                    BenchmarkR_cumprod_name=["date"];
                    for (g in total_group_list){{
                        if (not isNull(g)){{      // 防止组名为空
                            append!(total_select_list,string(g)); // 选取的列名
                            g=strReplace(string(g),"-","Div");
                            append!(result_group_list,g); // 最终放到Dict中的名称
                            append!(DailyR_group_name,"Return_"+g);  // 分组每日收益率
                            append!(DailyR_cumprod_name,"CumReturn_"+g);  // 分组累计收益率
                            append!(BenchmarkR_group_name,"Bench_"+g); // 分组基准每日收益率
                            append!(BenchmarkR_cumprod_name,"CumBench_"+g); // 分组基准累计收益率
                        }};
                    }};
            
                    // 统计每日收益率
                    result_pt=select nullFill(avg(DailyR),0) as DailyR,nullFill(avg(BenchmarkR),0) as BenchmarkR from R_pt group by date,group;
                    undef(`R_pt);  // 内存释放
                    return_df=select DailyR from result_pt pivot by date,group;
                    return_df=sql(sqlCol(colName=total_select_list),from=return_df).eval(); // 重新排序
                    rename!(return_df,DailyR_group_name); // 重新命名
                    return_df1=select BenchmarkR from result_pt pivot by date,group;
                    return_df1=sql(sqlCol(colName=total_select_list),from=return_df1).eval(); // 重新排序
                    rename!(return_df1,BenchmarkR_group_name); // 重新命名
                    return_df=select * from return_df left join return_df1 on return_df.date=return_df1.date;
                    undef(`return_df1);
                    
                    // 统计累计收益率
                    update result_pt set DailyR_cumprod=cumprod(1+DailyR) context by group;
                    update result_pt set BenchmarkR_cumprod=cumprod(1+BenchmarkR) context by group;
                    update result_pt set DailyR_cumprod=DailyR_cumprod.ffill() context by group;
                    update result_pt set BenchmarkR_cumprod=BenchmarkR_cumprod.ffill() context by group;
                    
                    cumprod_df=select DailyR_cumprod from result_pt pivot by date,group;
                    cumprod_df=sql(sqlCol(colName=total_select_list),from=cumprod_df).eval(); // 重新排序
                    rename!(cumprod_df,DailyR_cumprod_name); // 重新命名
                    cumprod_df1=select BenchmarkR_cumprod from result_pt pivot by date,group;
                    cumprod_df1=sql(sqlCol(colName=total_select_list),from=cumprod_df1).eval(); // 重新排序
                    rename!(cumprod_df1,BenchmarkR_cumprod_name); // 重新命名
                    cumprod_df=select * from cumprod_df left join cumprod_df1 on cumprod_df.date=cumprod_df1.date;
                    cumprod_df=cumprod_df.ffill();
                    undef(`cumprod_df1);
                    
                    //【新增】统计分组年化超额收益率
                    data=return_df.copy();
                    YearRet_list=[0.0];
                    MaxDrawdown_list=[0.0];
                    Sharpe_list=[0.0];
                    Calmar_list=[0.0]
                    for (g in result_group_list){{
                        // 年化超额收益率
                        data["excess_return"]=data["Return_"+g]-data["Bench_"+g];
                        YearRet=pow((prod(1+data["excess_return"])),(252.0/double(count(data[`date]))))-1;
                        append!(YearRet_list,YearRet); 
                        
                        // 最大回撤率
                        MaxDrawdown=maxDrawdown(cumprod(1+data["Return_"+g]),true);
                        append!(MaxDrawdown_list,MaxDrawdown);
                        
                        // 风险-收益指标
                        // 计算Sharpe比率(无风险利率假设为0)
                        Sharpe_ratio=(YearRet-0)/(std(data["Return_"+g])*sqrt(252))
                        append!(Sharpe_list,Sharpe_ratio);
                        
                        // 计算Calmar比率
                        Calmar_ratio=(YearRet-0)/abs(MaxDrawdown)
                        append!(Calmar_list,Calmar_ratio);
                    }};
                    Groupstats_df=table(result_group_list as group,
                                        YearRet_list[1:] as YearRet,
                                        MaxDrawdown_list[1:] as MaxDrawdown,
                                        Sharpe_list[1:] as Sharpe,
                                        Calmar_list[1:] as Calmar);
                     
                    // 输出为Dict
                    // 【新增】Dict中增加了组名，把total_group_list作为streamlit下拉框的输入即可
                    Dict=dict(["total_group_list","return","return_cumprod","strategy_stats"],[result_group_list,return_df,cumprod_df,Groupstats_df]);
                    Dict
                """)

                # 回测结果
                total_group_list=Dict["total_group_list"]       # return_(每日组合收益率)/Cumreturn_(累计组合收益率)/Bench_(每日基准收益率)/CumBench_(累计基准收益率)
                Return_df=Dict["return"]    # 每日收益率DataFrame
                Cumprod_df=Dict["return_cumprod"]   # 累计收益率DataFrame
                GroupStats=Dict["strategy_stats"]   # 策略收益率统计指标

                select_groups=st.multiselect(
                    key=group,   # 使用group作为索引使得复选框唯一
                    label="请选择组名",
                    options=(i for i in total_group_list),
                    default=total_group_list[:10],
                )
                st.title("分组等权收益率回测结果")
                st.subheader("PortFolio DailyReturn",divider=True)
                st.bar_chart(data=Return_df,x="date",y=["Return_"+str(i) for i in select_groups]+["Bench_"+str(i) for i in select_groups],stack=False)
                st.subheader("PortFolio CumprodReturn",divider=True)
                st.line_chart(data=Cumprod_df,x="date",y=["CumReturn_"+str(i) for i in select_groups]+["CumBench_"+str(i) for i in select_groups])
                st.subheader("Strategy Statistics",divider=True)
                st.dataframe(data=GroupStats)
        return Dict

    def OptimizeReturnPlot(self: ReturnModel_Backtest_mr):
        """组合优化收益率可视化"""
        benchmark=st.selectbox(
            label='请输入Benchmark',
            options=(i for i in self.benchmark_list),
            index=0,
            format_func=str,
            help='即ReturnModel中的Benchmark_list'
        )
        strategy_list=st.multiselect(
            label="请输入Strategy",
            options=(i for i in self.optstrategy_list),
            default=self.optstrategy_list       # 默认输入全部策略
        )
        string="".join([f"""
        update pt set {i}=opt1_{i} where isNull({i});
        update pt set {i}=nullFill({i}.ffill(),0) context by 辅助列;
        dropColumns!(pt,"opt1_"+"{i}");
        """ for i in strategy_list])

        Dict=self.session.run(rf"""
        // 
        optstrategy_list={strategy_list};
        
        // 模板数据库
        template_pt=select symbol,start_date,end_date,period from loadTable("{self.combine_database}","{self.template_individual_table}"); // 这里更严谨,假定每个symbol不一定共享整个period的区间
        
        // 行情数据库
        pt=select firstNot(open) as open,lastNot(close) as close,firstNot({benchmark}_open) as benchmark_open,lastNot({benchmark}_close) as benchmark_close,firstNot(marketvalue) as marketvalue,firstNot(industry) as industry,firstNot({self.label_pred}) as {self.label_pred} from loadTable("{self.combine_database}","{self.combine_table}") group by symbol,date,period;
        sortBy!(pt,[`symbol,`date],[1,1]);
        update pt set symbol_date =string(symbol)+string(date);
        update pt set DailyR={self.Asset_return_Algo};
        update pt set BenchmarkR={self.Bench_return_Algo};
        
        // 基础策略数据库
        update pt set EQweight=1.0/count(symbol) context by date;                   // 等权收益率
        update pt set MVweight=marketvalue/sum(marketvalue) context by date;       // 市值加权收益率
        update pt set industry=nullFill(industry.ffill(),"NA") context by symbol;
        update pt set 辅助列1=string(industry)+string(date);
        update pt set 行业市值=sum(marketvalue) context by 辅助列1;
        update pt set 行业权重=行业市值/sum(marketvalue) context by date;
        update pt set IDweight=行业权重/count(symbol) context by 辅助列1;     // 行业市值加权收益率
        dropColumns!(pt,`辅助列1`行业市值`行业权重`marketvalue`industry);
        
        // 组合优化结果数据库
        opt=select symbol,period,{','.join(strategy_list)} from loadTable("{self.optimize_database}","{self.optimize_result_table}") where Benchmark="{benchmark}";
        opt=select * from opt left join template_pt on template_pt.period=opt.period and opt.symbol=template_pt.symbol;
        undef(`template_pt);
                    
        // combination
        opt0=select symbol,start_date as date,{','.join(strategy_list)} from opt;
        pt=select * from pt left join opt0 on opt0.symbol=pt.symbol and opt0.date=pt.date;
        undef(`opt0);
        opt1=select symbol,end_date as date,{','.join(strategy_list)} from opt;
        pt=select * from pt left join opt1 on opt1.symbol=pt.symbol and opt1.date=pt.date;
        undef(`opt1`opt);
        
        // ffill weight
        update pt set 辅助列=string(period)+string(symbol);
        update pt set EQweight=nullFill(EQweight.ffill(),0) context by 辅助列;
        update pt set MVweight=nullFill(MVweight.ffill(),0) context by 辅助列;
        update pt set IDweight=nullFill(IDweight.ffill(),0) context by 辅助列;
        {string}
        
        // 收益率回测
        daily_pt=select firstNot(date) as date,DailyR**EQweight as Return_EQ,DailyR**MVweight as Return_MV,DailyR**IDweight as Return_ID,{','.join([f"DailyR**{i} as Return_{i}" for i in strategy_list])} from pt group by period;
        bench_pt=select firstNot(date) as date,BenchmarkR**EQweight as Bench_EQ,BenchmarkR**MVweight as Bench_MV,DailyR**IDweight as Bench_ID,{','.join([f"BenchmarkR**{i} as Bench_{i}" for i in strategy_list])} from pt group by period;
        cumprod_pt=select date,period,cumprod(1+Return_EQ) as CumReturn_EQ,cumprod(1+Return_MV) as CumReturn_MV,cumprod(1+Return_ID) as CumReturn_ID,{','.join([f"cumprod(1+Return_{i}) as CumReturn_{i}" for i in strategy_list])} from daily_pt;
        cumprod_bench_pt=select date,period,cumprod(1+Bench_EQ) as CumBench_EQ,cumprod(1+Bench_MV) as CumBench_MV,cumprod(1+Bench_ID) as CumBench_ID,{','.join([f"cumprod(1+Bench_{i}) as CumBench_{i}" for i in strategy_list])} from bench_pt;
        
        // 返回结果Dict
        dict(["daily_return","cumprod_return","bench_return","benchcumprod_return"],[daily_pt,cumprod_pt,bench_pt,cumprod_bench_pt])
        """)
        daily_return,cumprod_return,bench_return,benchcumprod_return=Dict["daily_return"],Dict["cumprod_return"],Dict["bench_return"],Dict["benchcumprod_return"]
        Return_df=pd.concat([daily_return.set_index(["date","period"]),
                             bench_return.set_index(["date","period"])],axis=1).reset_index()
        Cumprod_df=pd.concat([cumprod_return.set_index(["date","period"]),
                             benchcumprod_return.set_index(["date","period"])],axis=1).reset_index()
        strategy_list=["EQ","MV","ID"]+strategy_list    # 默认添加等权、市值加权、行业市值加权三种投资组合回测结果

        st.title("组合优化收益率回测结果")
        st.subheader("PortFolio DailyReturn (Optimization)",divider=True)
        st.bar_chart(data=Return_df,x="date",y=["Return_"+str(i) for i in strategy_list]+["Bench_"+str(i) for i in strategy_list],stack=False)
        st.subheader("PortFolio CumprodReturn (Optimization)",divider=True)
        st.line_chart(data=Cumprod_df,x="date",y=["CumReturn_"+str(i) for i in strategy_list]+["CumBench_"+str(i) for i in strategy_list])
        return Dict


if __name__=="__main__":
    with open(r"D:\DolphinDB\Project\FactorBrick\src\config\returnmodel_config0730.json5", mode="r", encoding="UTF-8") as file:
        cfg = json5.load(file)
    session=ddb.session()
    session.connect("172.16.0.184",8001,"maxim","dyJmoc-tiznem-1figgu")
    pool=ddb.DBConnectionPool("172.16.0.184",8001,10,"maxim","dyJmoc-tiznem-1figgu")
    P=ReturnModel_Plot(session=session,pool=pool,config=cfg,
    asset_return_Algo="nullFill((close-prev(close))\close, 0.0) context by symbol",
    bench_return_Algo="nullFill((benchmark_close-prev(benchmark_close))\prev(benchmark_close),0.0) context by symbol"
    )
    P.Summary_plot()
    # P.FactorR_plot()                # 绘制单因子模型回测结果
    # P.MultiFactorR_plot()         # 绘制多因子模型回测结果
    # P.Return_Fitted_plot()        # 绘制自定义ML&DL模型预测结果
    # P.GroupReturn_Plot()          # 绘制分组等权收益率回测结果
    # P.OptimizeReturnPlot()        # 绘制投资组合优化回测结果
