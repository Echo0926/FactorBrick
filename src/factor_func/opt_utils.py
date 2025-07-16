# import riskfolio as rp
import pandas as pd
import numpy as np

def opt_weights(return_matrix,  # 历史收益率矩阵（必需）
                cov_matrix=pd.DataFrame(),expect_returns=pd.DataFrame(),    # 收益率协方差矩阵、预期收益率矩阵
                constraints=pd.DataFrame(),asset_classes=pd.DataFrame(),    # 矩阵约束条件（详情见riskfolio主页）
                objective: str = "Utility",
                HCP: bool = False,   # 是否是分层聚类组合优化
                sht: bool = False,   # 是否允许做空
                card: int = None,   # 最大资产个数
                short_max: float = 0.5, # 空头端最大仓位(sht=True时)
                long_max: float = 0.5,  # 多头端最大仓位(sht=True时)
                w_min: bool = None,  # 最小权重
                w_max: bool = None,  # 最大权重
                lowerret: bool = None,
                budget: float = 1.0,
                rf: float = 0.0,
                lamda: float = 0.5
                ):
    """RiskFolio-lib core Function
    return_matrox: 历史收益率
    cov_matrix: 收益率协方差矩阵
    expect_returns: 预测收益率
    asset_classes&constraints: see https://www.wuzao.com/document/riskfolio-lib/constraints.html#ConstraintsFunctions.hrp_constraints
    HCP:True表示为分层聚类投资组合
    objective:HCP=True时{'MinRisk', 'Utility', 'Sharpe' or 'ERC'}; HCP=False时{'MinRisk', 'Utility', 'Sharpe' or 'MaxRet'}
    sht:True代表可以卖空
    uppersht:个股权重上限
    lowerret:目标最低收益率
    budget:权重之和
    rf: Risk free rate
    lamda: Risk aversion factor, only useful when obj is 'Utility'
    """
    # Step1. 建立投资组合对象(普通投资组合 or 分层聚类投资组合)
    if not HCP:
        port=rp.Portfolio(returns=return_matrix,sht=sht,card=card,lowerret=lowerret,
                          budget=budget,uppersht=short_max,upperlng=long_max)     # 普通投资组合
    else:
        port=rp.HCPortfolio(returns=return_matrix,w_max=w_max,w_min=w_min) # 分层聚类投资组合
    if not cov_matrix.empty or not expect_returns.empty:
        if expect_returns.empty and not cov_matrix.empty:   # 只传了expect_Return
            port.mu=expect_returns
            if not HCP:
                port.assets_stats(method_cov='hist') # 历史协方差矩阵估计
        elif cov_matrix.empty and not expect_returns.empty: # 只传了cov
            port.cov=cov_matrix
            if not HCP:
                port.assets_stats(method_mu='hist') # 历史收益率估计未来收益率
        else:   # expect_Return&cov都有
            port.cov=cov_matrix
            port.mu=expect_returns
    else:
        if not HCP:
            port.assets_stats(method_mu='hist',method_cov='hist') # 计算历史收益率和协方差矩阵

    # Step2. 添加矩阵约束
    if not constraints.empty:
        if not HCP: # 说明是普通投资组合的约束
            A,B=rp.assets_constraints(constraints,asset_classes)
            port.ainequality=A
            port.binequality=B
        else:       # 说明是分层聚类组合的约束
            W_MAX,W_MIN=rp.hrp_constraints(constraints,asset_classes)
            port.w_max=W_MAX
            port.w_min=W_MIN
    else:   # 说明没有矩阵约束条件
        pass

    # Step3. 求解组合优化
    if not HCP:
        w=port.optimization(model='Classic',rm="MV",obj=objective,rf=rf,l=lamda,hist=True)  # 普通投资组合优化求解
    else:
        w=port.optimization(model="HRP",rm="MV",obj=objective,rf=rf,l=lamda,covariance="custom_cov",custom_mu=port.mu,custom_cov=port.cov)  # HRP,HERC or HERC2
    try:
        weights=np.ravel(w.to_numpy())  # 求解权重
    except:
        print("求解失败,返回0向量")
        weights=np.array([0]*len(port.assetslist))

    # Step4. 统计汇报结果
    try:
        shp=rp.Sharpe(w,port.mu,       # 夏普比率
                cov=port.cov,
                returns=return_matrix,
                rm="MV",
                rf=0,
                alpha=0.05,
                solver='MOSEK')
    except:
        print("求解失败,返回None值")
        shp=np.nan
    # 可以进一步添加别的指标...
    return {"weight":weights,"sharpe":shp}