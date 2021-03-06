def fireengine(target, MVA_BASE, U0, admatrix, Umin, Umax, flow_PG, Pmin, Pmax, PLD, limit_min_Q, limit_max_Q, QLD):
    '''
    target为耗量特性矩阵
    MVA_BASE为功率基准值
    U0为潮流各个节点电压额定值的标幺值（1），一维列表/矩阵
    admatrix为节点导纳矩阵
    Umin，Umax，Pmin，Pmax为手动输入的电压，float类型
    flow_PG为发电机节点的注入有功，一维列表/矩阵
    PLD为从各个节点所带负荷有功，一维列表/矩阵
    limit_min_Q为各个发电机节点无功最小值限制，一维列表/矩阵
    limit_max_Q为各个发电机节点无功最大值限制，一维列表/矩阵
    QLD为各个节点负荷无功，一维列表/矩阵
    '''
    from scipy.optimize import minimize
    import numpy as np
    from lib_packages.calculate_pq import calculate_p, calculate_q

    k = target.shape[0]
    fun = lambda x: sum(target[:, 1] + target[:, 2] * x[0: k] * MVA_BASE + target[:, 3] * x[0: k] * x[0: k] * MVA_BASE * MVA_BASE)  # 构建待规划函数F
    cons = []   # 建立约束条件列表
    limit_min_U = list(np.array(U0) * Umin)     # 构建电压和发电机有功输出上下限的列表
    limit_max_U = list(np.array(U0) * Umax)
    limit_min_P = list(np.array(flow_PG) * Pmin)
    limit_max_P = list(np.array(flow_PG) * Pmax)
    # 匿名函数列表x中，前k个为发电机有功，之后k个为发电机无功，之后admtrix.shape[0]个为全部节点电压有效值，最后admtrix.shape[0]个为全部节点角度
    for i in range(k):  # 发电机的有功约束
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - limit_min_P[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: -x[i] + limit_max_P[i]})
    for i in range(k):  # 发电机无功约束
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[k + i] - limit_min_Q[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: -x[k + i] + limit_max_Q[i]})
    for i in range(admatrix.shape[0]):  # x的2k+1到2k+n个为所有节点的电压约束
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * k + i] - limit_min_U[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: -x[2 * k + i] + limit_max_U[i]})
    for i in range(k):      # 发电机节点的等式约束
        cons.append({'type': 'eq', 'fun': lambda x, i=i: x[i] - PLD[int(target[i, 0]) - 1] - calculate_p(admatrix.shape[0], x[2 * k: 2 * k + admatrix.shape[0]], x[2 * k + admatrix.shape[0]: 2 * k + 2 * admatrix.shape[0]], admatrix)[int(target[i, 0] - 1)]})
        cons.append({'type': 'eq', 'fun': lambda x, i=i: x[k + i] - QLD[int(target[i, 0]) - 1] + calculate_q(admatrix.shape[0], x[2 * k: 2 * k + admatrix.shape[0]], x[2 * k + admatrix.shape[0]: 2 * k + 2 * admatrix.shape[0]], admatrix)[int(target[i, 0] - 1)]})
    n = []
    for i in range(admatrix.shape[0]):  # 构建一个与节点总数相等的计数矩阵
        n.append(i + 1)
    for i in range(k):  # 将发电机节点的编号移除
        n.remove(target[i, 0])
    for i in n[:]:      # 非发电机节点的等式约束
        cons.append({'type': 'eq', 'fun': lambda x, i=i: PLD[i - 1] + calculate_p(admatrix.shape[0], x[2 * k: 2 * k + admatrix.shape[0]], x[2 * k + admatrix.shape[0]: 2 * k + 2 * admatrix.shape[0]], admatrix)[i - 1]})
        cons.append({'type': 'eq', 'fun': lambda x, i=i: QLD[i - 1] - calculate_q(admatrix.shape[0], x[2 * k: 2 * k + admatrix.shape[0]], x[2 * k + admatrix.shape[0]: 2 * k + 2 * admatrix.shape[0]], admatrix)[i - 1]})
    x0 = np.ones(2 * k + 2 * admatrix.shape[0])       # 设置初始值
    res = minimize(fun, x0, method='SLSQP', constraints=cons)   # 求解规划
    min_F = res.fun     # fun的最小值
    divide_PG = np.zeros((k, 3))
    for i in range(k):      # 发电机节点编号以及对应有功无功
        divide_PG[i, 0] = target[i, 0]
        divide_PG[i, 1] = res.x[i] * MVA_BASE
        divide_PG[i, 2] = res.x[k + i] * MVA_BASE
    data_pu_U = res.x[2 * k: 2 * k + admatrix.shape[0]]     # 各个节点电压标幺值
    data_deta = res.x[2 * k + admatrix.shape[0]: 2 * k + 2 * admatrix.shape[0]]     # 各个节点相角
    return res.success, min_F, divide_PG, data_pu_U, data_deta
    # 返回最优化求解成功符号“True”；总耗量最小值；各发电机节点编号以及对应有功无功；各个节点电压标幺值；各个节点相角；
