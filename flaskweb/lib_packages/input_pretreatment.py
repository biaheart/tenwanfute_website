def input_pretreatment(target, bus_num, a1, MVA_BASE, admatrix):
    '''
    :param target: 读取耗量特性文件返回的矩阵
    :param bus_num:参数bus_num为母线个数
    :param a1: 读取主键为‘BUS_NAMES’的数据
    :param MVA_BASE: 参数MVA_BASE为基准功率
    :param admatrix: 参数admatirx为建立的节点导纳矩阵
    :return: U0为潮流各个节点电压额定值的标幺值（1），一维列表/矩阵；flow_PG为发电机节点的注入有功，一维列表/矩阵；PLD为从各个节点所带负荷有功，一维列表/矩阵；
    limit_min_Q为各个发电机节点无功最小值限制，一维列表/矩阵；
    limit_max_Q为各个发电机节点无功最大值限制，一维列表/矩阵；
    QLD为各个节点负荷无功，一维列表/矩阵；
    '''
    from lib_packages.Load_Flow_Calculation import load_flow_calculation
    import numpy as np

    U, deta1, flowS = load_flow_calculation(admatrix, a1, bus_num, MVA_BASE)
    flowP = flowS.real
    U0 = np.ones(bus_num)
    flow_PG = np.zeros(int(target.shape[0]))
    i = 0
    while i < target.shape[0]:  # 将潮流计算得到的各节点有功有名值矩阵转化为标幺值列表
        flow_PG[i] = flowP[0, int(target[i, 0]) - 1] / MVA_BASE
        i += 1
    PLD = np.zeros(bus_num)
    i = 0
    while i < bus_num:  # 将各节点负载功率转为标幺值列表
        PLD[i] = a1[i, 5] / MVA_BASE
        i += 1
    limit_min_Q = []
    for i in range(target.shape[0]):  # 得到各节点无功下限值
        limit_min_Q.append(a1[int(target[i, 0]) - 1, 12])
    limit_max_Q = []
    for i in range(target.shape[0]):  # 得到各节点无功上限值
        limit_max_Q.append(a1[int(target[i, 0]) - 1, 11])
    QLD = np.zeros(bus_num)  # 得到各节点负载无功标幺值列表
    i = 0
    while i < bus_num:
        QLD[i] = a1[i, 6] / MVA_BASE
        i += 1
    return U0, flow_PG, PLD, limit_min_Q, limit_max_Q, QLD
