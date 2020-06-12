def calculate_p(bus_num, u_per_unit_value, angle_actual_value, admatrix):
    '''
    bus_num为总的母线个数
    u_per_unit_value为潮流计算输出的电压标幺值列表
    angle_actual_value为潮流计算输出的相角角度制列表
    admatrix为节点导纳矩阵
    '''
    import numpy as np
    import math

    RE = np.zeros((bus_num, bus_num))  # 实部矩阵
    IM = np.zeros((bus_num, bus_num))  # 虚部矩阵
    i = 0
    j = 0
    while i < bus_num:  # 获取实部虚部矩阵
        while j < bus_num:
            RE[i, j] = admatrix[i, j].real
            IM[i, j] = admatrix[i, j].imag
            j = j + 1
        i = i + 1
        j = 0
    angle_value = list(map(lambda x: x * math.pi / 180, angle_actual_value))  # 相角弧度制矩阵
    i = 0
    j = 0
    p_value = list(np.zeros(bus_num))
    while i < bus_num:      # 由潮流计算结果获得注入有功标幺值列表
        pi = 0
        while j < bus_num:
            angle_ij = angle_value[i] - angle_value[j]
            pi += u_per_unit_value[i] * u_per_unit_value[j] * (
                    RE[i, j] * math.cos(angle_ij) + IM[i, j] * math.sin(angle_ij))
            j = j + 1
        j = 0
        p_value[i] = pi
        i += 1
    return p_value


def calculate_q(bus_num, u_per_unit_value, angle_actual_value, admatrix):
    '''
    bus_num为总的母线个数
    u_per_unit_value为潮流计算输出的电压标幺值列表
    angle_actual_value为潮流计算输出的相角角度制列表
    admatrix为节点导纳矩阵
    '''
    import numpy as np
    import math

    RE = np.zeros((bus_num, bus_num))  # 实部矩阵
    IM = np.zeros((bus_num, bus_num))  # 虚部矩阵
    i = 0
    j = 0
    while i < bus_num:  # 获取实部虚部矩阵
        while j < bus_num:
            RE[i, j] = admatrix[i, j].real
            IM[i, j] = admatrix[i, j].imag
            j = j + 1
        i = i + 1
        j = 0
    angle_value = list(map(lambda x: x * math.pi / 180, angle_actual_value))  # 相角弧度制矩阵
    i = 0
    j = 0
    q_value = list(np.zeros(bus_num))
    while i < bus_num:     # 由潮流计算结果获得注入无功标幺值列表
        qi = 0
        while j < bus_num:
            angle_ij = angle_value[i] - angle_value[j]
            qi += u_per_unit_value[i] * u_per_unit_value[j] * (
                    RE[i, j] * math.sin(angle_ij) - IM[i, j] * math.cos(angle_ij))
            j = j + 1
        j = 0
        q_value[i] = qi
        i += 1
    return q_value



