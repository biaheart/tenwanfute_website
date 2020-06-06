from flask import Flask, render_template, request
import os, h5py, pickle
from lib_packages import *


app = Flask(__name__)


# 默认路由
@app.route('/')    # 路由默认使用GET方式进行路径访问，可以配置成methods=['GET', 'POST']等
def index():
    return render_template('index.html')


# 初始化路由
@app.route('/initialization')
def initialization():
    file_dict = {}
    address = './data/knot_admittance_matrix_data'
    for file in os.listdir(address):
        file_dict[file] =file[0:len(file)-4]
    return file_dict


# 数据预处理路由
@app.route('/dataPreprocess', methods=['POST'])  
def message():
    storage_folder = './data/original_data'
    file = request.files['file']
    if file.filename == '':
        return "no selected file"
    filename = file.filename
    path = os.path.join(storage_folder, filename)
    file.save(path)
    # 将原始文件存储下来
    temp = data_preprocessing.read_data(path)
    # 原始文件预处理
    f = h5py.File(temp, 'r')
    a1 = f['BUS_DATA'][()]
    bus_num = f['BUS_NAMES'].shape[0]
    a2 = f['BRANCH_DATA'][()]
    matrix = Admittance_Matrix_Class.Admittancematrix(bus_num)
    matrix.generate_matrix(a2[:, 0], a2[:, 1], a2[:, 6], a2[:, 7], a2[:, 8], a1[:, 13], a1[:, 14])
    # 支路追加建立矩阵
    matrix_storage_path = './data/knot_admittance_matrix_data/' + temp[21:len(temp[1]) - 4] + '.pkl'
    matrix_name = temp[21:len(temp[1]) - 4]
    f.close()
    with open(matrix_storage_path, 'wb') as f:
        pickle.dump(matrix, f)
    # 将节点导纳对象写入pkl文件，注意读取的时候要有类定义，否则无法解释
    data = list(map(lambda x: str(x), matrix.get_matrix().tolist()))
    return {
        matrix_name: data,
    }


# 读取存储的矩阵路由
@app.route('/get', methods=["POST"])
def get():
    key = request.values.get("key")
    path = './data/knot_admittance_matrix_data/' + key + '.pkl'
    with open(path, 'rb') as f:
        matrix = pickle.load(f)
    data = list(map(lambda x: str(x), matrix.get_matrix().tolist()))
    return {
        key: data,
    }


# 修改节点导纳矩阵的串联支路路由
@app.route('/reviseMatrixSeries', methods=["POST"])
def reviseSeries():
    if request.method == 'POST':
        bus1 = request.form.get('bus1')
        bus2 = request.form.get('bus2')
        resistance = request.form.get('resistance')
        reactance = request.form.get('reactance')
        susceptance = request.form.get('susceptance')
        matrix_name = request.form.get('select2') # 返回矩阵名称
        addOrminus = request.form.get('select3')    # 返回0或1，0表示删除，1表示增加
        matrix_address = './data/knot_admittance_matrix_data/' + matrix_name + '.pkl'
    if (matrix_name == 'null') or (not all([bus1, bus2, resistance, reactance, susceptance, matrix_name, addOrminus])):
        return "参数不完整修改失败"
    else:
        if int(bus1) > 0 and int(bus2) > 0:
            with open(matrix_address, 'rb') as f:
                matrix = pickle.load(f)

            if (addOrminus == '1'):
                matrix.add(int(bus1), int(bus2), float(resistance), float(reactance), float(susceptance))

            else:
                matrix.minus(int(bus1), int(bus2), float(resistance), float(reactance), float(susceptance))
            with open(matrix_address, 'wb') as f:
                pickle.dump(matrix, f)
            return "修改成功，如需查看，请重新选择矩阵（如修改的矩阵和当前预览的矩阵相同，请先选择null后再选择矩阵，因为不是实时更新的啦）"
        else:
            return "母线编号输入错误"


# 修改节点导纳矩阵的并联支路路由
@app.route('/reviseMatrixParallel', methods=["POST"])
def reviseParallel():
    if request.method == 'POST':
        bus = request.form.get('bus')
        conductance = request.form.get('conductance')
        susceptance = request.form.get('susceptance')
        matrix_name = request.form.get('select5')   # 返回矩阵名称
        addOrminus = request.form.get('select6')    # 返回0或1，0表示删除，1表示增加
        matrix_address = './data/knot_admittance_matrix_data/' + matrix_name + '.pkl'
    if (matrix_name == 'null') or (not all([bus, conductance, susceptance, matrix_name, addOrminus])):
        return "参数不完整修改失败"
    else:
        if int(bus) > 0:
            with open(matrix_address, 'rb') as f:
                matrix = pickle.load(f)

            if (addOrminus == '1'):
                matrix.set_self_ad(int(bus), float(conductance), float(susceptance))

            else:
                matrix.set_self_minus(int(bus), float(conductance), float(susceptance))
            with open(matrix_address, 'wb') as f:
                pickle.dump(matrix, f)
            return "修改成功，如需查看，请重新选择矩阵（如修改的矩阵和当前预览的矩阵相同，请先选择null后再选择矩阵，因为不是实时更新的啦）"
        else:
            return "母线编号输入错误"


# 执行潮流计算的路由
@app.route('/loadFlowCalculation', methods=["POST"])
def loadFlowCalculation():
    key = request.values.get("key")
    ori_data_path = './data/original_data/' + key + '.h5'
    knot_matrix_data_path = './data/knot_admittance_matrix_data/' + key + '.pkl'
    # 读取h5和pkl文件的数据来调用潮流程序，然后将计算结果存为csv文件放入./data/load_flow_calculation文件夹中，返回计算成功的结果
    f = h5py.File(ori_data_path, 'r')  # 正确读取对应的h5文件
    a1 = f['BUS_DATA'][()]  # 读取主键为‘BUS_NAMES’的数据
    bus_num = f['BUS_NAMES'].shape[0]
    MVA_BASE = f['MVA_BASE'][()]
    branch_data = f['BRANCH_DATA'][()]
    # 以下几行代码是为了获得节点连接关系的数据
    branch_connect = branch_data[:, 0:2].tolist()
    branch_connect = list(map(lambda x: list(map(int, x)), branch_connect)) # 将数据从numpy矩阵转成list且将浮点型转成整型
    f.close()
    with open(knot_matrix_data_path, 'rb') as file:
        matrix = pickle.load(file)
        admatrix = matrix.get_matrix()
    (U_actual_value, angle_actual_value, S_actual_value) = Load_Flow_Calculation.load_flow_calculation(admatrix, a1, bus_num, MVA_BASE)
    # 下面的语句是将结果转成json格式进行传输，因为json没有复数类型，所以S_actual_value转成了字符串
    result = {'U_actual_value': U_actual_value.tolist(), 'angle_actual_value': angle_actual_value.tolist(),
              'S_actual_value': list(map(str, S_actual_value)), 'branch_connect': branch_connect, 'bus_num': bus_num}
    return result


if __name__ == "__main__":
    app.run(debug=True)
