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
        file_dict[file] = address + '/' + file
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
    temp = data_preprocessing.read_data(path)
    # 原始文件预处理
    f = h5py.File(temp, 'r')
    a1 = f['BUS_DATA'][()]
    bus_num = f['BUS_NAMES'].shape[0]
    a2 = f['BRANCH_DATA'][()]
    matrix = Admittance_Matrix_Class.Admittancematrix(bus_num)
    matrix.generate_matrix(a2[:, 0], a2[:, 1], a2[:, 6], a2[:, 7], a1[:, 13], a1[:, 14])
    # 支路追加建立矩阵
    matrix_storage_path = './data/knot_admittance_matrix_data/' + temp[21:len(temp[1]) - 3] + 'pkl'
    f.close()
    with open(matrix_storage_path, 'wb') as f:
        pickle.dump(matrix, f)
    # 将节点导纳对象写入pkl文件，注意读取的时候要有类定义，否则无法解释
    data = list(map(lambda x: str(x), matrix.get_matrix().tolist()))
    return {
        matrix_storage_path: data,
    }

# 读取存储的矩阵路由
@app.route('/get', methods=["POST"])
def get():
    key = request.values.get("key")
    with open(key, 'rb') as f:
        matrix = pickle.load(f)
    data = list(map(lambda x: str(x), matrix.get_matrix().tolist()))
    return {
        key: data,
    }


@app.route('/reviseMatrix', methods=["POST"])
def revise():
    if request.method == 'POST':
        bus1 = request.form.get('bus1')
        bus2 = request.form.get('bus2')
        resistance = request.form.get('resistance')
        reactance = request.form.get('reactance')
        matrix_address = request.form.get('select2') # 返回矩阵地址
        addOrminus = request.form.get('select3')    # 返回0或1，0表示删除，1表示增加
    if (matrix_address == 'null') or (not all([bus1, bus2, resistance, reactance, matrix_address,addOrminus])):
        return "参数不完整修改失败"
    else:
        with open(matrix_address, 'rb') as f:
            matrix = pickle.load(f)

        if (addOrminus == '1'):
            matrix.add(float(bus1), float(bus2), float(resistance), float(reactance))

        else:
            matrix.minus(float(bus1), float(bus2), float(resistance), float(reactance))
        with open(matrix_address, 'wb') as f:
            pickle.dump(matrix, f)
        return "修改成功，如需查看，请重新选择矩阵（如修改的矩阵和当前预览的矩阵相同，请先选择null后再选择矩阵，因为不是实时更新的啦）"


if __name__ == "__main__":
    app.run(debug=True)
