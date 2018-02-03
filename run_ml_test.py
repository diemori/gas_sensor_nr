from ml_gas import *

gl = gas_learn(load_config=True)

gl.load_csv('gas.csv')

gl.train('co')
gl.train('c6h6')
gl.train('no2')
gl.train('nox')

# gl.load_models()

print("train done!!\n")

gl.predict_batch('gas2.txt', calc_error=True, csv_name='mixed_result.csv')
