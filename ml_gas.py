import os

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# 기본적으로 Linear Reg.와 앙상블의 하나인 RandomForestReg.를 제공한다
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing as preproc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json

from sklearn.externals import joblib
from pickle import dump
from pickle import load

import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Multi Layer Perceptron model
def mlp_model():
    model = Sequential()

    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

class gas_learn():
    def __init__(self, load_config=False):
        print("[__init__] This program requires UTF-8 encoding format.")
        self.co = {}
        self.c6h6 = {}
        self.nox = {}
        self.no2 = {}
        self.sensors = None

        if load_config == True:
            self.load_config()


    # config 문자열로 부터 해당 기체를 분석하기 위한 모델링 알고리즘을 파싱하여 저장한다
    def __get_model_type(self, dict_input, raw_input):
        dict_input['model_name'] = raw_input.replace('\n', '').split('=')[1]
        estimators = []
        estimators.append(('standardize', StandardScaler()))

        if dict_input['model_name'] == "LinearRegression":
            estimators.append(('lr', LinearRegression()))
        elif dict_input['model_name'] == "RandomForest":
            estimators.append(('rf', RandomForestRegressor()))
        elif dict_input['model_name'] == "DL_MLP":
            dict_input['model'] = mlp_model()
            return 0
            # estimators.append(('mlp', KerasRegressor(build_fn=mlp_model, nb_epoch=100, batch_size=10, verbose=0)))
        else:
            print("[__get_model_type][error] invalid option... %s" % raw_input)
            return -1

        dict_input['model'] = Pipeline(estimators)

        return 0

    # config.txt의 설정 정보를 로드하여 저장한다
    def load_config(self):
        self.co['model'] = None
        self.co['model_str'] = ""
        lines = None

        with open('config.txt', 'r', encoding='utf-8') as fc:
            lines = fc.readlines()

        for line in lines:
            if "co_model" in line:
                if self.__get_model_type(self.co, line) != 0:
                    exit()
            if "c6h6_model" in line:
                if self.__get_model_type(self.c6h6, line) != 0:
                    exit()
            if "nox_model" in line:
                if self.__get_model_type(self.nox, line) != 0:
                    exit()
            if "no2_model" in line:
                if self.__get_model_type(self.no2, line) != 0:
                    exit()

        print("[load_config] config file loaded.")

    # 로드한 데이터를 바탕으로 학습을 수행한다
    def train(self, target):
        if 'guide_output' not in self.co:
            print('[train] there is no guide data for ml')
            return -1

        X = self.sensors
        tar_data = None

        if target == 'co':
            tar_data = self.co
        elif target == 'c6h6':
            tar_data = self.c6h6
        elif target == 'no2':
            tar_data = self.no2
        elif target == 'nox':
            tar_data = self.nox
        else:
            print('[train] invalid target: %s' % target)
            return -2

        Y = tar_data['guide_output']

        if 'model' not in self.co:
            self.load_config()
            if 'model' not in self.co:
                print('[train_co] there is no model configuration..')
                return -2

        if tar_data['model_name'].startswith('DL_'):
            tar_data['model'].fit(X, Y, epochs=150, batch_size=10, verbose=0)

        else:
            tar_data['model'].fit(X, Y)
            results = tar_data['model'].score(X, Y)

            print("[train] %s : %0.04f" % (target, results))

        # check directory
        dir_name = 'saved_models'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        # save the model to disk
        if tar_data['model_name'].startswith('DL_'):
            self.save_model_json(target, tar_data['model'])
            # joblib.dump(tar_data['model'], './saved_models/%s_model.pkl' % target)
        else:
            filename = './%s/%s_model.sav' % (dir_name, target)
            dump(tar_data['model'], open(filename, 'wb'))

        return 0

    # DL 모델의 결과를 h5, json 형태로 저장한다
    # json : model 저장 파일
    # h5 : model에서 사용하는 weight 저장 파일 (HDF5 포맷)
    @staticmethod
    def save_model_json(file_name, model):
        # serialize model to JSON
        model_json = model.to_json()

        with open("./saved_models/%s_model.json" % file_name, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("./saved_models/%s_model.h5" % file_name)
        print("Saved model to disk")


    @staticmethod
    def load_model_json(json_path, h5_path):
        # load json and create model
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(h5_path)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        return loaded_model


    # 학습 시킬 데이터를 로드한다
    def load_csv(self, file_path):

        df = read_csv(file_path, delimiter='\t')
        array = df.values

        self.sensors = array[:, 0:8]
        self.co['guide_output'] = array[:, 8]
        self.c6h6['guide_output'] = array[:, 9]
        self.nox['guide_output'] = array[:, 10]
        self.no2['guide_output'] = array[:, 11]

        return 0

    def __load_model_dl(self, target, dir_path="./saved_models/"):
        json_path = dir_path + '%s_model.json' % target
        h5_path = dir_path + '%s_model.h5' % target

        json_exist = os.path.exists(json_path)
        h5_exist = os.path.exists(h5_path)

        if json_exist and h5_exist:
            return self.load_model_json(json_path, h5_path)
        else:
            print("[load_models] co_model not exist.")
            return -1

    # model을 총 4개 로드해야 한다 (가스 별로)
    def load_models(self, dir_path="./saved_models/"):
        result = 0

        # load co model
        if self.co['model_name'].startswith('DL_'):
            result = self.__load_model_dl("co")

            if result == -1:
                return result

            self.co['model'] = result

        elif os.path.exists(dir_path + 'co_model.sav'):
            self.co['model'] = load(open(dir_path + 'co_model.sav', 'rb'))
        else:
            print("[load_models] co_model.sav not exist.")
            result = -1

        # load c6h6 model
        if self.c6h6['model_name'].startswith('DL_'):
            result = self.__load_model_dl("c6h6")

            if result == -1:
                return result

            self.c6h6['model'] = result

        elif os.path.exists(dir_path + 'c6h6_model.sav'):
            self.c6h6['model'] = load(open(dir_path + 'c6h6_model.sav', 'rb'))
        else:
            print("[load_models] c6h6_model.sav not exist.")
            result = -1

        # load no2 model
        if self.no2['model_name'].startswith('DL_'):
            result = self.__load_model_dl("no2")

            if result == -1:
                return result

            self.no2['model'] = result
        elif os.path.exists(dir_path + 'no2_model.sav'):
            self.no2['model'] = load(open(dir_path + 'no2_model.sav', 'rb'))
        else:
            print("[load_models] no2_model.sav not exist.")
            result = -1


        # load nox model
        if self.nox['model_name'].startswith('DL_'):
            result = self.__load_model_dl("nox")

            if result == -1:
                return result

            self.nox['model'] = result

        elif os.path.exists(dir_path + 'nox_model.sav'):
            self.nox['model'] = load(open(dir_path + 'nox_model.sav', 'rb'))
        else:
            print("[load_models] nox_model.sav not exist.")
            result = -1

        return result

    def predict_batch(self, file_path, csv_name='predict_result.csv', calc_error=False):
        print("[predict_batch] start (calc_error=%r)" % calc_error)
        tar_df = read_csv(file_path, delimiter='\t')
        array = tar_df.values

        tar_inputs = array[:, 0:8]
        # min_max_scaler = preproc.MinMaxScaler()
        # tar_inputs = min_max_scaler.fit_transform(tar_inputs)

        # prediction
        tar_df['CO_PRED'] = self.co['model'].predict(tar_inputs)
        tar_df['C6H6_PRED'] = self.c6h6['model'].predict(tar_inputs)
        tar_df['NO2_PRED'] = self.no2['model'].predict(tar_inputs)
        tar_df['NOx_PRED'] = self.nox['model'].predict(tar_inputs)

        print("[predict_batch] predict done")

        # calculate error
        if calc_error == True:
            tar_df['CO_ERROR'] = tar_df['CO'] - tar_df['CO_PRED']
            tar_df['C6H6_ERROR'] = tar_df['C6H6'] - tar_df['C6H6_PRED']
            tar_df['NO2_ERROR'] = tar_df['NO2'] - tar_df['NO2_PRED']
            tar_df['NOx_ERROR'] = tar_df['NOx'] - tar_df['NOx_PRED']

            tar_df['CO_ERR_R'] = round(abs(tar_df['CO_ERROR'] / (tar_df['CO'] + tar_df['CO_PRED'])), 6)
            tar_df['C6H6_ERR_R'] = round(abs(tar_df['C6H6_ERROR'] / (tar_df['C6H6'] + tar_df['C6H6_PRED'])), 6)
            tar_df['NO2_ERR_R'] = round(abs(tar_df['NO2_ERROR'] / (tar_df['NO2'] + tar_df['NO2_PRED'])), 6)
            tar_df['NOx_ERR_R'] = round(abs(tar_df['NOx_ERROR'] / (tar_df['NOx'] + tar_df['NOx_PRED'])), 6)

            print("[predict_batch] error calculation done")

        file_name = file_path.split('/')[-1].split('.')[0]
        tar_df.to_csv(csv_name, sep='\t', encoding='utf-8', index=False)

        return 0


    def predict_input(self, input_list):
        return 0
