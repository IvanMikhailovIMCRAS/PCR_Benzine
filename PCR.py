import spectra as sp
import numpy as np


def parsing_names(name: str):
    proc = ''
    read_label = False
    for i in name[::-1]:
        if read_label:
            if i.isdigit():
                proc = proc + i
            else:
                break
        if i == '.':
            read_label = True
    return float(proc[::-1])


def add_ones_column(M):
    return np.insert(M, 0, values=np.ones(np.shape(M)[0]), axis=1)


class LinearRegression:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        assert self.y.shape[0] == self.X.shape[0], "check in data size (X,y)"
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.X = self.standardization(self.X)
        self.X = add_ones_column(self.X)
        self.w = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def standardization(self, M):
        return (M - self.mean)/self.std

    def predictions(self, M):
        X = np.array(M)
        X = self.standardization(X)
        X = add_ones_column(X)
        assert X.shape[1] == self.w.shape[0], "check in data size (X,w)"
        return X @ self.w


if __name__ == '__main__':
    train_set_1 = sp.Spectra('data_1')
    train_set_2 = sp.Spectra('data_2')
    select_intervals = [(3000, 2800), (1500, 1300)]
    # select_intervals = [(3999, 599)]
    train_set_1.norm()
    train_set_2.norm()
    ##
    train_set_1.select_freq(*select_intervals)
    train_set_2.select_freq(*select_intervals)
    ##
    train_set_1.show('spectra_set_1_norm', select_intervals)
    train_set_2.show('spectra_set_2_norm', select_intervals)
    # here we can make change of the principal components number
    n_components = len(train_set_1.names) - 1
    print(f'n_components: {n_components}')
    try:
        pc_set_1 = train_set_1.principle_components(n_components)
        pc_set_2 = train_set_2.principle_components(n_components)
        ##
        y1 = np.array([parsing_names(i) for i in train_set_1.names])
        y2 = np.array([parsing_names(i) for i in train_set_2.names])
        assert y1.shape == y2.shape, "y1 and y2 must have the same shape"
        assert set(y1) == set(y2), "y1 and y2 must have the same set of values"
        assert train_set_1.X[1:].shape[0] == y1.shape[0], "check in data size (pc1,y1)"
        assert train_set_2.X[1:].shape[0] == y2.shape[0], "check in data size (pc2,y2)"
        # sorting y1 and y2 and corresponding pc_sets
        indices = y1.argsort()
        y1 = y1[indices]
        pc_set_1 = pc_set_1[indices]
        ##
        indices = y2.argsort()
        y2 = y2[indices]
        pc_set_2 = pc_set_2[indices]
        # built the models of linear regression for both sets
        model_1 = LinearRegression(pc_set_1, y1)
        y12_predict = model_1.predictions(pc_set_2)
        model_2 = LinearRegression(pc_set_2, y2)
        y21_predict = model_2.predictions(pc_set_1)
        ##
        print(f'y_true_values: {[round(val,1) for val in y1]}')
        print(f'y_predict_12 : {[round(val,1) for val in y12_predict]}')
        print(f'y_predict_21 : {[round(val,1) for val in y21_predict]}')
        ##
        error12 = np.sqrt(np.mean((y1 - y12_predict) ** 2))
        error21 = np.sqrt(np.mean((y2 - y21_predict) ** 2))
        print(f'error12: {round(error12,1)}, error21: {round(error21,1)}')
    except Exception as err:
        print(f'Error: {err}! Please change number of components!')
