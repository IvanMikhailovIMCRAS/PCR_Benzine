from brukeropusreader import read_file
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np
import os
from sklearn.decomposition import PCA


class Spectra:
    def __init__(self, directory: str):
        self.directory = directory  # from which we load the spectra
        self.names = list()  # names of files
        self.paths = list()
        self.data = [[0, ], [0, ]]  # data storage: data[0] - wave-numbers; data[1:] - intensities
        self.X = np.zeros(shape=(1, 1))  # usable data-matrix for cleaning and processing
        self.freq_mask = np.array([True])  # mask for data clipping by frequency
        self.obs_mask = np.array([True])  # mask for data clipping by observations
        self.empty = True  # label that downloading from {directory} is failed
        for current_dir, _, files in os.walk(self.directory):
            for file in files:
                self.download_one_data_file(current_dir, file)
        if not self.empty:
            self.freq_mask = np.ones(len(self.data[0]), dtype=bool)
            self.obs_mask = np.ones(len(self.data), dtype=bool)
            self.recalc_X_under_masks()
            self.X = np.array(self.data)

    def recalc_X_under_masks(self):
        """ Recalculation of X-data-matrix taking into account selected restrictions"""
        try:
            self.X = np.array(self.data)[np.ix_(self.obs_mask, self.freq_mask)]
        except IndexError:
            self.X = np.zeros(shape=(1, 1))

    def download_one_data_file(self, current_dir: str, file: str):
        path = f'{current_dir}/{file}'
        len_main_name = file.rfind('.')
        if file[-1].isdigit() and file[(len_main_name + 1):].isdigit():
            try:
                opus_file = read_file(path)
                if self.empty:
                    x_data = opus_file.get_range("AB")
                    # x_data = [3998.2646619038087 - i * 0.96436677807 for i in range(3525)]
                    self.data = list()
                    self.data.append(x_data)
                    self.empty = False
                x_data = list(opus_file.get_range("AB"))
                y_data = list(opus_file["AB"][:])
                if len(x_data) != len(self.data[0]):
                    print(f'Warning! Uncorrected length: {path}')
                    # for i in range(3525):
                    #     if x_data[i] - self.data[0][i] > 0.48218338903:
                    #         y_data[i] = y_data[i + 1]
                    # y_data.pop(-1)
                self.data.append(y_data)
                self.names.append(file)
                self.paths.append(path)
            except Exception as err:
                print(f'Warning: file {path} cannot be opened! Err: {err}')

    def select_freq(self, *intervals: tuple, invert: bool = False, reset: bool = True):
        if reset:
            self.freq_mask[:] = invert  # reset of the previous mask
        for i, freq in enumerate(self.data[0]):
            for interval in intervals:
                if interval[1] < freq < interval[0]:
                    self.freq_mask[i] = not invert
        self.recalc_X_under_masks()

    def select_obs(self, *observances: str, invert: bool = False, reset: bool = True):
        if reset:
            self.obs_mask[:] = invert
        for obs in observances:
            try:
                self.obs_mask[self.names.index(obs)] = not invert
            except ValueError as err:
                print(f'Observance \'{obs}\' cannot be selected! Err: {err} of names')
        self.recalc_X_under_masks()

    def norm(self):
        for i in range(1, self.X.shape[0]):
            self.X[i] = self.X[i] / np.sum(self.X[i])

    def derivative(self, n=1):
        for _ in range(n):
            for i in range(1, self.X.shape[0]):
                self.X[i] = np.gradient(self.X[i], self.X[0], edge_order=2)

    def principle_components(self, nc):
        pca = PCA(n_components=nc)
        return pca.fit_transform(self.X[1:])

    def show(self, name_picture, intervals=None):
        if intervals is None:
            intervals = [(3999, 599)]
        name_picture = name_picture + '.jpg'
        br_axes = brokenaxes(xlims=intervals, hspace=1)
        for y in self.X[1:]:
            br_axes.plot(self.X[0], y)
        plt.savefig(name_picture, dpi='figure', format="jpg")
        plt.close()


if __name__ == '__main__':
    pass
