import numpy as np
from scipy import optimize
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal
from scipy.stats import skewnorm
from numba import jit
from scipy.io import loadmat
import scipy.stats as stats
# import pygpufit.gpufit as gf
from scipy.signal import find_peaks
import joblib
from tabulate import tabulate
from sklearn.metrics import r2_score
import plotly.express as px

class Input_Object():
    def __init__(self, filepath, true_values, TEMs = None, description = None, significant_spheres = None):
        self.significant_spheres = significant_spheres
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.wavelengths = self.data.iloc[:, 0].values  # wavelength values
        # print(self.wavelengths)
        self.intens = self.data.iloc[:, 1].values - min(self.data.iloc[:, 1].values)  # intensities and
        # baseline subtraction
        period_index = self.filepath.index('.')
        self.name = ''
        count = period_index - 1
        while filepath[count] != '/':
            self.name = self.name + filepath[count]
            count = count - 1
        self.name = self.name[::-1]
        print('name = ' + self.name)

        self.true_values = true_values
        self.TEMs = TEMs
        self.description = description

        self.spectrum = [self.wavelengths, self.intens]


    def plot_spectrum(self):
        plt.figure(figsize=(7.0, 6.0))
        plt.plot(self.spectrum[0], self.spectrum[1])
        plt.title(self.name, fontsize = 16)
        plt.xlabel('Wavelength (nm)', fontsize = 14)
        plt.ylabel('Absorption', fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)


    def visualize_TEM_sizes(self, bins=20):

        self.lengths_to_hist = []
        self.diameters_to_hist = []
        self.ars_to_hist = []

        self.loaded_tems = loadmat(self.TEMs)

        mean_ar_temp = None
        if 'lengths' in self.loaded_tems:
            for particle in self.loaded_tems['lengths']:
                if particle[3] == 2:
                    self.lengths_to_hist.append(particle[0])
                    self.diameters_to_hist.append(particle[1])
                    self.ars_to_hist.append(particle[2])
            if np.mean(self.ars_to_hist) < 1.5:
                mean_ar_temp = np.mean(self.ars_to_hist)
                self.lengths_to_hist = []
                self.diameters_to_hist = []
                self.ars_to_hist = []
                for particle in self.loaded_tems['lengths']:
                    if particle[3] == 1:
                        self.lengths_to_hist.append(particle[0])
                        self.diameters_to_hist.append(particle[1])
                        self.ars_to_hist.append(particle[2])
            if mean_ar_temp != None:
                if mean_ar_temp > np.mean(self.ars_to_hist):
                    print('Really short rods, need to fix this')

        if 'majlen' in self.loaded_tems:
            for length in self.loaded_tems['majlen'][0]:
                self.lengths_to_hist.append(length)
            for dia in self.loaded_tems['minlen'][0]:
                self.diameters_to_hist.append(dia)
            for i in range(0, len(self.lengths_to_hist)):
                self.ars_to_hist.append(self.lengths_to_hist[i] / self.diameters_to_hist[i])
        labels = ['Length', 'Diameter', 'Aspect Ratio']
        count = 0
        for size_dim in [self.lengths_to_hist, self.diameters_to_hist, self.ars_to_hist]:
            plt.figure(figsize=(7.0, 6.0))
            plt.hist(size_dim, bins=bins, density=True, color='grey', label='TEM Analysis')


            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(labels[count] + '(nm)', fontsize=24)
            plt.ylabel("Proportion", fontsize=24)
            plt.title(labels[count] + " Distribution", fontsize=30)
            # plt.legend([, distribution_label], fontsize = 13)
            plt.legend(fontsize = 14)
            plt.show()
            count += 1


class test_spectra_fit():
    def __init__(self, fit_summary, fit_description, output_objects):
        self.fit_summary = fit_summary
        self.fit_description = fit_description
        self.output_objects = output_objects

        print(len(self.output_objects))


        self.true_len_means = []
        self.true_len_stds = []
        self.true_dia_means = []
        self.true_dia_stds = []
        self.true_ar_means = []
        self.true_ar_stds = []

        self.predicted_len_means = []
        self.predicted_len_stds = []
        self.predicted_dia_means = []
        self.predicted_dia_stds = []
        self.predicted_ar_means = []
        self.predicted_ar_stds = []



        self.names = []
        table_names = ['Model Type']
        """
        for obj in self.output_objects:
            if type(obj) != list:
                if type(obj.true_values) != type(None):
                    self.names.append(obj.name)
                    print(obj.name)
                    print(obj.true_values)
                    self.true_len_means.append(obj.true_values[0])
                    self.true_len_stds.append(obj.true_values[1])
                    self.true_dia_means.append(obj.true_values[2])
                    self.true_dia_stds.append(obj.true_values[3])
                    self.true_ar_means.append(obj.true_values[4])
                    self.true_ar_stds.append(obj.true_values[5])
        """
            # want table where each row is a model type and each column is a spectrum. entries are predicted values.
            # need one of these for each parameter.

        self.predicted_sizes = [self.predicted_len_means,
        self.predicted_len_stds,
        self.predicted_dia_means,
        self.predicted_dia_stds,
        self.predicted_ar_means,
        self.predicted_ar_stds]


        self.ht_lengths = []
        self.lit_lengths = []
        self.ht_ars = []
        self.lit_ars = []

        self.ht_lengths_sd = []
        self.lit_lengths_sd = []
        self.ht_ars_sd = []
        self.lit_ars_sd = []

        self.sorted_sizes = [[self.ht_lengths, self.lit_lengths],
                             [self.ht_ars, self.lit_ars],
                             [self.ht_lengths_sd, self.lit_lengths_sd],
                             [self.ht_ars_sd, self.lit_ars_sd]]

        self.predicted_ht_lengths = []
        self.predicted_lit_lengths = []
        self.predicted_ht_ars = []
        self.predicted_lit_ars = []

        self.predicted_ht_lengths_sd = []
        self.predicted_lit_lengths_sd = []
        self.predicted_ht_ars_sd = []
        self.predicted_lit_ars_sd = []

        self.predicted_sorted_sizes = [[self.predicted_ht_lengths, self.predicted_lit_lengths],
                                       [self.predicted_ht_ars, self.predicted_lit_ars],
                                       [self.predicted_ht_lengths_sd, self.predicted_lit_lengths_sd],
                                       [self.predicted_ht_ars_sd, self.predicted_lit_ars_sd]]

        count = 0
        keys = ['len_dia_correlation']
        for size_param in self.predicted_sizes:
            # size_param.append(table_names)
            for key in keys:
                table_row = []
                table_row.append(key)
                for obj in self.output_objects:
                    if type(obj) != list:
                        try:
                            if type(obj.fit_results[key][0][0]) != str:
                                if obj.true_values != None:
                                    print(obj.name)
                                    fitted_parameters = obj.fit_results[key][0][0][4]
                                    print(fitted_parameters)
                                    # print(round(fitted_parameters[count], 1))
                                    table_row.append(round(fitted_parameters[count], 1))
                                    if obj.name not in table_names:
                                        table_names.append(obj.name)
                                    if count == 0:
                                        print(obj.name)
                                        print(obj.true_values)
                                        self.true_len_means.append(obj.true_values[0])
                                        self.true_len_stds.append(obj.true_values[1])
                                        self.true_dia_means.append(obj.true_values[2])
                                        self.true_dia_stds.append(obj.true_values[3])
                                        self.true_ar_means.append(obj.true_values[4])
                                        self.true_ar_stds.append(obj.true_values[5])
                                        if 'Clean' in obj.name or 'nimbus' in obj.name:
                                            self.predicted_sorted_sizes[0][0].append(float(fitted_parameters[0]))
                                            self.predicted_sorted_sizes[1][0].append(float(fitted_parameters[4]))

                                            self.predicted_sorted_sizes[2][0].append(float(fitted_parameters[1]))
                                            self.predicted_sorted_sizes[3][0].append(float(fitted_parameters[5]))

                                            self.sorted_sizes[0][0].append(float(obj.true_values[0]))
                                            self.sorted_sizes[1][0].append(float(obj.true_values[4]))

                                            self.sorted_sizes[2][0].append(float(obj.true_values[1]))
                                            self.sorted_sizes[3][0].append(float(obj.true_values[5]))
                                        else:
                                            self.predicted_sorted_sizes[0][1].append(float(fitted_parameters[0]))
                                            self.predicted_sorted_sizes[1][1].append(float(fitted_parameters[4]))
                                            self.predicted_sorted_sizes[2][1].append(float(fitted_parameters[1]))
                                            self.predicted_sorted_sizes[3][1].append(float(fitted_parameters[5]))

                                            self.sorted_sizes[0][1].append(float(obj.true_values[0]))
                                            self.sorted_sizes[1][1].append(float(obj.true_values[4]))

                                            self.sorted_sizes[2][1].append(float(obj.true_values[1]))
                                            self.sorted_sizes[3][1].append(float(obj.true_values[5]))
                        except KeyError:
                            if type(obj.fit_results['len_dia_correlation_rel_std'][0][0]) != str:
                                if obj.true_values != None:
                                    print(obj.name)
                                    fitted_parameters = obj.fit_results['len_dia_correlation_rel_std'][0][0][4]
                                    print(fitted_parameters)
                                    # print(round(fitted_parameters[count], 1))
                                    table_row.append(round(fitted_parameters[count], 1))
                                    if obj.name not in table_names:
                                        table_names.append(obj.name)
                                    if count == 0:
                                        print(obj.name)
                                        print(obj.true_values)
                                        self.true_len_means.append(obj.true_values[0])
                                        self.true_len_stds.append(obj.true_values[1])
                                        self.true_dia_means.append(obj.true_values[2])
                                        self.true_dia_stds.append(obj.true_values[3])
                                        self.true_ar_means.append(obj.true_values[4])
                                        self.true_ar_stds.append(obj.true_values[5])
                                        if 'Clean' in obj.name or 'nimbus' in obj.name:
                                            self.predicted_sorted_sizes[0][0].append(float(fitted_parameters[0]))
                                            self.predicted_sorted_sizes[1][0].append(float(fitted_parameters[4]))

                                            self.predicted_sorted_sizes[2][0].append(float(fitted_parameters[1]))
                                            self.predicted_sorted_sizes[3][0].append(float(fitted_parameters[5]))

                                            self.sorted_sizes[0][0].append(float(obj.true_values[0]))
                                            self.sorted_sizes[1][0].append(float(obj.true_values[4]))

                                            self.sorted_sizes[2][0].append(float(obj.true_values[1]))
                                            self.sorted_sizes[3][0].append(float(obj.true_values[5]))
                                        else:
                                            self.predicted_sorted_sizes[0][1].append(float(fitted_parameters[0]))
                                            self.predicted_sorted_sizes[1][1].append(float(fitted_parameters[4]))
                                            self.predicted_sorted_sizes[2][1].append(float(fitted_parameters[1]))
                                            self.predicted_sorted_sizes[3][1].append(float(fitted_parameters[5]))

                                            self.sorted_sizes[0][1].append(float(obj.true_values[0]))
                                            self.sorted_sizes[1][1].append(float(obj.true_values[4]))

                                            self.sorted_sizes[2][1].append(float(obj.true_values[1]))
                                            self.sorted_sizes[3][1].append(float(obj.true_values[5]))
                    else:
                        print(obj.name + ' being passed for some reason')
                        # pass
                        # table_row.append('NA')
                size_param.append(table_row)
            count += 1
        print(len(self.predicted_len_means[0]))
        print(self.true_len_means)
        self.predicted_len_means = pd.DataFrame(self.predicted_len_means, columns=table_names)
        self.predicted_len_means.set_index('Model Type', inplace=True)

        self.predicted_len_stds = pd.DataFrame(self.predicted_len_stds, columns=table_names)
        self.predicted_len_stds.set_index('Model Type', inplace=True)

        self.predicted_dia_means = pd.DataFrame(self.predicted_dia_means, columns=table_names)
        self.predicted_dia_means.set_index('Model Type', inplace=True)

        self.predicted_dia_stds = pd.DataFrame(self.predicted_dia_stds, columns=table_names)
        self.predicted_dia_stds.set_index('Model Type', inplace=True)

        self.predicted_ar_means = pd.DataFrame(self.predicted_ar_means, columns=table_names)
        self.predicted_ar_means.set_index('Model Type', inplace=True)

        self.predicted_ar_stds = pd.DataFrame(self.predicted_ar_stds, columns=table_names)
        self.predicted_ar_stds.set_index('Model Type', inplace=True)

        self.names = table_names

    def r_squared(self, size_param, show = 'all', show_plot = True):

        self.predicted_size_params = [self.predicted_len_means, self.predicted_len_stds,
                                 self.predicted_dia_means, self.predicted_dia_stds,
                                 self.predicted_ar_means, self.predicted_ar_stds]
        self.true_size_params = [self.true_len_means, self.true_len_stds,
                                 self.true_dia_means, self.true_dia_stds,
                                 self.true_ar_means, self.true_ar_stds]
        self.param_codes = ['len_mean','len_std','dia_mean','dia_std','ar_mean','ar_std']


        location = self.param_codes.index(size_param)
        # print(self.predicted_len_means.index)
        if show == 'all':
            for model in self.predicted_len_means.index:
                to_plot_predicted = []
                to_plot_true = []
                names = []
                # print(self.predicted_len_means.T)
                # self.predicted_len_means = self.predicted_len_means.T
                # self.predicted_ar_means = self.predicted_ar_means.T
                # print(len(self.true_len_means))
                for i in range(0, len(self.true_len_means)):
                    to_plot_true.append(float(self.true_size_params[location][i]))
                    to_plot_predicted.append(self.predicted_size_params[location].loc[model][i])
                # print(to_plot_predicted)
                # print(to_plot_true)
                # print(self.names)

                df = pd.DataFrame(np.asarray([to_plot_true, to_plot_predicted]),columns=self.names[1:len(self.names)])
                print(df.T)
                # df.set_index(['Label'], inplace=True)
                # df = df.transpose()
                # df.set_index(['Label'], inplace=True)
                # print(df)
                # plt.plot(to_plot_true, to_plot_true)
                plt.figure(figsize=(7.0, 6.0))
                # plt.scatter(to_plot_true, to_plot_predicted)
                if size_param == 'len_mean':
                    print('ht_num ' + str(len(self.sorted_sizes[0][0])))
                    print('lit_num ' + str(len(self.sorted_sizes[0][1])))
                    plt.scatter(self.sorted_sizes[0][0], self.predicted_sorted_sizes[0][0], label = 'High Throughput', s=50)
                    plt.scatter(self.sorted_sizes[0][1], self.predicted_sorted_sizes[0][1], label = 'Literature', s=50)
                    plt.plot(np.arange(20, 140, 1), np.arange(20, 140, 1), color='k', linewidth = 3)
                    plt.plot(np.arange(20, 140, 1), np.arange(20, 140, 1) * 1.2, color='k', linestyle='--',
                             label='20% error', linewidth = 3)
                    plt.plot(np.arange(20, 140, 1), np.arange(20, 140, 1) * 0.8, color='k', linestyle='--', linewidth = 3)
                    plt.ylim([20, 135])
                    plt.xlim([20, 135])
                    # plt.title('Predicted vs True Mean Length', fontsize = 26)
                    plt.xlabel('True ' +  r'$\tilde{L}$', fontsize=26)
                    plt.ylabel('Predicted ' +  r'$\tilde{L}$', fontsize=26)

                if size_param == 'ar_mean':
                    plt.scatter(self.sorted_sizes[1][0], self.predicted_sorted_sizes[1][0], label = 'High Throughput', s=50)
                    plt.scatter(self.sorted_sizes[1][1], self.predicted_sorted_sizes[1][1], label = 'Literature', s=50)
                    plt.plot(np.arange(2.5, 9.5, 1), np.arange(2.5, 9.5, 1), color='k', linewidth = 3)
                    plt.plot(np.arange(2.5, 9.5, 1), np.arange(2.5, 9.5, 1) * 1.2, color='k', linestyle='--',
                             label='20% error', linewidth = 3)
                    plt.plot(np.arange(2.5, 9.5, 1), np.arange(2.5, 9.5, 1) * 0.8, color='k', linestyle='--',
                             linewidth = 3)
                    plt.ylim([2.5, 8])
                    plt.xlim([2.5, 8])

                    # plt.title('Predicted vs True Mean Aspect Ratio', fontsize = 26)
                    plt.xlabel('True ' +  r'$\tilde{AR}$', fontsize=26)
                    plt.ylabel('Predicted ' +  r'$\tilde{AR}$', fontsize=26)

                    print(r2_score(self.sorted_sizes[1][1], self.predicted_sorted_sizes[1][1]))
                    print(sum(np.abs(np.asarray(self.predicted_sorted_sizes[1][1]) - np.asarray(self.sorted_sizes[1][1])))/len(self.sorted_sizes[1][1]))

                if size_param == 'len_std':

                    print(self.sorted_sizes[2][0])
                    print(len(self.sorted_sizes[2][0]))
                    print(self.predicted_sorted_sizes[2][0])
                    print(len(self.predicted_sorted_sizes[2][0]))

                    plt.scatter(self.sorted_sizes[2][0], self.predicted_sorted_sizes[2][0], label='High Throughput',
                                s=50)
                    plt.scatter(self.sorted_sizes[2][1], self.predicted_sorted_sizes[2][1], label='Literature', s=50)
                    plt.plot(np.arange(2, 27, 1), np.arange(2, 27, 1), color='k', linewidth = 3)
                    plt.plot(np.arange(2, 27, 1), np.arange(2, 27, 1) * 1.2, color='k', linestyle='--',
                             label='20% error', linewidth = 3)
                    plt.plot(np.arange(2, 27, 1),np.arange(2, 27, 1) * 0.8, color='k', linestyle='--',
                             linewidth = 3)
                    plt.ylim([2, 26])
                    plt.xlim([2, 26])

                    # plt.title('Predicted vs True SD Length', fontsize=26)
                    plt.xlabel('True ' +  '\u03C3' + 'L', fontsize=26)
                    plt.ylabel('Predicted ' +  '\u03C3' + 'L', fontsize=26)

                if size_param == 'ar_std':
                    plt.scatter(self.sorted_sizes[3][0], self.predicted_sorted_sizes[3][0], label='High Throughput',
                                s=50)
                    to_plot_true_use = []
                    to_plot_predicted_use = []
                    for i in range(len(self.sorted_sizes[3][1])):
                        print(self.sorted_sizes[3][1][i], i)
                        if np.isnan(self.sorted_sizes[3][1][i]) == False:
                            to_plot_predicted_use.append(self.predicted_sorted_sizes[3][1][i])
                            to_plot_true_use.append(self.sorted_sizes[3][1][i])
                    plt.scatter(to_plot_true_use, to_plot_predicted_use, label='Literature', s=50)

                    for i in range(len(self.sorted_sizes[3][0])):
                        if np.isnan(self.sorted_sizes[3][0][i]) == False:
                            to_plot_predicted_use.append(self.predicted_sorted_sizes[3][0][i])
                            to_plot_true_use.append(self.sorted_sizes[3][0][i])


                    plt.plot(np.arange(0.3, 10, 1), np.arange(0.3, 10, 1), color='k', linewidth = 3)
                    plt.plot(np.arange(0.3, 10, 1), np.arange(0.3, 10, 1) * 1.2, color='k', linestyle='--',
                             label='20% error', linewidth = 3)
                    plt.plot(np.arange(0.3, 10, 1), np.arange(0.3, 10, 1) * 0.8, color='k', linestyle='--',
                             linewidth = 3)
                    plt.ylim([0.3, 1.55])
                    plt.xlim([0.3, 1.55])

                    # plt.title('Predicted vs True SD Aspect Ratio', fontsize = 26)
                    plt.xlabel('True ' +  '\u03C3' + 'AR', fontsize=26)
                    plt.ylabel('Predicted ' +  '\u03C3' + 'AR', fontsize=26)
                    # print(r2_score(to_plot_true, to_plot_predicted))
                    # print(sum(np.abs(np.asarray(to_plot_predicted) - np.asarray(to_plot_true)))/len(to_plot_true))


                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.legend(fontsize = 16)
                plt.savefig('Fig_4_'+size_param+'.pdf', bbox_inches='tight', transparent=True)

                plt.show()
                if size_param != 'ar_std':
                    print(r2_score(to_plot_true, to_plot_predicted))
                    print(sum(np.abs(np.asarray(to_plot_predicted) - np.asarray(to_plot_true)))/len(to_plot_true))
                else:
                    print(r2_score(to_plot_true_use, to_plot_predicted_use))
                    print(sum(np.abs(np.asarray(to_plot_predicted_use) - np.asarray(to_plot_true_use)))/len(to_plot_true_use))
        if show == 'catagory':
            for model in self.predicted_len_means.index:
                to_plot_true = []
                to_plot_predicted = []

                to_plot_predicted_not_murphy = []
                to_plot_true_not_murphy = []

                to_plot_predicted_murphy = []
                to_plot_true_murphy = []
                names = []
                for i in range(0, len(self.true_len_means)):
                    if type(self.predicted_len_means.loc[model][i]) == str:
                        print(self.predicted_len_means.loc[model][i])
                    else:
                        if 'Murphy' in self.names[i]:

                            to_plot_true_murphy.append(self.true_size_params[location][i])
                            to_plot_predicted_murphy.append(self.predicted_size_params[location].loc[model][i])
                        else:
                            to_plot_true_not_murphy.append(self.true_size_params[location][i])
                            to_plot_predicted_not_murphy.append(self.predicted_size_params[location].loc[model][i])

                        to_plot_true.append(self.true_size_params[location][i])
                        to_plot_predicted.append(self.predicted_size_params[location].loc[model][i])
                        names.append(self.names[i])
                # df = pd.DataFrame(np.asarray([to_plot_true, to_plot_predicted]),columns=names)
                # df.set_index(['Label'], inplace=True)
                # df = df.transpose()
                # df.set_index(['Label'], inplace=True)
                # print(df)
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(to_plot_true, to_plot_true)
                plt.scatter(to_plot_true_murphy, to_plot_predicted_murphy, color = 'red', label = 'Murphy')
                plt.scatter(to_plot_true_not_murphy, to_plot_predicted_not_murphy, color = 'blue', label = 'Other Spectra')
                plt.xlabel('True Value', fontsize = 16)
                plt.ylabel('Predicted Value', fontsize = 16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title('Predicted vs True Mean Aspect Ratio', fontsize = 18)
                plt.legend(fontsize = 14)
                plt.show()
                print('all', r2_score(to_plot_true, to_plot_predicted))
                print('not murphy', r2_score(to_plot_true_not_murphy, to_plot_predicted_not_murphy))

                print(sum(np.abs(np.asarray(to_plot_predicted) - np.asarray(to_plot_true)))/len(to_plot_true))

    def visualize_overlaps(self, hist_title, show_plots = False, overlap_method = 'lit'):
        overlaps = []
        overlaps_650 = []
        overlaps_variable = []
        names = []
        names_650 = []
        names_variable = []

        overlaps_650_rel_std = []
        overlaps_650_normal = []

        overlaps_650_rel_std_names = []
        overlaps_650_normal_names = []
        # print(len(self.output_objects))
        for obj in self.output_objects:
            # print(obj.name)
            if np.isnan(obj.true_values[1]) == False:
                table = obj.tabulate_results(overlap_method = overlap_method, show_table = False)
                if obj.name == 'Lai_2014_Figure_3D_Blue':
                    print(table)
                if table[1][1] != 'NA':
                    names.append(obj.name)
                    overlaps.append(float(table[1][1]))

                    if '650' in obj.fit_description:
                        overlaps_650.append(float(table[1][1]))
                        names_650.append(obj.name)
                        if 'rel_std' in list(obj.fit_results.keys())[0]:
                            overlaps_650_rel_std.append(float(table[1][1]))
                            overlaps_650_rel_std_names.append(obj.name)

                        else:
                            overlaps_650_normal.append(float(table[1][1]))
                            overlaps_650_normal_names.append(obj.name)

                    else:
                        overlaps_variable.append(float(table[1][1]))
                        names_variable.append(obj.name)
        # print(overlaps)
        mean_overlap = np.mean(overlaps)
        std_overlap = np.std(overlaps)
        median_overlap = np.median(overlaps)
        print('mean overlap = ' + str(mean_overlap))
        print('std overlap = ' + str(std_overlap))
        print('median overlap = ' + str(median_overlap))
        plt.figure(figsize=(7,6))
        plt.hist(overlaps, bins = 15)
        plt.vlines(mean_overlap, 0, 5, color = 'red', linestyles='--', label = 'Mean Overlap', linewidth=3)
        plt.legend(fontsize = 20)
        plt.title(hist_title, fontsize = 30)
        plt.xlabel('Overlap', fontsize = 26)
        plt.ylabel('Count', fontsize = 26)
        plt.xticks(fontsize = 26)
        plt.yticks(fontsize=26)
        plt.savefig('Fig_4_overlaps.pdf', bbox_inches='tight', transparent=True)

        plt.show()
        plt.hist(overlaps_650, bins = 15, color = 'blue')
        plt.hist(overlaps_variable, bins = 15, color = 'r')
        plt.title(hist_title, fontsize = 14)
        plt.show()

        return [mean_overlap, std_overlap, median_overlap, overlaps, names, overlaps_650, names_650, overlaps_variable,
                names_variable, overlaps_650_rel_std, overlaps_650_normal, overlaps_650_rel_std_names,
                overlaps_650_normal_names]
