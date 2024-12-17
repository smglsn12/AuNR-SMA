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
import pandas as pd
import matplotlib.ticker as ticker

from   scipy.stats import (multivariate_normal as mvn,
                           norm)
from   scipy.stats._multivariate import _squeeze_output

@jit
def compute_ar_dist(ars, ar_matrix, population_matrix):
    # print('calculating ar dist')
    # print(ar_matrix.shape)
    # print(ars)
    # print(population_matrix.shape)
    ar_dist = np.zeros(len(ars))
    for i in range(0, len(ars)):
        count = 0
        prob_temp = np.zeros(len(ar_matrix))
        probs = 0
        for j in range(0, len(ar_matrix)):
            for k in range(0, len(ar_matrix[0])):
                if ar_matrix[j][k] == np.round(ars[i], 2):
                    prob_temp[count] = population_matrix[j][k]
                    count += 1
        for prob in prob_temp:
            probs += prob
        ar_dist[i] = probs

    return ar_dist

@jit
def ar_matrix_correction(population_matrix, ar_matrix):
    # print('calculating ar matrix')
    for i in range(0, len(population_matrix)):
        for j in range(0, len(population_matrix[0])):
            population_matrix[i][j] = population_matrix[i][j]*ar_matrix[i][j]

    return population_matrix

@jit
def calculate_spectrum(profiles_nrs, population_matrix, spectrum):
    # print('calculating spectrum')
    for i in range(0, len(profiles_nrs)):
        for j in range(0, len(profiles_nrs[0])):
            spectrum = spectrum + profiles_nrs[i][j] * population_matrix[i][j]
    return spectrum

def gaussian(x,B,mu,s):
    return B*np.exp(-0.5*(np.square((x-mu)/s)))

class multivariate_skewnorm:

    def __init__(self, shape, means, cov=None):
        self.dim = len(shape)
        self.shape = np.asarray(shape)
        self.mean = means
        #self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        x = mvn._process_quantiles(x, self.dim)
        pdf = mvn(self.mean, self.cov).logpdf(x)
        cdf = norm(self.mean[0], self.cov[0][0]**2).logcdf(np.dot(x, self.shape))
        # print('done')
        return _squeeze_output(np.log(2) + pdf + cdf)


class Absorption_decon():

    def __init__(self, filepath, long_dims = None, short_dims = None, ars = None, profile_ars = None, sphere_ars = None,
                 smooth_spectrum = True, pick_longitudinal=False, NS_sizes = None, longitudinal_threshold=None,
                 blue_baseline = False, blue_edge_location = None, red_edge_threshold = None, tem_data = None,
                 true_values = None, min_longitudinal_threshold = 0.12, show_sim = True, smoothing_parameters = (13,3)):
        '''
        Initializing a Absorption_decon object requires a csv file with the following structure:
        1st column: wavelength values (nm)
        2nd column: measured intensities

        filepath - string:
        path to the csv file containing data in the above structure

        long dims, short dims, and ars - arrays:
        arrays of the lengths, diameters, and aspect ratios contained in the profiles matrix. This will need to
        be exact, as these are used to build the population matrix which is then multipled by the single particle
        spectra.

        Profiles - 2d array with dims lengths x diameters

        Smooth_spectrum - Bool:
        whether or not to smooth the inputted spectrum

        Pick_longitudinal - Bool:
        whether to extract the longitudinal peak or fit the full spectrum

        Longitudinal_threshold - Float (between 0 and 1):
        The absorption threshold to cut the longitudinal peak (post normalization). For example, a threshold of 0.2
        includes all of the longitudinal peak above 0.2

        x0 - list or list of lists
        initial guess(es) of size distribution parameters [length mean, length sd, AR mean, AR sd, AR skew,
        length/AR correlation]. If no value is provided, a function runs to produce the default values (see
        create_x0s)
        '''

        # initialize attributes
        self.profiles = joblib.load("C:/Users/smgls/PycharmProjects/Data_Models_and_Spectra/Au NR Profiles (for size prediction model)/AuNR_Profiles_1_23_21.joblib")
        self.ar_profiles = joblib.load('C:/Users/smgls/PycharmProjects/Data_Models_and_Spectra/Au NR Profiles (for size prediction model)/AR_len_profile_matrix_spacing_1_decimal.joblib')
        print(self.profiles.shape)
        self.ar_profiles_spheres = joblib.load("C:/Users/smgls/PycharmProjects/Data_Models_and_Spectra/Au NR Profiles (for size prediction model)/ar_len_profiles_matrix_spheres.joblib")
        self.sphere_profiles = joblib.load('C:/Users/smgls/PycharmProjects/Data_Models_and_Spectra/Au NR Profiles (for size prediction model)/sphere_profiles_091321.joblib')
        self.simulation_wavelengths = np.round(np.arange(401, 1700, 1), 0)
        self.true_distributions = None
        self.smoothed_intens_old = None
        self.wavelengths_old = None
        self.fitted_params = None
        self.peak_indicies = None
        self.blue_baseline_intens = None
        self.blue_baseline_wavelengths = None
        self.lengths_to_hist = []
        self.diameters_to_hist = []
        self.ars_to_hist = []
        self.filepath = filepath
        self.tem_data = tem_data
        self.longitudinal_threshold = longitudinal_threshold
        self.true_values = true_values
        self.bad_spectrum = False
        self.pick_longitudinal = pick_longitudinal
        self.blue_edge_location = blue_edge_location
        self.red_edge_threshold = red_edge_threshold
        self.min_longitudinal_threshold = min_longitudinal_threshold
        self.show_sim = show_sim
        self.smoothing_params = smoothing_parameters
        self.already_set_h12 = False
        # set size ranges to default values if none are provided
        if long_dims == None:
            self.long_dims = np.arange(10, 501, 1)
        else:
            self.long_dims = long_dims

        if short_dims == None:
            self.short_dims = np.arange(5, 51, 1)
        else:
            self.short_dims = short_dims

        if ars == None:
            self.ars = np.arange(1.5, 10.01, 0.01) # AR values for when fit is in len/dia space
            self.profile_ars = np.arange(1.5, 10.1, 0.1) # AR values for when fit is in AR/len space (scaling changed
            # to 0.1 to make the population matrix less unwieldy)
            self.sphere_ars = np.arange(1,10.1,0.1) # AR values extend down to one when population matrix is set up to
            # include spheres

        else:
            self.ars = ars
            self.profile_ars = profile_ars
            self.sphere_ars = sphere_ars

        if NS_sizes == None:
            self.NS_Sizes = np.arange(5,101,1)
        else:
            self.NS_sizes = NS_sizes

        self.normal_ar_distribution = None
        self.normal_diameter_distribution = None
        self.normal_length_distribution = None


        # load the spectrum from csv
        if type(self.filepath) == str: # this block loads the spectrum if the spectrum has been provided via a filepath
            self.data = pd.read_csv(self.filepath)
            self.wavelengths = self.data.iloc[:,0].values  # wavelength values
            # print(self.wavelengths)
            self.intens = self.data.iloc[:, 1].values - min(self.data.iloc[:, 1].values)  # intensities and
            # baseline subtraction
            period_index = self.filepath.index('.')
            self.name = ''
            count = period_index - 1
            while filepath[count] not in '\/':
                # print(filepath[count])
                self.name = self.name + filepath[count]
                count = count-1
            self.name = self.name[::-1]
            print(self.name)
        else: # this block loads the spectrum if it has been provided as a list of points
            self.name = self.filepath[0]
            self.wavelengths = self.filepath[1]
            self.intens = self.filepath[2]
            self.intens = self.intens - min(self.intens)  # intensities and
            # baseline subtraction

        for i in range(0, len(self.wavelengths)):
            self.wavelengths[i] = round(self.wavelengths[i])  # this should be fixing issues where my wavelength points
            # are not integers (450.999999 etc)
            # TODO put in something that checks that the wavelengths are all one integer apart (I've had issues with
            # this not being the case for some of the extractions)
        big_issue = False
        for i in range(1, len(self.wavelengths)):
            # print(self.wavelengths[i])
            # print(self.wavelengths[i]-self.wavelengths[i-1])
            if self.wavelengths[i]-self.wavelengths[i-1] != 1:
                print('BIG ISSUE STOPPPPPP')
                big_issue = True
        if big_issue == True:
            f = interp1d(self.wavelengths, self.intens)
            one_int_wavelengths = np.arange(min(self.wavelengths), max(self.wavelengths)+1)
            self.interp_intens = f(one_int_wavelengths)
            # plt.plot(one_int_wavelengths, self.interp_intens)
            # plt.plot(self.wavelengths, self.intens)
            self.wavelengths = one_int_wavelengths
            self.intens = self.interp_intens

        # plt.plot(self.wavelengths, self.intens)
        # plt.show()
        self.inputted_intens = self.intens
        self.inputted_wavelengths = self.wavelengths
        self.intens = self.intens / max(self.intens)   # normalization, although be careful with this normalization!
        # it will normalize to the highest point, whatever that is. So if there is noise around the longitudinal peak,
        # it could normalize to the highest noise location, rather than the true location of the peak. This will cause
        # significant issues for the fit if the noise is significant enough
        # plt.figure(figsize=(8,6))
        # plot spectrum
        # plt.plot(self.wavelengths, self.intens, label = 'Original Spectrum')




        if smooth_spectrum == True:
            self.spectrum_smoothed = True
            smoothing_ouput = self.smooth_spectrum(self.intens, self.wavelengths)
            self.smoothed_intens = smoothing_ouput[0]
            # TODO convert these print statements into tests
            # print(max(self.smoothed_intens))
            self.err = smoothing_ouput[1]
            # print(max(self.smoothed_intens))
            peak_index_max = list(self.smoothed_intens).index(max(self.smoothed_intens)) # the index of the maximum point
            # NOT necessarily the longitudinal peak (perhaps this is why I was having issues with samples with low yield)
            print(self.wavelengths[peak_index_max])
            # plt.show()
        if smooth_spectrum == False:
            self.spectrum_smoothed = False
            self.err = []

        self.peak_indicies_old = self.set_wavelength_scale()
        self.wavelengths_old = self.wavelengths
        self.smoothed_intens_old = self.smoothed_intens

        # print(self.peak_indicies_old, [min(self.wavelengths_old), max(self.wavelengths_old)])
        # print(self.wavelengths)




        ar_predictions = self.guess_AR()
        print(ar_predictions)
        self.ar_prediction = ar_predictions[0]
        peak_wavelength = ar_predictions[1]
        longitudinal = ar_predictions[2]
        # if peak_wavelength < 800:
            # self.bad_spectrum = True
        if self.ar_prediction == 'NA':
            self.bad_spectrum = True

        elif longitudinal < 1.0:
            self.bad_spectrum = True

        transverse_peak_wavelength = ar_predictions[3]
        print("Guessed AR " + str(self.ar_prediction))

        if self.bad_spectrum == True:
            print('Spectrum will not be fit, longitudinal peak is too low intensity')


        # transverse = np.max(self.smoothed_intens[blue_edge:blue_edge+75])
        # transverse_index = list(self.smoothed_intens).index(transverse)

        # for i in range(transverse_index, len(self.smoothed_intens)):
        # blue_edge_longitudinal = blue_edge + 50
        # longitudinal = np.max(self.smoothed_intens[blue_edge_longitudinal:len(self.smoothed_intens)])
        # longitudinal_index = list(self.smoothed_intens).index(longitudinal)


        # print('self.bad_spectrum', self.bad_spectrum)


        if self.bad_spectrum == False:
            self.transverse_peak_wavelength = transverse_peak_wavelength
            self.longitudinal_peak_wavelength = peak_wavelength
            longitudinal_index = list(self.smoothed_intens_old).index(longitudinal)
            # plt.scatter(self.wavelengths[longitudinal_index], self.smoothed_intens[longitudinal_index], color='k')
            # plt.plot(self.wavelengths, self.smoothed_intens)
            # plt.show()
            if blue_baseline == True:
                self.set_baseline()
                self.blue_baseline_wavelengths = self.wavelengths
                self.blue_baseline_intens = self.smoothed_intens

            if pick_longitudinal == True:

                # finds the location of the longitudinal peak, cuts it out, and plots it on top of the original spectrum
                # so the user can confirm the process has been done correctly
                if self.spectrum_smoothed == True:
                    if longitudinal_threshold == None:
                        longitudinal_threshold = self.find_longitudinal_threshold(self.wavelengths[longitudinal_index])
                        self.longitudinal_threshold = longitudinal_threshold
                        print('self.longitudinal_threshold = ' + str(self.longitudinal_threshold))
                    # plt.plot(self.wavelengths, self.smoothed_intens, color = 'k', linewidth = 3,
                             # label = 'Full Smoothed Spectrum')
                    longitudinal_calc = self.find_longitudinal(longitudinal_threshold, self.wavelengths[longitudinal_index])
                    self.wavelengths = longitudinal_calc[1]
                    self.smoothed_intens = longitudinal_calc[0]
                    self.peak_indicies = longitudinal_calc[2] # this allows the fit to compare only to the wavelength
                # points of the longitudinal peak
                    self.err = longitudinal_calc[3]
                    # plt.plot(self.wavelengths, self.smoothed_intens, label = 'Extracted Longitudinal Peak',
                             # color = 'darkorange')
                    # plt.xlabel("Wavelength (nm)", fontsize=14)
                    # plt.ylabel("Absorption (normalized)", fontsize=14)
                    # plt.title('Longitudinal Extraction', fontsize = 16)
                    # plt.xticks(fontsize=14)
                    # plt.yticks(fontsize=14)
                    # plt.legend()
                    # plt.show()
            if pick_longitudinal == False:
                print('setting peaks')
                # if not extracting the longitudinal peak, it is necessary to determine the wavelength scale of the inputted
                # spectrum and then determine what points of the simulated spectrum (which runs from 401 nm to 1699 nm)
                # create a simulated spectrum on the same wavelength scale
                print(min(self.wavelengths))
                self.peak_indicies = self.set_wavelength_scale()
                print(self.peak_indicies)

        # print(self.simulation_wavelengths[self.peak_indicies[0]:self.peak_indicies[1]])

            self.wavelengths_list = []
            self.intens_list = []
            self.population_matrix = None
            self.reduced_chi = None
            self.fits = []
            self.best_fit = None
            if self.blue_edge_location != None:
                print('Starting')
                blue_edge_index = list(self.wavelengths).index(self.blue_edge_location)
                len_smoothed_intens_old = len(self.smoothed_intens)
                self.wavelengths = self.wavelengths[blue_edge_index:len(self.wavelengths)]
                self.smoothed_intens = self.smoothed_intens[blue_edge_index:len(self.smoothed_intens)]
                print(self.peak_indicies, 'peak indicies')
                if self.peak_indicies == None:
                    self.peak_indicies = [0,0]
                    self.peak_indicies[0] = blue_edge_index
                    self.peak_indicies[1] = len_smoothed_intens_old
                if self.peak_indicies != None:
                    self.peak_indicies[0] = blue_edge_index
                    self.peak_indicies[1] = len_smoothed_intens_old

            if self.red_edge_threshold != None:
                longitudinal_index = list(self.smoothed_intens).index(longitudinal)
                print(longitudinal_index)
                first_nm_over = None
                for i in range(0, len(self.smoothed_intens) - longitudinal_index):
                    # print(self.smoothed_intens[longitudinal + i], first_nm_over)
                    if self.smoothed_intens[longitudinal_index + i] < red_edge_threshold and first_nm_over == None:
                        first_nm_over = longitudinal_index + i
                        print(self.smoothed_intens[longitudinal_index + i])
                        print(first_nm_over)

                if first_nm_over != None:
                    self.smoothed_intens = self.smoothed_intens[0:first_nm_over]
                    self.wavelengths = self.wavelengths[0:first_nm_over]

                elif first_nm_over == None:
                    first_nm_over = len(self.wavelengths)

                print([max(self.wavelengths), 'max wavelength'])
                self.peak_indicies[1] = self.peak_indicies[0] + first_nm_over

            plt.plot(self.wavelengths, self.smoothed_intens)
            plt.show()
            print('comparision peak indicies')
            # if pick_longitudinal == True:
                # if self.peak_indicies_old[2] == 'Red':
                    # self.peak_indicies[0] = self.peak_indicies[0] + self.peak_indicies_old[0]
                    # self.peak_indicies[1] = self.peak_indicies[1] + self.peak_indicies_old[0]


            # self.peak_indicies[0] = self.peak_indicies[0]
            # self.peak_indicies[1] = self.peak_indicies[0] + 800

            # self.wavelengths = self.wavelengths[self.peak_indicies[0]:self.peak_indicies[1]]
            # self.smoothed_intens = self.smoothed_intens[self.peak_indicies[0]:self.peak_indicies[1]]

            print(self.peak_indicies)
            print(min(self.wavelengths), max(self.wavelengths))
            print(self.simulation_wavelengths[self.peak_indicies[0]], self.simulation_wavelengths[self.peak_indicies[1]])
            if min(self.wavelengths) != self.simulation_wavelengths[self.peak_indicies[0]]:
                # pass
                diff = min(self.wavelengths) - self.simulation_wavelengths[self.peak_indicies[0]]
                diff = int(diff)
                self.peak_indicies[0] = self.peak_indicies[0] + diff
                self.peak_indicies[1] = self.peak_indicies[1] + diff
                print(self.peak_indicies)
                print('printing new results')
                print(self.simulation_wavelengths[self.peak_indicies[0]], self.simulation_wavelengths[self.peak_indicies[1]])
                print(min(self.wavelengths), max(self.wavelengths))

    def find_longitudinal_threshold(self, longitudinal_peak_wavelength):
        # should come after smoothing and blue baseline logic
        longitudinal = list(self.wavelengths).index(longitudinal_peak_wavelength)
        if self.smoothed_intens[longitudinal] != max(self.smoothed_intens):
            max_longitudinal = self.smoothed_intens[longitudinal]
            sample = self.smoothed_intens/max_longitudinal
        else:
            sample = self.smoothed_intens


        blue_threshold = None
        red_threshold = None

        for i in range(0, len(sample) - longitudinal):
            # print(sample[longitudinal - i])
            if sample[longitudinal + i] < self.min_longitudinal_threshold and red_threshold == None:
                red_threshold = self.min_longitudinal_threshold
        for i in range(0, longitudinal):
            if sample[longitudinal - i] < self.min_longitudinal_threshold and blue_threshold == None:
                blue_threshold = self.min_longitudinal_threshold
        if blue_threshold == None:
            blue_threshold = min(sample[0:longitudinal])
            blue_threshold = round(blue_threshold + 0.01, 2)
        if red_threshold == None:
            red_threshold = round(sample[len(sample)-1]+0.01,2)
        """
        if blue_threshold == None:
            peaks = find_peaks(self.smoothed_intens, height=0.15, distance=50)
            peaks_list = []
            for peak in peaks[0]:
                peaks_list.append(peak)
            if len(peaks_list) != 2:
                for peak in peaks_list:
                    print(peak)
            if len(peaks_list) == 2:
                between_peaks = self.smoothed_intens[peaks_list[0]:peaks_list[1]]
                blue_threshold = round(min(between_peaks) + 0.01, 2)
        """
        print(['thresholds', blue_threshold, red_threshold])
        if max([blue_threshold, red_threshold]) < 1.0:
            true_threshold = max([blue_threshold, red_threshold])
        else:
            true_threshold = min([blue_threshold, red_threshold])

        print(true_threshold, blue_threshold, red_threshold)
        self.blue_threshold = blue_threshold
        self.red_threshold = red_threshold
        return true_threshold


    def set_wavelength_scale(self):
        min_wavelength = min(self.wavelengths)
        start_type = None
        if min_wavelength >= 401:
            index_first_wavelength = np.where(self.simulation_wavelengths == np.round(min_wavelength))
            # print('index first wavelength')
            # print(index_first_wavelength)
            start_type = "Red"
        if min_wavelength < 401:
            gap_to_401 = 401 - min_wavelength
            gap_to_401 = int(gap_to_401)
            self.smoothed_intens = self.smoothed_intens[gap_to_401:len(self.smoothed_intens)]
            self.wavelengths = self.wavelengths[gap_to_401:len(self.wavelengths)]
            index_first_wavelength = [[0]]
            start_type = "Blue"
        min_wavelength = min(self.wavelengths)
        max_wavelength = max(self.wavelengths)
        print(min_wavelength,max_wavelength)
        index_last_wavelength = np.where(self.simulation_wavelengths == np.round(max_wavelength,0))
        return [index_first_wavelength[0][0], index_last_wavelength[0][0]+1, start_type]


    def set_baseline(self):
        # find low point between peaks
        # end spectrum at the last point with intensity higher than this point
        transverse_index = list(self.wavelengths).index(self.transverse_peak_wavelength)
        longitudinal_index = list(self.wavelengths).index(self.longitudinal_peak_wavelength)

        peaks_list = [transverse_index, longitudinal_index]
        # plt.scatter(self.wavelengths[peaks_list], self.smoothed_intens[peaks_list], color = 'r')
        # plt.plot(self.wavelengths, self.smoothed_intens)
        # plt.show()
        if len(peaks_list) == 2:
            between_peaks = self.smoothed_intens[peaks_list[0]:peaks_list[1]]
            baseline = min(between_peaks)
            # print(baseline)
            after_peaks = self.smoothed_intens[peaks_list[1]:len(self.smoothed_intens)]
            index_of_cutoff = None
            for i in after_peaks:
                if i < baseline and index_of_cutoff == None:
                    index_of_cutoff = np.where(self.smoothed_intens == i)[0][0]
            if index_of_cutoff != None:
                scaled_intens = self.smoothed_intens[0:index_of_cutoff] - baseline
                scaled_wavelengths = self.wavelengths[0:index_of_cutoff]
                scaled_intens = scaled_intens/max(scaled_intens)
                # plt.plot(scaled_wavelengths, scaled_intens, label = 'Scaled Spectrum')
                # plt.title("Scaled Baseline", fontsize = 16)
                # plt.legend(['Scaled Spectrum', 'Peaks'], fontsize = 12)
                # plt.xlabel("Wavelength (nm)", fontsize=14)
                # plt.ylabel("Absorption (normalized)", fontsize=14)
                # plt.xticks(fontsize=14)
                # plt.yticks(fontsize=14)
                # plt.legend()
                # plt.show()
                self.smoothed_intens = scaled_intens
                self.wavelengths = scaled_wavelengths
            if index_of_cutoff == None:
                print('baseline already blue')
                pass
        else:
            print("no worko")



    def guess_AR(self):
        """
        :param sample: list or array of intensities
        :param sample_wavelengths: list or array of wavelengths
        :return: predicted mean aspect ratio of the rods in the sample to be used in the initial guess(es). Uses
        equation found here: https://pubs.acs.org/doi/10.1021/cm303661d
        """
        if 500 in self.wavelengths:
            blue_edge = list(self.wavelengths).index(500)
            transverse_peak = find_peaks(self.smoothed_intens[blue_edge:blue_edge+75], height=0.15, distance=50)
            print(transverse_peak[0])

            if len(transverse_peak[0]) == 0:
                print('starting')
                transverse_peak_location = max(self.smoothed_intens[blue_edge:blue_edge+75])
                transverse_peak_index = list(self.smoothed_intens).index(transverse_peak_location)
                transverse_peak_wavelength = self.wavelengths[transverse_peak_index]
                print(transverse_peak_wavelength, 'transverse peak wavelength')
                start = transverse_peak_index + 5
            else:
                print(transverse_peak[0][0]+500)
                start = blue_edge+transverse_peak[0][0]+5
                transverse_peak_wavelength = transverse_peak[0][0] + 500

            transverse_peak_index = list(self.wavelengths).index(transverse_peak_wavelength)
            self.transverse_peak_intensity = self.smoothed_intens[transverse_peak_index]
            start = int(start)
            longitudinal_peak = find_peaks(self.smoothed_intens[start:len(self.smoothed_intens)], height=0.15, distance=50, width = 20)
            print('printing longitudinal')
            print(longitudinal_peak[0])
            print('max post transverse')
            print(start)
            print(transverse_peak_index)
            print(max(self.smoothed_intens[start:len(self.smoothed_intens)]))
            if len(longitudinal_peak[0]) != 0:
                peak_location = longitudinal_peak[0][0] + start
                longitudinal = self.smoothed_intens[peak_location]

                peak_wavelength = self.wavelengths[peak_location]
                print('peak location = ' + str(peak_wavelength))
                # plt.scatter(peak_location, 1.0)
                ar = (peak_wavelength-420)/95
                if longitudinal > 0.2:
                    return [ar, peak_wavelength, longitudinal, transverse_peak_wavelength]
                elif longitudinal < 0.2:
                    self.bad_spectrum = True
                    return ['NA', 'NA', 'NA', 'NA']


            elif max(self.smoothed_intens[start:len(self.smoothed_intens)]) == 1:
                peak_location = list(self.smoothed_intens).index(1)
                peak_wavelength = self.wavelengths[peak_location]
                print('peak location = ' + str(peak_wavelength))
                # plt.scatter(peak_location, 1.0)
                ar = (peak_wavelength - 420) / 95
                longitudinal = self.smoothed_intens[peak_location]
                return [ar, peak_wavelength, longitudinal, transverse_peak_wavelength]

            else:
                self.bad_spectrum = True
                return ['NA', 'NA', 'NA', 'NA']
        elif 600 in self.wavelengths:
            start = list(self.wavelengths).index(600)
            longitudinal_peak = find_peaks(self.smoothed_intens[start:len(self.smoothed_intens)], height=0.15,
                                           distance=50, width=20)
            print('printing longitudinal')
            print(longitudinal_peak[0])
            print('max post transverse')
            print(start)
            print(max(self.smoothed_intens[start:len(self.smoothed_intens)]))
            if len(longitudinal_peak[0]) != 0:
                peak_location = longitudinal_peak[0][0] + start
                longitudinal = self.smoothed_intens[peak_location]

                peak_wavelength = self.wavelengths[peak_location]
                print('peak location = ' + str(peak_wavelength))
                # plt.scatter(peak_location, 1.0)
                ar = (peak_wavelength - 420) / 95
                if longitudinal > 0.2:
                    return [ar, peak_wavelength, longitudinal, 'NA']
                elif longitudinal < 0.2:
                    self.bad_spectrum = True
                    return ['NA', 'NA', 'NA', 'NA']


            elif max(self.smoothed_intens[start:len(self.smoothed_intens)]) == 1:
                peak_location = list(self.smoothed_intens).index(1)
                peak_wavelength = self.wavelengths[peak_location]
                print('peak location = ' + str(peak_wavelength))
                # plt.scatter(peak_location, 1.0)
                ar = (peak_wavelength - 420) / 95
                longitudinal = self.smoothed_intens[peak_location]
                return [ar, peak_wavelength, longitudinal, 'NA']

            else:
                self.bad_spectrum = True
                return ['NA', 'NA', 'NA', 'NA']
        else:
            self.bad_spectrum = True
            return ['NA', 'NA', 'NA', 'NA']

    def find_longitudinal(self, threshold, longitudinal_peak_wavelength):

        print('Wavelength mins, must be 401 for proper comparison to simulations')
        print(min(self.wavelengths))
        """
        :param sample: list or array of intensities, full spectrum
        :param sample_wavelengths: list or array of wavelengths
        :param threshold: absorption threshold (normalized) to extract longitudinal peak. Extraction will occur from the
        bluest wavelength where the intensity is greater than the threshold and run to the reddest point where this is
        still true. TODO this function will probably get messed up if significant impurities are present and the
        TODO longitudinal peak is not the highest peak
        :return: list
        1st entry - intensities of longitudinal peak
        2nd entry - corresponding wavelengths
        3rd entry - list containing indicies for simulated spectrum to put it on the same wavelength scale
        """

        longitudinal = list(self.wavelengths).index(longitudinal_peak_wavelength)
        if self.smoothed_intens[longitudinal] != max(self.smoothed_intens):
            max_longitudinal = self.smoothed_intens[longitudinal]
            sample = self.smoothed_intens/max_longitudinal
        else:
            sample = self.smoothed_intens


        if sample[longitudinal] != max(sample):
            max_longitudinal = sample[longitudinal]
            sample = sample/max_longitudinal



        # peak = sample.index(1.0)
        # peak_index = peak
        peak_location = longitudinal_peak_wavelength
        # plt.scatter(peak_location, 1)
        longitudinal_peak = []
        longitudinal_peak_wavelengths = []
        err = []
        # print(sample)
        peak_begining = None
        peak_end = None
        peak_index_end = None
        peak_index_begining = None
        for i in range(0, len(self.wavelengths)):
            if sample[longitudinal - i] < threshold and peak_begining == None:
                peak_index_begining = longitudinal - i
                peak_begining = peak_location - i
            if longitudinal + i < len(self.wavelengths):
                if sample[longitudinal + i] < threshold and peak_end == None:
                    peak_index_end = longitudinal + i
                    peak_end = peak_location + i
        if peak_index_end == None:
            peak_index_end = len(self.smoothed_intens)
        # plt.scatter([peak_begining, peak_end], [0.4, 0.4])
        print([peak_index_begining, peak_index_end])
        for i in range(peak_index_begining, peak_index_end):
            longitudinal_peak.append(sample[i])
            longitudinal_peak_wavelengths.append(self.wavelengths[i])
            err.append(self.err[i])

        return [longitudinal_peak, longitudinal_peak_wavelengths, [peak_index_begining, peak_index_end], err]

    def smooth_spectrum(self, sample, sample_wavelengths):
        """
        :param sample: list or array of intensities to be smoothed
        :param sample_wavelengths: list or array of wavelengths
        :return: smoothed spectrum and the absolute value of the difference between smoothed and original spectrum
        """


        window = self.smoothing_params[0] # will eventually want a function that estimates the proper size of this, but fitting the smoothed
        # spectrum doesn't seem to be nearly as sensitive to the smoothing parameters
        window_size, poly_order = window, self.smoothing_params[1]
        yy_sg = savgol_filter(sample, window_size, poly_order) # smooth the spectrum and plot on top of original
        # print(max(yy_sg))
        # print(min(yy_sg))

        # plt.plot(sample_wavelengths, self.intens-(yy_sg/max(yy_sg)), label = 'diff sample and normalized smooth spectrum')
        # plt.plot(sample_wavelengths, yy_sg-(yy_sg/max(yy_sg)), label = 'diff smoothed spectra')
        # plt.legend()
        # plt.show()
        yy_sg = yy_sg - min(yy_sg)
        yy_sg = yy_sg/max(yy_sg)
        # print(max(yy_sg))
        # print(min(yy_sg))
        plt.plot(sample_wavelengths, sample, color='k', label='measured',linewidth=3)
        plt.plot(sample_wavelengths, yy_sg/max(yy_sg), linestyle='-', label='Smoothed',color='orange')
        plt.title("Spectrum with Smoothing", fontsize = 16)
        plt.legend(fontsize = 12)
        plt.xlabel("Wavelength (nm)", fontsize = 14)
        plt.ylabel("Absorption (normalized)", fontsize = 14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

        # determine difference between smoothed spectrum and sample and plot
        err = yy_sg - sample
        # plt.plot(sample_wavelengths, np.abs(err), label = 'window size = ' + str(window))
        # plt.title("Error Function", fontsize = 16)
        # plt.legend(fontsize = 12)
        # plt.xlabel('Wavelength', fontsize = 14)
        # plt.ylabel('Error Abs', fontsize = 14)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.show()

        # take derivatives of smoothed and sample spectrum and plot
        deriv = np.gradient(yy_sg)
        deriv2 = np.gradient(sample)
        # plt.plot(sample_wavelengths, deriv2, 'k', label="Deriv True Spectrum")
        # plt.plot(sample_wavelengths, deriv, 'orange', label='Deriv Smoothed Spectrum')
        # plt.legend(fontsize=12)
        # plt.xlabel('Wavelength', fontsize = 14)
        # plt.ylabel('First Deriv', fontsize = 14)
        # plt.title("Derivative Spectra", fontsize=16)

        return [yy_sg, np.abs(err)]

    """
    from here through sim are helper functions that various parts of the fitting code call 
    TODO add documentation to these
    """

    def calc_diameter_stdev(self, means, cov):
        # Diameter = Length/AR
        AR = means[0]  # make sure first entry is AR
        length = means[1]

        AR_sd = np.sqrt(cov[0][0])
        length_sd = np.sqrt(cov[1][1])

        covariance = cov[1][0]

        Diameter_SD = np.sqrt((((length / (AR ** 2)) * AR_sd) ** 2) + (((1 / AR) * length_sd) ** 2) + (
                (-2 * length / (AR ** 3)) * covariance))

        # print((-2 * length / (AR ** 3)) * covariance)

        return Diameter_SD

    def lin_scaling(self, mean_len, mean_dia, std_len, std_dia):
        default_len = [-5, 5]
        default_dia = [-5, 5]
        len_scale = (255 - mean_len) / 48.652
        dia_scale = (27.5 - mean_dia) / 4.5
        scaled_len = [default_len[0] + len_scale, default_len[1] + len_scale]
        scaled_dia = [default_dia[0] + dia_scale, default_dia[1] + dia_scale]

        percentage_neg = np.abs(scaled_len[0]) / (np.abs(scaled_len[0]) + scaled_len[1])
        percentage_pos = 1 - percentage_neg
        std_1 = [-245, 245]
        std_scale_len = np.asarray(std_1) / std_len
        std_scale_len[0] = std_scale_len[0] * percentage_neg * 2
        std_scale_len[1] = std_scale_len[1] * percentage_pos * 2

        percentage_neg = np.abs(scaled_dia[0]) / (np.abs(scaled_dia[0]) + scaled_dia[1])
        percentage_pos = 1 - percentage_neg
        std_2 = [-22.5, 22.5]
        std_scale_dia = np.asarray(std_2) / std_dia
        std_scale_dia[0] = std_scale_dia[0] * percentage_neg * 2
        std_scale_dia[1] = std_scale_dia[1] * percentage_pos * 2

        return [std_scale_len, std_scale_dia]

    def bivariate_gaussian_skew_len_dia(self, mean,std,corr,skews):
        diameters = np.arange(5, 51, 1)
        lengths = np.arange(10, 501, 1)

        scaled_params = self.lin_scaling(mean[0], mean[1], std[0], std[1])

        xx = np.linspace(scaled_params[1][0], scaled_params[1][1], 46)
        yy = np.linspace(scaled_params[0][0], scaled_params[0][1], 491)
        X, Y = np.meshgrid(xx, yy)
        pos = np.dstack((X, Y))

        cov = corr

        Z = multivariate_skewnorm(shape=skews, means=[0, 0], cov=[[1, cov], [cov, 1]]).pdf(pos)
        # plt.figure(figsize=(7.0, 6.0))
        # X = mean[0] + var[0]*X
        # Y = mean[1] + var[1]*Y
        # print(Y)
        # plt.contour(diameters, lengths, Z)
        # plt.ylim(30,200)
        # plt.xlim(10,40)
        # plt.show()

        return Z

    def bivariate_gaussian_len_dia(self, lengths, diameters, means, cov):

        first_num_length = lengths[0]
        step_size_length = lengths[1] - lengths[0]
        last_num_length = lengths[len(lengths) - 1] + step_size_length

        first_num_dia = diameters[0]
        step_size_dia = diameters[1] - diameters[0]
        last_num_dia = diameters[len(diameters) - 1] + step_size_dia

        x, y = np.mgrid[first_num_dia:last_num_dia:step_size_dia, first_num_length:last_num_length:step_size_length]
        pos = np.dstack((x, y))
        rv = multivariate_normal(means, cov)
        """
        length_distribution = []
        AR_distribution = []

        
        for length in rv.pdf(pos).T:
            length_distribution.append(sum(length))
        for AR in rv.pdf(pos):
            AR_distribution.append(sum(AR))

        calc_diameters = np.zeros((len(aspect_ratios), len(lengths)))
        for i in range(0, len(aspect_ratios)): 
            for j in range(0, len(lengths)):
                calc_diameters[i][j] = round(lengths[j]/aspect_ratios[i])

        """
        ar_std = self.calc_diameter_stdev(means, cov)
        """
        population_len = []
        population_aspect = []
        population_diameter = []
        for length in lengths:
            population_len.append(sizeFxn.gaussian(length, 1, means[1], np.sqrt(cov[1][1])))
        for ar in aspect_ratios: 
            population_aspect.append(sizeFxn.gaussian(ar, 1, means[0], np.sqrt(cov[0][0])))
        for dia in diameters:
            population_diameter.append(sizeFxn.gaussian(dia, 1, means[1]/means[0], diameter_sd))
        """
        population_matrix = rv.pdf(pos)
        """
        plt.contourf(x, y, rv.pdf(pos))
        plt.xlabel("Diameter (nm)", fontsize = 16)
        plt.ylabel("Length (nm)", fontsize = 16)
        plt.title("Length and Diameter Bivariate Normal", fontsize = 18)
        plt.xlim([15,35])
        plt.ylim([45,85])
        plt.show()
        """
        # plt.contourf(x,y,biv)
        # plt.show()
        return [population_matrix, means[1] / means[0], ar_std]

    def ar_population_bivariate_gaussian_len_dia(self, ars, lengths, means, cov):

        first_num_length = lengths[0]
        step_size_length = lengths[1] - lengths[0]
        last_num_length = lengths[len(lengths) - 1] + step_size_length
        first_num_ars = ars[0]
        step_size_ars = ars[1] - ars[0]
        last_num_ars = ars[len(ars) - 1] + step_size_ars
        # print(first_num_ars, last_num_ars)
        x, y = np.mgrid[first_num_length:last_num_length:step_size_length, first_num_ars:last_num_ars:step_size_ars]
        pos = np.dstack((x, y))
        rv = multivariate_normal(means, cov)
        """
        length_distribution = []
        AR_distribution = []


        for length in rv.pdf(pos).T:
            length_distribution.append(sum(length))
        for AR in rv.pdf(pos):
            AR_distribution.append(sum(AR))

        calc_diameters = np.zeros((len(aspect_ratios), len(lengths)))
        for i in range(0, len(aspect_ratios)): 
            for j in range(0, len(lengths)):
                calc_diameters[i][j] = round(lengths[j]/aspect_ratios[i])

        """
        ar_std = self.calc_diameter_stdev(means, cov)
        """
        population_len = []
        population_aspect = []
        population_diameter = []
        for length in lengths:
            population_len.append(sizeFxn.gaussian(length, 1, means[1], np.sqrt(cov[1][1])))
        for ar in aspect_ratios: 
            population_aspect.append(sizeFxn.gaussian(ar, 1, means[0], np.sqrt(cov[0][0])))
        for dia in diameters:
            population_diameter.append(sizeFxn.gaussian(dia, 1, means[1]/means[0], diameter_sd))
        """
        population_matrix = rv.pdf(pos)
        """
        plt.contourf(x, y, rv.pdf(pos))
        plt.xlabel("Diameter (nm)", fontsize = 16)
        plt.ylabel("Length (nm)", fontsize = 16)
        plt.title("Length and Diameter Bivariate Normal", fontsize = 18)
        plt.xlim([15,35])
        plt.ylim([45,85])
        plt.show()
        """
        # plt.contourf(x,y,biv)
        # plt.show()
        return [population_matrix, means[1] / means[0], ar_std]



    def create_ar_matrix(self, short_dims, long_dims, ar_shape_parameter, ar_vals, ar_mean, ar_sd):
        for i in range(0, len(ar_vals)):
            ar_vals[i] = round(ar_vals[i], 2)
        ar_matrix = np.zeros((len(short_dims), len(long_dims)))
        for i in range(len(ar_matrix)):
            for j in range(len(ar_matrix[0])):
                if long_dims[j] > short_dims[i] * 1.5 and long_dims[j] < 10 * short_dims[i]:
                    ar_matrix[i][j] = np.round(long_dims[j] / short_dims[i], 2)

        delta = ar_shape_parameter / (np.sqrt(1 + ar_shape_parameter ** 2))
        ar_scale_parameter = np.sqrt((ar_sd ** 2) / (1 - (2 * (delta ** 2) / np.pi)))
        ar_location_parameter = ar_mean - ar_scale_parameter * delta * np.sqrt(2 / np.pi)
        ar_pop = skewnorm.pdf(ar_vals, ar_shape_parameter, ar_location_parameter, ar_scale_parameter)

        # ar_pop =stats.cauchy.pdf(ar_vals,ar_mean, ar_sd)

        # plt.plot(ar_vals, ar_pop)
        # plt.vlines(ar_mean, 0, 1)
        # plt.show()
        # print(gaussian(ar_vals, 1, ar_mean, ar_sd))
        for i in range(0, len(ar_vals)):
            # print(ar_vals[i])
            ar_matrix[ar_matrix == ar_vals[i]] = ar_pop[i]
            # ar_matrix[ar_matrix == ar_vals[i]] = 1
        return ar_matrix

    def create_spectrum_bivariate_longitudinal(self, population_matrix, profiles_nrs, ar_matrix=[], peak_range=None,
                                               simulation_baseline=False, show_distributions=False):

        if ar_matrix != []:
            population_matrix = ar_matrix_correction(population_matrix, ar_matrix)
        # diameters = np.arange(10, 30, 1)
        # lengths = np.arange(31, 150, 1)
        if show_distributions == True:
            old_length_distribution = []
            old_diameter_distribution = []
            for length in population_matrix.T:
                old_length_distribution.append(sum(length))
            for diameter in population_matrix:
                old_diameter_distribution.append(sum(diameter))


            diameters = np.arange(5, 51, 1)
            lengths = np.arange(10, 501, 1)
            ars = np.arange(1.5, 10.01, 0.01)
            if ar_matrix != []:
                plt.contourf(lengths, diameters, ar_matrix, cmap='viridis')
                plt.title("AR Distribution 2D", fontsize=18)
                plt.colorbar()
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.show()

                plt.contourf(lengths, diameters, population_matrix, cmap='viridis')
                plt.title("Rods Population Post AR Adjustment", fontsize=18)
                plt.ylim(5, 40)
                plt.xlim(10, 200)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.colorbar()
                plt.show()

            """
            length_distribution = []
            diameter_distribution = []
            for length in population_matrix.T:
                length_distribution.append(sum(length))
            for diameter in population_matrix:
                diameter_distribution.append(sum(diameter))
            normalize_length = np.trapz(y=length_distribution, x=lengths)
            normalize_diameter = np.trapz(y=diameter_distribution, x=diameters)
            plt.figure(figsize=(7.0, 6.0))
            plt.plot(diameters, diameter_distribution / max(diameter_distribution), label='Post AR dist')
            plt.plot(diameters, old_diameter_distribution / max(old_diameter_distribution), label='Pre AR dist')

            plt.title("Diameter Distributions", fontsize=18)
            plt.legend(fontsize=12)
            plt.xlabel("Diameter (nm)", fontsize=16)
            plt.ylabel("Proportion (Normalized)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.vlines(30, 0, 1)
            plt.show()

            plt.figure(figsize=(7.0, 6.0))
            plt.plot(lengths, length_distribution / max(length_distribution), label='Post AR dist')
            plt.plot(lengths, old_length_distribution / max(old_length_distribution), label='Pre AR dist')
            plt.title("Length Distributions", fontsize=18)
            plt.legend(fontsize=12)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("Length (nm)", fontsize=16)
            plt.ylabel("Proportion (Normalized)", fontsize=16)
            # plt.vlines(150, 0, 1)
            plt.xlim(50, 200)
            plt.show()
            """

        """
        new_len_mean = np.sum((length_distribution/normalize_length)*lengths)
        new_dia_mean = np.sum((diameter_distribution/normalize_diameter)*diameters)
        new_len_std = np.sqrt(np.dot(((lengths - new_len_mean)**2),length_distribution/normalize_length))
        new_dia_std = np.sqrt(np.dot(((diameters - new_dia_mean)**2),diameter_distribution/normalize_diameter))
        ar_values_long = np.arange(1.5, 10.1, 0.1)
        ar_matrix = np.zeros((len(diameters), len(lengths)))
        for i in range(len(ar_matrix)):
            for j in range(len(ar_matrix[0])):
                if lengths[j] >= diameters[i] * 1.5 and lengths[j] <= 10 * diameters[i]:
                    ar_matrix[i][j] = np.round(lengths[j] / diameters[i], 1)


        ar_dist = []
        for i in range(0, len(ar_values_long)):
            # print(ar_vals[i])
            prob_temp = []
            for j in range(0, len(ar_matrix)):
                for k in range(0, len(ar_matrix[0])):
                    if ar_matrix[j][k] == np.round(ar_values_long[i], 1):
                        prob_temp.append(population_matrix[j][k])
            if sum(prob_temp) == 0.0:
                print(ar_values_long[i])
            ar_dist.append(sum(prob_temp))

        ar_values_long_useful = np.arange(15, 101, 1)

        normalize_ar = np.trapz(y = ar_dist, x = ar_values_long_useful)
        plt.plot(ar_values_long, ar_dist/normalize_ar)
        print(np.trapz(y = ar_dist/normalize_ar, x = ar_values_long))
        plt.show()

        # new_len_mean = np.sum((length_distribution/normalize_length)*lengths)

        new_ar_mean = np.sum((ar_dist/normalize_ar)*ar_values_long_useful)
        new_ar_std = np.sqrt(np.dot(((ar_values_long_useful - new_ar_mean)**2),ar_dist/normalize_ar))



        print([new_len_mean, new_len_std, new_dia_mean, new_dia_std, new_ar_mean/10, new_ar_std/10])
        """

        # population_matrix = population_matrix/max(population_matrix)
        # print(np.sum(lengths * length_distribution/quad(length_distribution, min(lengths),max(lengths))))
        # print(np.sum(diameters*diameter_distribution/quad(diameter_distribution, min(diameters),max(diameters))))
        # num_points = peak_range[1] - peak_range[0]
        spectrum = np.zeros(len(profiles_nrs[0][0]))
        # plt.plot(self.simulation_wavelengths, spectrum)
        # plt.show()
        # print(spectrum)
        # spectrum = calculate_spectrum(profiles_nrs, population_matrix, spectrum)

        if profiles_nrs.shape[0] == population_matrix.shape[0]:
            # print('all good')
            spectrum = calculate_spectrum(profiles_nrs, population_matrix, spectrum)
        elif profiles_nrs.shape[0] == population_matrix.shape[1]:
            # print('fixing')
            spectrum = calculate_spectrum(profiles_nrs, population_matrix.transpose(), spectrum)
        else:
            raise ValueError(
                'Profiles and population matrices must have the same first two dimensions, but Profiles has ' + str(
                    profiles_nrs.shape) + ' and population has ' + str(population_matrix.shape))

        # plt.plot(self.simulation_wavelengths, spectrum)
        # plt.show()
        # print(max(spectrum))
        spectrum = spectrum/max(spectrum)
        # plt.plot(self.simulation_wavelengths, spectrum)
        # plt.show()
        if simulation_baseline == True:
            baseline = min(spectrum[peak_range[0]:peak_range[1]])
            spectrum = spectrum - baseline
            spectrum = spectrum / max(spectrum)
        spectrum = spectrum[peak_range[0]:peak_range[1]]



        # print(min(spectrum))
        return [spectrum, population_matrix]


    def ar_population_create_spectrum_bivariate_longitudinal(self, population_matrix, profiles_nrs, peak_range=None,
                                               simulation_baseline=False, show_distributions=False):

        # diameters = np.arange(10, 30, 1)
        # lengths = np.arange(31, 150, 1)


        # population_matrix = population_matrix/max(population_matrix)
        # print(np.sum(lengths * length_distribution/quad(length_distribution, min(lengths),max(lengths))))
        # print(np.sum(diameters*diameter_distribution/quad(diameter_distribution, min(diameters),max(diameters))))
        # num_points = peak_range[1] - peak_range[0]
        spectrum = np.zeros(len(profiles_nrs[0][0]))
        spectrum = calculate_spectrum(profiles_nrs, population_matrix, spectrum)
        # spectrum = spectrum - min(spectrum)

        if simulation_baseline == True:
            baseline = min(spectrum[peak_range[0]:peak_range[1]])
            spectrum = spectrum - baseline
            spectrum = spectrum / max(spectrum)
        spectrum = spectrum[peak_range[0]:peak_range[1]]

        # print(spectrum)
        return [spectrum, population_matrix]


    def sim(self, fit_type, population_matrix = None, simulation_baseline = False, fit_parameters = None,
            print_params = False, show_distributions = False, show_plot = False, fit_vals_from_pop = None):

        # plot population matrix in len/dia space
        if type(population_matrix) != type(None):
            matrix_sum = 0
            for i in range(0, len(population_matrix)):
                for j in range(0, len(population_matrix[0])):
                    matrix_sum += population_matrix[i][j]
            print(matrix_sum)
            if fit_type in ['len_ar_rel_std', 'len_ar_rel_std_skew', 'len_dia_correlation_rel_std', 'ar_len_matrix',
                            'ar_len', 'len_dia_ar_matrix_std', 'len_dia_ar_matrix', 'len_dia_correlation_skew',
                            'len_dia_correlation', 'len_dia_correlation_ar_matrix', 'len_dia_correlation_ar_matrix_skew']:
                plt.figure(figsize=(8.0, 6.0))

                plt.contourf(self.long_dims, self.short_dims, population_matrix, cmap='Purples')
                plt.colorbar(ticks=[])
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=34)
                plt.ylabel("Diameters (nm)", fontsize=34)
                plt.xticks([50, 100, 150], fontsize=30)
                plt.yticks([15,25,35], fontsize=30)
                plt.title("Predicted AuNR Population", fontsize=38)
                plt.scatter(100, 21, s = 200, color = 'g')
                plt.scatter(115, 21, s = 200, color = 'orange')
                plt.scatter(130, 21, s = 200, color = 'r')

                if fit_vals_from_pop != None:
                    plt.hlines(fit_vals_from_pop[2], 10, fit_vals_from_pop[0], linestyle = ':', color = 'k', linewidth = 5)
                    plt.vlines(fit_vals_from_pop[0], fit_vals_from_pop[2], 40, linestyle = '--', color ='k', linewidth = 5)
                    ys = np.linspace(11.75, 26.75, 10)
                    xs = np.linspace(62.5, 150, 10)
                    plt.plot(xs, ys, color = 'k', linewidth = 3)
                plt.savefig('fig_1_predicted_aunr.pdf', transparent = True, bbox_inches = 'tight')
                plt.show()
                spectrum = np.zeros((len(self.profiles[0][0])))
                spectrum = calculate_spectrum(self.profiles, population_matrix, spectrum)

            if fit_type in ['ar_len_matrix_rel_std', 'ar_len_matrix_rel_std_normal_x0',
                            'sphere_first_ar_len_matrix']:
                plt.contourf(self.ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 200)
                plt.xlim(2, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()
                spectrum = np.zeros((len(self.ar_profiles[0][0])))
                spectrum = calculate_spectrum(self.ar_profiles, population_matrix, spectrum)

            if fit_type in ['guided_ar_len_matrix_spheres', 'ar_len_matrix_spheres']:
                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 200)
                plt.xlim(1, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()
                spectrum = np.zeros((len(self.ar_profiles_spheres[0][0])))
                spectrum = calculate_spectrum(self.ar_profiles_spheres, population_matrix, spectrum)

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()
            """
            print(self.peak_indicies)
            # simulate spectrum
            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, skew_avg_AR,
                                              skew_std_AR)
    
            if simulation_baseline == True:
                output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                          self.peak_indicies, simulation_baseline=True)
            if simulation_baseline == False:
                output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                          self.peak_indicies, simulation_baseline=False)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix
            """
            spectrum_full = spectrum
            spectrum_full = spectrum_full/max(spectrum_full)
            if simulation_baseline == True:
                spectrum_full = spectrum_full - min(spectrum_full[0:899])
                spectrum_full = spectrum_full/max(spectrum_full)


            spectrum_new = spectrum_full[self.peak_indicies[0]:self.peak_indicies[1]]
            spectrum_final = spectrum_new / max(spectrum_new) # normalize simulated spectrum
            # plot simulated spectra with unsmoothed sample

            # print(min(spectrum_final))
            # print(simulation_baseline)

            plt.figure(figsize=(7.0, 6.0))

            plt.plot(self.wavelengths, spectrum_final, linewidth=4)
            plt.plot(self.wavelengths, self.intens[self.peak_indicies[0]:self.peak_indicies[1]], linewidth=4)
            plt.legend(["Prediction", "Sample"], fontsize=16)
            plt.title("Predicted VS Unsmoothed Sample", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()

            plt.figure(figsize=(7.0, 6.0))
            plt.plot(self.wavelengths_old, spectrum_full[0:len(self.wavelengths_old)], linewidth=4)
            plt.plot(self.wavelengths_old, self.intens[0:len(self.wavelengths_old)], linewidth=4)
            plt.legend(["Prediction", "Sample"], fontsize=16)
            plt.title("Predicted VS Unsmoothed Sample", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()

            plt.figure(figsize=(7.0, 6.0))
            plt.plot(self.wavelengths_old, spectrum_full[0:len(self.wavelengths_old)], linewidth=4)
            plt.plot(self.wavelengths_old, self.smoothed_intens_old[0:len(self.wavelengths_old)], linewidth=4)
            plt.legend(["Prediction", "Sample"], fontsize=16)
            plt.title("Predicted VS smoothed Sample", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()

            res = spectrum_final - self.intens[self.peak_indicies[0]:self.peak_indicies[1]]
            # and residuals between them
            plt.figure(figsize=(7.0, 6.0))

            plt.plot(self.wavelengths, res, linewidth=2)
            plt.title("Residuals", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()

            # plot simulated spectra with smoothed sample (if any smoothing was done)
            if self.spectrum_smoothed == True:
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Smoothed Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

                res_smoothed = spectrum_final - self.smoothed_intens
                # and residuals
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, res_smoothed, linewidth=2)
                plt.title("Residuals", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)

            self.simulated = spectrum_final
        if type(population_matrix) == type(None):
            if fit_type == 'len_dia_correlation':
                avg_len = fit_parameters[0]
                std_len = fit_parameters[1]
                avg_dia = fit_parameters[2]
                std_dia = fit_parameters[3]
                correlation = fit_parameters[4]
                cov = correlation * std_len * std_dia
                distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                                [[std_dia ** 2, cov],
                                                                 [cov, std_len ** 2]])
            if fit_type == 'len_dia_correlation_rel_std':
                avg_len = fit_parameters[0]
                std_len = fit_parameters[2]*avg_len
                avg_dia = fit_parameters[1]
                std_dia = fit_parameters[2]*avg_dia
                correlation = fit_parameters[3]
                cov = correlation * std_len * std_dia
                distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                                [[std_dia ** 2, cov],
                                                                 [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
            # ar_std)

            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix=[],
                                                                   peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_final = spectrum
            self.simulated = spectrum_final
            # spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.axhline(y=0.0, color='r', linestyle='-', linewidth=3)
                plt.show()
            res = spectrum_final - self.intens[self.peak_indicies[0]:self.peak_indicies[1]]
            res_smoothed = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            sigmas = []
            for i in range(0, len(self.wavelengths)):
                sigmas.append(0.01)
            sigmas = np.asarray(sigmas)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(fit_parameters))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi)


        if self.spectrum_smoothed == False:
            return [self.simulated, res]
        if self.spectrum_smoothed == True:
            return [self.simulated, res, res_smoothed]


    def fit_len_dia_correlation(self, x0, bounds=([15, 1, 10, 0.5, 0.2], [180, 50, 45, 20, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False, store_fits = False, rebaseline = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        fits = []
        print(store_fits)
        if rebaseline == True:
            self.smoothed_intens = self.smoothed_intens - min(self.smoothed_intens)
            self.smoothed_intens = self.smoothed_intens/max(self.smoothed_intens)
        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, correlation):

            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            # print(population_matrix.shape)
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_final = spectrum
            # spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.axhline(y=0.0, color='r', linestyle='-', linewidth = 3)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if store_fits == True:
                fits.append([avg_len, std_len, avg_dia, std_dia, correlation, reduced_chi])
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi)

            # print(min(spectrum_final))
            # print(simulation_baseline)
            return spectrum_final
        # self.smoothed_intens = self.smoothed_intens - min(self.smoothed_intens)
        # self.smoothed_intens = self.smoothed_intens/max(self.smoothed_intens)
        # plt.plot(self.wavelengths, self.smoothed_intens)
        # plt.show()
        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))
            if self.show_sim == True:
                self.sim('len_dia_correlation', self.population_matrix, simulation_baseline)
            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0, fits]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'

    def fit_len_dia_correlation_spheres(self, x0, bounds=([15, 1, 10, 0.5, 0.2], [180, 50, 45, 20, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        self.sphere_dias = np.arange(10,101,1)
        self.sphere_matrix_ars = np.arange(1,1.6,0.1)

        avg_len = 47.2
        avg_dia = 9.8

        def skew_gauss_profiles(Es, std_len, std_dia, correlation, avg_sphere_dia, std_sphere_dia,
                                avg_sphere_ar, std_sphere_ar, sphere_correlation, sphere_ratio):
            cov = correlation * std_len * std_dia
            sphere_cov = sphere_correlation*std_sphere_dia*std_sphere_ar

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])

            sphere_distributions = self.bivariate_gaussian_len_dia(self.sphere_matrix_ars, self.sphere_dias, [avg_sphere_dia, avg_sphere_ar],
                                                            [[std_sphere_dia ** 2, sphere_cov],
                                                             [sphere_cov, std_sphere_ar ** 2]])

            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]

            population_matrix_sphere = sphere_distributions[0]
            """
            plt.contourf(self.sphere_matrix_ars, self.sphere_dias, population_matrix_sphere, cm='viridis')
            # plt.xlim(10, 200)
            # plt.ylim(5, 40)
            plt.xlabel("Lengths (nm)", fontsize=16)
            plt.ylabel("Diameters (nm)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title("Rods Population Pre AR Adjustment", fontsize=18)
            plt.colorbar()
            plt.show()
            """
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            # print(population_matrix_sphere.shape)
            # print(self.sphere_profiles.shape)
            spectrum_sphere = self.create_spectrum_bivariate_longitudinal(population_matrix_sphere, self.sphere_profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_sphere_extracted = spectrum_sphere[0]

            # plt.plot(self.wavelengths, spectrum_sphere_extracted)
            # plt.plot(self.wavelengths, spectrum)

            spectrum_final = spectrum + np.asarray(spectrum_sphere_extracted)*sphere_ratio

            spectrum_final = spectrum_final / max(spectrum_final)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.axhline(y=0.0, color='r', linestyle='-', linewidth = 3)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, avg_sphere_dia, std_sphere_dia,
                                avg_sphere_ar, std_sphere_ar, sphere_correlation, sphere_ratio, reduced_chi)

            return spectrum_final
        # self.smoothed_intens = self.smoothed_intens - min(self.smoothed_intens)
        # self.smoothed_intens = self.smoothed_intens/max(self.smoothed_intens)
        # plt.plot(self.wavelengths, self.smoothed_intens)
        # plt.show()

        self.smoothed_intens = self.smoothed_intens - min(self.smoothed_intens)
        self.smoothed_intens = self.smoothed_intens/max(self.smoothed_intens)
        plt.plot(self.wavelengths, self.smoothed_intens)
        plt.show()

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=10000, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))
            if self.show_sim == True:
                self.sim('len_dia_correlation', self.population_matrix, simulation_baseline)
            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'







    def fit_len_dia_correlation_skew(self, x0, bounds=([15, 1, 10, 0.5, 0.2, -5, -5], [180, 50, 45, 20, 0.8, 5, 5]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value




        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, correlation, skew_len, skew_dia):
            population_matrix = self.bivariate_gaussian_skew_len_dia([avg_len, avg_dia], [std_len, std_dia], correlation,
                                                                [skew_len, skew_dia])
            population_matrix = population_matrix.T
            ar_mean = avg_len/avg_dia
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, correlation, skew_len, skew_dia, reduced_chi)

            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'

    def fit_len_dia_correlation_rel_std(self, x0, bounds=([15, 10, 0.025, 0.2], [180, 45, 0.4, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False, rebaseline = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        # self.smoothed_intens = self.smoothed_intens - min(self.smoothed_intens)
        # self.smoothed_intens = self.smoothed_intens / max(self.smoothed_intens)
        if rebaseline == True:
            self.smoothed_intens = self.smoothed_intens - min(self.smoothed_intens)
            self.smoothed_intens = self.smoothed_intens/max(self.smoothed_intens)

        def skew_gauss_profiles(Es, avg_len, avg_dia, rel_std, correlation):
            std_len = rel_std*avg_len
            std_dia = rel_std*avg_dia
            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            # print(population_matrix.shape)
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.hlines(avg_dia, 10, 200, color = 'white', label = 'Diameter Mean', linestyle = ':')
                plt.vlines(avg_len, 5, 40, color = 'white', label = 'Length Mean', linestyle = '--')
                plt.plot([avg_len - 35, avg_len, avg_len + 35], [avg_dia - 6, avg_dia, avg_dia + 6], color = 'white', label = 'Correlation', linestyle = '-')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Fitted AuNR Population", fontsize=18)
                plt.legend()
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi)
            # print(min(spectrum_final))
            # print(simulation_baseline)
            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'





    def fit_len_dia_ar_matrix(self, x0, bounds=([15, 1, 10, 0.5], [180, 50, 45, 20]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value



        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia):
            cov = 0 * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, 0, self.ars, ar_mean,
                                                       ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix =ar_matrix,
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline,
                                                                   show_distributions = show_distributions)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, reduced_chi)

            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'


    def fit_len_dia_ar_matrix_std(self, x0, bounds=([15, 1, 10, 0.5, 0.1], [180, 50, 45, 20, 2]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        # time_start = datetime.datetime.now()
        # timeout_sec = datetime.timedelta(seconds=120)

        self.count = 0
        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, ar_std):
            self.count += 1
            # time_elapsed = datetime.datetime.now() - time_start
            # print(time_elapsed)
            # if time_elapsed > timeout_sec:
                # print("over run time")
            cov = 0 * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, 0, self.ars, ar_mean,
                                                       ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix =ar_matrix,
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline,
                                                                   show_distributions = show_distributions)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, reduced_chi, self.count)

            return spectrum_final
        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200,
                               bounds=bounds,
                               p0=x0)
            self.fitted_params = kopt
            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
            print('reduced chisq = ' + str(self.reduced_chi))

            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population()  # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]

        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'






    def fit_len_dia_correlation_ar_matrix(self, x0, bounds=([15, 1, 10, 0.5, 0.2], [180, 50, 45, 20, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value


        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, correlation):
            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, 0, self.ars, ar_mean,
                                                       ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = ar_matrix,
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline,
                                                                   show_distributions=show_distributions)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi)

            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'



    def fit_len_dia_correlation_ar_matrix_skew(self, x0, bounds=([15, 1, 10, 0.5, -3, 0.2], [180, 50, 45, 20, 3, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value



        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, ar_skew, correlation):
            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = ar_matrix,
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline,
                                                                   show_distributions=show_distributions)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, ar_skew, correlation, reduced_chi)

            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'



    def fit_ar_len_population_matrix(self, x0, bounds=([15, 1, 3, 0.1, 0.2], [180, 50, 9, 2, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es, avg_len, std_len, avg_AR, std_AR, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.profile_ars, self.long_dims, [avg_len, avg_AR],
                                                            [[std_len ** 2, cov],
                                                             [cov, std_AR ** 2]])

            population_matrix = distributions[0]
            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 200)
                plt.xlim(2, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix, self.ar_profiles,
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("(mean len, std len, mean AR, std AR, correlation)")
                print(self.reduced_chi)

            print('reduced chisq = ' + str(self.reduced_chi))
            cov = kopt[4] * kopt[1] * kopt[3]
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [kopt[2], kopt[0]],
                                                            [[kopt[3] ** 2, cov],
                                                             [cov, kopt[1] ** 2]])
            dia_mean = distributions[1]
            dia_std = distributions[2]
            self.true_distributions = [kopt[0], kopt[1], dia_mean, dia_std, kopt[2], kopt[3]]
            print(self.true_distributions)
            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'


    def fit_ar_len_population_matrix_rel_std(self, x0, bounds=([15, 1, 3, 0.2], [180, 50, 9, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es, avg_len, std_len, avg_AR, correlation):
            std_AR = (std_len/avg_len)*avg_AR
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.profile_ars, self.long_dims, [avg_len, avg_AR],
                                                            [[std_len ** 2, cov],
                                                             [cov, std_AR ** 2]])

            population_matrix = distributions[0]
            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.profile_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 200)
                plt.xlim(2, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix, self.ar_profiles,
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("(mean len, std len, mean AR, std AR, correlation)")
                print(self.reduced_chi)

            print('reduced chisq = ' + str(self.reduced_chi))
            cov = kopt[3] * kopt[1] * kopt[2] * (kopt[1] / kopt[0])
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [kopt[2], kopt[0]],
                                                            [[(kopt[2] * (kopt[1] / kopt[0])) ** 2, cov],
                                                             [cov, kopt[1] ** 2]])
            dia_mean = distributions[1]
            dia_std = distributions[2]
            self.true_distributions = [kopt[0], kopt[1], dia_mean, dia_std, kopt[2], kopt[2] * (kopt[1] / kopt[0])]
            print(self.true_distributions)

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'






    def fit_spheres_ar_len_population_matrix(self, x0, bounds=([15, 2, 0.1, 0.1, 0, 0.2],
                                                               [180, 9, 2, 25, 9, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value
        avg_spheres = 19
        def skew_gauss_profiles_fit_ars_directly(Es, avg_len, avg_AR, std_AR, std_spheres, ratio, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            std_len = avg_len*(std_AR/avg_AR)
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.sphere_ars, self.long_dims, [avg_len, avg_AR],
                                                            [[std_len ** 2, cov],
                                                             [cov, std_AR ** 2]])
            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = distributions[0]
            sphere_pop = []
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))
            sphere_pop = np.asarray(sphere_pop)*ratio

            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 100)
                plt.xlim(0.5, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix, self.ar_profiles_spheres,
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, avg_spheres, std_spheres, ratio, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("(mean len, std len, mean AR, std AR, correlation)")
                print(self.reduced_chi)

            print(kopt)
            print('reduced chisq = ' + str(self.reduced_chi))
            self.true_distributions = kopt
            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'

    def fit_spheres_first_ar_len_population_matrix(self, x0, sphere_ratio, bounds=([15, 2, 0.1, 0.2],
                                                               [180, 9, 2, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):

        show_final_plot = False
        avg_spheres = 20
        std_spheres = 0.18

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es, avg_len, avg_AR, std_AR, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            std_len = avg_len*(std_AR/avg_AR)
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.sphere_ars, self.long_dims, [avg_len, avg_AR],
                                                            [[std_len ** 2, cov],
                                                             [cov, std_AR ** 2]])
            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = distributions[0]
            sphere_pop = []
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))
            sphere_pop = np.asarray(sphere_pop)*sphere_ratio

            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 100)
                plt.xlim(0.5, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix, self.ar_profiles_spheres,
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, avg_spheres, std_spheres, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=80000, bounds=bounds,
                                   p0=x0)
        self.fitted_params = kopt

        if print_kopt == True:
            print("(mean len, std len, mean AR, std AR, correlation)")
            print(self.reduced_chi)

        print(kopt)
        print('reduced chisq = ' + str(self.reduced_chi))
        self.true_distributions = kopt

        if show_final_plot == True:
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(self.population_matrix,
                                                                                    self.ar_profiles_spheres,
                                                                                    peak_range=self.peak_indicies,
                                                                                    simulation_baseline=simulation_baseline,
                                                                                    show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            max_simulated = max(spectrum_new)
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            spheres_distribution = np.zeros(len(self.long_dims))

            # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
            # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
            for i in range(0, len(self.population_matrix)):
                spheres_distribution[i] = self.population_matrix[i][0]

            sphere_spectrum = np.zeros(len(self.wavelengths_old))
            for i in range(0, len(spheres_distribution)):
                sphere_spectrum += self.ar_profiles_spheres[i][0][self.peak_indicies_old[0]:self.peak_indicies_old[1]] * \
                                   spheres_distribution[i]
            sphere_spectrum = sphere_spectrum / max_simulated

            spectrum_rods_only = self.ar_population_create_spectrum_bivariate_longitudinal(
                self.population_matrix[:, 5:len(self.population_matrix[0])],
                self.ar_profiles_spheres[:, 5:len(self.population_matrix[0])],
                peak_range=self.peak_indicies_old,
                simulation_baseline=simulation_baseline,
                show_distributions=show_distributions)
            print(self.peak_indicies_old)
            spectrum_rods_only = spectrum_rods_only[0]
            spectrum_rods_only = spectrum_rods_only / max_simulated
            # plot simulated spectrum on top of sample if requested
            plt.figure(figsize=(7.0, 6.0))

            plt.plot(self.wavelengths_old, spectrum_final, linewidth=4, label='Prediction', color='#1f77b4')
            plt.plot(self.wavelengths_old, self.smoothed_intens_old, linewidth=4, label='Sample', color='red')

            plt.plot(self.wavelengths_old, sphere_spectrum, color='darkorange', linestyle='--', label='Pure Spheres',
                     linewidth=3)
            plt.plot(self.wavelengths_old, spectrum_rods_only, color='black', linestyle='--', label='Pure Rods',
                     linewidth=3)
            plt.legend(fontsize=16)
            # plt.legend(["Prediction", "Sample"], fontsize=16)
            plt.title("Predicted VS Sample", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()

        return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]



    def fit_jakob_sphere_method(self, x0, sphere_ratio, sphere_sizes, fit_h12 = False, bounds=([15, 1, 7, 0.5, 0.2],
                                                                              [180, 50, 45, 20, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):




        # show_plot = True
        # print_params = True
        show_final_plot = False
        avg_sphere_dia = sphere_sizes[0]
        std_sphere_dia = sphere_sizes[1]
        avg_sphere_ar = sphere_sizes[2]
        std_sphere_ar = sphere_sizes[3]
        sphere_correlation = sphere_sizes[4]
        sphere_sizes = list(sphere_sizes)
        sphere_sizes.append(sphere_ratio)
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # print(len(sigmas))
        # print(len(self.smoothed_intens))
        # print(len(self.wavelengths))
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        if fit_h12 == True:
            # h12_wavelengths = np.arange(401, 1300, 1)
            # print(max(h12_wavelengths))
            # plt.plot(h12_wavelengths, self.h12)
            # plt.title('h12 test')
            # plt.show()
            if self.already_set_h12 == False:
                h12_begining = 0
                h12_end = 899
                self.h12 = list(self.h12)
                if self.peak_indicies[1] > 899:
                    diff = self.peak_indicies[1] - 899
                    for i in range(0, diff):
                        self.h12.append(min(self.h12))
                    h12_end = h12_end + diff
                if self.peak_indicies[1] < 899:
                    diff = 899 - self.peak_indicies[1]
                    h12_end = h12_end - diff

                if self.peak_indicies[0] != 0:
                    h12_begining = h12_begining + self.peak_indicies[0]

                print(h12_begining, h12_end)
                self.h12 = self.h12[h12_begining:h12_end]
                self.h12 = self.h12 - min(self.h12)
                self.h12 = self.h12 / max(self.h12)
                # plt.plot(self.wavelengths_old, self.h12)
                # plt.show()
                self.already_set_h12 = True
        print(['x0 = ', x0])
        def skew_gauss_profiles_h12(Es, avg_len, std_len, avg_dia, std_dia, correlation, h12_ratio):

            sphere_cov = sphere_correlation * std_sphere_dia * std_sphere_ar

            sphere_distributions = self.bivariate_gaussian_len_dia(self.sphere_matrix_ars, self.sphere_dias,
                                                                   [avg_sphere_dia, avg_sphere_ar],
                                                                   [[std_sphere_dia ** 2, sphere_cov],
                                                                    [sphere_cov, std_sphere_ar ** 2]])

            population_matrix_sphere = sphere_distributions[0]
            spectrum_sphere = self.create_spectrum_bivariate_longitudinal(population_matrix_sphere,
                                                                          self.sphere_profiles, ar_matrix=[],
                                                                          peak_range=self.peak_indicies,
                                                                          simulation_baseline=simulation_baseline)

            spectrum_final = spectrum_sphere[0]
            sphere_spectrum_final = (spectrum_final / max(spectrum_final))*sphere_ratio  # normalize simulated spectrum


            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]
            h12_spectrum = self.h12*h12_ratio
            spectrum_final = spectrum + sphere_spectrum_final + h12_spectrum
            spectrum_final = spectrum_final/max(spectrum_final)
            spectrum_final = spectrum_final
            # print(simulation_baseline)
            if simulation_baseline == True:
                spectrum_final = spectrum_final - min(spectrum_final)
                spectrum_final = spectrum_final/max(spectrum_final)
            # spectrum_final = spectrum_final - (max(spectrum_final) - 1)
            # spectrum_final = spectrum_final / max(spectrum_final)
            # print(min(spectrum_final))
            # print(max(spectrum_final))
            self.population_matrix = population_matrix

            if show_plot == True:
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths_old, spectrum_final, linewidth=4, label='Prediction', color='#1f77b4')
                plt.plot(self.wavelengths_old, self.smoothed_intens, linewidth=4, label='Sample',
                         color='red')

                plt.plot(self.wavelengths_old, sphere_spectrum_final, color='darkorange',
                         linestyle='--', label='Pure Spheres',
                         linewidth=3)
                plt.plot(self.wavelengths_old, spectrum, color='black', linestyle='--',
                         label='Pure Rods',
                         linewidth=3)
                plt.plot(self.wavelengths_old, h12_spectrum, color='purple', linestyle='--',
                         label='Growth Solution',
                         linewidth=3)
                plt.legend(fontsize=16)
                # plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi, h12_ratio, max(spectrum_final))

            return spectrum_final

        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, correlation):


            sphere_cov = sphere_correlation * std_sphere_dia * std_sphere_ar

            sphere_distributions = self.bivariate_gaussian_len_dia(self.sphere_matrix_ars, self.sphere_dias,
                                                                   [avg_sphere_dia, avg_sphere_ar],
                                                                   [[std_sphere_dia ** 2, sphere_cov],
                                                                    [sphere_cov, std_sphere_ar ** 2]])

            population_matrix_sphere = sphere_distributions[0]
            spectrum_sphere = self.create_spectrum_bivariate_longitudinal(population_matrix_sphere,
                                                                          self.sphere_profiles, ar_matrix=[],
                                                                          peak_range=self.peak_indicies,
                                                                          simulation_baseline=simulation_baseline)

            spectrum_final = spectrum_sphere[0]
            sphere_spectrum_final = (spectrum_final / max(spectrum_final))*sphere_ratio  # normalize simulated spectrum


            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]
            spectrum_final = spectrum + sphere_spectrum_final
            spectrum_final = spectrum_final/max(spectrum_final)
            spectrum_final = spectrum_final
            # print(simulation_baseline)
            if simulation_baseline == True:
                spectrum_final = spectrum_final - min(spectrum_final)
                spectrum_final = spectrum_final/max(spectrum_final)
            # spectrum_final = spectrum_final - (max(spectrum_final) - 1)
            # spectrum_final = spectrum_final / max(spectrum_final)
            # print(min(spectrum_final))
            # print(max(spectrum_final))
            self.population_matrix = population_matrix

            if show_plot == True:
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths_old, spectrum_final, linewidth=4, label='Prediction', color='#1f77b4')
                plt.plot(self.wavelengths_old, self.smoothed_intens, linewidth=4, label='Sample',
                         color='red')

                plt.plot(self.wavelengths_old, sphere_spectrum_final, color='darkorange',
                         linestyle='--', label='Pure Spheres',
                         linewidth=3)
                plt.plot(self.wavelengths_old, spectrum, color='black', linestyle='--',
                         label='Pure Rods',
                         linewidth=3)
                plt.plot(self.wavelengths_old, h12_spectrum, color='purple', linestyle='--',
                         label='Growth Solution',
                         linewidth=3)
                plt.legend(fontsize=16)
                # plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            self.reduced_chi = reduced_chi
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi, max(spectrum_final))

            return spectrum_final


        if fit_h12 == False:
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=80000, bounds=bounds,
                                       p0=x0)
        if fit_h12 == True:
            kopt, kcov = curve_fit(skew_gauss_profiles_h12, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=80000, bounds=bounds,
                                       p0=x0)
        self.fitted_params = kopt

        if print_kopt == True:
            print("(mean len, std len, mean AR, std AR, correlation)")
            print(self.reduced_chi)

        print('reduced chisq = ' + str(self.reduced_chi))
        print("Calculating true distributions from population matrix")
        self.true_distributions = self.calc_size_distributions_from_population()  # calculate the true size distributions
        # by using population matrix, as these are not the same as the parameters which went into curve fit!
        print(self.true_distributions)
        if show_final_plot == True:
            sphere_cov = sphere_correlation * std_sphere_dia * std_sphere_ar

            sphere_distributions = self.bivariate_gaussian_len_dia(self.sphere_matrix_ars, self.sphere_dias,
                                                                   [avg_sphere_dia, avg_sphere_ar],
                                                                   [[std_sphere_dia ** 2, sphere_cov],
                                                                    [sphere_cov, std_sphere_ar ** 2]])

            population_matrix_sphere = sphere_distributions[0]
            spectrum_sphere = self.create_spectrum_bivariate_longitudinal(population_matrix_sphere,
                                                                          self.sphere_profiles, ar_matrix=[],
                                                                          peak_range=self.peak_indicies,
                                                                          simulation_baseline=simulation_baseline)

            spectrum_final = spectrum_sphere[0]
            sphere_spectrum_final = (spectrum_final / max(
                spectrum_final)) * sphere_ratio  # normalize simulated spectrum

            spectrum = self.create_spectrum_bivariate_longitudinal(self.population_matrix, self.profiles, ar_matrix=[],
                                                                   peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)
            spectrum = spectrum[0]
            if fit_h12 == True:
                spectrum_final = spectrum + sphere_spectrum_final + self.h12*kopt[len(kopt)-1]
            else:
                spectrum_final = spectrum + sphere_spectrum_final

            spectrum_final = spectrum_final/max(spectrum_final)


            if simulation_baseline == True:
                spectrum_final = spectrum_final - min(spectrum_final)
            # spectrum_final = spectrum_final - (max(spectrum_final) - 1)
                spectrum_final = spectrum_final / max(spectrum_final)



            plt.figure(figsize=(7.0, 6.0))

            plt.plot(self.wavelengths_old, spectrum_final, linewidth=4, label='Prediction', color='#1f77b4')
            plt.plot(self.wavelengths_old, self.smoothed_intens, linewidth=4, label='Sample',
                     color='red')

            plt.plot(self.wavelengths_old, sphere_spectrum_final, color='darkorange',
                     linestyle='--', label='Pure Spheres',
                     linewidth=3)
            plt.plot(self.wavelengths_old, spectrum, color='black', linestyle='--',
                     label='Pure Rods',
                     linewidth=3)
            if fit_h12 == True:
                plt.plot(self.wavelengths_old, self.h12*kopt[len(kopt)-1], color='purple', linestyle='--',
                         label='Growth Solution',
                         linewidth=3)
            plt.legend(fontsize=16)
            # plt.legend(["Prediction", "Sample"], fontsize=16)
            plt.title("Predicted VS Sample", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
            print(max(self.smoothed_intens))
            plt.figure(figsize=(7.0, 6.0))
            plt.plot(self.wavelengths, spectrum_final, linewidth=4)
            plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
            plt.legend(["Prediction", "Sample"], fontsize=16)
            plt.title("Predicted VS Sample", fontsize=28)
            plt.xlabel("Wavelength (nm)", fontsize=22)
            plt.ylabel("Absorbance", fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()



        return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0, sphere_sizes]





    def fit_spheres_distributions(self, x0, area_around_peak = 75, bounds=([15, 0.1],[100, 25]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):



        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        top = area_around_peak*2 + 1
        # sigmas = np.asarray(sigmas[0:top])
        center_index = list(self.wavelengths).index(self.transverse_peak_wavelength)
        low_point = center_index - area_around_peak
        high_point = center_index + area_around_peak + 1
        low_point = 100
        high_point = 175
        sigmas = np.asarray(sigmas[100:175])
        self.sphere_dias = np.arange(10,101,1)
        self.sphere_matrix_ars = np.arange(1,1.6,0.1)
        def skew_gauss_profiles(Es, avg_sphere_dia, std_sphere_dia,
                                avg_sphere_ar, std_sphere_ar, sphere_correlation):

            sphere_cov = sphere_correlation*std_sphere_dia*std_sphere_ar


            sphere_distributions = self.bivariate_gaussian_len_dia(self.sphere_matrix_ars, self.sphere_dias, [avg_sphere_dia, avg_sphere_ar],
                                                            [[std_sphere_dia ** 2, sphere_cov],
                                                             [sphere_cov, std_sphere_ar ** 2]])

            population_matrix_sphere = sphere_distributions[0]

            if show_distributions == True:
                plt.contourf(self.sphere_dias, self.sphere_matrix_ars, population_matrix_sphere.T, cm='viridis')
                # plt.xlim(10, 200)
                # plt.ylim(5, 40)
                plt.xlabel("Diameter (nm)", fontsize=16)
                plt.ylabel("AR", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Sphere Size Distribution", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            # print(population_matrix_sphere.shape)
            # print(self.sphere_profiles.shape)
            spectrum_sphere = self.create_spectrum_bivariate_longitudinal(population_matrix_sphere, self.sphere_profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies,
                                                                   simulation_baseline=simulation_baseline)



            spectrum_final = spectrum_sphere[0]
            spectrum_final = spectrum_final[low_point:high_point]
            spectrum_final = spectrum_final/max(spectrum_final)

            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths[low_point:high_point], spectrum_final, linewidth=4)
                plt.plot(self.wavelengths[low_point:high_point], self.smoothed_intens[low_point:high_point]/max(self.smoothed_intens[low_point:high_point]), linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens[low_point:high_point]
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_sphere_dia, std_sphere_dia,
                                avg_sphere_ar, std_sphere_ar, sphere_correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(bounds)
        print(x0)
        kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths[low_point:high_point],
                               self.smoothed_intens[low_point:high_point]/max(self.smoothed_intens[low_point:high_point]), sigma=sigmas, maxfev=80000, bounds=bounds,p0=x0)
        self.fitted_params = kopt

        if print_kopt == True:
            print("(mean len, std len, mean AR, std AR, correlation)")
            print(self.reduced_chi)

        print(kopt)
        print('reduced chisq = ' + str(self.reduced_chi))
        self.true_distributions = kopt


        return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]








    def fit_spheres(self, x0, area_around_peak = 5, bounds=([15, 0.1],[100, 25]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):



        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        top = area_around_peak*2 + 1
        sigmas = np.asarray(sigmas[0:top])
        center_index = list(self.wavelengths).index(self.transverse_peak_wavelength)
        low_point = center_index - area_around_peak
        high_point = center_index + area_around_peak + 1
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es, avg_spheres, std_spheres):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values


            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = np.zeros((len(self.ar_profiles_spheres), len(self.ar_profiles_spheres[0])))
            sphere_pop = []
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))
            sphere_pop = np.asarray(sphere_pop)

            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 100)
                plt.xlim(0.5, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix, self.ar_profiles_spheres,
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum[low_point:high_point]
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths[low_point:high_point], spectrum_final, linewidth=4)
                plt.plot(self.wavelengths[low_point:high_point], self.smoothed_intens[low_point:high_point]/max(self.smoothed_intens[low_point:high_point]), linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens[low_point:high_point]
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_spheres, std_spheres, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(bounds)
        print(x0)
        kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths[low_point:high_point],
                               self.smoothed_intens[low_point:high_point]/max(self.smoothed_intens[low_point:high_point]), sigma=sigmas, maxfev=80000, bounds=bounds,p0=x0)
        self.fitted_params = kopt

        if print_kopt == True:
            print("(mean len, std len, mean AR, std AR, correlation)")
            print(self.reduced_chi)

        print(kopt)
        print('reduced chisq = ' + str(self.reduced_chi))
        self.true_distributions = kopt

        output_list = self.ar_population_create_spectrum_bivariate_longitudinal(self.population_matrix,
                                                                                self.ar_profiles_spheres,
                                                                                peak_range=self.peak_indicies,
                                                                                simulation_baseline=simulation_baseline,
                                                                                show_distributions=show_distributions)
        population_matrix = output_list[1]
        spectrum = output_list[0]
        self.population_matrix = population_matrix

        spectrum_new = spectrum
        max_simulated = max(spectrum_new)
        spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
        spheres_distribution = np.zeros(len(self.long_dims))

        # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
        # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
        for i in range(0, len(self.population_matrix)):
            spheres_distribution[i] = self.population_matrix[i][0]

        sphere_spectrum = np.zeros(len(self.wavelengths_old))
        for i in range(0, len(spheres_distribution)):
            sphere_spectrum += self.ar_profiles_spheres[i][0][self.peak_indicies_old[0]:self.peak_indicies_old[1]] * \
                               spheres_distribution[i]
        sphere_spectrum = sphere_spectrum / max_simulated

        spectrum_rods_only = self.ar_population_create_spectrum_bivariate_longitudinal(
            self.population_matrix[:, 5:len(self.population_matrix[0])],
            self.ar_profiles_spheres[:, 5:len(self.population_matrix[0])],
            peak_range=self.peak_indicies_old,
            simulation_baseline=simulation_baseline,
            show_distributions=show_distributions)
        print(self.peak_indicies_old)
        spectrum_rods_only = spectrum_rods_only[0]
        spectrum_rods_only = spectrum_rods_only / max_simulated
        # plot simulated spectrum on top of sample if requested
        plt.figure(figsize=(7.0, 6.0))

        plt.plot(self.wavelengths, spectrum_final, linewidth=4, label='Prediction', color='#1f77b4')
        plt.plot(self.inputted_wavelengths, self.inputted_intens, linewidth=4, label='Sample', color='red')

        plt.plot(self.wavelengths, sphere_spectrum, color='darkorange', linestyle='--', label='Pure Spheres',
                 linewidth=3)
        plt.plot(self.wavelengths, spectrum_rods_only, color='black', linestyle='--', label='Pure Rods',
                 linewidth=3)
        plt.legend(fontsize=16)
        plt.legend(["Prediction", "Sample"], fontsize=16)
        plt.title("Predicted VS Sample", fontsize=28)
        plt.xlabel("Wavelength (nm)", fontsize=22)
        plt.ylabel("Absorbance", fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]


    def fit_spheres_std(self, x0, avg_spheres, area_around_peak = 5, bounds=([0.1],[25]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):



        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        top = area_around_peak*2 + 1
        sigmas = np.asarray(sigmas[0:top])
        center_index = list(self.wavelengths).index(self.transverse_peak_wavelength)
        low_point = center_index - area_around_peak
        high_point = center_index + area_around_peak + 1
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es, std_spheres):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values


            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = np.zeros((len(self.ar_profiles_spheres), len(self.ar_profiles_spheres[0])))
            sphere_pop = []
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))
            sphere_pop = np.asarray(sphere_pop)

            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 100)
                plt.xlim(0.5, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix, self.ar_profiles_spheres,
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum[low_point:high_point]
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths[low_point:high_point], spectrum_final, linewidth=4)
                plt.plot(self.wavelengths[low_point:high_point], self.smoothed_intens[low_point:high_point]/max(self.smoothed_intens[low_point:high_point]), linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens[low_point:high_point]
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_spheres, std_spheres, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(bounds)
        print(x0)
        kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths[low_point:high_point],
                               self.smoothed_intens[low_point:high_point]/max(self.smoothed_intens[low_point:high_point]), sigma=sigmas, maxfev=80000, bounds=bounds,p0=x0)
        self.fitted_params = kopt

        if print_kopt == True:
            print("(mean len, std len, mean AR, std AR, correlation)")
            print(self.reduced_chi)

        print(kopt)
        print('reduced chisq = ' + str(self.reduced_chi))
        self.true_distributions = kopt

        output_list = self.ar_population_create_spectrum_bivariate_longitudinal(self.population_matrix,
                                                                                self.ar_profiles_spheres,
                                                                                peak_range=self.peak_indicies,
                                                                                simulation_baseline=simulation_baseline,
                                                                                show_distributions=show_distributions)
        population_matrix = output_list[1]
        spectrum = output_list[0]
        self.population_matrix = population_matrix

        spectrum_new = spectrum
        max_simulated = max(spectrum_new)
        spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
        spheres_distribution = np.zeros(len(self.long_dims))

        # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
        # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
        for i in range(0, len(self.population_matrix)):
            spheres_distribution[i] = self.population_matrix[i][0]

        sphere_spectrum = np.zeros(len(self.wavelengths_old))
        for i in range(0, len(spheres_distribution)):
            sphere_spectrum += self.ar_profiles_spheres[i][0][self.peak_indicies_old[0]:self.peak_indicies_old[1]] * \
                               spheres_distribution[i]
        sphere_spectrum = sphere_spectrum / max_simulated

        spectrum_rods_only = self.ar_population_create_spectrum_bivariate_longitudinal(
            self.population_matrix[:, 5:len(self.population_matrix[0])],
            self.ar_profiles_spheres[:, 5:len(self.population_matrix[0])],
            peak_range=self.peak_indicies_old,
            simulation_baseline=simulation_baseline,
            show_distributions=show_distributions)
        print(self.peak_indicies_old)
        spectrum_rods_only = spectrum_rods_only[0]
        spectrum_rods_only = spectrum_rods_only / max_simulated
        # plot simulated spectrum on top of sample if requested

        plt.figure(figsize=(7.0, 6.0))

        plt.plot(self.wavelengths_old, spectrum_final, linewidth=4, label='Prediction', color='#1f77b4')
        plt.plot(self.wavelengths_old, self.smoothed_intens_old, linewidth=4, label='Sample', color='red')

        plt.plot(self.wavelengths_old, sphere_spectrum, color='darkorange', linestyle='--', label='Pure Spheres',
                 linewidth=3)
        plt.plot(self.wavelengths_old, spectrum_rods_only, color='black', linestyle='--', label='Pure Rods',
                 linewidth=3)
        plt.legend(fontsize=16)
        # plt.legend(["Prediction", "Sample"], fontsize=16)
        plt.title("Predicted VS Sample", fontsize=28)
        plt.xlabel("Wavelength (nm)", fontsize=22)
        plt.ylabel("Absorbance", fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]



    def guided_fit_spheres_ar_len_population_matrix(self, x0, avg_len, avg_AR, bounds=([0.1, 15, 0.1, 0, 0.2],
                                                               [2, 90, 25, 9, 0.8]),
                                             print_kopt=True, show_plot=False,
                                             print_params=False, simulation_baseline=False, show_distributions=False):

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths_old)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)



        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es, std_AR, avg_spheres, std_spheres, ratio,
                                                 correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            std_len = avg_len * (std_AR / avg_AR)
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.sphere_ars, self.long_dims,
                                                                          [avg_len, avg_AR],
                                                                          [[std_len ** 2, cov],
                                                                           [cov, std_AR ** 2]])
            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = distributions[0]

            sphere_pop = []
            pop_sum = 0
            for row in population_matrix:
                pop_sum += sum(row)
            # print(pop_sum)
            population_matrix = population_matrix/pop_sum
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))
            sphere_pop = np.asarray(sphere_pop)
            # print(sum(sphere_pop), 'pre adjustment')
            # plt.plot(self.long_dims[0:91], sphere_pop, label = 'original')
            sphere_pop = np.asarray(sphere_pop)/sum(sphere_pop)
            # print(sum(sphere_pop), 'normalized')
            # plt.plot(self.long_dims[0:91], sphere_pop, label = 'normalized')
            sphere_pop = sphere_pop * ratio
            # print(sum(sphere_pop), 'scaled')
            # plt.plot(self.long_dims[0:91], sphere_pop, label = 'scaled')
            # plt.legend()
            # plt.show()


            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            final_sum = 0
            for row in population_matrix:
                final_sum += sum(row)
            # print(final_sum, 'total sum population matrix')
            # print(sum(sphere_pop)/final_sum, 'ratio spheres')
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                print(len(population_matrix), len(population_matrix[0]))
                plt.figure(figsize=(7.0, 6.0))
                plt.contourf(self.long_dims, self.sphere_ars, population_matrix.T, cm='viridis')
                plt.xlim(10, 100)
                plt.ylim(2, 10)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix,
                                                                                    self.ar_profiles_spheres,
                                                                                    peak_range=self.peak_indicies_old,
                                                                                    simulation_baseline=simulation_baseline,
                                                                                    show_distributions=show_distributions)
            # print(self.peak_indicies_old)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            max_simulated = max(spectrum_new)
            spectrum_final = spectrum_new / max_simulated  # normalize simulated spectrum
            if show_plot == True:
                spheres_distribution = np.zeros(len(self.long_dims))


                # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
                for i in range(0, len(population_matrix)):
                    spheres_distribution[i] = population_matrix[i][0]

                sphere_spectrum = np.zeros(len(self.wavelengths_old))
                for i in range(0, len(spheres_distribution)):
                    sphere_spectrum += self.ar_profiles_spheres[i][0][self.peak_indicies_old[0]:self.peak_indicies_old[1]]*spheres_distribution[i]
                sphere_spectrum = sphere_spectrum/max_simulated

                spectrum_rods_only = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix[:, 5:len(population_matrix[0])],
                                                                                    self.ar_profiles_spheres[:, 5:len(population_matrix[0])],
                                                                                    peak_range=self.peak_indicies_old,
                                                                                    simulation_baseline=simulation_baseline,
                                                                                    show_distributions=show_distributions)
                print(self.peak_indicies_old)
                spectrum_rods_only = spectrum_rods_only[0]
                spectrum_rods_only = spectrum_rods_only/max_simulated
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths_old, spectrum_final, linewidth=4, label = 'Prediction', color = '#1f77b4')
                plt.plot(self.wavelengths_old, self.smoothed_intens_old, linewidth=4, label = 'Sample', color = 'red')

                plt.plot(self.wavelengths_old, sphere_spectrum, color = 'darkorange', linestyle = '--', label = 'Pure Spheres', linewidth = 3)
                plt.plot(self.wavelengths_old, spectrum_rods_only, color = 'black', linestyle = '--', label = 'Pure Rods', linewidth = 3)
                plt.legend(fontsize = 16)
                # plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens_old
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, avg_spheres, std_spheres, ratio, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths_old, self.smoothed_intens_old, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("(mean len, std len, mean AR, std AR, correlation)")
                print(self.reduced_chi)

            print(kopt)
            print('reduced chisq = ' + str(self.reduced_chi))
            self.true_distributions = kopt
            print(len(self.population_matrix), len(self.population_matrix[0]))
            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'

    def fix_spheres_fit_ar_len_population_matrix(self, x0, avg_spheres, std_spheres, ratio, bounds=([0.1, 15, 0.1, 0, 0.2],
                                                               [2, 90, 25, 9, 0.8]),
                                             print_kopt=True, show_plot=False,
                                             print_params=False, simulation_baseline=False, show_distributions=False):

        # kopts = [] eventually set up a way to store all the values the fit tries
        # return [longitudinal_peak, longitudinal_peak_wavelengths, [peak_index_begining, peak_index_end], err]

        # longitudinal_set = self.find_longitudinal(self.smoothed_intens_old, self.wavelengths_old, 0.3, old = True)
        # self.smoothed_intens = longitudinal_set[0]
        # self.wavelengths = longitudinal_set[1]
        # self.peak_indicies = longitudinal_set[2]

        sigmas = []
        for i in range(0, len(self.wavelengths_old)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)



        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es,avg_len, avg_AR, std_AR, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            std_len = avg_len * (std_AR/avg_AR)

            # calculate diameter values
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.sphere_ars, self.long_dims,
                                                                          [avg_len, avg_AR],
                                                                          [[std_len ** 2, cov],
                                                                           [cov, std_AR ** 2]])
            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = distributions[0]
            sphere_pop = []
            pop_sum = 0
            for row in population_matrix:
                pop_sum += sum(row)
            population_matrix = population_matrix/pop_sum
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))

            sphere_pop = np.asarray(sphere_pop)/sum(sphere_pop)
            sphere_pop = sphere_pop * ratio

            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                print(len(population_matrix), len(population_matrix[0]))
                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 100)
                plt.xlim(2, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix,
                                                                                    self.ar_profiles_spheres,
                                                                                    peak_range=self.peak_indicies_old,
                                                                                    simulation_baseline=simulation_baseline,
                                                                                    show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                spheres_distribution = np.zeros(len(self.long_dims))


                # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
                for i in range(0, len(population_matrix)):
                    spheres_distribution[i] = population_matrix[i][0]

                sphere_spectrum = np.zeros(len(self.wavelengths_old))
                for i in range(0, len(spheres_distribution)):
                    sphere_spectrum += self.ar_profiles_spheres[i][0][self.peak_indicies_old[0]:self.peak_indicies_old[1]]*spheres_distribution[i]
                sphere_spectrum = sphere_spectrum/max(sphere_spectrum)
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))
                plt.plot(self.wavelengths_old, sphere_spectrum, color = 'red', linestyle = '--')
                plt.plot(self.wavelengths_old, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths_old, self.smoothed_intens_old, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens_old
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, avg_spheres, std_spheres, ratio, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'


    def half_guided_fit_spheres_ar_len_population_matrix(self, x0, bounds=([0.1, 15, 0.1, 0, 0.2],
                                                               [2, 90, 25, 9, 0.8]),
                                             print_kopt=True, show_plot=False,
                                             print_params=False, simulation_baseline=False, show_distributions=False):

        print('starting half guided')
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths_old)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)



        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_fit_ars_directly(Es,avg_len, avg_AR, std_len, std_AR, avg_spheres, std_spheres, ratio,
                                                 correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            # std_len = avg_len * (std_AR / avg_AR)
            cov = correlation * std_len * std_AR

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.ar_population_bivariate_gaussian_len_dia(self.sphere_ars, self.long_dims,
                                                                          [avg_len, avg_AR],
                                                                          [[std_len ** 2, cov],
                                                                           [cov, std_AR ** 2]])
            # need to update:
            # profiles matrix to include spheres (right now just adding a bunch of AR=1.5 rods)
            population_matrix = distributions[0]
            sphere_pop = []
            for val in self.long_dims[0:91]:
                sphere_pop.append(gaussian(val, 1, avg_spheres, std_spheres))
            sphere_pop = np.asarray(sphere_pop) * ratio

            for i in range(0, 90):
                population_matrix[i][0] = sphere_pop[i]

            # print(self.profiles.shape)
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment

                plt.contourf(self.sphere_ars, self.long_dims, population_matrix, cm='viridis')
                plt.ylim(10, 200)
                plt.xlim(0.5, 8)
                plt.ylabel("Lengths (nm)", fontsize=16)
                plt.xlabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.ar_population_create_spectrum_bivariate_longitudinal(population_matrix,
                                                                                    self.ar_profiles_spheres,
                                                                                    peak_range=self.peak_indicies_old,
                                                                                    simulation_baseline=simulation_baseline,
                                                                                    show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths_old, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths_old, self.smoothed_intens_old, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens_old
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, avg_spheres, std_spheres, ratio, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final

        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'

    def fit_ar_len_rel_std_skew(self, x0, bounds=([15, 2, 0.025, -3], [180, 9, 0.4, 3]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_ar_len(Es, avg_len, skew_avg_AR, rel_std, ar_skew):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)
            avg_AR = skew_avg_AR + (skew_avg_AR*rel_std) * (ar_skew / (np.sqrt(1 + ar_skew ** 2))) * np.sqrt(2 / np.pi)
            std_AR = np.sqrt(((skew_avg_AR*rel_std) ** 2) * (1 - (2 * (ar_skew / (np.sqrt(1 + ar_skew ** 2)))) / np.pi))

            std_len = avg_len * rel_std

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            avg_dia = avg_len / avg_AR
            std_dia = avg_dia*rel_std

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            cov_len_dia = 0 * std_len * std_dia  # treating correlation between length and diameter as zero and allowing
            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov_len_dia],
                                                             [cov_len_dia, std_len ** 2]])

            population_matrix = distributions[0]
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, 0, self.ars, skew_avg_AR,
                                              std_AR)

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                      self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_dia, std_dia, avg_AR, std_AR, ar_skew, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))

            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(max(self.smoothed_intens))
        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_ar_len, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'


    def fit_ar_len_rel_std(self, x0, bounds=([15, 2, 0.025], [180, 9, 0.4]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_ar_len(Es, avg_len, avg_AR, rel_std):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)
            std_AR = avg_AR*rel_std

            std_len = avg_len * rel_std

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            avg_dia = avg_len / avg_AR
            std_dia = avg_dia*rel_std

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            cov_len_dia = 0 * std_len * std_dia  # treating correlation between length and diameter as zero and allowing
            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov_len_dia],
                                                             [cov_len_dia, std_len ** 2]])

            population_matrix = distributions[0]
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, 0, self.ars, avg_AR,
                                              std_AR)

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                      self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_dia, std_dia, avg_AR, std_AR, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))

            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(max(self.smoothed_intens))
        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_ar_len, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'



    def fit_ar_len_one_rel_std(self, x0, bounds=([15, 0.1, 2], [180, 50, 9]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_ar_len(Es, avg_len, std_len, avg_AR):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)
            std_AR = avg_AR*std_len/avg_len

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            avg_dia = avg_len / avg_AR
            std_dia = avg_dia*std_len/avg_len

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            cov_len_dia = 0 * std_len * std_dia  # treating correlation between length and diameter as zero and allowing
            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov_len_dia],
                                                             [cov_len_dia, std_len ** 2]])

            population_matrix = distributions[0]
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, 0, self.ars, avg_AR,
                                              std_AR)

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                      self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_dia, std_dia, avg_AR, std_AR, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))

            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(max(self.smoothed_intens))
        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_ar_len, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'




    def fit_ar_len(self, x0, bounds=([15, 1, 3, 0.1, -3, 0.2], [180, 50, 9, 2, 3, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, simulation_baseline=False, show_distributions = False):
        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
        # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value

        def skew_gauss_profiles_ar_len(Es, avg_len, std_len, skew_avg_AR, skew_std_AR, ar_skew, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)
            avg_AR = skew_avg_AR + skew_std_AR * (ar_skew / (np.sqrt(1 + ar_skew ** 2))) * np.sqrt(2 / np.pi)
            std_AR = np.sqrt((skew_std_AR ** 2) * (1 - (2 * (ar_skew / (np.sqrt(1 + ar_skew ** 2)))) / np.pi))

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            cov = correlation * std_len * std_AR
            avg_dia = avg_len / avg_AR
            std_dia = self.calc_diameter_stdev([avg_AR, avg_len], [[std_AR ** 2, cov], [cov, std_len ** 2]])

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            cov_len_dia = 0 * std_len * std_dia  # treating correlation between length and diameter as zero and allowing
            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov_len_dia],
                                                             [cov_len_dia, std_len ** 2]])

            population_matrix = distributions[0]
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, skew_avg_AR,
                                              skew_std_AR)

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                      self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_dia, std_dia, avg_AR, std_AR, ar_skew, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # print('reduced chisq = ' + str(self.reduced_chi))

            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final



        print(max(self.smoothed_intens))
        try:
            kopt, kcov = curve_fit(skew_gauss_profiles_ar_len, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=200, bounds=bounds,
                                       p0=x0)
            self.fitted_params = kopt

            if print_kopt == True:
                print("Fitted Parameters, NOT true distributions from population matrix")
                print(kopt)
                print(self.reduced_chi)
            print("Calculating true distributions from population matrix")
            self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
            # by using population matrix, as these are not the same as the parameters which went into curve fit!
            print(self.true_distributions)
            print('reduced chisq = ' + str(self.reduced_chi))

            return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
        except RuntimeError:
            print("Error - curve_fit failed")
            return 'runtime error'





    def create_x0s(self, fit_style = 'len_dia_correlation', x0s = None):
        # if x0s are specified at the beginning, sets initial guess attribute to them. Otherwise, uses default values
        if type(x0s) != type(None):
            self.x0 = x0s


        elif x0s == None:

            if fit_style == 'fit_spheres':
                self.x0 = ([12, 2],
                           [50,5])
            if fit_style == 'fit_spheres_temp':
                self.x0 = ([20, 2, 1.2, 0.2, 0.4],
                           [40,4, 1.2, 0.2, 0.4])

            if fit_style == 'guided_ar_len_matrix_spheres':
                print('creating x0')
                self.x0 = ([self.best_fit[4][5], 30, 5, 0.0, 0.4],
                           [self.best_fit[4][5], 30, 5, 0.01, 0.4])
                           # [round(self.ar_prediction*0.2, 3), 30, 5, 0.01,  0.4],
                           # [round(self.ar_prediction*0.2, 3), 30, 5, 0.1, 0.4],
                           # [round(self.ar_prediction*0.1, 3), 60, 10, 0.01, 0.4],
                           # [round(self.ar_prediction*0.1, 3), 60, 10, 0.01, 0.4])
            if fit_style == 'ar_len_one_rel_std':
                self.x0 = ([50, 2.5, round(self.ar_prediction, 2)],
                           [50, 5, round(self.ar_prediction, 2)],
                           [50, 10, round(self.ar_prediction, 2)],
                           [100, 5, round(self.ar_prediction, 2)],
                           [100, 10, round(self.ar_prediction, 2)],
                           [100, 20, round(self.ar_prediction, 2)],
                           [150, 7.5, round(self.ar_prediction, 2)],
                           [150, 15, round(self.ar_prediction, 2)],
                           [150, 30, round(self.ar_prediction, 2)])

            if fit_style == 'ar_len_matrix_spheres':
                self.x0 = ([30, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 5, 0.0, 0.4],
                           [60, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 5, 0.0, 0.4],
                           [90, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 5, 0.0,  0.4],
                           [120, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 5, 0.0, 0.4],
                           [150, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 5, 0.0, 0.4])

            if fit_style == 'sphere_first_ar_len_matrix':
                self.x0 = ([30, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [60, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [90, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [120, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [150, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4])

            if fit_style == 'len_ar_rel_std':
                self.x0 = ([50, round(self.ar_prediction, 2), 0.05],
                    [50, round(self.ar_prediction, 2), 0.1],
                           [50, round(self.ar_prediction, 2), 0.2],
                       [100, round(self.ar_prediction, 2), 0.05],
                    [100, round(self.ar_prediction, 2), 0.1],
                           [100, round(self.ar_prediction, 2), 0.2],
                       [150, round(self.ar_prediction, 2), 0.05],
                       [150, round(self.ar_prediction, 2), 0.1],
                           [150, round(self.ar_prediction, 2), 0.2])

            if fit_style == 'ar_len_matrix_rel_std':
                self.x0 = ([30, 3, round(self.ar_prediction, 2),0.4],
                           [60, 6, round(self.ar_prediction, 2), 0.4],
                           [90, 9, round(self.ar_prediction, 2), 0.4],
                           [120, 12, round(self.ar_prediction, 2), 0.4],
                           [150, 15, round(self.ar_prediction, 2), 0.4])

            if fit_style == 'ar_len_matrix_rel_std_normal_x0':
                self.x0 = ([50, 2.5, round(self.ar_prediction, 2),0.4],
                           [50, 5, round(self.ar_prediction, 2), 0.4],
                           [50, 10, round(self.ar_prediction, 2), 0.4],
                           [100, 5, round(self.ar_prediction, 2), 0.4],
                           [100, 10, round(self.ar_prediction, 2), 0.4],
                           [100, 15, round(self.ar_prediction, 2), 0.4],
                           [150, 7.5, round(self.ar_prediction, 2), 0.4],
                           [150, 15, round(self.ar_prediction, 2), 0.4],
                           [150, 30, round(self.ar_prediction, 2), 0.4])

            if fit_style == 'simultaneous_len_dia_sphere':
                self.x0 = ([[5, 1, 0.21,
                             20, 2, 1.2, 0.3, 0.2, 0.0001],
                            [75, 9, round(75 / self.ar_prediction, 2), round(75 / self.ar_prediction, 2) * 0.2, 0.4,
                             20, 2, 1.2, 0.2, 0.2, 0.0001],
                            [150, 24, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.2, 0.4,
                             20, 2, 1.2, 0.2, 0.2, 0.0001]])


            if fit_style == 'len_ar_rel_std_skew':
                self.x0 = ([50, round(self.ar_prediction, 2), 0.05, 0],
                    [50, round(self.ar_prediction, 2), 0.1, 0],
                           [50, round(self.ar_prediction, 2), 0.2, 0],
                       [100, round(self.ar_prediction, 2), 0.05, 0],
                    [100, round(self.ar_prediction, 2), 0.1, 0],
                           [100, round(self.ar_prediction, 2), 0.2, 0],
                       [150, round(self.ar_prediction, 2), 0.05, 0],
                       [150, round(self.ar_prediction, 2), 0.1, 0],
                           [150, round(self.ar_prediction, 2), 0.2, 0])


            if fit_style == 'len_dia_correlation_rel_std':
                self.x0 = ([50, round(50/self.ar_prediction, 2), 0.05, 0.4],
                    [50, round(50/self.ar_prediction, 2), 0.1, 0.4],
                           [50, round(50 / self.ar_prediction, 2), 0.2, 0.4],
                       [100, round(100/self.ar_prediction, 2), 0.05, 0.4],
                    [100, round(100/self.ar_prediction, 2), 0.1, 0.4],
                           [100, round(100 / self.ar_prediction, 2), 0.2, 0.4],
                       [150, round(150/self.ar_prediction, 2), 0.05, 0.4],
                       [150, round(150/self.ar_prediction, 2), 0.1, 0.4],
                           [150, round(150 / self.ar_prediction, 2), 0.2, 0.4])

            if fit_style == 'ar_len_matrix':
                self.x0 = ([50, 2.5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [50, 5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [50, 10, round(self.ar_prediction, 2), round(self.ar_prediction * 0.2, 3), 0.4],
                           [100, 5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [100, 10, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [100, 20, round(self.ar_prediction, 2), round(self.ar_prediction * 0.2, 3), 0.4],
                           [150, 7.5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [150, 15, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0.4],
                           [150, 30, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 0.4])

            if fit_style == 'ar_len':
                self.x0 = ([50, 2.5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0, 0.4],
                           [50, 5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0, 0.4],
                           [50, 10, round(self.ar_prediction, 2), round(self.ar_prediction * 0.2, 3), 0, 0.4],
                           [100, 5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0, 0.4],
                           [100, 10, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0, 0.4],
                           [100, 20, round(self.ar_prediction, 2), round(self.ar_prediction * 0.2, 3), 0, 0.4],
                           [150, 7.5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0, 0.4],
                           [150, 15, round(self.ar_prediction, 2), round(self.ar_prediction*0.1, 3), 0, 0.4],
                           [150, 30, round(self.ar_prediction, 2), round(self.ar_prediction*0.2, 3), 0, 0.4])

            if fit_style == 'len_dia_ar_matrix_std':
                self.x0 = ([50, 2.5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1,
                            round(self.ar_prediction*0.1, 2)],
                           [50, 5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1,
                            round(self.ar_prediction*0.1, 2)],
                           [50, 10, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.2,
                            round(self.ar_prediction * 0.2, 2)],
                           [100, 5, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1,
                            round(self.ar_prediction*0.1, 2)],
                           [100, 10, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1,
                            round(self.ar_prediction*0.1, 2)],
                           [100, 20, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.2,
                            round(self.ar_prediction * 0.2, 2)],
                           [150, 7.5, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1,
                            round(self.ar_prediction*0.1, 2)],
                           [150, 15, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1,
                            round(self.ar_prediction*0.1, 2)],
                           [150, 30, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.2,
                            round(self.ar_prediction * 0.2, 2)])

            if fit_style == 'len_dia_ar_matrix':
                self.x0 = ([50, 2.5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1],
                           [50, 5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1],
                           [50, 10, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.2],
                           [100, 5, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1],
                           [100, 10, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1],
                           [100, 20, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.2],
                           [150, 7.5, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1],
                           [150, 15, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1],
                           [150, 30, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1])

            if fit_style == 'len_dia_correlation_skew':
                self.x0 = ([50, 2.5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1, 0.4,0,0],
                            [50, 5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1, 0.4,0,0],
                            [50, 10, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.2, 0.4, 0,0],
                            [100, 5, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1, 0.4,0,0],
                            [100, 10, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1,
                             0.4,0,0],
                            [100, 20, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.2,
                             0.4, 0, 0],
                            [150, 7.5, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1,
                             0.4,0,0],
                            [150, 15, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1,
                             0.4,0,0],
                            [150, 30, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.2,
                             0.4, 0, 0])
            if fit_style == 'len_dia_correlation' or fit_style == 'len_dia_correlation_ar_matrix':
                self.x0 = ([50, 2.5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1, 0.4],
                            [50, 5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1, 0.4],
            [50, 10, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.2, 0.4],
                [100, 5, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1, 0.4],
                [100, 10, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1, 0.4],
            [100, 20, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.2, 0.4],
                [150, 7.5, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1, 0.4],
            [150, 15, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1, 0.4],
            [150, 30, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.2, 0.4])


            if fit_style == 'len_dia_correlation_h12':
                self.x0 = ([50, 2.5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1, 0.4, 0],
                            [50, 5, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.1, 0.4, 0],
            [50, 10, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.2, 0.4, 0],
                [100, 5, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1, 0.4, 0],
                [100, 10, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.1, 0.4, 0],
            [100, 20, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.2, 0.4, 0],
                [150, 7.5, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1, 0.4, 0],
            [150, 15, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.1, 0.4, 0],
            [150, 30, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.2, 0.4, 0])

            if fit_style == 'len_dia_correlation_ar_matrix_skew':
                self.x0 = ([50, 2.5, round(50/self.ar_prediction, 2), round(50/self.ar_prediction, 2)*0.1, 0, 0.4],
                    [50, 5, round(50/self.ar_prediction, 2), round(50/self.ar_prediction, 2)*0.1, 0, 0.4],
                        [50, 10, round(50 / self.ar_prediction, 2), round(50 / self.ar_prediction, 2) * 0.2, 0, 0.4],
                       [100, 5, round(100/self.ar_prediction, 2), round(100/self.ar_prediction, 2)*0.1, 0, 0.4],
                    [100, 10, round(100/self.ar_prediction, 2), round(100/self.ar_prediction, 2)*0.1, 0, 0.4],
                        [100, 20, round(100 / self.ar_prediction, 2), round(100 / self.ar_prediction, 2) * 0.2, 0,0.4],
                           [150, 7.5, round(150/self.ar_prediction, 2), round(150/self.ar_prediction, 2)*0.1, 0, 0.4],
                       [150, 15, round(150/self.ar_prediction, 2), round(150/self.ar_prediction, 2)*0.1, 0, 0.4],
                    [150, 30, round(150 / self.ar_prediction, 2), round(150 / self.ar_prediction, 2) * 0.2, 0, 0.4])


        return self.x0


    def create_bounds(self, fit_style = 'len_dia_correlation', low_ar = 2, high_ar = 9, bounds = None):

        if bounds != None:
            bounds = bounds

        else:
            if fit_style == 'fit_spheres':
                bounds = ([10, 1], [100, 25])

            if fit_style == 'guided_ar_len_matrix_spheres':
                bounds = [[0.1, 15, 0.1, 0, 0.2], [2, 90, 25, 9, 0.8]]

            if fit_style == 'ar_len_one_rel_std':
                bounds = ([15, 1, low_ar], [180, 50, high_ar])

            if fit_style == 'ar_len_matrix_spheres':
                bounds = ([15, low_ar, 0.1, 2, 0, 0.2], [180, high_ar, 2, 6.3, 9, 0.8])

            if fit_style == 'sphere_first_ar_len_matrix':
                bounds = ([15, low_ar, 0.1, 0.2], [180, high_ar, 2, 0.8])

            if fit_style == 'len_ar_rel_std':
                bounds = ([15, low_ar, 0.025], [180, high_ar, 0.4])

            if fit_style in ['ar_len_matrix_rel_std', 'ar_len_matrix_rel_std_normal_x0']:
                bounds = ([15, 1, low_ar, 0.2], [180, 50, high_ar, 0.8])

            if fit_style == 'simultaneous_len_dia_sphere':
                bounds = ([1, 0.5, 0.2, 15, 0.6, 1.0, 0.25, 0.2, 0], [50, 20, 0.8, 95, 25, 1.6, 0.8, 0.8, 10])

            if fit_style == 'len_ar_rel_std_skew':
                bounds = ([15, low_ar, 0.025, -3], [180, high_ar, 0.4, 3])

            if fit_style == 'len_dia_correlation_rel_std':
                bounds = ([12, 6, 0.025, 0.2], [180, 45, 0.4, 0.8])

            if fit_style == 'ar_len_matrix':
                bounds = ([15, 1, low_ar, 0.1, 0.2], [180, 50, high_ar, 2, 0.8])

            if fit_style == 'ar_len':
                bounds = [[15, 1, low_ar, 0.1, -3, 0.2], [180, 50, high_ar, 2, 3, 0.8]]

            if fit_style == 'len_dia_ar_matrix_std':
                bounds = ([15, 1, 5, 0.5, 0.1], [180, 50, 50, 20, 2])

            if fit_style == 'len_dia_ar_matrix':
                bounds = ([15, 1, 5, 0.5], [180, 50, 50, 20])

            if fit_style == 'len_dia_correlation_skew':
                bounds = ([12, 0.1, 6, 0.1, 0.2, -5, -5], [180, 50, 50, 20, 0.8, 5, 5])

            if fit_style in ['len_dia_correlation', 'len_dia_correlation_ar_matrix']:
                bounds = ([12, 1, 6, 0.5, 0.2], [180, 50, 50, 20, 0.8])

            if fit_style == 'len_dia_correlation_ar_matrix_skew':
                bounds = ([15, 1, 5, 0.5, -3, 0.2], [180, 50, 50, 20, 3, 0.8])
        return bounds


    def check_x0s(self, x0, bounds):
        bad_bound = False
        count = 0
        for param in x0:
            if bad_bound == False:
                if param < bounds[0][count] or param > bounds[1][count]:
                    bad_bound = True
            count += 1

        return bad_bound



    def full_fit(self,
                 print_kopt=False, show_plot=False,
                 print_params=False, fit_style='len_dia_correlation', simulation_baseline=False, show_distributions=False,
                 store_fits = False, sphere_ratio = None, rebaseline_spectrum = False, fit_h12 = False):
        """
        An outer function for fit which runs the fit method over all initial guesses and stores all those that don't
        run into any of the bounds (besides correlation between len and AR since every fit seems to hit this bound
        no matter what pretty much). Finds the fit with the lowest chisq and labeles that as the best fit, but returns
        all non bound hitting fits. I suspect this will end up being important, as a couple fits have very similar chisqs
        so it may not be accurate to only keep the lowest one. TODO may want some logic here to check for very strange
        TODO distributions, (ie length distribution lower than diameter) even if any bounds aren't hit
        :return:
        """
        if self.bad_spectrum == True:
            return 'Fit not attempted, longitudinal too small'
        self.fits = []
        self.bad_fits = []
        chisqs = []
        if self.ar_prediction > 5:
            low_ar = round(self.ar_prediction - 3, 1)
        else:
            low_ar = 2

        if self.ar_prediction < 6:
            high_ar = round(self.ar_prediction + 3, 1)
        else:
            high_ar = 9
        print(self.x0)
        print([low_ar, high_ar])

        if fit_style == 'guided_ar_len_matrix_spheres':
            self.create_x0s('len_dia_correlation')
            self.fits = ['first pass, len_dia_correlation']
            for x0 in self.x0:
                hit_bound = False
                bad_bound = False
                bounds = ([12, 1, 6, 0.5, 0.2], [180, 50, 50, 20, 0.8])
                fit_temp = self.fit_len_dia_correlation(x0, bounds, print_kopt, show_plot, print_params,
                                                                     simulation_baseline, show_distributions)
                # print("chisq = " + str(fit_temp[3]))
                for i in range(0, len(bounds[0]) - 1):
                    if round(fit_temp[1][i], 2) == bounds[0][i] or round(fit_temp[1][i], 2) == bounds[1][i]:
                        print('Hit bound' + str(i))
                        hit_bound = True
                print(fit_temp[3])
                if fit_temp[3] > 1000:
                    hit_bound = True
                if hit_bound == False:
                    self.fits.append(fit_temp)
                    chisqs.append(fit_temp[3])
                    print("hit bound = " + str(hit_bound))
                if hit_bound == True:
                    self.bad_fits.append(fit_temp)


            if len(chisqs) != 0:
                best_fit_index = chisqs.index(min(chisqs))
                self.best_fit = self.fits[best_fit_index]
                print("true parameters of best fit")
                print(self.best_fit)
                print(self.best_fit[4])
                self.create_x0s('guided_ar_len_matrix_spheres')
                chisqs = []
                self.fits = []
                print(self.x0)
                for x0 in self.x0:
                    hit_bound = False
                    # bounds = [self.best_fit[4][0]-0.2*self.best_fit[4][0],
                              # self.best_fit[4][4]-0.2*self.best_fit[4][4], 1, 0.1, 15, 0.1, 0, 0.2], \
                             # [self.best_fit[4][0]+0.2*self.best_fit[4][0], self.best_fit[4][4]+0.2*self.best_fit[4][4],
                             # 30, 2, 90, 25, 9, 0.8]
                    bounds = [[0.1, 15, 0.1, 0, 0.2], [2, 90, 25, 9, 0.8]]
                    # print(bounds)
                    print(self.best_fit[4][0], self.best_fit[4][4])
                    fit_temp = self.guided_fit_spheres_ar_len_population_matrix(x0, self.best_fit[4][0], self.best_fit[4][4],
                                                                                bounds, print_kopt, show_plot = False,
                                                                         print_params = False,
                                                                         simulation_baseline = False, show_distributions = False)
                    # fit_temp = self.guided_fit_spheres_ar_len_population_matrix(x0, self.best_fit[4][0], self.best_fit[4][4],
                                                                                # bounds, print_kopt, show_plot=True, print_params=True,
                                                                         # simulation_baseline=False, show_distributions=False)
                # print("chisq = " + str(fit_temp[3]))
                    # for i in range(0, len(bounds[0]) - 4):
                        # if round(fit_temp[1][i], 5) == bounds[0][i] or round(fit_temp[1][i], 5) == bounds[1][i]:
                            # hit_bound = True
                    # print(fit_temp)
                    if fit_temp[3] > 200:
                        hit_bound = True
                    avg_ar = self.best_fit[4][4]
                    avg_len = self.best_fit[4][0]
                    fit_temp_new = [avg_len, avg_ar]
                    for i in range(0, len(fit_temp[1])):
                        fit_temp_new.append(fit_temp[1][i])

                    fit_temp[1] = fit_temp_new
                    if hit_bound == False:
                        self.fits.append(fit_temp)
                        chisqs.append(fit_temp[3])
                    print("hit bound = " + str(hit_bound))

                # print(len(chisqs))

                # avg_ar = self.best_fit[4][4]
                # avg_len = self.best_fit[4][0]

                # print(avg_ar, avg_len, std_len)

            if len(chisqs) != 0:
                best_fit_index = chisqs.index(min(chisqs))
                self.best_fit = self.fits[best_fit_index]
                print(self.best_fit)
                print("true parameters of best fit")

                print(self.best_fit[4])
                return [self.best_fit, self.fits, self.bad_fits]

            if len(chisqs) == 0:
                print("all fits hit bounds, fit unsuccessful")
                return ['Failed Fit, all fits hit bounds', self.bad_fits]


            """
            if len(chisqs) != 0:
                best_fit_index = chisqs.index(min(chisqs))
                self.best_fit = self.fits[best_fit_index]
                std_ar = self.best_fit[4][0]
                self.create_x0s(x0s=[[avg_len, avg_ar, std_ar, 0.8]])
                print("true parameters of best fit")
                print(self.best_fit[4])
                print('made it here')
                print(self.x0)
                hit_bound = False
                # self.create_x0s('fix_spheres_ar_len_matrix_rel_std')
                chisqs = []
                for x0 in self.x0:
                    bounds = ([15, low_ar, 0.1, 0.2], [180, high_ar, 2, 0.8])
                    fit_temp = self.fix_spheres_fit_ar_len_population_matrix(x0, self.best_fit[4][1], self.best_fit[4][2],
                                                                              self.best_fit[4][3], bounds, print_kopt,
                                                                             show_plot = True, print_params = True,
                                                                         simulation_baseline = False, show_distributions = False)
                    # print("chisq = " + str(fit_temp[3]))
                    # for i in range(0, len(bounds[0]) - 1):
                        # if round(fit_temp[1][i], 5) == bounds[0][i] or round(fit_temp[1][i], 5) == bounds[1][i]:
                            # hit_bound = True
                    if fit_temp[3] > 200:
                        hit_bound = True
                    if hit_bound == False:
                        self.fits.append(fit_temp)
                        chisqs.append(fit_temp[3])
                        print("hit bound = " + str(hit_bound))
            """
        if fit_style == 'fit_spheres_temp':
            chisqs = []
            for x0 in self.x0:
                bounds = ([12, 0.5, 1, 0.1, 0.2], [95, 25, 1.6, 0.3, 0.8])
                self.bounds = bounds
                print(x0)
                fit_temp = self.fit_spheres_distributions(x0, 30, bounds, print_kopt, show_plot, print_params, simulation_baseline,
                                            show_distributions)

                self.fits.append(fit_temp)
                chisqs.append(fit_temp[3])
            best_fit_index = chisqs.index(min(chisqs))
            sphere_distribution = self.fits[best_fit_index][1]
            print(sphere_distribution)

            chisqs = []
            self.fits = []
            if fit_h12 == False:
                print('creating x0')
                self.create_x0s('len_dia_correlation')
                bounds = ([12, 1, 6, 0.5, 0.2], [180, 50, 50, 20, 0.8])
            if fit_h12 == True:
                print('creating x0')
                self.create_x0s('len_dia_correlation_h12')
                bounds = ([12, 1, 6, 0.5, 0.2, 0], [180, 50, 50, 20, 0.8, 1])

            for x0 in self.x0:
                hit_bound = False  # tracks whether the fit hit any of the bound
                count = 0
                bad_bound = False
                for param in x0:
                    if bad_bound == False:
                        if param < bounds[0][count] or param > bounds[1][count]:
                            bad_bound = True
                    count += 1
                if bad_bound == False:
                    fit_temp = self.fit_jakob_sphere_method(x0, sphere_ratio, sphere_distribution, fit_h12, bounds, show_plot=show_plot,
                                                            print_params=print_params,
                                                            simulation_baseline=simulation_baseline, show_distributions=show_distributions)
                    # print("chisq = " + str(fit_temp[3]))
                    if fit_temp[3] > 200:
                        hit_bound = True
                    for i in range(0, len(bounds[0]) - 2):
                        if round(fit_temp[1][i], 3) == bounds[0][i] or round(fit_temp[1][i], 3) == bounds[1][i]:
                            hit_bound = True
                if bad_bound == True:
                    fit_temp = [x0, 'X0 outside of bounds']

                if hit_bound == False and bad_bound == False:
                    self.fits.append(fit_temp)
                    chisqs.append(fit_temp[3])
                    print("hit bound = " + str(hit_bound))

                if hit_bound == False and bad_bound == True:
                    self.fits.append(fit_temp)
                    chisqs.append(10000)
                if hit_bound == True:
                    print(fit_temp[1])
                    self.bad_fits.append(fit_temp)
                print(len(chisqs))
            if len(chisqs) != 0:
                if min(chisqs) < 1000:
                    best_fit_index = chisqs.index(min(chisqs))
                    self.best_fit = self.fits[best_fit_index]
                    print("true parameters of best fit")
                    # print(self.best_fit)
                    print(self.best_fit[4])
            elif len(chisqs) == 0:
                self.best_fit = None
                print("true parameters of best fit")
                # print(self.best_fit)
                print('No best fit, all fits hit bounds')
            return [self.best_fit, self.fits, self.bad_fits, self.bounds]

        if fit_style == 'fit_spheres':
            chisqs = []
            for x0 in self.x0:
                bounds = ([10, 1], [100, 25])
                print(x0)
                fit_temp = self.fit_spheres(x0, 5, bounds, print_kopt, show_plot, print_params, simulation_baseline,
                                                show_distributions)

                self.fits.append(fit_temp)
                chisqs.append(fit_temp[3])
            best_fit_index = list(chisqs).index(min(chisqs))
            best_fit_mean = self.fits[best_fit_index][1][0]
            self.x0 = [best_fit_mean*0.3]
            bounds = ([2], [best_fit_mean/3])
            fit_temp = self.fit_spheres_std(self.x0, best_fit_mean, 100, bounds, print_kopt, False, False, simulation_baseline,
                                        show_distributions)
            print(fit_temp[1][0])
            sphere_distribution = [best_fit_mean, fit_temp[1][0]]
            print("sphere mean and std", sphere_distribution)

            self.create_x0s('len_dia_correlation')
            for x0 in self.x0:
                hit_bound = False  # tracks whether the fit hit any of the bounds
                bounds = ([12, 1, 6, 0.5, 0.2], [180, 50, 50, 20, 0.8])
                count = 0
                bad_bound = False
                for param in x0:
                    if bad_bound == False:
                        if param < bounds[0][count] or param > bounds[1][count]:
                            bad_bound = True
                    count += 1
                if bad_bound == False:
                    fit_temp = self.fit_jakob_sphere_method(x0, 0.7, sphere_distribution, bounds, show_plot = False, print_params = True,
                                                            simulation_baseline = False, show_distributions = False)
                    # print("chisq = " + str(fit_temp[3]))
                    if fit_temp[3] > 200:
                        hit_bound = True
                    for i in range(0, len(bounds[0]) - 1):
                        if round(fit_temp[1][i], 3) == bounds[0][i] or round(fit_temp[1][i], 3) == bounds[1][i]:
                            hit_bound = True
                if bad_bound == True:
                    fit_temp = [x0, 'X0 outside of bounds']


        if fit_style == 'sphere_first_ar_len_matrix':
            sphere_ratios = np.arange(0, 2.005, 0.001)
            # sphere_ratios = [0.0448]
            for ratio in sphere_ratios:
                print("sphere ratio = " + str(ratio))
                fit_temps = []
                for x0 in self.x0:
                    hit_bound = False  # tracks whether the fit hit any of the bounds
                    bounds = ([15, low_ar, 0.1, 0.2], [180, high_ar, 2, 0.8])
                    count = 0
                    bad_bound = False
                    for param in x0:
                        if bad_bound == False:
                            if param < bounds[0][count] or param > bounds[1][count]:
                                bad_bound = True
                        count += 1
                    if bad_bound == False:
                        fit_temp = self.fit_spheres_first_ar_len_population_matrix(x0, ratio, bounds, print_kopt, show_plot, print_params,
                                                               simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))
                        if type(fit_temp) != str:
                            # print(type(fit_temp))
                            for i in range(0, len(bounds[0])):
                                if round(fit_temp[1][i], 5) == bounds[0][i] or round(fit_temp[1][i], 5) == bounds[1][i]:
                                    hit_bound = True
                            if fit_temp[3] > 200:
                                hit_bound = True
                        if type(fit_temp) == str:
                            hit_bound = True
                    if bad_bound == True:
                        fit_temp = [x0, 'X0 outside of bounds']


                    if hit_bound == False and bad_bound == False:
                        fit_temps.append(fit_temp)
                        chisqs.append(fit_temp[3])
                        print("hit bound = " + str(hit_bound))
                    if hit_bound == False and bad_bound == True:
                        self.fits.append(fit_temp)
                        chisqs.append(100000)
                    if hit_bound == True:
                        self.bad_fits.append(fit_temp)


                # if len(chisqs) != 0:
                   # best_fit_index = chisqs.index(min(chisqs))
                    # self.best_fit = self.fits[best_fit_index]

                    # print("true parameters of best fit for ratio " + str(ratio))
                    # print(self.best_fit[4])
                    # print("chisq = " + str(self.best_fit[3]))
                self.fits.append(fit_temps)

            return [self.best_fit, self.fits, self.bad_fits]


        if fit_style not in ['guided_ar_len_matrix_spheres', 'sphere_first_ar_len_matrix', 'fit_spheres',
                             'fit_spheres_temp']:
            bounds = self.create_bounds(fit_style, low_ar, high_ar)
            self.bounds = bounds
            print(['bounds =', self.bounds])
            for x0 in self.x0:
                hit_bound = False  # tracks whether the fit hit any of the bounds
                bad_bound = self.check_x0s(x0, bounds)
                if bad_bound == True:
                    fit_temp = [x0, 'X0 outside of bounds']
                if bad_bound == False:

                    if fit_style == 'ar_len_one_rel_std':
                        fit_temp = self.fit_ar_len_one_rel_std(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)

                    if fit_style == 'simultaneous_len_dia_sphere':
                        # bounds = ([1, 0.5, 0.2, 15, 0.6, 1.0, 0.25, 0.2, 0], [50, 20, 0.8, 95, 25, 1.6, 0.8, 0.8, 10])


                        fit_temp = self.fit_len_dia_correlation_spheres(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)
                            # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'len_dia_ar_matrix_std':
                        # bounds = ([15, 1, 5, 0.5, 0.1], [180, 50, 50, 20, 2])


                        fit_temp = self.fit_len_dia_ar_matrix_std(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)
                            # print("chisq = " + str(fit_temp[3]))

                    if fit_style == 'len_ar_rel_std_skew':
                        # bounds = ([15, low_ar, 0.025, -3], [180, high_ar, 0.4, 3])


                        fit_temp = self.fit_ar_len_rel_std_skew(x0, bounds, print_kopt, show_plot, print_params,
                                                                            simulation_baseline, show_distributions)
                                # print("chisq = " + str(fit_temp[3]))

                    if fit_style == 'len_ar_rel_std':
                        # bounds=([15, low_ar, 0.025], [180, high_ar, 0.4])

                        fit_temp = self.fit_ar_len_rel_std(x0, bounds, print_kopt, show_plot, print_params,
                                                                    simulation_baseline, show_distributions)

                    if fit_style == 'len_dia_ar_matrix':
                        # bounds = ([15, 1, 5, 0.5], [180, 50, 50, 20])



                        fit_temp = self.fit_len_dia_ar_matrix(x0, bounds, print_kopt, show_plot, print_params,
                                                        simulation_baseline, show_distributions)
                            # print("chisq = " + str(fit_temp[3]))

                    if fit_style == 'len_dia_correlation_rel_std':
                        # bounds=([12, 6, 0.025, 0.2], [180, 45, 0.4, 0.8])



                        fit_temp = self.fit_len_dia_correlation_rel_std(x0, bounds, print_kopt, show_plot, print_params,
                                                          simulation_baseline, show_distributions, rebaseline_spectrum)


                    if fit_style == 'len_dia_correlation' or  fit_style == 'len_dia_correlation_temp':
                    # bounds = ([12, 1, 6, 0.5, 0.2], [180, 50, 50, 20, 0.8])

                        fit_temp = self.fit_len_dia_correlation(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions, store_fits, rebaseline_spectrum)
                        # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'len_dia_correlation_skew':
                    # bounds = ([12, 0.1, 6, 0.1, 0.2, -5, -5], [180, 50, 50, 20, 0.8,5,5])

                        fit_temp = self.fit_len_dia_correlation_skew(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'len_dia_correlation_ar_matrix':
                    # bounds = ([15, 1, 5, 0.5, 0.2], [180, 50, 50, 20, 0.8])


                        fit_temp = self.fit_len_dia_correlation_ar_matrix(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'len_dia_correlation_ar_matrix_skew':
                    # bounds = ([15, 1, 5, 0.5, -3, 0.2], [180, 50, 50, 20, 3, 0.8])


                        fit_temp = self.fit_len_dia_correlation_ar_matrix_skew(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'ar_len_matrix':
                    # bounds = ([15, 1, low_ar, 0.1, 0.2], [180, 50, high_ar, 2, 0.8])

                        fit_temp = self.fit_ar_len_population_matrix(x0, bounds, print_kopt, show_plot, print_params,
                                                    simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'ar_len_matrix_spheres':
                    # bounds = ([15, low_ar, 0.1, 2, 0, 0.2], [180, high_ar, 2, 6.3, 9, 0.8])



                        fit_temp = self.fit_spheres_ar_len_population_matrix(x0, bounds, print_kopt, show_plot,
                                                                                         print_params, simulation_baseline,
                                                                                         show_distributions)


                    if fit_style == 'ar_len_matrix_rel_std' or fit_style == 'ar_len_matrix_rel_std_normal_x0':
                    # bounds = ([15, 1, low_ar, 0.2], [180, 50, high_ar, 0.8])



                        fit_temp = self.fit_ar_len_population_matrix_rel_std(x0, bounds, print_kopt, show_plot, print_params,
                                                    simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))


                    if fit_style == 'ar_len':
                    # bounds = [[15, 1, low_ar, 0.1, -3, 0.2], [180, 50, high_ar, 2, 3, 0.8]]

                        fit_temp = self.fit_ar_len(x0, bounds, print_kopt, show_plot, print_params,
                                                      simulation_baseline, show_distributions)
                        # print("chisq = " + str(fit_temp[3]))


                    if type(fit_temp) != str:
                        if fit_style in ['ar_len_one_rel_std', 'len_ar_rel_std', 'len_dia_ar_matrix_std', 'len_dia_ar_matrix']:
                            for i in range(0, len(bounds[0])):
                                if round(fit_temp[1][i], 2) == bounds[0][i] or round(fit_temp[1][i], 2) == bounds[1][i]:
                                    hit_bound = True
                                if fit_temp[3] > 200:
                                    hit_bound = True
                        if fit_style in ['sphere_first_ar_len_matrix', 'ar_len_matrix_rel_std', 'ar_len_matrix_rel_std_normal_x0',
                     'len_ar_rel_std_skew', 'len_dia_correlation_rel_std', 'ar_len_matrix', 'len_dia_correlation',
                     'len_dia_correlation_ar_matrix']:
                            for i in range(0, len(bounds[0])-1):
                                if round(fit_temp[1][i], 2) == bounds[0][i] or round(fit_temp[1][i], 2) == bounds[1][i]:
                                    hit_bound = True
                                if fit_temp[3] > 200:
                                    hit_bound = True
                        if fit_style in ['ar_len', 'len_dia_correlation_ar_matrix_skew', 'len_dia_correlation_skew']:
                            for i in range(0, len(bounds[0])-2):
                                if round(fit_temp[1][i], 2) == bounds[0][i] or round(fit_temp[1][i], 2) == bounds[1][i]:
                                    hit_bound = True
                                if fit_temp[3] > 200:
                                    hit_bound = True
                    if type(fit_temp) == str:
                        hit_bound = True

        # uncomment to fit just one initial guess
        # print(self.ar_prediction)
        # self.best_fit = self.fit([50, 2.5, round(self.ar_prediction, 2), round(self.ar_prediction*0.1,3), 0, 0.4], bounds, print_kopt, show_plot, print_params, fit_style,
                             # simulation_baseline, show_distributions)
        # return self.best_fit
                if fit_style != 'sphere_first_ar_len_matrix':
                    if hit_bound == False and bad_bound == False:
                        self.fits.append(fit_temp)
                        chisqs.append(fit_temp[3])
                        print("hit bound = " + str(hit_bound))
                    if hit_bound == False and bad_bound == True:
                        self.fits.append(fit_temp)
                        chisqs.append(10000)
                    if hit_bound == True:
                        print(fit_temp[1])
                        self.bad_fits.append(fit_temp)
                    print(len(self.fits))
            if len(chisqs) != 0:
                if min(chisqs) < 1000:
                    best_fit_index = chisqs.index(min(chisqs))
                    self.best_fit = self.fits[best_fit_index]


                    print("true parameters of best fit")
                    # print(self.best_fit)
                    print(self.best_fit[4])
                    return [self.best_fit, self.fits, self.bad_fits, self.bounds]

                else:
                    print("all fits hit bounds, fit unsuccessful")
                    return ['Failed Fit, all fits hit bounds', self.fits, self.bad_fits, self.bounds]

            if len(chisqs) == 0:
                print("all fits hit bounds, fit unsuccessful")
                return ['Failed Fit, all fits hit bounds', self.bad_fits, self.bounds]




            # if len(chisqs) == 0:
                # print("all fits hit bounds, fit unsuccessful")
                # return ['Failed Fit, all fits hit bounds', self.bad_fits]



    def fit_frontend(self, fit_type = 'all', show_plot=False, print_params=False, show_distributions=False,
                     simulation_baseline=False, store_fits = False, sphere_ratio = None, rebaseline_spectrum = False,
                     fit_h12 = False):
        if fit_type != 'all':
            output_file = {fit_type: []}
            print(fit_type)
            if fit_type == 'guided_ar_len_matrix_spheres':
                self.create_x0s(fit_style='len_dia_correlation')
            else:
                self.create_x0s(fit_style=fit_type)

            output = self.full_fit(show_plot=show_plot,
                                   print_params=print_params,
                                   show_distributions=show_distributions,
                                   simulation_baseline=simulation_baseline,
                                   fit_style=fit_type,
                                   store_fits=store_fits,
                                   sphere_ratio = sphere_ratio,
                                   rebaseline_spectrum = rebaseline_spectrum,
                                   fit_h12 = fit_h12)
            # print(len(output[0][0]), len(output[0][0][0]))
            output_file[fit_type].append(output)

            output_object = Deconvolution_Output(output_file, self.inputted_intens, self.inputted_wavelengths,
                                             self.long_dims, self.short_dims, self.ars, self.profile_ars,
                                             self.name, self.tem_data, self.true_values, self.blue_baseline_intens,
                                             self.blue_baseline_wavelengths, self.smoothed_intens, self.wavelengths,
                                             self.longitudinal_threshold, smoothing_parameters=self.smoothing_params,
                                             red_edge_threshold=self.red_edge_threshold,
                                             blue_edge_location=self.blue_edge_location,
                                             simulation_baseline=simulation_baseline,
                                             rebaseline_spectrum = rebaseline_spectrum,
                                             fit_h12 = fit_h12)
            return output_object

        if fit_type == 'all':
            print('WARNING THIS HAS NOT BEEN UPDATED TO INCLUDE REBASELINEING SAMPLE SPECTRUM')
            # removed len_ar_rel_std_skew because it was taking forever to fit
            fit_styles = ['len_dia_correlation',
                          'ar_len',
                          'len_dia_correlation_ar_matrix',
                          'len_dia_correlation_ar_matrix_skew',
                          'len_dia_ar_matrix_std',
                          'len_dia_ar_matrix',
                          'len_dia_correlation_rel_std',
                          'len_ar_rel_std',
                          'ar_len_one_rel_std',
                          'ar_len_matrix_rel_std',
                          'ar_len_matrix_rel_std_normal_x0',
                          'ar_len_matrix']

            self.output_file = {}
            for key in fit_styles:
                self.output_file[key] = []

            for fit_type in fit_styles:
                print(fit_type)
                self.create_x0s(fit_style=fit_type)
                output = self.full_fit(show_plot=show_plot,
                                            print_params=print_params,
                                            show_distributions=show_distributions,
                                            simulation_baseline=simulation_baseline,
                                            fit_style=fit_type)
                self.output_file[fit_type].append(output)

            output_object = Deconvolution_Output(self.output_file, self.inputted_intens, self.inputted_wavelengths,
                                                 self.long_dims, self.short_dims, self.ars, self.profile_ars, self.name,
                                                 self.tem_data, self.true_values, self.blue_baseline_intens,
                                                 self.blue_baseline_wavelengths, self.smoothed_intens, self.wavelengths,
                                                 self.longitudinal_threshold, smoothing_parameters=self.smoothing_params,
                                             red_edge_threshold=self.red_edge_threshold,
                                             blue_edge_location=self.blue_edge_location,
                                             simulation_baseline=simulation_baseline)
            return output_object

    def calc_size_distributions_from_population(self):
        # helper method which calculates the true distributions of the size parameters directly from the population
        # matrix TODO add documentation
        length_distribution = []
        diameter_distribution = []
        for length in self.population_matrix.T:
            length_distribution.append(sum(length))
        for diameter in self.population_matrix:
            diameter_distribution.append(sum(diameter))
        normalize_length = np.trapz(y=length_distribution, x=self.long_dims)
        normalize_diameter = np.trapz(y=diameter_distribution, x=self.short_dims)

        self.normal_length_distribution = length_distribution / normalize_length
        self.normal_diameter_distribution = diameter_distribution / normalize_diameter


        new_len_mean = np.sum((length_distribution / normalize_length) * self.long_dims)
        new_dia_mean = np.sum((diameter_distribution / normalize_diameter) * self.short_dims)
        new_len_std = np.sqrt(np.dot(((self.long_dims - new_len_mean) ** 2), length_distribution / normalize_length))
        new_dia_std = np.sqrt(np.dot(((self.short_dims - new_dia_mean) ** 2), diameter_distribution / normalize_diameter))

        ar_matrix = np.zeros((len(self.short_dims), len(self.long_dims)))
        for i in range(len(ar_matrix)):
            for j in range(len(ar_matrix[0])):
                if self.long_dims[j] >= self.short_dims[i] * 1.5 and self.long_dims[j] <= 10 * self.short_dims[i]:
                    ar_matrix[i][j] = np.round(self.long_dims[j] / self.short_dims[i], 2)

        """
        ar_dist = []
        for i in range(0, len(self.ars)):
            # print(ar_vals[i])
            prob_temp = []
            for j in range(0, len(ar_matrix)):
                for k in range(0, len(ar_matrix[0])):
                    if ar_matrix[j][k] == np.round(self.ars[i], 2):
                        prob_temp.append(self.population_matrix[j][k])
                # print(ar_values_long[i])
            ar_dist.append(sum(prob_temp))
        """

        ar_dist = compute_ar_dist(self.ars, ar_matrix,self.population_matrix)
        ar_values_long_useful = np.arange(150, 1001, 1)

        normalize_ar = np.trapz(y=ar_dist, x=ar_values_long_useful)
        # plt.plot(ar_values_long, ar_dist / normalize_ar)
        # print(np.trapz(y=ar_dist / normalize_ar, x=ar_values_long))
        # plt.show()

        # new_len_mean = np.sum((length_distribution/normalize_length)*lengths)

        new_ar_mean = np.sum((ar_dist / normalize_ar) * ar_values_long_useful)
        new_ar_std = np.sqrt(np.dot(((ar_values_long_useful - new_ar_mean) ** 2), ar_dist / normalize_ar))
        self.normal_ar_distribution= ar_dist/normalize_ar
        # if self.true_distributions == None:
            # self.true_distributions = [new_len_mean, new_len_std, new_dia_mean, new_dia_std, new_ar_mean / 100,
                                       # new_ar_std / 100]
        return [new_len_mean, new_len_std, new_dia_mean, new_dia_std, new_ar_mean / 100, new_ar_std / 100]



    def fit_bootstrap(self, n_random = 100, fit_type = 'len_dia_correlation', show_plot = False, print_params = False,
                      show_distributions = False):
        results = []
        i = 0
        # n_failed = 0
        #if type(error_func)==type(None):
            #error_func = self.error
        # if type(sample) != list:
            # uncertainties = self.error(wavelengths=self.q_vals, intens=self.intens)

        self.create_x0s(fit_type)
        smoothed_spectrum_temp = self.smoothed_intens
        while i < n_random:
            self.smoothed_intens = smoothed_spectrum_temp
            error_dat = np.random.normal(loc = self.smoothed_intens,scale = self.err,
                                         size = len(self.wavelengths)) # calculate boostrapped sample using
            # differences between true and smoothed spectrum as the standard deviations for each wavelength point
            # error_dat = error_dat/max(error_dat) # re normalize
            self.smoothed_intens = error_dat # TODO make some logic to reset self.smoothed_intens to the smoothed
            # TODO original spectrum after the bootstrapping is done. Right now it will be stuck as whatever the
            # TODO last bootstrapped spectrum was (AND the spectrum will get steadily noisier as the boostrap is run)


            # plot boostrapped sample and difference between it and the smoothed spectrum
            plt.plot(self.wavelengths, np.abs(error_dat), label = "Bootstrapped Spectrum")
            plt.plot(self.wavelengths, np.abs(self.smoothed_intens-error_dat), label = "abs diff to smoothed sample")
            plt.legend()
            plt.show()

            # this block contained an idea that I had a while back - bootstrapping and re smoothing - but turned out
            # to not really change the spectrum at all in the end
            """
            itp = interp1d(self.q_vals, error_dat, kind='linear')
            window_size, poly_order = 101, 3
            yy_sg = savgol_filter(itp(self.q_vals), window_size, poly_order)
            plt.plot(self.q_vals, yy_sg, 'k', linestyle='--', label='Smoothed')
            plt.scatter(self.q_vals, error_dat, s=5, alpha=0.75, label="Spectrum")
            plt.title("Spectrum with Smoothing", fontsize=16)
            plt.legend(fontsize=12)
            plt.xlabel("Wavelength (nm)", fontsize=14)
            plt.ylabel("Absorption (normalized)", fontsize=14)
            plt.show()
            err = np.abs(yy_sg - error_dat)
            plt.plot(self.q_vals, err)
            plt.show()
            #plt.plot(self.q_vals,error_dat)
            #print(max(error_dat))
            """

            results_temp = []
            # run full fit (including all initial guesses) on bootstrapped spectrum.

            output = self.full_fit(fit_style=fit_type, show_plot = show_plot, print_params=print_params,
                                           show_distributions=show_distributions)
            count = 0
            # Store all non bound hitting fits, their true distributions, their chisq, and the initial guess which
            # produced them
            fit = output[0]
            prediction = fit[4]
            print(prediction, fit[3])
            # print(fit, self.reduced_chi)
            results.append(output)
            #if verbose:
                #print('Iteration ' + str(i) + ' results: ' + str(self.x0))
            #if n_failed > 100:
                #print('This fitting procedure has failed too often. Try using a more reasonable starting guess, or a more reasonable error approximation')
                #break
            # print(results_temp)
            i+=1
        self.bootstrap_results = results

        return results




class Deconvolution_Output():
    def __init__(self, fit_results, inputted_spectrum, wavelengths, long_dims, short_dims, ars, ar_len_matrix_ars,
                 name, tem_data = None, true_values = None, blue_baseline_intens = None,
                 blue_baseline_wavelengths = None, smoothed_spectrum = None, extracted_longitudinal = None,
                 longitudinal_wavelengths = None, longitudinal_threshold = None, fit_description = None,
                 smoothing_parameters = None, red_edge_threshold = None, blue_edge_location = None,
                 simulation_baseline = None, rebaseline_spectrum = False, fit_h12 = False):
        self.name = name
        self.fit_results = fit_results
        self.wavelengths = wavelengths
        self.longitudinal_wavelengths = longitudinal_wavelengths
        self.inputted_spectrum = inputted_spectrum
        self.blue_baseline_wavelengths = blue_baseline_wavelengths
        self.blue_baseline_intens = blue_baseline_intens
        self.smoothed_spectrum = smoothed_spectrum
        self.extracted_longitudinal = extracted_longitudinal
        self.longitudinal_threshold = longitudinal_threshold
        self.lengths = long_dims
        self.diameters = short_dims
        self.ars = ars
        self.ar_len_matrix_ars = ar_len_matrix_ars
        self.true_values = []
        self.true_pop_matrix = None
        self.fit_description = fit_description
        self.smoothing_parameters = smoothing_parameters
        self.red_edge_threshold = red_edge_threshold
        self.blue_edge_location = blue_edge_location
        self.simulation_baseline = simulation_baseline
        self.rebaseline_spectrum = rebaseline_spectrum
        self.fit_h12 = fit_h12
        if tem_data == None:
            self.TEM_results = None
        else:
            self.TEM_results = loadmat(tem_data)
        self.lengths_to_hist = []
        self.diameters_to_hist = []
        self.ars_to_hist = []
        if true_values != None:
            self.true_length_mean = true_values['length_mean']
            self.true_length_std = true_values['length_std']
            self.true_diameter_mean = true_values['diameter_mean']
            self.true_diameter_std = true_values['diameter_std']
            self.true_AR_mean = true_values['AR_mean']
            self.true_AR_std = true_values['AR_std']
            self.true_values = [self.true_length_mean, self.true_length_std, self.true_diameter_mean, self.true_diameter_std,
                            self.true_AR_mean, self.true_AR_std]
        else:
            self.true_values = None

    def add_true_vals(self, true_values):
        self.true_length_mean = true_values['length_mean']
        self.true_length_std = true_values['length_std']
        self.true_diameter_mean = true_values['diameter_mean']
        self.true_diameter_std = true_values['diameter_std']
        self.true_AR_mean = true_values['AR_mean']
        self.true_AR_std = true_values['AR_std']
        self.true_values = [self.true_length_mean, self.true_length_std, self.true_diameter_mean,
                            self.true_diameter_std,
                            self.true_AR_mean, self.true_AR_std]
    def add_tems(self, tems_filepath, tem_type):
        if tem_type == 'mat':
            self.TEM_results = loadmat(tems_filepath)
        if tem_type == 'csv':
            self.TEM_results = pd.read_csv(tems_filepath)
    def compare_to_true_vals(self):
        self.models = []
        for key in self.fit_results.keys():
            self.models.append(key)

        if self.true_values == None:
            raise Exception('true_values = None, update true values to compare')
        self.parameter_labels = self.true_values.keys()
        print(self.parameter_labels)
        self.parameters = []
        for parameter in self.parameter_labels:
            self.parameters.append(parameter)
        predicted_lengths_mean = []
        predicted_lengths_std = []
        predicted_diameter_mean = []
        predicted_diameter_std = []
        predicted_AR_mean = []
        predicted_AR_std = []

        for model in self.models:
            result = self.output[model]
            best_fit = result[0][0][4]
            predicted_lengths_mean.append(best_fit[0])
            predicted_lengths_std.append(best_fit[1])
            predicted_diameter_mean.append(best_fit[2])
            predicted_diameter_std.append(best_fit[3])
            predicted_AR_mean.append(best_fit[4])
            predicted_AR_std.append(best_fit[5])
        count = -1

        self.predictions = [predicted_lengths_mean, predicted_lengths_std, predicted_diameter_mean,
                            predicted_diameter_std, predicted_AR_mean, predicted_AR_std]
        for val in self.true_values:
            count += 1
            if val != None:
                plt.scatter(self.models, self.predictions[count])
                plt.plot(self.models, self.predictions[count], linestyle='--')
                plt.hlines(self.true_values[count], xmin=self.models[0], xmax=self.models[len(self.models) - 1])
                plt.xticks(rotation=90)
                plt.xlabel('Model Type', fontsize=22)
                plt.ylabel(self.parameters[count], fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

    def show_best_fits(self):
        pass

    def generate_true_predicted_distributions(self, population_matrix):

        print('starting')

        length_distribution = []
        diameter_distribution = []
        for length in population_matrix.T:
            length_distribution.append(sum(length))
        for diameter in population_matrix:
            diameter_distribution.append(sum(diameter))
        normalize_length = np.trapz(y=length_distribution, x=self.lengths)
        normalize_diameter = np.trapz(y=diameter_distribution, x=self.diameters)

        self.normal_length_distribution = length_distribution / normalize_length
        self.normal_diameter_distribution = diameter_distribution / normalize_diameter
        # print(self.normal_length_distribution)
        new_len_mean = np.sum((length_distribution / normalize_length) * self.lengths)
        new_dia_mean = np.sum((diameter_distribution / normalize_diameter) * self.diameters)
        new_len_std = np.sqrt(np.dot(((self.lengths - new_len_mean) ** 2), length_distribution / normalize_length))
        new_dia_std = np.sqrt(np.dot(((self.diameters - new_dia_mean) ** 2), diameter_distribution / normalize_diameter))

        ar_matrix = np.zeros((len(self.diameters), len(self.lengths)))
        for i in range(len(ar_matrix)):
            # print(i)
            for j in range(len(ar_matrix[0])):
                if self.lengths[j] >= self.diameters[i] * 1.5 and self.lengths[j] <= 10 * self.diameters[i]:
                    ar_matrix[i][j] = np.round(self.lengths[j] / self.diameters[i], 2)

        """
        ar_dist = []
        for i in range(0, len(self.ars)):
            # print(ar_vals[i])
            prob_temp = []
            for j in range(0, len(ar_matrix)):
                for k in range(0, len(ar_matrix[0])):
                    if ar_matrix[j][k] == np.round(self.ars[i], 2):
                        prob_temp.append(self.population_matrix[j][k])
                # print(ar_values_long[i])
            ar_dist.append(sum(prob_temp))
        """
        self.ars = np.arange(1.5, 10.01, 0.01)
        ar_dist = compute_ar_dist(self.ars, ar_matrix, population_matrix)
        ar_values_long_useful = np.arange(150, 1001, 10)

        normalize_ar = np.trapz(y=ar_dist, x=ar_values_long_useful)
        plt.plot(ar_values_long_useful, ar_dist / normalize_ar)
        print(np.trapz(y=ar_dist / normalize_ar, x=ar_values_long_useful))
        plt.show()

        # new_len_mean = np.sum((length_distribution/normalize_length)*lengths)

        new_ar_mean = np.sum((ar_dist / normalize_ar) * ar_values_long_useful)
        new_ar_std = np.sqrt(np.dot(((ar_values_long_useful - new_ar_mean) ** 2), ar_dist / normalize_ar))
        self.normal_ar_distribution= ar_dist/normalize_ar
        self.true_distributions = None
        if self.true_distributions == None:
            self.true_distributions = [new_len_mean, new_len_std, new_dia_mean, new_dia_std, new_ar_mean / 100,
                                       new_ar_std / 100]
        return [new_len_mean, new_len_std, new_dia_mean, new_dia_std, new_ar_mean / 100, new_ar_std / 100]

    def process_tems(self, tem_type):
        self.lengths_to_hist = []
        self.diameters_to_hist = []
        self.ars_to_hist = []
        mean_ar_temp = None
        if tem_type == 'mat':
            if 'lengths' in self.TEM_results:
                for particle in self.TEM_results['lengths']:
                    if particle[3] == 2:
                        self.lengths_to_hist.append(particle[0])
                        self.diameters_to_hist.append(particle[1])
                        self.ars_to_hist.append(particle[2])
                if np.mean(self.ars_to_hist) < 1.5:
                    mean_ar_temp = np.mean(self.ars_to_hist)
                    self.lengths_to_hist = []
                    self.diameters_to_hist = []
                    self.ars_to_hist = []
                    for particle in self.TEM_results['lengths']:
                        if particle[3] == 1:
                            self.lengths_to_hist.append(particle[0])
                            self.diameters_to_hist.append(particle[1])
                            self.ars_to_hist.append(particle[2])
                if mean_ar_temp != None:
                    if mean_ar_temp > np.mean(self.ars_to_hist):
                        print('Really short rods, need to fix this')

            if 'majlen' in self.TEM_results:
                for length in self.TEM_results['majlen'][0]:
                    self.lengths_to_hist.append(length)
                for dia in self.TEM_results['minlen'][0]:
                    self.diameters_to_hist.append(dia)
                for i in range(0, len(self.lengths_to_hist)):
                    self.ars_to_hist.append(self.lengths_to_hist[i]/self.diameters_to_hist[i])

            bin_lengths = np.arange(min(self.lengths)-0.5, max(self.lengths)+1.5, 1)
            bin_diameters = np.arange(min(self.diameters)-0.5, max(self.diameters)+1.5, 1)
            bin_ars = np.arange(min(self.ar_len_matrix_ars)-0.05, max(self.ar_len_matrix_ars)+0.05, 0.1)
            self.true_pop_matrix = plt.hist2d(self.lengths_to_hist, self.diameters_to_hist, bins=(bin_lengths, bin_diameters),
                                              density=True)[0]
            self.true_pop_matrix_ar_len = plt.hist2d(self.lengths_to_hist, self.ars_to_hist, bins=(bin_lengths, bin_ars),
                                              density=True)[0]

            matrix_sum = 0
            for row in self.true_pop_matrix_ar_len:
                matrix_sum = matrix_sum + sum(row)
            self.true_pop_matrix_ar_len = self.true_pop_matrix_ar_len/matrix_sum

        if tem_type == 'csv':
            count = 0
            for column in self.TEM_results.columns:
                # take lengths, diameters, ars from columns and add to sizes_to_hist
                for size in self.TEM_results[column]:
                    if count == 0:
                        self.lengths_to_hist.append(size)
                    if count == 1:
                        self.diameters_to_hist.append(size)
                    if count == 2:
                        self.ars_to_hist.append(size)
                count += 1

            bin_lengths = np.arange(min(self.lengths)-0.5, max(self.lengths)+1.5, 1)
            bin_diameters = np.arange(min(self.diameters)-0.5, max(self.diameters)+1.5, 1)
            bin_ars = np.arange(min(self.ar_len_matrix_ars)-0.05, max(self.ar_len_matrix_ars)+0.05, 0.1)
            self.true_pop_matrix = plt.hist2d(self.lengths_to_hist, self.diameters_to_hist, bins=(bin_lengths, bin_diameters),
                                              density=True)[0]
            self.true_pop_matrix_ar_len = plt.hist2d(self.lengths_to_hist, self.ars_to_hist, bins=(bin_lengths, bin_ars),
                                              density=True)[0]

            matrix_sum = 0
            for row in self.true_pop_matrix_ar_len:
                matrix_sum = matrix_sum + sum(row)
            self.true_pop_matrix_ar_len = self.true_pop_matrix_ar_len/matrix_sum

        return self.true_pop_matrix

    def calculate_overlap(self, fit, fit_type, show_plot = False):
        population_matrix = fit[0][0][0]
        # print(population_matrix)
        # print(fit[0][0])
        # print(fit[0])

        print(fit_type)

        if type(population_matrix) != str:
            # print('predicted pop matrix', len(population_matrix), len(population_matrix[0]))
            if fit_type in ['guided_ar_len_matrix_spheres', 'half_guided_ar_len_matrix_spheres', 'ar_len_matrix_spheres']:
                matrix_sum = 0
                for i in range(5, len(population_matrix.T)):
                    matrix_sum = matrix_sum + sum(population_matrix.T[i])
                # print(matrix_sum)
                population_matrix = population_matrix / matrix_sum

                matrix_sum = 0

                for row in self.true_pop_matrix_ar_len:
                    matrix_sum = matrix_sum + sum(row)
            else:
                matrix_sum = 0
                for row in population_matrix:
                    matrix_sum = matrix_sum + sum(row)
                # print(matrix_sum)
                population_matrix = population_matrix/matrix_sum

                matrix_sum = 0

                for row in self.true_pop_matrix:
                    matrix_sum = matrix_sum + sum(row)
                print(matrix_sum)
            # put in some code to extract population matrix from output dictionary
            # print(fit_type)
            if fit_type in ['guided_ar_len_matrix_spheres', 'half_guided_ar_len_matrix_spheres', 'ar_len_matrix_spheres']:
                overlap = 0
                overlap_matrix = np.zeros((len(self.lengths), len(self.ar_len_matrix_ars)))
                spheres_distribution = np.zeros(len(self.lengths))

                # print(len(population_matrix), len(population_matrix[0][5:len(population_matrix[0])]))
                # print(len(self.true_pop_matrix_ar_len), len(self.true_pop_matrix_ar_len[0]))


                # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                    # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
                for i in range(0, len(population_matrix)):
                    spheres_distribution[i] = population_matrix[i][0]
                if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                    if len(population_matrix[0])-5 == len(self.true_pop_matrix_ar_len[0]):
                        for i in range(0, len(population_matrix)):
                            for j in range(0, len(population_matrix[0])-5):
                                if population_matrix[i][j+5] > self.true_pop_matrix_ar_len[i][j]:
                                    overlap += self.true_pop_matrix_ar_len[i][j]
                                    overlap_matrix[i][j] = self.true_pop_matrix_ar_len[i][j]
                                if population_matrix[i][j+5] < self.true_pop_matrix_ar_len[i][j]:
                                    overlap += population_matrix[i][j+5]
                                    overlap_matrix[i][j] = population_matrix[i][j+5]
                plt.figure(figsize=(7.0, 6.0))
                # plt.plot(self.lengths, spheres_distribution)
                plt.show()

            elif fit_type in ['ar_len_matrix_rel_std', 'ar_len_matrix_rel_std_normal_x0', 'ar_len_matrix']:
                overlap = 0
                overlap_matrix = np.zeros((len(self.lengths), len(self.ar_len_matrix_ars)))

                if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                    if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
                        for i in range(0, len(population_matrix)):
                            for j in range(0, len(population_matrix[0])):
                                if population_matrix[i][j] > self.true_pop_matrix_ar_len[i][j]:
                                    overlap += self.true_pop_matrix_ar_len[i][j]
                                    overlap_matrix[i][j] = self.true_pop_matrix_ar_len[i][j]
                                if population_matrix[i][j] < self.true_pop_matrix_ar_len[i][j]:
                                    overlap += population_matrix[i][j]
                                    overlap_matrix[i][j] = population_matrix[i][j]
            else:
                overlap = 0
                overlap_matrix = np.zeros((len(self.lengths), len(self.diameters)))
                if len(population_matrix) == len(self.true_pop_matrix.T):
                    if len(population_matrix[0]) == len(self.true_pop_matrix.T[0]):
                        for i in range(0, len(population_matrix)):
                            for j in range(0, len(population_matrix[0])):
                                if population_matrix[i][j] > self.true_pop_matrix.T[i][j]:
                                    overlap += self.true_pop_matrix.T[i][j]
                                    overlap_matrix.T[i][j] = self.true_pop_matrix.T[i][j]
                                if population_matrix[i][j] < self.true_pop_matrix.T[i][j]:
                                    overlap += population_matrix[i][j]
                                    overlap_matrix.T[i][j] = population_matrix[i][j]

                def fmt(x, pos):
                    a, b = '{:.2e}'.format(x).split('e')
                    b = int(b)
                    return r'${} \times 10^{{{}}}$'.format(a, b)
                if show_plot:
                    plt.figure(figsize=(4.0, 3.0))
                    plt.contourf(self.lengths, self.diameters, population_matrix, cm='viridis')
                    plt.ylim(10, 40)
                    plt.xlim(10, 200)
                    plt.xlabel("Lengths (nm)", fontsize=20)
                    plt.ylabel("Diameters (nm)", fontsize=20)
                    plt.xticks(fontsize=18)
                    plt.yticks([15,25,35], fontsize=18)
                    plt.title("Predicted Population", fontsize=20)
                    cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                    tick_font_size = 14
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    plt.rcParams['pdf.fonttype'] = 'truetype'
                    plt.savefig('Fig_3_predicted_population.pdf', bbox_inches='tight', transparent=True)
                    plt.show()

                    plt.figure(figsize=(4.0, 3.0))
                    plt.contourf(self.lengths, self.diameters, self.true_pop_matrix.T, cm='viridis')
                    plt.ylim(10, 40)
                    plt.xlim(10, 200)
                    plt.xlabel("Lengths (nm)", fontsize=20)
                    plt.ylabel("Diameters (nm)", fontsize=20)
                    plt.xticks(fontsize=18)
                    plt.yticks([15,25,35], fontsize=18)
                    plt.title("Measured Population", fontsize=20)
                    cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                    tick_font_size = 14
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    plt.show()

                    plt.figure(figsize=(4.0, 3.0))
                    plt.contourf(self.lengths, self.diameters, overlap_matrix.T, cm='viridis')
                    plt.ylim(10, 40)
                    plt.xlim(10, 200)
                    plt.xlabel("Lengths (nm)", fontsize=20)
                    plt.ylabel("Diameters (nm)", fontsize=20)
                    plt.xticks(fontsize=18)
                    plt.yticks([15,25,35], fontsize=18)
                    plt.title("Overlap Matrix", fontsize=20)
                    cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                    tick_font_size = 14
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    plt.show()


            """
            plt.contourf(self.lengths, self.ar_len_matrix_ars, overlap_matrix.T, cm='viridis')
            # plt.ylim(1, 10)
            # plt.xlim(10, 100)
            plt.xlabel("Lengths (nm)", fontsize=16)
            plt.ylabel("Diameters (nm)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title("Overlap Matrix", fontsize=18)
            plt.colorbar()
            plt.show()
    
            bin_lengths = np.arange(9.5, 501.5, 1)
            bin_ars = np.arange(1.5, 10.1, 0.1)
            plt.hist2d(self.lengths_to_hist, self.ars_to_hist, bins=(bin_lengths, bin_ars), density=True)
            plt.show()
    
            print(len(self.lengths), len(self.ar_len_matrix_ars))
            print(len(population_matrix),len(population_matrix[0])-5)
            population_matrix_adj = population_matrix[:, 5:len(population_matrix[0])]
            print(len(population_matrix_adj),len(population_matrix_adj[0]))
            plt.contourf(self.lengths, self.ar_len_matrix_ars, population_matrix_adj.T, cm='viridis')
            # plt.ylim(1, 10)
            # plt.xlim(10, 100)
            plt.xlabel("Lengths (nm)", fontsize=16)
            plt.ylabel("Diameters (nm)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title("Overlap Matrix", fontsize=18)
            plt.colorbar()
            plt.show()
            """




            # for row in true_pop_matrix:
            # print(max(row))

            # plt.hist2d(lengths_to_hist, diameters_to_hist, bins = (bin_lengths, bin_diameters))
            # plt.show()
            return overlap

        else:
            return 'NA'

    def get_true_vals(self):
        if self.true_values == None:
            test_vars = [self.lengths_to_hist, self.ars_to_hist, self.diameters_to_hist]
            if [] in test_vars:
                return self.true_values
            else:
                self.true_values = [np.mean(self.lengths_to_hist), np.std(self.lengths_to_hist),
                                    np.mean(self.diameters_to_hist), np.std(self.diameters_to_hist),
                                    np.mean(self.ars_to_hist), np.std(self.ars_to_hist)]
                return self.true_values
        else:
            return self.true_values

    def plot_distributions(self, TEM_sizes, mu, sigma, num_bins=20, distribution_type='length',
                           color='darkorange', yticks = [], show_TEM=True,
                           distribution_label="Fitted Distribution", skew=None, savefig=False):
        plt.figure(figsize=(3.0, 3.0))

        if show_TEM == True:
            plt.hist(TEM_sizes, bins=num_bins, density=True, color='grey', label='TEM Analysis')
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 5000)
        if skew == None:
            plt.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=3, color=color, label=distribution_label)
        else:
            plt.plot(x, skewnorm.pdf(x, skew, mu, sigma), linewidth=3, color=color)
        plt.xticks(fontsize=16)
        plt.yticks(yticks, fontsize=16)
        plt.xlabel(distribution_type + ' (nm)', fontsize=18)
        plt.ylabel("Proportion", fontsize=18)
        plt.title(distribution_type + " Distribution", fontsize=18)
        if savefig:
            plt.savefig('Fig_4_'+ distribution_type +'.pdf', bbox_inches = 'tight', transparent = True)
        # plt.legend(['TEM Size Analysis', distribution_label], fontsize = 24)
        # plt.xlim(0, 10)


    def plot_all_distributions(self, num_bins = 20, savefig = True):
        types = ['Length', 'Diameter', 'Aspect Ratio']
        colors = ['red', 'blue', 'purple']
        yticks = [[0.00, 0.01, 0.02, 0.03], [0.00, 0.04, 0.08, 0.12], [0.0, 0.2, 0.4]]
        for key in self.fit_results.keys():
            fit_key = key
        print(fit_key)
        best_fit_from_pop = self.fit_results[fit_key][0][0][4]
        fit_results_extracted = [[best_fit_from_pop[0], best_fit_from_pop[1]], [best_fit_from_pop[2], best_fit_from_pop[3]], [best_fit_from_pop[4], best_fit_from_pop[5]]]
        tems = [self.lengths_to_hist, self.diameters_to_hist, self.ars_to_hist]
        for i in range(0, len(types)):
            self.plot_distributions(tems[i], fit_results_extracted[i][0], fit_results_extracted[i][1], num_bins = num_bins,
                                    distribution_type = types[i], color = colors[i], yticks = yticks[i],
                                    savefig=savefig)



    def plot_true_distributions(self, distribution_type, TEM_sizes, distribution,num_bins):
        plt.hist(TEM_sizes, bins=num_bins, density=True, color='grey', label='TEM Analysis')
        if distribution_type == 'Length':
            plt.plot(self.lengths, distribution, linewidth = 3)
            plt.xlim(self.true_distributions[0]-self.true_distributions[1]*5,
                     self.true_distributions[0]+self.true_distributions[1]*5)
        if distribution_type == 'Diameter':
            plt.plot(self.diameters, distribution, linewidth = 3)
            plt.xlim(self.true_distributions[2]-self.true_distributions[3]*5,
                     self.true_distributions[2]+self.true_distributions[3]*5)

        if distribution_type == 'Aspect Ratio':
            plt.plot(self.ars, distribution*10)
            plt.xlim(self.true_distributions[4]-self.true_distributions[5]*5,
                     self.true_distributions[4]+self.true_distributions[5]*5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(distribution_type + '(nm)', fontsize=24)
        plt.ylabel("Proportion", fontsize=24)
        plt.title(distribution_type + " Distribution", fontsize=30)
        # plt.legend([, distribution_label], fontsize = 13)
        plt.show()


    def compare_distributions_to_tem(self,num_bins = 20):
        print('starting')
        for key in self.fit_results.keys():
            population_matrix = self.fit_results[key][0][0][0]

            self.generate_true_predicted_distributions(population_matrix)

            self.plot_true_distributions('Length', self.lengths_to_hist, self.normal_length_distribution, num_bins)
            self.plot_true_distributions('Diameter', self.diameters_to_hist, self.normal_diameter_distribution, num_bins)
            self.plot_true_distributions('Aspect Ratio', self.ars_to_hist, self.normal_ar_distribution, num_bins)

    def display_predicted_values(self, fit, fit_type='len_dia_correlation'):


        if type(fit[0][0]) != str:
            if fit_type == 'guided_ar_len_matrix_spheres':
                fitted_parameters = fit[0][0][1]
                fitted_len_mean = round(fitted_parameters[0], 1)
                fitted_ar_mean = round(fitted_parameters[1], 1)
                fitted_ar_std = round(fitted_parameters[2], 1)
                fitted_len_std = round((fitted_ar_std/fitted_ar_mean)*fitted_len_mean, 1)
                table_row = str(fitted_len_mean) + '(' + str(fitted_len_std) + ')' + ' ' + \
                            str(fitted_ar_mean) + '(' + str(fitted_ar_std) + ')'

            else:
                fitted_parameters = fit[0][0][4]
            # print(fitted_parameters)
                fitted_len_mean = round(fitted_parameters[0], 1)
                fitted_len_std = round(fitted_parameters[1], 1)
                fitted_dia_mean = round(fitted_parameters[2], 1)
                fitted_dia_std = round(fitted_parameters[3], 1)
                fitted_ar_mean = round(fitted_parameters[4], 1)
                fitted_ar_std = round(fitted_parameters[5], 1)

                table_row = str(fitted_len_mean) + '(' + str(fitted_len_std) + ')' + ' ' + \
                            str(fitted_dia_mean) + '(' + str(fitted_dia_std) + ')' + ' ' +\
                            str(fitted_ar_mean) + '(' + str(fitted_ar_std) + ')'

            return table_row
        if type(fit[0][0]) == str:
            table_row = fit[0][0]
            return table_row

    def bivariate_gaussian_len_dia(self, lengths, diameters, means, cov):

        first_num_length = lengths[0]
        step_size_length = lengths[1] - lengths[0]
        last_num_length = lengths[len(lengths) - 1] + step_size_length

        first_num_dia = diameters[0]
        step_size_dia = diameters[1] - diameters[0]
        last_num_dia = diameters[len(diameters) - 1] + step_size_dia

        x, y = np.mgrid[first_num_dia:last_num_dia:step_size_dia, first_num_length:last_num_length:step_size_length]
        pos = np.dstack((x, y))
        rv = multivariate_normal(means, cov)

        population_matrix = rv.pdf(pos)

        return [population_matrix]

    def calc_overlap_from_lit(self, fit, fit_type, show_plot = False):
        population_matrix = fit[0][0][0]
        fit_results = fit[0][0][1]
        # print(fit_results)
        # print('starting')
        # print(population_matrix)
        # print(fit[0][0])
        # print(fit[0])
        if type(population_matrix) != str:
            # print('made it here')
            # print('predicted pop matrix', len(population_matrix), len(population_matrix[0]))

            # TODO currently not set up
            """
            if fit_type in ['guided_ar_len_matrix_spheres', 'half_guided_ar_len_matrix_spheres',
                            'ar_len_matrix_spheres']:
                matrix_sum = 0
                for i in range(5, len(population_matrix.T)):
                    matrix_sum = matrix_sum + sum(population_matrix.T[i])
                # print(matrix_sum)
                population_matrix = population_matrix / matrix_sum

                matrix_sum = 0

                for row in self.true_pop_matrix_ar_len:
                    matrix_sum = matrix_sum + sum(row)
            """
            # else:
            long_dims = np.arange(10, 501, 1)
            short_dims = np.arange(5, 51, 1)
            avg_len = float(self.true_values[0])
            std_len = float(self.true_values[1])
            avg_dia = float(self.true_values[2])
            std_dia = float(self.true_values[3])
            correlation = 0
            cov = 0
            distributions = self.bivariate_gaussian_len_dia(long_dims, short_dims, [avg_dia, avg_len],
                                                                [[std_dia ** 2, cov],
                                                                 [cov, std_len ** 2]])


            true_matrix_from_means = distributions[0]
            # print([avg_len, std_len, avg_dia, std_dia])
            matrix_sum = 0
            for row in population_matrix:
                matrix_sum = matrix_sum + sum(row)
            # print(matrix_sum)
            population_matrix = population_matrix / matrix_sum

            matrix_sum = 0

            for row in true_matrix_from_means:
                matrix_sum = matrix_sum + sum(row)
            true_matrix_from_means = true_matrix_from_means/matrix_sum

            # put in some code to extract population matrix from output dictionary
            # print(fit_type)
            # TODO currently not set up
            """
            if fit_type in ['guided_ar_len_matrix_spheres', 'half_guided_ar_len_matrix_spheres',
                            'ar_len_matrix_spheres']:
                overlap = 0
                overlap_matrix = np.zeros((len(self.lengths), len(self.ar_len_matrix_ars)))
                spheres_distribution = np.zeros(len(self.lengths))

                # print(len(population_matrix), len(population_matrix[0][5:len(population_matrix[0])]))
                # print(len(self.true_pop_matrix_ar_len), len(self.true_pop_matrix_ar_len[0]))

                # if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                # if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
                for i in range(0, len(population_matrix)):
                    spheres_distribution[i] = population_matrix[i][0]
                if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                    if len(population_matrix[0]) - 5 == len(self.true_pop_matrix_ar_len[0]):
                        for i in range(0, len(population_matrix)):
                            for j in range(0, len(population_matrix[0]) - 5):
                                if population_matrix[i][j + 5] > self.true_pop_matrix_ar_len[i][j]:
                                    overlap += self.true_pop_matrix_ar_len[i][j]
                                    overlap_matrix[i][j] = self.true_pop_matrix_ar_len[i][j]
                                if population_matrix[i][j + 5] < self.true_pop_matrix_ar_len[i][j]:
                                    overlap += population_matrix[i][j + 5]
                                    overlap_matrix[i][j] = population_matrix[i][j + 5]
                plt.figure(figsize=(7.0, 6.0))
                # plt.plot(self.lengths, spheres_distribution)
                plt.show()

            elif fit_type in ['ar_len_matrix_rel_std', 'ar_len_matrix_rel_std_normal_x0', 'ar_len_matrix']:
                overlap = 0
                overlap_matrix = np.zeros((len(self.lengths), len(self.ar_len_matrix_ars)))

                if len(population_matrix) == len(self.true_pop_matrix_ar_len):
                    if len(population_matrix[0]) == len(self.true_pop_matrix_ar_len[0]):
                        for i in range(0, len(population_matrix)):
                            for j in range(0, len(population_matrix[0])):
                                if population_matrix[i][j] > self.true_pop_matrix_ar_len[i][j]:
                                    overlap += self.true_pop_matrix_ar_len[i][j]
                                    overlap_matrix[i][j] = self.true_pop_matrix_ar_len[i][j]
                                if population_matrix[i][j] < self.true_pop_matrix_ar_len[i][j]:
                                    overlap += population_matrix[i][j]
                                    overlap_matrix[i][j] = population_matrix[i][j]
            """
            # else:
            overlap = 0
            overlap_matrix = np.zeros((len(self.lengths), len(self.diameters)))
            if len(population_matrix) == len(true_matrix_from_means):
                if len(population_matrix[0]) == len(true_matrix_from_means[0]):
                    for i in range(0, len(population_matrix)):
                        for j in range(0, len(population_matrix[0])):
                            if population_matrix[i][j] > true_matrix_from_means[i][j]:
                                overlap += true_matrix_from_means[i][j]
                                overlap_matrix.T[i][j] = true_matrix_from_means[i][j]
                            if population_matrix[i][j] < true_matrix_from_means[i][j]:
                                overlap += population_matrix[i][j]
                                overlap_matrix.T[i][j] = population_matrix[i][j]
            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)

            if show_plot:
                plt.figure(figsize=(4.0, 3.0))
                plt.contourf(self.lengths, self.diameters, population_matrix, cm='viridis')
                plt.yticks([5,15, 25, 35], fontsize=18)
                plt.ylim(5, 20)
                plt.xlim(10, 80)
                plt.xlabel("Lengths (nm)", fontsize=20)
                plt.ylabel("Diameters (nm)", fontsize=20)
                plt.xticks(fontsize=18)
                plt.title("Predicted Population", fontsize=20)
                cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                tick_font_size = 14
                cbar.ax.tick_params(labelsize=tick_font_size)
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig('overlap_illustration_predicted_population_matrix.pdf', bbox_inches='tight', transparent=True)
                plt.show()

                plt.figure(figsize=(4.0, 3.0))
                plt.contourf(self.lengths, self.diameters, self.true_pop_matrix.T, cm='viridis')
                plt.yticks([5,15, 25, 35], fontsize=18)
                plt.ylim(5, 20)
                plt.xlim(10, 80)
                plt.xlabel("Lengths (nm)", fontsize=20)
                plt.ylabel("Diameters (nm)", fontsize=20)
                plt.xticks(fontsize=18)
                plt.title("Measured Population", fontsize=20)
                cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                tick_font_size = 14
                cbar.ax.tick_params(labelsize=tick_font_size)
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig('overlap_illustration_true_dist.pdf', bbox_inches='tight', transparent=True)
                plt.show()

                plt.figure(figsize=(4.0, 3.0))
                plt.contourf(self.lengths, self.diameters, true_matrix_from_means, cm='viridis')
                plt.yticks([5,15, 25, 35], fontsize=18)
                plt.ylim(5, 20)
                plt.xlim(10, 80)
                plt.xlabel("Lengths (nm)", fontsize=20)
                plt.ylabel("Diameters (nm)", fontsize=20)
                plt.xticks(fontsize=18)
                plt.title("Measured Population Projected", fontsize=20)
                cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                tick_font_size = 14
                cbar.ax.tick_params(labelsize=tick_font_size)
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig('overlap_illustration_projected_true_population_matrix.pdf', bbox_inches='tight', transparent=True)
                plt.show()

                plt.figure(figsize=(4.0, 3.0))
                plt.contourf(self.lengths, self.diameters, overlap_matrix.T, cm='viridis')
                plt.yticks([5,15, 25, 35], fontsize=18)
                plt.ylim(5, 20)
                plt.xlim(10, 80)
                plt.xlabel("Lengths (nm)", fontsize=20)
                plt.ylabel("Diameters (nm)", fontsize=20)
                plt.xticks(fontsize=18)
                plt.title("Overlap Matrix", fontsize=20)
                cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
                tick_font_size = 14
                cbar.ax.tick_params(labelsize=tick_font_size)
                plt.rcParams['pdf.fonttype'] = 'truetype'
                plt.savefig('overlap_illustration_overlap_matrix.pdf', bbox_inches='tight', transparent=True)
                plt.show()

            return overlap

        else:
            return 'NA'

    def tabulate_results(self, show_table = True, overlap_method = 'TEMs', show_plot = False):
        # print('testing')
        # print(self.true_values)
        test_vars = [self.lengths_to_hist, self.ars_to_hist, self.diameters_to_hist]
        # print(test_vars)
        # print([] in test_vars)
        if [] in test_vars:
            if type(self.true_values) == type(None):
                # print('no worko')
                output_table = [
                    ['Model Type', 'Overlap', 'Predictions ' + str(round(np.mean(self.lengths_to_hist), 1)) +
                     '(' + str(round(np.std(self.lengths_to_hist), 1)) + ')' + ' ' + \
                     str(round(np.mean(self.diameters_to_hist), 1)) + '(' +
                     str(round(np.std(self.diameters_to_hist), 1)) + ')' + ' ' + \
                     str(round(np.mean(self.ars_to_hist), 1)) + '(' + str(round(np.std(self.ars_to_hist), 1)) + ')']]
            else:
                # print('I hope this is running')
                output_table = [
                    ['Model Type', 'Overlap', 'Predictions ' + str(self.true_values[0]) +
                     '(' + str(self.true_values[1]) + ')' + ' ' + \
                     str(self.true_values[2]) + '(' +
                     str(self.true_values[3]) + ')' + ' ' + \
                     str(self.true_values[4]) + '(' + str(self.true_values[5]) + ')']]
        else:
            output_table = [['Model Type', 'Overlap', 'Predictions ' + str(round(np.mean(self.lengths_to_hist), 1)) +
                         '(' + str(round(np.std(self.lengths_to_hist), 1)) + ')' + ' ' +\
                    str(round(np.mean(self.diameters_to_hist),1)) + '(' +
                         str(round(np.std(self.diameters_to_hist), 1)) + ')' + ' ' +\
                    str(round(np.mean(self.ars_to_hist), 1)) + '(' + str(round(np.std(self.ars_to_hist), 1)) + ')']]
        for key in self.fit_results.keys():
            table_list = []
            table_list.append(key)
            # print(self.fit_results[key])
            # if type(self.true_pop_matrix) == type(None):
                # table_list.append('NA')
            if self.fit_results[key][0][0] != None:
                if overlap_method == 'TEMs':
                    if [] in test_vars:
                        table_list.append('NA')
                    else:
                        table_list.append(self.calculate_overlap(self.fit_results[key], key, show_plot=show_plot))
                else:
                    # print('else')
                    table_list.append(self.calc_overlap_from_lit(self.fit_results[key], key, show_plot=show_plot))
            else:
                table_list.append('NA')

            if key == 'guided_ar_len_matrix_spheres':
                table_list.append(self.display_predicted_values(self.fit_results[key], 'guided_ar_len_matrix_spheres'))
            else:
                if self.fit_results[key][0][0] != None:
                    table_list.append(self.display_predicted_values(self.fit_results[key]))
                else:
                    table_list.append('All Fits Hit Bounds')
                if self.true_values == None:
                    true_vals = []
                    val = ''
                    # print(output_table)
                    # print(len(output_table))
                    # print(output_table[0][2][12:15])
                    if output_table[0][2][12:15] != 'nan':
                        for char in output_table[0][2]:
                            if char not in 'Predictions':
                                if char not in '( )':
                                    val += (char)
                                elif len(val) != 0:
                                    true_vals.append(float(val))
                                    val = ''
                    # print(true_vals)
                    self.true_values = true_vals
            # table_list.append('Temp')

            output_table.append(table_list)

        # print(table_list)
        self.output_table = output_table
        if show_table == True:
            print(tabulate(self.output_table, headers='firstrow', tablefmt='fancy_grid'))
        return self.output_table

class TEM_sizes():
    def __init__(self, TEM_results, name = None, tem_sizes_type = 'mat'):
        self.name = name
        self.TEM_results = TEM_results

        if type(self.name) == type(None):

            period_index = self.TEM_results.index('.')
            self.name = ''
            count = period_index - 1
            while self.TEM_results[count] not in '\/':
                # print(self.TEM_results[count])
                self.name = self.name + self.TEM_results[count]
                count = count-1
            self.name = self.name[::-1]
            print(self.name)
        self.tem_sizes_type = tem_sizes_type

        self.lengths_to_hist = []
        self.diameters_to_hist = []
        self.ars_to_hist = []
        self.lengths = np.arange(10, 501, 1)

        self.diameters = np.arange(5, 51, 1)

        self.ars = np.arange(1.5, 10.1, 0.1) # AR values for when fit is in len/dia space
        self.ar_len_matrix_ars = np.arange(1.5, 10.1, 0.1) # AR values for when fit is in AR/len space (scaling changed
            # to 0.1 to make the population matrix less unwieldy)
        self.sphere_ars = np.arange(1,10.1,0.1) # AR values extend down to one when population matrix is set up to
            # include spheres


        if type(self.TEM_results) == str:
            self.TEM_results = loadmat(self.TEM_results)




        mean_ar_temp = None
        if self.tem_sizes_type == 'mat':
            if 'lengths' in self.TEM_results:
                for particle in self.TEM_results['lengths']:
                    if particle[3] == 2:
                        if 200 > particle[0] > 10:
                            if 50 > particle[1] > 5:
                                self.lengths_to_hist.append(particle[0])
                                self.diameters_to_hist.append(particle[1])
                                self.ars_to_hist.append(particle[2])
                if np.mean(self.ars_to_hist) < 1.5:
                    mean_ar_temp = np.mean(self.ars_to_hist)
                    self.lengths_to_hist = []
                    self.diameters_to_hist = []
                    self.ars_to_hist = []
                    for particle in self.TEM_results['lengths']:
                        if particle[3] == 1:
                            if 200 > particle[0] > 10:
                                if 50 > particle[1] > 5:
                                    self.lengths_to_hist.append(particle[0])
                                    self.diameters_to_hist.append(particle[1])
                                    self.ars_to_hist.append(particle[2])
                if mean_ar_temp != None:
                    if mean_ar_temp > np.mean(self.ars_to_hist):
                        print('Really short rods, need to fix this')

            if 'majlen' in self.TEM_results:
                for length in self.TEM_results['majlen'][0]:
                    self.lengths_to_hist.append(length)
                for dia in self.TEM_results['minlen'][0]:
                    self.diameters_to_hist.append(dia)
                for i in range(0, len(self.lengths_to_hist)):
                    self.ars_to_hist.append(self.lengths_to_hist[i] / self.diameters_to_hist[i])



            # print(self.lengths_to_hist)
            # print(self.diameters_to_hist)
            # print(self.ars_to_hist)

            print('Number of Rods Measured ' + str(len(self.lengths_to_hist)))



        if self.tem_sizes_type == 'csv':
            count = 0
            for column in self.TEM_results.columns:
                # take lengths, diameters, ars from columns and add to sizes_to_hist
                for size in self.TEM_results[column]:
                    if count == 0:
                        self.lengths_to_hist.append(size)
                    if count == 1:
                        self.diameters_to_hist.append(size)
                    if count == 2:
                        self.ars_to_hist.append(size)
                count += 1

            bin_lengths = np.arange(min(self.lengths) - 0.5, max(self.lengths) + 1.5, 1)
            bin_diameters = np.arange(min(self.diameters) - 0.5, max(self.diameters) + 1.5, 1)
            bin_ars = np.arange(min(self.sphere_ars) - 0.05, max(self.sphere_ars) + 0.05, 0.1)
            self.true_pop_matrix = \
            plt.hist2d(self.lengths_to_hist, self.diameters_to_hist, bins=(bin_lengths, bin_diameters),
                       density=True)[0]
            self.true_pop_matrix_ar_len = plt.hist2d(self.lengths_to_hist, self.ars_to_hist, bins=(bin_lengths, bin_ars),
                                                     density=True)[0]

            matrix_sum = 0
            for row in self.true_pop_matrix_ar_len:
                matrix_sum = matrix_sum + sum(row)
            self.true_pop_matrix_ar_len = self.true_pop_matrix_ar_len / matrix_sum


    def print_distributions(self):
        self.output_table = [['Length', 'Diameter', 'Aspect Ratio '],
                        [str(round(np.mean(self.lengths_to_hist), 1)) +
                        '(' + str(round(np.std(self.lengths_to_hist), 1)) + ')' ,
                        str(round(np.mean(self.diameters_to_hist), 1)) + '(' +
                        str(round(np.std(self.diameters_to_hist), 1)) + ')',
                        str(round(np.mean(self.ars_to_hist), 1)) + '(' + str(round(np.std(self.ars_to_hist), 1)) + ')']]

        print(tabulate(self.output_table, headers='firstrow', tablefmt='fancy_grid'))
        return self.output_table


    def visualize_1d_histograms(self, ax_lims=None, ax_fontsize = 24, title_fontsize = 30, tick_fontsize = 16,
                                fig_size = (7.0, 6.0) , num_bins = 20):
        self.distributions = [self.lengths_to_hist, self.diameters_to_hist, self.ars_to_hist]
        self.distribution_names = ['Lengths', 'Diameters', 'Aspect Ratios']
        if ax_lims == None:
            length_min = min(self.lengths_to_hist)
            length_max = max(self.lengths_to_hist)

            diameter_min = min(self.diameters_to_hist)
            diameter_max = max(self.diameters_to_hist)

            ar_min = min(self.ars_to_hist)
            ar_max = max(self.ars_to_hist)

            ax_lims = [[length_min,length_max], [diameter_min, diameter_max], [ar_min, ar_max]]
        for i in range(0,3):
            plt.figure(figsize=fig_size)
            plt.hist(self.distributions[i], bins=num_bins, density=True, color='grey', label='TEM Analysis')

            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.xlabel(self.distribution_names[i] + ' (nm)', fontsize=ax_fontsize)
            plt.ylabel("Proportion", fontsize=ax_fontsize)
            plt.title(self.distribution_names[i] + " Distribution", fontsize=title_fontsize)
            plt.xlim(ax_lims[i])
            plt.show()
            # plt.legend([, distribution_label], fontsize = 13)
            # plt.xlim(0, 10)


    def visualize_2d_histograms(self):
        bin_lengths = np.arange(min(self.lengths) - 0.5, max(self.lengths) + 1.5, 1)
        bin_diameters = np.arange(min(self.diameters) - 0.5, max(self.diameters) + 1.5, 1)
        bin_ars = np.arange(min(self.ar_len_matrix_ars) - 0.05, max(self.ar_len_matrix_ars) + 0.05, 0.1)
        self.true_pop_matrix = \
            plt.hist2d(self.lengths_to_hist, self.diameters_to_hist, bins=(bin_lengths, bin_diameters),
                       density=True)[0]
        self.true_pop_matrix_ar_len = \
            plt.hist2d(self.lengths_to_hist, self.ars_to_hist, bins=(bin_lengths, bin_ars),
                       density=True)[0]

        matrix_sum = 0
        for row in self.true_pop_matrix_ar_len:
            matrix_sum = matrix_sum + sum(row)
        self.true_pop_matrix_ar_len = self.true_pop_matrix_ar_len / matrix_sum

    """
    old fit function 
    def fit(self, x0, bounds=([15, 1, 3, 0.1, 0.2], [180, 50, 9, 2, 0.8]),
            print_kopt=True, show_plot=False,
            print_params=False, fit_style='ar_len_matrix', simulation_baseline=False, show_distributions = False):
        
        """
    """
        :param x0: list - initial guess [mean len, std len, mean ar, std ar, skew ar, correlation len/ar]
        :param bounds: list of list - boundary values for the fitted parameters
        :param print_kopt: bool - whether the function should print the fitted parameters upon completion
        :param show_plot: bool - whether the function should plot the simulated spectrum on top of the sample for
        every step in the calculation. Can be very useful for debugging
        :param print_params: bool - whether the function should print out the fitted parameters for every step in the
        calculation
        :param fit_style: string - len_ars or len_dia. Determines whether the fit is directly fitting length and AR
        and extracting diameter, or fitting length and diameter and extracting aspect ratio
        TODO len_dia hasn't been maintained very thoroughly and probably doesn't work anymore
        :param simulation_baseline: bool - baseline subtracts simulations (not using currently, so remains false)
        :param show_distributions: bool - whether to show the population matrix for each step in the calculation, and
        how applying aspect ratios changes it. Takes quite a bit of time to run the fit if = True, but can be very
        useful for debugging

        :return: list
        1st entry - population matrix
        2nd and 3rd entry - kopt and kcov from curve fit
        4th entry - reduced chisq of fit
        5th entry - true values of size parameters calculated from population matrix in the form of
        [mean len, std len, mean diameter, std diameter, mean ar, std ar]
        """
    """

        # kopts = [] eventually set up a way to store all the values the fit tries
        sigmas = []
        for i in range(0, len(self.wavelengths)):
            sigmas.append(0.01)
        sigmas = np.asarray(sigmas)
         # set all uncertainties to the same value (0.01) since fitting the smoothed spectrum
        # with no uncertainty function. 0.01 chosen arbitrarily to give the reduced chisq a reasonable value



        Function for fitting length and diameter

        # TODO I've changed a lot of the functions this calls, it very well may no longer work. Probably worth ignoring
        # for now
        def skew_gauss_profiles(Es, avg_len, std_len, avg_dia, std_dia, correlation):
            cov = correlation * std_len * std_dia
            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov],
                                                             [cov, std_len ** 2]])
            population_matrix = distributions[0]
            ar_mean = distributions[1]
            ar_std = distributions[2]
            if show_distributions == True:
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("Rods Population Pre AR Adjustment", fontsize=18)
                plt.colorbar()
                plt.show()

            # ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, ar_mean,
                                                       # ar_std)


            spectrum = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix = [],
                                                              peak_range=self.peak_indicies)
            self.population_matrix = spectrum[1]
            spectrum = spectrum[0]

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)
            if show_plot == True:
                print(max(self.smoothed_intens))
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            if print_params == True:
                print(avg_len, std_len, avg_dia, std_dia, ar_mean, ar_std, correlation, reduced_chi)

            return spectrum_final


        Function for fitting length and AR


        def skew_gauss_profiles_fit_ars_directly(Es, avg_len, std_len, avg_AR, std_AR, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum


            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)


            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            cov = correlation * std_len * std_AR


            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

        # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
        # for this correlation.

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.ars, [avg_AR, avg_len],
                                                            [[std_AR ** 2, cov],
                                                             [cov, std_len ** 2]])

            population_matrix = distributions[0]
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.long_dims, self.ars, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(2, 8)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Aspect Ratio", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()


            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()


            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix= [],
                                                                      peak_range=self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new) # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final-self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res**2/sigmas**2)/(len(res)-len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_AR, std_AR, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final

        def skew_gauss_profiles_ar_len(Es, avg_len, std_len, skew_avg_AR, skew_std_AR, ar_skew, correlation):
            # This function returns a simulated spectrum for a particular set of conditions. Interfaces with curve fit
            # to find the parameters which minimize the difference between the sample spectrum and simulated spectrum

            # calculate the actual mean and std AR from skew norm distribution (purely to observe while printing out
            # parameters)
            avg_AR = skew_avg_AR + skew_std_AR * (ar_skew / (np.sqrt(1 + ar_skew ** 2))) * np.sqrt(2 / np.pi)
            std_AR = np.sqrt((skew_std_AR ** 2) * (1 - (2 * (ar_skew / (np.sqrt(1 + ar_skew ** 2)))) / np.pi))

            # uncomment for relative standard deviation (and will need to change a couple parameters and bounds)
            # std_len = rel_std_len*avg_len

            # calculate diameter values
            cov = correlation * std_len * std_AR
            avg_dia = avg_len / avg_AR
            std_dia = self.calc_diameter_stdev([avg_AR, avg_len], [[std_AR ** 2, cov], [cov, std_len ** 2]])

            # avg_dia = np.round(avg_dia, 2)
            # std_dia = np.round(std_dia, 2)
            # correlation_2 = self.calc_len_diameter_correlation(avg_dia, std_dia, avg_len, std_len, avg_AR, std_AR)

            # cov = correlation* std_len * std_dia
            # ar_std_temp = self.calc_diameter_stdev([avg_dia, avg_len], [[std_dia ** 2, cov],[cov, std_len ** 2]])
            # print(ar_std_temp)

            cov_len_dia = 0 * std_len * std_dia  # treating correlation between length and diameter as zero and allowing
            # aspect ratio scaling to enforce this. Commented out code above is from when I had an additional parameter
            # for this correlation.

            distributions = self.bivariate_gaussian_len_dia(self.long_dims, self.short_dims, [avg_dia, avg_len],
                                                            [[std_dia ** 2, cov_len_dia],
                                                             [cov_len_dia, std_len ** 2]])

            population_matrix = distributions[0]
            if show_distributions == True:
                # if user called for show distributions, this will plot the population matrix before the AR adjustment
                # plotting post adjustment comes in the helper functions that compute the adjustment
                plt.contourf(self.long_dims, self.short_dims, population_matrix, cm='viridis')
                plt.xlim(10, 200)
                plt.ylim(5, 40)
                plt.xlabel("Lengths (nm)", fontsize=16)
                plt.ylabel("Diameters (nm)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.title("AuNR Population", fontsize=18)
                plt.colorbar()
                plt.show()

            ar_matrix = self.create_ar_matrix(self.short_dims, self.long_dims, ar_skew, self.ars, skew_avg_AR,
                                              skew_std_AR)

            # plt.plot(short_dims, population_short)
            # plt.show()
            # plt.plot(long_dims, population_long)
            # plt.show()

            # simulate spectrum using helper method create_spectrum_bivariate_longitudinal
            output_list = self.create_spectrum_bivariate_longitudinal(population_matrix, self.profiles, ar_matrix,
                                                                      self.peak_indicies,
                                                                      simulation_baseline=simulation_baseline,
                                                                      show_distributions=show_distributions)
            population_matrix = output_list[1]
            spectrum = output_list[0]
            self.population_matrix = population_matrix

            spectrum_new = spectrum
            spectrum_final = spectrum_new / max(spectrum_new)  # normalize simulated spectrum
            if show_plot == True:
                # plot simulated spectrum on top of sample if requested
                plt.figure(figsize=(7.0, 6.0))

                plt.plot(self.wavelengths, spectrum_final, linewidth=4)
                plt.plot(self.wavelengths, self.smoothed_intens, linewidth=4)
                plt.legend(["Prediction", "Sample"], fontsize=16)
                plt.title("Predicted VS Sample", fontsize=28)
                plt.xlabel("Wavelength (nm)", fontsize=22)
                plt.ylabel("Absorbance", fontsize=22)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.show()

            # calculate reduced chisq
            res = spectrum_final - self.smoothed_intens
            # print(res)
            # print(sigmas)
            # print(spectrum_final)
            # print(self.smoothed_intens)
            reduced_chi = np.sum(res ** 2 / sigmas ** 2) / (len(res) - len(x0))
            # print(reduced_chi)
            if print_params == True:
                # print out parameters if requested
                print(avg_len, std_len, avg_dia, std_dia, avg_AR, std_AR, ar_skew, correlation, reduced_chi)
            self.reduced_chi = reduced_chi
            # this function returns a simulated spectrum for a particular set of conditions. interfaces with curve fit
            # to find the parameters which minimize difference between the sample spectrum and simulated spectrum
            return spectrum_final

        # set of logic allows specification of the fitting method by using a different function to create the
        # simulated spectrum
        if fit_style == 'len_ars':

            kopt, kcov = curve_fit(skew_gauss_profiles_fit_ars_directly, self.wavelengths, self.smoothed_intens, sigma=sigmas,
                                   maxfev=80000,
                                   bounds=bounds, p0=x0)
            self.fitted_params = kopt

        if fit_style == 'len_dia':
            kopt, kcov = curve_fit(skew_gauss_profiles, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=80000, bounds=bounds,
                                   p0=x0)
            self.fitted_params = kopt

        if fit_style == 'ar_len_matrix':
            kopt, kcov = curve_fit(skew_gauss_profiles_ar_len, self.wavelengths, self.smoothed_intens, sigma=sigmas, maxfev=80000, bounds=bounds,
                                   p0=x0)
            self.fitted_params = kopt

        # after fit has found optimal conditions, calls sim method to plot the simulated and sample spectrum and show
        # the residuals
        # self.sim(simulation_baseline=simulation_baseline)

        if print_kopt == True:
            print("Fitted Parameters, NOT true distributions from population matrix")
            print(kopt)
        print("Calculating true distributions from population matrix")
        self.true_distributions = self.calc_size_distributions_from_population() # calculate the true size distributions
        # by using population matrix, as these are not the same as the parameters which went into curve fit!
        print(self.true_distributions)
        return [self.population_matrix, kopt, kcov, self.reduced_chi, self.true_distributions, x0]
    """