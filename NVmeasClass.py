import visa, os, sys, click, json
import time as tm
import numpy as np
import matplotlib.pyplot as plt
import shutil as sh
from lmfit.models import LorentzianModel


class NVMeas:
    # private:
    # variables
    # Generator parameters:
    __generator_instr = None  # Generator control variable
    __freq_start = None  # Start frequency, GHz
    __freq_stop = None  # Stop frequency, GHz
    __freq_step = None  # Frequency step, GHz
    __freq = None  # Frequency range
    __freq_num = None  # Number of points
    __gen_level = None  # Generator power level, dBm
    __gen_con_status = False  # Status of generator connection

    # Power sensor parameters:
    __power_sensor_instr = None  # Power sensor control variable
    __power_sensor_con_status = False  # Status of power sensor connection

    # Measurements parameters
    __meas_title = None  # Measurements title
    __pump_laser_power = None  # Pump laser power, %
    __pump_laser_wavelength = None  # Pump laser wavelength, nm
    __heat_laser_power = None  # Heat laser power, mW
    __heat_laser_wavelength = None  # Heat laser wavelength, nm
    __meas_point_time = None  # Measurement time of LabRam (one point measurement), s
    __estimated_meas_time = None  # Estimated mesurement time, s
    __exec_time = None  # Execusion time, s
    __n_average = None  # Number of averaging

    # Path to working folders
    __path_to_labram_folder = None  # Path to LabRam Folder
    __path_to_labram_backup_folder = None  # Path to LabRam Backup Folder
    __path_to_result_folder = None  # Path to Result Folder

    # Measurements variables
    __reference_intensity = None  # Reference level intensity (no MW fields)
    __intensity = None  # Array of not normalized intensities

    # Fitting parameters
    # Lorentzian parameters (amplitude: 'amplitude', center frequency: 'center', sigma: 'sigma', fwhm: 'fwhm', height: 'height')
    __fit_param = None
    __LorentzianFunc = None  # Lorentzian lambda function
    __Temp = None  # Temperature, K

    # .json data
    __add_info_json = None
    __labram_files_first_last = None
# functions

    # Constructor
    def __init__(self, FRGen=True, RFSensor=False):
        print("Hello")

    # Destructor
    def __del__(self):
        # Close all connections
        if self.__gen_con_status:
            self.__generator_instr.close()  # Generator
        if self.__power_sensor_con_status:
            self.__power_sensor_instr.close()  # Power sensor

    def __LorentzianFit(self):
        """
        Fitting by Lorentzian

        Lambda function will written in __LorentzianFunc

        Lorentzian parameter will be written in __fit_param

        amplitude: 'amplitude', center frequency: 'center', sigma: 'sigma', fwhm: 'fwhm', height: 'height'
        """

        x, y = self.__freq,  self.__intensity / self.__reference_intensity

        mod = LorentzianModel()
        pars = mod.guess(1-y, x=x)
        out = mod.fit(1-y, pars, x=x)

        def loren(a, x, x0, sigma): return a/np.pi * sigma / \
            ((x-x0)**2+sigma**2)  # Lorentzian template function

        self.__LorentzianFunc = lambda x: 1 - \
            loren(out.values['amplitude'], x,
                  out.values['center'], out.values['sigma'])
        self.__fit_param = out.values
        self.__CalcTemp(out.values['center'] * 1000)

    def __CalcTemp(self, f_center):
        """
        Calculation of temperature based on ODMR spectra

        Value will be written in __Temp
        """
        self.__Temp = -11.25 * f_center + 32545

    def __ConnectInstrument(self, instrument_adress):
        """Create a connection to an instrument via VISA protocol"""
        rm = visa.ResourceManager()  # Intializing VISA device manager
        try:
            # Connection to an instrument
            instr = rm.open_resource(instrument_adress)
            print("Conected to", instr.query("*IDN?"))
            return instr
        except:
            print(
                "No connection to an instrument\nCheck the address and/or the connection")
            exit()

    def __PlotSpectrumIntens(self, plot_fit=True):
        """
        Plotting measured and fitted ODMR spectra

        In case of plotting fitted spectra in the legend additional information (temerature in Kelvin, central frequency and FWHM in MHz ) will be given
        """
        print("Plotting")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams.update({'font.size': 18})
        plt.title('ODMR Spectra')

        # Plotting measured spectra
        intens = self.__intensity / self.__reference_intensity
        plt.plot(self.__freq, intens, '-o', label="Measured")
        plt.xlim(self.__freq[0], self.__freq[-1])

        # Plotting fitted spectra
        if plot_fit:
            fit_freq = np.linspace(
                self.__freq[0], self.__freq[-1], self.__freq_num * 10)
            fit_intens = self.__LorentzianFunc(fit_freq)
            plt.plot(fit_freq, fit_intens, label="\nFit\nT = " + str(round(self.__Temp)) + ' K\n' + "Center Frequency = " + str(
                round(1000 * self.__fit_param['center'])) + " MHz\n" + "FWHM = " + str(round(1000 * self.__fit_param['fwhm'])) + " MHz")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            if min(fit_intens) < min(intens):
                plt.ylim(min(fit_intens), 1)
            else:
                plt.ylim(min(intens), 1)
        else:
            plt.ylim(min(intens), 1)

        plt.xlabel('Frequency, GHz')
        plt.ylabel('Intensity, a.u.')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def __BackupLabRamFiles(self):
        """Move all files from one folder to another"""
        print("Backuping LabRam files")
        files_list = os.listdir(self.__path_to_labram_folder)
        self.__labram_files_first_last = [files_list[0], files_list[self.__freq_num]]
        for i in files_list:
            sh.move(self.__path_to_labram_folder + i,
                    self.__path_to_labram_backup_folder + i)

    def __CalcIntensity(self, path_to_file):
        """
        Integration of spectral intensity

        File in format (only numbers)::

            lambda      intensity

             530        20000
              .           .
              .           .
              .           .
        """
        data = np.loadtxt(path_to_file)
        sum_ = np.sum(data[:, 1])
        leng = len(data[:, 1])
        return sum_/leng

    def __CalcIntegIntesity(self):
        """
        Integrating full spectra intensities
        """

        print("Integrating intensities")
        # Files for integration
        files_list = os.listdir(self.__path_to_labram_folder)
        # self.__reference_intensity = self.__CalcIntensity(self.__path_to_labram_folder + files_list[0]) # Reference intensity

        # Calculation of intensities
        # Zero array of intensities
        self.__intensity = np.zeros(self.__freq_num)
        for n in range(self.__n_average):  # Averaging cycle
            for i in range(0, self.__freq_num):
                self.__intensity[i] += self.__CalcIntensity(
                    self.__path_to_labram_folder + files_list[1+i+n * self.__freq_num])

        self.__intensity /= self.__n_average

        self.__reference_intensity = max(
            self.__intensity)  # SHOULD BE DELETED!!!!

    def __FileNameCreate(self):
        """
        Return file name in format::        
            yy_mm_dd_HH_MM_SS
        """
        return tm.strftime('%y') + '_' + tm.strftime('%m') + '_' + tm.strftime('%d') + '_' + tm.strftime('%H') + '_' + tm.strftime('%M') + '_' + tm.strftime('%S')

    def __MeasODMR(self):
        """
        Start Measurements of ODMR spectra

        RF frequency will be changed after a file is created in specific folder by LabRaM

        """
        try:
            # Control generator output off
            self.__generator_instr.write('OUTPut OFF')
            self.__generator_instr.write(
                "LEVEL " + str(self.__gen_level) + "dBm")  # Generator power level

            num_files0 = len(os.listdir(self.__path_to_labram_folder))

            print("Ready to start measurments")
            print("Waiting for LabRam")

            # Creating total and current progress bars
            current_progress_step = 100 / self.__freq_num
            total_progress_step = 100 / (self.__n_average * self.__freq_num)
            total_progress = total_progress_step  # Creating total progress bar

            # Trigger
            # Measuring reference intensity level (no MW fields)
            while True:
                num_files = len(os.listdir(self.__path_to_labram_folder))
                if(num_files - num_files0 == 1):
                    # if(num_files0 != num_files):
                    break
            num_files0 = num_files

            print("Measurments Start")
            time0 = tm.time()  # Start time

            self.__generator_instr.write('OUTPut ON')  # Generator output on

            for n in range(self.__n_average):
                current_progress = current_progress_step
                for i in self.__freq:
                    
                    self.__generator_instr.write('FREQ ' + str(i) + 'GHz')  # Generator frequency
                    
                    
                    # Waiting untill a new file created
                    while True:
                        num_files = len(os.listdir(
                            self.__path_to_labram_folder))
                        if(num_files - num_files0 == 1):
                            # if(num_files0 != num_files):
                            break
                    num_files0 = num_files
                    sys.stdout.write("\rTotal progress: {0}% : Current progress: {1}%".format(
                        round(total_progress), round(current_progress)))
                    current_progress += current_progress_step
                    total_progress += total_progress_step
            self.__generator_instr.write('OUTP OFF')  # Generator output off
            print("\n----------------------------------------------------")
            print("Measurments Finished")

            self.__exec_time = tm.time() - time0  # Total measurement time, s

            print("Execusion time:",  round(self.__exec_time, 3), "sec")

        except:
            print("Measurments Aborted")
            # self.__generator_instr.write('OUTP OFF')
            self.__del__()  # destructor
            exit()

    def __CheckSystems(self):
        """
        Checking::
            - folders(Labram folder, Labram backup folder, Result folder)
            - connection to instruments (genearator)
        """
        status_check = True
        print("System check:")
        print("----------------------------------------------------")

        print("Folders:")
        if os.path.isdir(self.__path_to_labram_folder):
            print("Labram folder exists")
            file_list = os.listdir(self.__path_to_labram_folder)
            if len(file_list) != 0:
                print(" Labram Folder is not empty")
                print(" All files will be deleted")
                print(" Press any key to continue")
                on_click_button = click.getchar()
                # os.rmdir(self.__path_to_labram_folder)
                sh.rmtree(self.__path_to_labram_folder)
                os.mkdir(self.__path_to_labram_folder)
        else:
            print("Error!!! Labram folder does not exist")
            status_check = False

        if os.path.isdir(self.__path_to_labram_backup_folder):
            print("Labram backup folder exists")
        else:
            print("Error!!! Labram backup folder does not exist")
            status_check = False

        if os.path.isdir(self.__path_to_result_folder):
            print("Result folder exists")
        else:
            print("Error!!! Labram backup folder does not exist")
            status_check = False

        print("Instruments:")
        if self.__gen_con_status:
            print("Generator ready")
        else:
            print("Error!!! No connection to a generator")
            status_check = False
        print("----------------------------------------------------")

        if not status_check:
            print("Check errors and try again!!!")
            exit()

    def __PrintParam(self):
        print("----------------------------------------------------")
        print("Labram folder: " + self.__path_to_labram_folder)
        print("Labram backup folder: " + self.__path_to_labram_backup_folder)
        print("Results folder: " + self.__path_to_result_folder)
        print("Start frequency:", self.__freq_start, "GHz")
        print("Stop frequency:", self.__freq_stop, "GHz")
        print("Number of points:", self.__freq_num)
        print("Frequency step:", self.__freq_step, "GHz")
        print("RF Output Power:", self.__gen_level, "dBm")
        print("Number of averaging:", self.__n_average)
        print("Pump Laser Output Power:", self.__pump_laser_power, "%")
        print("Heat Laser Output Power:",
              self.__heat_laser_power, "mW")
        print("Measurment time of a step:", self.__meas_point_time, "sec")

        m, s = divmod(self.__estimated_meas_time, 60)
        print("Approximate measurment time:",
              int(m), "min", round(s, 3), "sec")
        print("----------------------------------------------------")

    def __WriteJSON(self):
        print("Generating JSON file")
        data_json = {
        "Measurements data": {
            "Frequency": list(np.round(self.__freq, 5)),
            "Frequency Units": "GHz",
            
            "Reference intensity": self.__reference_intensity,
            "Intensity": list(np.round(self.__intensity, 5)),
            "Intensity units": "a.u.",

            "Temperature": self.__Temp,
            "Temperature units": "K"
        },

        "Measurements parameters": {
            "Title": self.__meas_title,
            "Labram folder": self.__path_to_labram_folder,
            "Labram backup folder": self.__path_to_labram_backup_folder,
            "Results folder": self.__path_to_result_folder,
            "Start frequency": self.__freq_start,
            "Stop frequency": self.__freq_stop,
            "Frequency step": self.__freq_step,
            "Frequency units": "GHz",
            "Number of points": self.__freq_num,
            "RF output power": self.__gen_level,
            "RF output power units": "dBm",
            "Pump laser output power": self.__pump_laser_power,
            "Pump laser output power units": "%",
            "Heat laser output power": self.__heat_laser_power,
            "Heat laser output power units": "mW",
            "Measurment time step ": self.__meas_point_time,
            "Number of repition": self.__n_average,
            "Approximate measurement time": self.__estimated_meas_time,
            "Execution time": self.__exec_time,
            "Time units": "s",
            "LabRam Files": {
                "from": self.__labram_files_first_last[0],
                "to": self.__labram_files_first_last[1]
            }
        },

        "Fitting Parameters": {
            "Type": "Lorentzian",
            "Units": "GHz",
            "Center frequency": self.__fit_param['center'],
            "FWHM":  self.__fit_param['fwhm'],
            "Sigma":  self.__fit_param['sigma'],
            "Amplitude":  self.__fit_param['amplitude'],
            "Height": self.__fit_param['height']
        },

        "Additional information": self.__add_info_json

        }

        print("Saving JSON file")
        with open(self.__path_to_result_folder + self.__FileNameCreate() + ".json", "w") as json_file:
            json.dump(data_json, json_file)

# public:

    def SetGen(self, address, f_start, f_stop, f_step, level):
        """
        Setting parameters of the RF Generator:
        Input parameters::
            address - VISA address to a generator
            f_start - Start frequency
            f_stop - Stop frequency
            f_step - Frequency step
            level - Generator power level
        """
        self.__generator_instr = self.__ConnectInstrument(address)
        self.__generator_instr.write('OUTPut OFF')
        self.__gen_con_status = True
        self.__freq_start = f_start
        self.__freq_stop = f_stop
        self.__freq_step = f_step
        self.__freq = np.arange(
            self.__freq_start, self.__freq_stop + self.__freq_step, self.__freq_step)
        self.__freq_num = len(self.__freq)
        self.__gen_level = level

    def SetParamMeas(self, meas_title, n_average, pump_laser_power, heat_laser_power, meas_point_time):
        """
        Setting mesurements parameters::
            meas_title - Measurements title
            n_average - Number of averaging
            pump_laser_power - Pump laser power, %
            heat_laser_power- heat laser power, mW 
            meas_point_time - Measurement time of LabRam, s
        """
        self.__meas_title = meas_title  # Measurements title
        self.__pump_laser_power = pump_laser_power  # Pump laser power, %
        self.__heat_laser_power = heat_laser_power  # Heat laser power, mW
        self.__meas_point_time = meas_point_time  # Measurement time of LabRam, s
        self.__estimated_meas_time = meas_point_time * n_average * \
            self.__freq_num  # Estimated mesurement time, s
        self.__n_average = n_average  # Number of averaging

    def SetPathToFolders(self, path_to_labram_folder, path_to_labram_backup_folder, path_to_result_folder):
        self.__path_to_labram_folder = path_to_labram_folder  # Path to LabRam Folder
        # Path to LabRam Backup Folder
        self.__path_to_labram_backup_folder = path_to_labram_backup_folder
        self.__path_to_result_folder = path_to_result_folder  # Path to Result Folder

    def RFPowerSpectra(self, power_sens_adr):
        """
        Getting RF Power spectra

        Firstly, connect to a generator (call SetGen function)

        Input parameters::
            address - VISA address to a power sensor

        Requirements::
            -RF generator
            -Power Sensor    
        """

        self.__power_sensor_instr = self.__ConnectInstrument(
            power_sens_adr)  # Connection to power sensor
        self.__power_sensor_con_status = True

        try:
            print("Reading RF Power Spectra")
            meas_spec_level = np.empty(0)

            # Setting parameters of power sensor
            # continious/trigger measurements
            self.__power_sensor_instr.write("INIT:CONT:ALL ON")
            self.__power_sensor_instr.write('UNIT:POW DBM')  # Level units

            # Setting start frequency
            self.__power_sensor_instr.write(
                'FREQ ' + str(self.__freq[0]) + 'GHz')  # Generator
            self.__generator_instr.write(
                'FREQ ' + str(self.__freq[0]) + 'GHz')  # Power sensor

            # Generator
            self.__generator_instr.write(
                "LEVEL " + str(self.__gen_level) + "DBM")  # Generator power level
            self.__generator_instr.write("OUTP ON")  # Generator output on

            tm.sleep(2)

            for i in self.__freq:
                # Setting start frequency
                self.__power_sensor_instr.write(
                    'FREQ ' + str(i) + 'GHz')  # Power sensor
                self.__generator_instr.write(
                    'FREQ ' + str(i) + 'GHz')  # Generator
                tm.sleep(1e-1)  # Awating for setting parameters
                meas_spec_level = np.append(meas_spec_level, float(
                    self.__power_sensor_instr.query('FETC?')))  # Adding measurements

            self.__generator_instr.write("OUTP OFF")  # Generator output off
            print("Reading RF Power Spectra Finished")

            # Plotting Measurments

            plt.rcParams["figure.figsize"] = (8, 6)
            plt.rcParams.update({'font.size': 18})
            plt.grid()
            plt.plot(self.__freq, meas_spec_level)
            plt.xlabel('Frequency, GHz')
            plt.ylabel('Power, dBm')
            plt.tight_layout()
            plt.show()

        except:
            print("Reading RF Power Spectra Aborted")
            self.__generator_instr.write("OUTP OFF")
            self.__generator_instr.close()
            print(self.__power_sensor_instr.query('ERR?'))
            self.__power_sensor_instr.write("*CLS?")
            self.__power_sensor_instr.close()

    def Comment(self, comment):
        """
        Creates a comment which will be parsed and added in the .json file in "Additional information"

        Write comment in the form string::
            ' "key1": value1, "key2":value2, ...  '

        if value is::
            number value1 = 1
            string value1 = "string"
            array = [1, 2, 3]

        """

        self.__add_info_json = json.loads(u'{' + comment + '}')

    def Start(self):
        self.__PrintParam()
        self.__CheckSystems()
        print("Press any key to start measurements")
        on_click_button = click.getchar()
        print("----------------------------------------------------")
        self.__MeasODMR()
        print("----------------------------------------------------")
        self.__CalcIntegIntesity()
        self.__BackupLabRamFiles()
        self.__LorentzianFit()
        self.__WriteJSON()
        self.__PlotSpectrumIntens()
