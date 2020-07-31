# %%
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import math
from sklearn.metrics import confusion_matrix
from .data_retrieval import ventilator_sql_data_retrieval
from datetime import datetime, timedelta
from math import log
import os.path
import logging


class ventilator_prediction:
    # the average pf and standard deviation stored to avoid bogging down the response calculating them every time
    pf_average = 0
    pf_std_dev = 0
    # and then to get the
    pf_average_and_std_dev_calculated = 0

    def __init__(self):

        global sess
        global saver
        global get_sql_data
        self.logger = logging.getLogger("ventilator_prediction." + __name__)
        get_sql_data = ventilator_sql_data_retrieval()

        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()

        currentPath = os.path.dirname(os.path.realpath(__file__))

        saver = tf.compat.v1.train.import_meta_graph(
            currentPath + "/VET_June27_model/VET_model.meta"
        )
        saver.restore(
            sess, tf.train.latest_checkpoint(currentPath + "/VET_June27_model/")
        )

    # checks that a specified enc_csn_id exists with good data
    def check_pat_enc_csn_id(self, pat_enc_csn_id):
        get_pat_csn_result = get_sql_data.retrieve_pat_csn(
            pat_enc_csn_id, num_hours_ahead=6
        )
        if get_pat_csn_result.empty:
            return False
        else:
            return True

    # gets a list of paitent csn id to get for the demo
    def get_pat_csn_list(self):

        pat_csn_list = []

        get_pat_csn_results = get_sql_data.retrieve_pat_csn(
            1, patients=100000, num_hours_ahead=6
        )
        random_pat_csn = get_pat_csn_results.sample(n=10)
        for row in random_pat_csn.itertuples():
            # adds the pat_enc_csn_id to the list as a string so it can be selected from a dropdown list
            pat_csn_list.append(str(int(row[1])))
        return pat_csn_list

    def retriveData(
        self, rnn=12, patients=2, hours_ahead=0, n_meds=1, n_covs=1, enc_csn_id="1"
    ):
        # arrays to be populated and returned
        num_obs_time = []
        num_obs_values = []
        num_rnn_grid_times = []
        rnn_grid_times = []
        labels = []
        T = []
        Y = []
        ind_kf = []
        ind_kt = []
        meds_on_grid = []
        baseline_covs = []

        print("Starting retrive Lahey data...")
        # first selects the distinct encounters

        pat_enc = get_sql_data.retrieve_pat_csn(
            enc_csn_id=enc_csn_id, patients=patients, num_hours_ahead=hours_ahead
        )

        baseline_covs = np.zeros((len(pat_enc), n_covs))

        i = 0
        for row in pat_enc.itertuples():

            pats_enc_id = row[1]

            # then for each encounter get a list of all data points taken

            pat_enc = get_sql_data.retrive_enc_data(
                pat_enc_csn_id=enc_csn_id,
                num_hours_ahead=hours_ahead,
                trim_hours_ahead=True,
            )
            # and tranforms them into the proper format
            pat_y, pat_ind_kf, pat_ind_kt, pat_label, pat_T = self.transformData(
                pat_enc
            )

            # before appending them to the relavent mesures
            pat_T = list(sorted(set(pat_T), key=lambda x: pat_T.index(x)))
            Y.append(np.array(pat_y))
            ind_kf.append(np.array(pat_ind_kf).astype(int))
            ind_kt.append(np.array(pat_ind_kt).astype(int))
            # then goes on to calcualted or retried other info needed

            # the T values for just this paitent
            T.append(np.array(pat_T))
            # number of times this paitent was observed
            num_obs_time.append(len(pat_T))
            # and how many mesurements were developed
            num_obs_values.append(len(pat_y))
            # calculates end time
            end_time = max(pat_T)
            # calculets the num_rnn_gird times observed by dividing the end time by 12 and rounding down

            num_rnn_grid_times.append(math.floor(end_time / rnn) + 1)
            # rnn_grid_times.append(np.arange(num_rnn_grid_times[i]))
            rnn_grid_times.append(
                (np.arange(start=0, stop=np.floor(end_time) + 1, step=rnn, dtype="int"))
            )
            # adds label to grid
            labels.append(int(pat_label))
            # files in the med grid with zeros
            # meds_on_grid.append(np.zeros((num_rnn_grid_times[i],n_meds),dtype="int"))

            meds = np.array(np.zeros((num_rnn_grid_times[i], n_meds), dtype="int"))
            meds_on_grid.append(meds)

            # baseline_covs.append(np.zeros(n_covs,dtype="int"))

            # grabs the label for the

            i += 1
            # converts all arrays to numpy arrays before returning them

        num_obs_time = np.array(num_obs_time)
        num_obs_values = np.array(num_obs_values)
        num_rnn_grid_times = np.array(num_rnn_grid_times)
        rnn_grid_times = np.array(rnn_grid_times)
        labels = np.array(labels)
        T = np.array(T)
        Y = np.array(Y)
        ind_kf = np.array(ind_kf)
        ind_kt = np.array(ind_kt)
        meds_on_grid = np.array(meds_on_grid)
        baseline_covs = np.array(baseline_covs)

        return (
            num_obs_time,
            num_obs_values,
            num_rnn_grid_times,
            rnn_grid_times,
            labels,
            T,
            Y,
            ind_kf,
            ind_kt,
            meds_on_grid,
            baseline_covs,
        )

    # takes the dataframe from the database and tranforms its into the correct 2 dimensional array
    def transformData(self, patDataFrame):
        # the array the holds the value of the measurements
        y = []
        # array that holds what value the measurements was
        ind_kf = []
        # array that holds the time the measurements was taken at
        ind_kt = []
        T = []
        pat_label = 0

        features = [
            "RESPIRATIONS",
            "TEMPERATURE",
            "MAP",
            "PULSE",
            "P/F RATIO",
            "BMI",
            "C-REACTIVE PROTEIN",
            "FERRITIN",
            "ABSOLUTE LYMPH CT",
            "R LH SEPSIS WATCH SCORE",
            "LD",
            "age",
            "POTASSIUM",
            "CREATININE",
            "ANION GAP",
            "HEMATOCRIT",
            "WBC",
            "PLATELET COUNT",
            "POLYS",
            "ABSOLUTE GRAN CT",
            "LYMPHOCYTE",
            "MONOCYTE",
            "EOSINOPHIL",
            "BASOPHIL",
            "ABSOLUTE MONO CT",
            "ABSOLUTE EOS CT",
            "ABSOLUTE BASO CT",
            "ALBUMIN",
            "AST SGOT",
            "GFR NON AFRICAN-AMER (CKD-EPI)",
            "MCHC",
            "CREATINE KINASE",
            "PO2",
            "PH",
            "PCO2",
            "BICARBONATE",
            "O2HB",
            "PROCALCITONIN",
            "LACTIC ACID",
            "D-DIMER",
        ]
        # components that need to have the log base 2 taken before z score is calcualted
        log_base_2_components = ["FERRITIN", "ABSOLUTE LYMPH CT", "LD"]
        # components that need to have the log base 10 taken before z score is calcualted
        log_base_10_components = ["C-REACTIVE PROTEIN"]
        feature_indices = list(range(len(features)))
        # We need to index the features by
        measurementNameKF = dict(zip(features, feature_indices))
        avg_and_std_deviation = get_sql_data.retrive_avg_and_std_deviation()

        # only calcuuatles the pf average if either the pf_average is blank aka 0 or if more then a hour has passed since it has last been calcualted
        if self.pf_average == 0:
            self.pf_std_dev, self.pf_average = self.calc_pf_average_and_std_dev()
            self.pf_average_and_std_dev_calculated = datetime.now()
        elif (
            self.pf_average_and_std_dev_calculated - datetime.now()
        ).total_seconds() / 3600 > 1:
            self.pf_std_dev, self.pf_average = self.calc_pf_average_and_std_dev()
            self.pf_average_and_std_dev_calculated = datetime.now()
        # itterates through the dataframe passed in to populate the array
        for row in patDataFrame.itertuples():
            y_val = row[1]
            # takes log of values if they are values that need to have that done
            if row[2] in log_base_2_components:
                y_val = log(y_val, 2)
            elif row[2] in log_base_10_components:
                y_val = log(y_val, 10)

            # if the row is pf/ratio does aditional calcuation to calculate pf ratio before calculating the z scored value
            if row[2] == "P/F RATIO":
                fio2 = self.estimate_fio2(row[8], float(row[9]))
                pf_ratio = self.estimate_pf_ratio(
                    self.estimate_pao2(float(row[7])), fio2
                )
                y.append((pf_ratio - self.pf_average) / self.pf_std_dev)
            else:
                y.append(
                    (y_val - avg_and_std_deviation[row[2]][0])
                    / avg_and_std_deviation[row[2]][1]
                )
            pat_label = row[4]

            # converts the measurement name to a number before adding it
            ind_kf.append(measurementNameKF[row[2]])
            ind_kt.append(row[5])
            T.append(row[3])
        return y, ind_kf, ind_kt, pat_label, T

    # gets the overall average and standar deviation for pf ratios
    def calc_pf_average_and_std_dev(self):
        pf_ratios = []
        raw_data = get_sql_data.retrive_pf_data()
        for row in raw_data.itertuples():
            fio2 = self.estimate_fio2(row[4], float(row[5]))
            pf_ratios.append(
                self.estimate_pf_ratio(self.estimate_pao2(float(row[3])), fio2)
            )
        raw_data["pf_ratio"] = pf_ratios
        std_dev = raw_data.pf_ratio.std()
        average = raw_data.pf_ratio.mean()
        return std_dev, average

    def sao2_fio2_ratio(self, pulse_oximetry, fio2):
        """ Simple ratio of spo2/fio2.
            Parameters: pulse_oximetry, fio2 """

        if (pulse_oximetry < 0) or (pulse_oximetry > 100):
            raise ValueError(
                f"{pulse_oximetry} Not a valid SPO2 (pulse oximeter) value. Has to be between 0 and 100."
            )

        if (fio2 < 0) or (fio2 > 1):
            raise ValueError(
                f"{fio2} Not a valid fraction of inspired oxygen FiO2. Has to be between 0 and 1"
            )

        return pulse_oximetry / fio2

    def estimate_fio2(self, oxygen_device, oxygen_flow_rate):

        """Estimates the fio2 from the device being used to supply oxygen and fio2.
        For devices like nasal cannulas, the commonest device, we have to estimate the fio2
        In devices such as ventilators, this is set by the  therapists. 
        Parameters: oxygen_device: there are a plethora of devices by which oxygen can be given. 
        The commonest one is Nasal cannula. """

        fio2_set_on_device = [
            "T-Piece",
            "Bi-PAP",
            "High flow nasal cannula",
            "Transtracheal catheter",
            "CPAP",
            "High Flow",
            "Ventilator",
            "Aerosol mask",
            "Trach mask",
            "Bubble CPAP",
            "Venturi mask",
        ]
        # 'Other (Comment)',

        if oxygen_device in fio2_set_on_device:
            return oxygen_flow_rate / 100

        if oxygen_device == "Other (Comment)":
            return 0.21
        # with a nasal cannula, we assume that the fraction of oxygen that is inspired
        # (above the normal atmospheric level or 20%)
        # increases by 4% for every additional liter of oxygen flow administered

        if oxygen_device == "Nasal cannula":
            return 0.21 + 0.03 * oxygen_flow_rate

        # commonest case
        if oxygen_device == "None (Room air)":
            return 0.21

        if oxygen_device == "MistyOx":
            # 60 percent - 96 percent FiO2 range
            # Primary jet flow range 42-80 LPM
            # 0.6 = a + b*42
            # 0.96= a +b*80
            # 0.36 = b*38 => b == (0.36/38)
            # 0.6 - (0.36/38)*42 == 0.2021 == a
            return 0.2021 + 0.00947 * oxygen_flow_rate

        if oxygen_device == "Face tent":
            # Delivers only 40% Oxygen at 10-15 liters per minute
            # just use the face mask equation
            return 0.1 + 0.4167 * oxygen_flow_rate

        # Simple face mask ~6-12 L/min supplying 35-60%*
        # 0.35 = a +b*6, 0.6 = a + b*12 => b = (0.25/6), a = 0.1
        if oxygen_device == "Simple mask":
            return 0.1 + 0.4167 * oxygen_flow_rate

        # 25% at 1.5L, 90% at 15L
        # parameters derived
        if oxygen_device == "Oxymask":
            return 0.17777 + 0.048 * oxygen_flow_rate

        if oxygen_device == "Oxymizer Pendant Nasal Cannula":
            if oxygen_flow_rate <= 0.5:
                return 0.26
            if oxygen_flow_rate <= 1:
                return 0.32
            if oxygen_flow_rate < 10:
                return 0.32 + (oxygen_flow_rate - 1) * 0.0445
            # 10.0 LPM = 72% FiO2
            # 11.0 LPM = 77% FiO2
            # 12.0 LPM = 82% FiO2
            if oxygen_flow_rate >= 10 and oxygen_flow_rate <= 12:
                return 0.72 + (oxygen_flow_rate - 10) * 0.02

            if oxygen_flow_rate > 12:
                return 0.82 + (oxygen_flow_rate - 10) * 0.01

        if oxygen_device == "Blow-by":

            if oxygen_flow_rate > 6:
                return 0.5
            if oxygen_flow_rate > 2.9:
                return 0.3
            if oxygen_flow_rate < 2.9:
                return 0.21

            # use https://www.nejm.org/doi/suppl/10.1056/NEJMoa2012410/suppl_file/nejmoa2012410_appendix.pdf
        if oxygen_device == "Non-rebreather":
            return 0.8

        # if (oxygen_device == "Oxymizer Pendant Nasal Cannula"):
        #    return a + b*oxygen_flow_rate
        # FiO2 is variable with ambu bags, fortunately few cases

        if oxygen_device == "Ambu Bag":
            return 0.8

        raise ValueError(f"{oxygen_device} not found in list of devices")

    ## Taken from https://www.intensive.org/epic2/Documents/Estimation%20of%20PO2%20and%20FiO2.pdf
    def estimate_pao2(self, sao2):
        if sao2 < 75:
            return sao2 / 2
        if sao2 > 99:  #  an error
            return 145

        # SO2:(%):PaO2:(mmHg)
        table = {
            75: 40,
            76: 41,
            77: 41.5,
            78: 42,
            79: 43,
            80: 44,
            81: 45,
            82: 46,
            83: 47,
            84: 49,
            85: 50,
            86: 52,
            87: 53,
            88: 55,
            89: 57,
            90: 60,
            91: 62,
            92: 65,
            93: 69,
            94: 73,
            95: 79,
            96: 86,
            97: 96,
            98: 112,
            99: 145,
        }
        return table[sao2]

    def estimate_pf_ratio(self, sao2, fio2):
        """Estimates the pf_ratio from  pulse oximetry data and fio2.
        Parameters: sao2 measured by pulse oximetry; typical values are in the 80 to 100 range. 
        fio2: the fraction of oxygen in inspired air. In devices such as ventilators, this is set by the  therapists. For devices like nasal cannulas, the commonest device, we have to estimate the fio2"""
        if fio2 == 0:
            return 0
        return self.estimate_pao2(sao2) / fio2

    def pad_rawdata(self, T, Y, ind_kf, ind_kt, X, meds_on_grid, covs):
        """ 
        Helper func. Pad raw data so it's in a padded array to be fed into the graph,
        since we can't pass in arrays of arrays directly.
        
        Inputs:
            arrays of data elements:
                T: array of arrays, with raw observation times
                Y,ind_kf,ind_kt: array of arrays;
                    observed lab/vitals,
                    indices into Y (same size)
                X: grid points
                meds_on_grid: list of arrays, each is grid_size x num_meds
                covs: matrix of baseline covariates for each patient. 
                    to be tiled at each time and combined w meds
        Returns:
            Padded 2d np arrays of data, now of dim batchsize x batch_maxlen
        """
        N = np.shape(T)[0]  # num in batch
        num_meds = np.shape(meds_on_grid[0])[1]
        num_covs = np.shape(covs)[1]

        T_lens = np.array([len(t) for t in T])
        T_maxlen = np.max(T_lens)
        T_pad = np.zeros((N, T_maxlen))

        Y_lens = np.array([len(y) for y in Y])
        Y_maxlen = np.max(Y_lens)
        Y_pad = np.zeros((N, Y_maxlen))
        ind_kf_pad = np.zeros((N, Y_maxlen))
        ind_kt_pad = np.zeros((N, Y_maxlen))
        grid_lens = np.array([np.shape(m)[0] for m in meds_on_grid])
        grid_maxlen = np.max(grid_lens)
        meds_cov_pad = np.zeros((N, grid_maxlen, num_meds + num_covs))
        X_pad = np.zeros((N, grid_maxlen))

        for i in range(N):

            T_pad[i, : T_lens[i]] = T[i]
            Y_pad[i, : Y_lens[i]] = Y[i]
            ind_kf_pad[i, : Y_lens[i]] = ind_kf[i]
            ind_kt_pad[i, : Y_lens[i]] = ind_kt[i]
            X_pad[i, : grid_lens[i]] = X[i]
            meds_cov_pad[i, : grid_lens[i], :num_meds] = meds_on_grid[i]
            meds_cov_pad[i, : grid_lens[i], num_meds:] = np.tile(
                covs[i], (grid_lens[i], 1)
            )

        return T_pad, Y_pad, ind_kf_pad, ind_kt_pad, X_pad, meds_cov_pad

    # gets and prints the patents discharge disposition
    def get_vent_label(self, enc_csn_id):

        get_discharge_disposition_res = get_sql_data.retrive_vent_label(enc_csn_id)

        for row in get_discharge_disposition_res.itertuples():
            return str(row[1])

    def get_training_label(self, enc_csn_id):

        get_discharge_disposition_res = get_sql_data.retrive_training_label(enc_csn_id)

        for row in get_discharge_disposition_res.itertuples():
            return str(int(row[1]))

    def get_prediction(self, n_hours_ahead=6, s_enc_csn_id="218348727"):

        #####
        ##### Setup ground truth and sim some data from a GP
        #####
        num_encs = 1
        M = 50
        n_covs = 1
        n_meds = 1
        Ntr = 1
        batch_size = 1

        (
            num_obs_times,
            num_obs_values,
            num_rnn_grid_times,
            rnn_grid_times,
            labels,
            times,
            values,
            ind_lvs,
            ind_times,
            meds_on_grid,
            covs,
        ) = self.retriveData(6, num_encs, n_hours_ahead, enc_csn_id=s_enc_csn_id)

        (
            T_pad_te,
            Y_pad_te,
            ind_kf_pad_te,
            ind_kt_pad_te,
            X_pad_te,
            meds_cov_pad_te,
        ) = self.pad_rawdata(
            times, values, ind_lvs, ind_times, rnn_grid_times, meds_on_grid, covs
        )

        num_obs_times_te = num_obs_times
        num_obs_values_te = num_obs_values
        num_rnn_grid_times_te = num_rnn_grid_times
        labels_te = labels

        preds = sess.graph.get_tensor_by_name("preds:0")
        probs = sess.graph.get_tensor_by_name("probs:0")
        Y = sess.graph.get_tensor_by_name("Y:0")
        T = sess.graph.get_tensor_by_name("T:0")
        ind_kf = sess.graph.get_tensor_by_name("ind_kf:0")
        ind_kt = sess.graph.get_tensor_by_name("ind_kt:0")
        X = sess.graph.get_tensor_by_name("X:0")
        med_cov_grid = sess.graph.get_tensor_by_name("med_cov_grid:0")
        O = sess.graph.get_tensor_by_name("O:0")
        num_obs_times = sess.graph.get_tensor_by_name("num_obs_times:0")
        num_obs_values = sess.graph.get_tensor_by_name("num_obs_values:0")
        num_rnn_grid_times = sess.graph.get_tensor_by_name("num_rnn_grid_times:0")

        feed_dict = {
            Y: Y_pad_te,
            T: T_pad_te,
            ind_kf: ind_kf_pad_te,
            ind_kt: ind_kt_pad_te,
            X: X_pad_te,
            med_cov_grid: meds_cov_pad_te,
            num_obs_times: num_obs_times_te,
            num_obs_values: num_obs_values_te,
            num_rnn_grid_times: num_rnn_grid_times_te,
        }
        # ,O:labels_te}

        te_probs = sess.run([probs], feed_dict)

        prediction = float(te_probs[0])
        actual_result = self.get_training_label(s_enc_csn_id)
        ventilator_used = self.get_vent_label(s_enc_csn_id)
        results = {
            "Predicted need for respiratory support": "{0:.0%}".format(prediction),
            "Type of  respiratory support used": ventilator_used,
        }

        return results
