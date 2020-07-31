# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import math
from sklearn.metrics import confusion_matrix
from .data_retrieval import EWS_sql_data_retrieval
import os.path
import logging


class EWS_prediction:

    # constructor that restores the saved sessions
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        global sess
        global saver
        global get_sql_data
        sess = tf.compat.v1.Session()

        currentPath = os.path.dirname(os.path.realpath(__file__))
        saver = tf.compat.v1.train.import_meta_graph(
            currentPath + "/EWS_model/EWS_model.meta"
        )
        saver.restore(sess, tf.train.latest_checkpoint(currentPath + "/EWS_model/"))
        get_sql_data = EWS_sql_data_retrieval()
        self.logger = logging.getLogger("EWS_prediction." + __name__)

    # checks that a paitent id exists
    def check_pat_enc_csn_id(self, pat_enc_csn_id):
        get_pat_csn_result = get_sql_data.retrieve_pat_csn(
            pat_enc_csn_id, num_hours_ahead=72
        )
        if get_pat_csn_result.empty:
            return False
        else:
            return True

    # gets a list of paitent csn id to get for the demo
    def get_pat_csn_list(self):

        pat_csn_list = []

        get_pat_csn_result = get_sql_data.retrieve_pat_csn(
            1, patients=100000, num_hours_ahead=72
        )
        random_pat_csn = get_pat_csn_result.sample(n=10)
        for row in random_pat_csn.itertuples():
            # adds the pat_enc_csn_id to the list as a string so it can be selected from a dropdown list
            pat_csn_list.append(str(int(row[1])))
        return {"patient encounter ids": pat_csn_list}

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
        # Create the connection
        # first selects the distinct encounters

        pat_enc = get_sql_data.retrieve_pat_csn(
            enc_csn_id, patients=patients, num_hours_ahead=hours_ahead
        )

        baseline_covs = np.zeros((len(pat_enc), n_covs))

        i = 0
        for row in pat_enc.itertuples():

            pats_enc_id = row[1]

            # then for each encounter get a list of all data points taken
            pat_enc = get_sql_data.retrive_enc_data(pats_enc_id, hours_ahead)
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
        # large dict to convert name of mesurement to number
        #    features = [  "ABSOLUTE LYMPH CT",
        #        "BMI",  # static feature
        #        "C-REACTIVE PROTEIN",
        #        "FERRITIN",  # very important dynamic feature
        #        "HEIGHT", # static feature
        #        "MAP",
        #        "PULSE",
        #        "PULSE OXIMETRY",
        #        "R FIO2",
        #        "R LH SEPSIS WATCH SCORE", # USE AS LABEL
        #        "R OXYGEN FLOW RATE":11,
        #        "R OXYGEN DELIVERY DEVICE":12,
        #        "R VENT PRESS SUPPORT":13,
        #        "R VENT RESP RATE (SET)":14,
        #        "R VENT TIDAL VOLUME OBSERVED":15,
        #        "R VENT TIDAL VOLUME SET":16,
        #        "RESPIRATIONS":17,
        #        "TEMPERATURE":18,
        #        "WEIGHT/SCALE":19
        # ]
        features = [
            "RESPIRATIONS",
            "TEMPERATURE",
            "MAP",
            "PULSE",
            "PULSE OXIMETRY",
            "BMI",
            "R FIO2",
            "R OXYGEN FLOW RATE",
            "R OXYGEN DEVICE",
            "C-REACTIVE PROTEIN",
            "FERRITIN",
            "ABSOLUTE LYMPH CT",
            "R LH SEPSIS WATCH SCORE",
        ]
        feature_indices = list(range(len(features)))
        # We need to index the features by
        measurementNameKF = dict(zip(features, feature_indices))
        avg_and_std_deviation = get_sql_data.retrive_avg_and_std_deviation()

        # itterates through the dataframe passed in to populate the array
        i = 0
        for row in patDataFrame.itertuples():

            if i == 0:
                pat_label = row[4]

            # calculates z score for the mesurement and appends it, the z score is calculated by subtracting the raw score and dividing by the standard deviation
            y.append(
                (row[1] - avg_and_std_deviation[row[2]][0])
                / avg_and_std_deviation[row[2]][1]
            )
            # converts the measurement name to a number before adding it
            ind_kf.append(measurementNameKF[row[2]])
            ind_kt.append(row[5])
            T.append(row[3])
        return y, ind_kf, ind_kt, pat_label, T

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

    def get_prediction(self, n_hours_ahead=24, s_enc_csn_id="218348727"):
        seed = 8675309
        #####
        ##### Setup ground truth and sim some data from a GP
        #####
        num_encs = 1

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

        te_probs = sess.run([probs], feed_dict)
        # gathers prediction and actual results
        prediction = float(te_probs[0])
        discharge_disposition = get_sql_data.retrive_discharge_disposition(s_enc_csn_id)
        epic_discharge_disposition = get_sql_data.retrive_discharge_disposition_epic(
            s_enc_csn_id
        )
        # puts in dict to be returned
        results = {
            "Predicted Mortality": "{0:.0%}".format(prediction),
            "Condition at discharge": epic_discharge_disposition,
        }
        return results


# %%
