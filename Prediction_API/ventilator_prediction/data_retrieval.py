import numpy as np
import pandas as pd
import sys
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import logging


# class used by ventilator prediction model to retrive data from sql database
#
class ventilator_sql_data_retrieval:
    _SERVERNAME = ""
    _DATABASE = ""
    _USERPWD = ""
    _DRIVER = ""
    _ENGINE = create_engine(
        "mssql+pyodbc://"
        + _USERPWD
        + "@"
        + _SERVERNAME
        + "/"
        + _DATABASE
        + "?"
        + _DRIVER
    )

    def __init__(self):
        global logger
        logger = logging.getLogger("ventilator_prediction." + __name__)

    # retrives either a singular or list of paitent csn id
    def retrieve_pat_csn(self, enc_csn_id, patients=1, num_hours_ahead=6):
        # paramters for prepared statement
        statement_params = []
        # most of the sql statement that stays the same
        sql_statement_base = f"""select TOP {patients} [pat_enc_csn_id] FROM <covid_feature table see detail in spec sheet> """
        if patients == 1:
            sql_statement = sql_statement_base + "WHERE pat_enc_csn_id = ?"
            statement_params.append(enc_csn_id)
        results = pd.read_sql(sql_statement, self._ENGINE, params=statement_params)
        return results
    #retrives all of the data about a paitents encounter, including all mesurements taken
    def retrive_enc_data(
        self, pat_enc_csn_id, num_hours_ahead=6, trim_hours_ahead=True
    ):
        enc_data_sql = f""" select [ord_num_value]                            
                            ,[component_name]
                            ,x.[hours_since_admitted]
                            ,v_label
                            ,[PULSE OXIMETRY]
                            ,[R OXYGEN DEVICE]
                            ,[R OXYGEN FLOW RATE]
                            from <covid_feature column table see detail in spec sheet>
                            ORDER BY x.[hours_since_admitted]"""
        enc_data_params = [pat_enc_csn_id]
        enc_data = pd.read_sql(enc_data_sql, self._ENGINE, params=enc_data_params)
        # dense ranks based on time
        enc_data.insert(
            4,
            "ind_kt",
            (enc_data.hours_since_admitted.rank(method="dense").astype(int) - 1),
        )
        # finds length of paitents stay
        length_of_stay = enc_data.hours_since_admitted.max(axis=0)
        # and adds it to each row typo is intentionaly carrying over typo from old code to avoid breaking things
        enc_data.insert(5, "lengh_of_stay", length_of_stay)

        # unless told otherwise trims values from the end of dataset within num_hours_ahead from the length of stay
        # eg if length of stay is 60 and num_hours ahead is 12 it will trim the last twelve hours of data only returning
        # data from 0-48 hours
        if trim_hours_ahead == True:
            enc_data = enc_data.query(
                "hours_since_admitted<=@length_of_stay-@num_hours_ahead"
            )
        return enc_data

    # retrives a list of the avg and std deviation for a series of mesuremetns and puts them in a dictionary where the key is the mesurement name
    # and the value is a array of the avgerage score followed by the standard devation the exeption is the pulse oximetry score as that needs to be calculated
    def retrive_avg_and_std_deviation(self):
        sql_statement = """SELECT [component_name],[avg_score],[stdev_score]
                        FROM <feature_z_lookup table see detail in spec sheet > """
        data = pd.read_sql(sql_statement, self._ENGINE)
        avg_and_std_deviation = dict()
        for row in data.itertuples():
            avg_and_std_deviation.update({row[1]: [row[2], row[3]]})
        return avg_and_std_deviation

    # retives all the data related to calculat pf values for all encounters, this is done to get the average pf ratio and the standard devation to calculate z scores
    def retrive_pf_data(self):
        select_pf_average_and_std_dev = """SELECT [pat_enc_csn_id]
                        ,[hours_since_admitted]
                        ,[PULSE OXIMETRY]
                        ,[R OXYGEN DEVICE]
                        ,[R OXYGEN FLOW RATE]
                        FROM <pf_ratio table see spec for details>"""
        raw_data = pd.read_sql(select_pf_average_and_std_dev, self._ENGINE)
        return raw_data

    def retrive_vent_label(self, pat_enc_csn_id):
        get_discharge_disposition = """select   a.meas_value 
        from <covid patient flowsheet see spec for details>
        where a.pat_enc_csn_id=?
 
        """
        parameters = [pat_enc_csn_id]
        results = pd.read_sql(
            get_discharge_disposition, self._ENGINE, params=parameters
        )
        return results

    def retrive_training_label(self, pat_enc_csn_id):
        get_discharge_disposition = """SELECT v_label
                                    FROM <ventilator label table see spec sheet for details>
                                        WHERE PAT_ENC_CSN_ID=?"""
        parameters = [pat_enc_csn_id]
        results = pd.read_sql(
            get_discharge_disposition, self._ENGINE, params=parameters
        )
        return results
