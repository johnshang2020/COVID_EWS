import numpy as np
import pandas as pd
import sys
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine


# This is backend class used by sepsis prediction (and others) to handle the
# retrival of the data, the processing of the data
# is still often done in the frontedn class
class EWS_sql_data_retrieval:
    _SERVERNAME = "WVECLADBP01SS.ad.laheyhealth.org"
    _DATABASE = "Datarepo_DB"
    _USERPWD = "data_exploration:Lahey123#"
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

    # retrives either a singular or list of paitent csn id
    def retrieve_pat_csn(self, enc_csn_id, patients=1, num_hours_ahead=24):
        # paramters for prepared statement
        statement_params = [num_hours_ahead]
        # most of the sql statement that stays the same
        sql_statement_base = f"""
                            SELECT DISTINCT TOP {patients} [pat_enc_csn_id] 
                            FROM <covid feature col table see spec sheet for detail>  """
        if patients == 1:
            sql_statement = sql_statement_base + "Where pat_enc_csn_id = ?"
            statement_params = [num_hours_ahead, enc_csn_id]
        else:
            sql_statement = sql_statement_base + "\n)"
        results = pd.read_sql(sql_statement, self._ENGINE, params=statement_params)
        return results

    def retrive_enc_data(
        self, pat_enc_csn_id, num_hours_ahead=24, trim_hours_ahead=True
    ):
        enc_data_sql = f"""
                            SELECT [ord_num_value], [component_name],[hours_since_admitted],
                            discharge_disposition
                            FROM <covid feature col table see spec sheet for detail> 
                            WHERE 
                            a.[pat_enc_csn_id]=? 
                            ORDER BY [hours_since_admitted]"""
        enc_data_params = [pat_enc_csn_id]
        enc_data = pd.read_sql(enc_data_sql, self._ENGINE, params=enc_data_params)
        # dense ranks based on time
        enc_data["ind_kt"] = (
            enc_data.hours_since_admitted.rank(method="dense").astype(int) - 1
        )
        # finds length of paitents stay
        length_of_stay = enc_data.hours_since_admitted.max(axis=0)
        # then adds it to each row typo is intentionaly carrying over typo from old code to avoid breaking things
        enc_data["lengh_of_stay"] = length_of_stay
        # unless told otherwise trims values from the end of dataset within num_hours_ahead from the length of stay
        # eg if length of stay is 60 and num_hours ahead is 12 it will trim the last twelve hours of data only returning
        # data from 0-48 hours
        if trim_hours_ahead == True:
            enc_data = enc_data.query(
                "hours_since_admitted<=@length_of_stay-@num_hours_ahead"
            )
        return enc_data

    # retrives a list of the avg and std deviation for a series of mesuremetns and puts them in a dictionary where the key is the mesurement name
    # and the value is a array of the avgerage score followed by the standard devation
    def retrive_avg_and_std_deviation(self):
        sql_statement = """SELECT [component_name],[avg_score],[stdev_score]
                        FROM  <feature z lookup table see spec sheet for detail>"""
        data = pd.read_sql(sql_statement, self._ENGINE)
        avg_and_std_deviation = dict()
        for row in data.itertuples():
            avg_and_std_deviation.update({row[1]: [row[2], row[3]]})
        return avg_and_std_deviation

    # returns a patents discharge dispostion
    def retrive_discharge_disposition(self, enc_csn_id):
        get_discharge_disposition = """SELECT discharge_disposition
                                FROM <covid feature col table see spec sheet for detail>
                                    WHERE PAT_ENC_CSN_ID=?"""
        get_discharge_disposition_res = pd.read_sql(
            get_discharge_disposition, self._ENGINE, params=[enc_csn_id]
        )
        for row in get_discharge_disposition_res.itertuples():
            return str(int(row[1]))

    # returns a paitents epic discharge disposition
    def retrive_discharge_disposition_epic(self, enc_csn_id):
        get_discharge_disposition = """SELECT name
                                        FROM <covid feature col table see spec sheet for detail>
                                            WHERE PAT_ENC_CSN_ID=?"""
        get_discharge_disposition_res = pd.read_sql(
            get_discharge_disposition, self._ENGINE, params=[enc_csn_id]
        )

        for row in get_discharge_disposition_res.itertuples():
            return str(row[1])
