from __future__ import print_function
import os

import numpy as np
import pandas as pd
import sys
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import time
import  matplotlib.pyplot as plt
import seaborn as sns

ServerName = "xxxx"
Database = "xxxx
UserPwd = "xxxxx"
Driver = "DRIVER={sqlsrv}"

_engine = None

def get_data(sql):
	''' Enter sql, get back data. Parameter: sql string Usage: training, test = get_data('select * from foo')'''    

	if (_engine == None):  
		engine = create_engine('mssql+pyodbc://' + UserPwd + '@' + ServerName + '/' + Database + "?" + Driver)

	return pd.read_sql(sql, engine)


def get_connect_string():
    ''' Returns a connection string: useful when we want to isssue queries from the notebook using %sql magic '''
    return 'mssql+pyodbc://' + UserPwd + '@' + ServerName + '/' + Database + "?" + Driver

def getCovidFeatures(table_name):
    ''' There are three covid tables created by John - this simplifies access to them '''
    sql = f'''select a.pat_id,  
    cast(cast(datediff(minute, b.admit_date, a.[result_time]) as DECIMAL(8,2)) /60 as  DECIMAL(8,2))  hours_since_admitted, 
    ord_num_value [value], ord_value [text_value],
    component_name
        from [COVID].[{table_name}] a 
        left join 
        (select pat_id, min( [admit_date]) admit_date from  [COVID].[covid_admits] group by pat_id) b on a.pat_id=b.pat_id 
        where 
        [component_name] in( 'Ferritin' ,'C-REACTIVE PROTEIN') and 
        cast(cast(datediff(minute, b.admit_date , a.[result_time]) as  DECIMAL(8,2)) /60 as  DECIMAL(8,2)) < 3*7*24
        and ord_num_value< 99999'''
    
    return get_data(sql)


def getCovidFeatures_w_disposition(table_name):
    ''' There are three covid tables created by John - this simplifies access to them '''
    sql = f'''select a.pat_id,  
    cast(cast(datediff(minute, b.admit_date, a.[result_time]) as DECIMAL(8,2)) /60 as  DECIMAL(8,2))  hours_since_admitted, 
    ord_num_value [value], ord_value [text_value],
    component_name,discharge_disposition
        from [COVID].[{table_name}] a 
        left join 
        (select pat_id, min( [admit_date]) admit_date from  [COVID].[covid_admits] group by pat_id) b on a.pat_id=b.pat_id 
        where 
        [component_name] in( 'Ferritin' ,'C-REACTIVE PROTEIN') and 
        cast(cast(datediff(minute, b.admit_date , a.[result_time]) as  DECIMAL(8,2)) /60 as  DECIMAL(8,2)) < 3*7*24
        and ord_num_value< 99999'''
    
    return get_data(sql)


def sao2_fio2_ratio(pulse_oximetry, fio2):
    ''' Simple ratio of spo2/fio2.
        Parameters: pulse_oximetry, fio2 '''
  
    if (pulse_oximetry < 0) or (pulse_oximetry > 100):
        raise ValueError(f"{pulse_oximetry} Not a valid SPO2 (pulse oximeter) value. Has to be between 0 and 100.")

    if (fio2 < 0) or (fio2 > 1):
        raise ValueError(f"{fio2} Not a valid fraction of inspired oxygen FiO2. Has to be between 0 and 1")

    return pulse_oximetry/fio2

def estimate_fio2 (oxygen_device, oxygen_flow_rate):

    '''Estimates the fio2 from the device being used to supply oxygen and fio2.
    For devices like nasal cannulas, the commonest device, we have to estimate the fio2
    In devices such as ventilators, this is set by the  therapists. 
    Parameters: oxygen_device: there are a plethora of devices by which oxygen can be given. 
    The commonest one is Nasal cannula. '''

    fio2_set_on_device = ['T-Piece', 'Bi-PAP','High flow nasal cannula',
            'Transtracheal catheter', 'CPAP', 'High Flow','Ventilator','Aerosol mask',
            'Trach mask',
            'Bubble CPAP',
            'Venturi mask']
    # 'Other (Comment)',

    if oxygen_device in fio2_set_on_device:
        return oxygen_flow_rate/100

  
    if (oxygen_device == 'Other (Comment)'): return 0.21
   # with a nasal cannula, we assume that the fraction of oxygen that is inspired 
   # (above the normal atmospheric level or 20%) 
   #increases by 4% for every additional liter of oxygen flow administered

    if (oxygen_device == 'Nasal cannula'): 
        return 0.21 + 0.03*oxygen_flow_rate
    
    # commonest case
    if (oxygen_device == 'None (Room air)'): return 0.21

    if (oxygen_device == 'MistyOx'):
    # 60 percent - 96 percent FiO2 range
    # Primary jet flow range 42-80 LPM
    # 0.6 = a + b*42
    # 0.96= a +b*80
    # 0.36 = b*38 => b == (0.36/38)
    # 0.6 - (0.36/38)*42 == 0.2021 == a
        return 0.2021 + 0.00947*oxygen_flow_rate

    if (oxygen_device == 'Face tent'):
    # Delivers only 40% Oxygen at 10-15 liters per minute
   # just use the face mask equation
        return 0.1+ 0.4167*oxygen_flow_rate

    # Simple face mask ~6-12 L/min supplying 35-60%*
    # 0.35 = a +b*6, 0.6 = a + b*12 => b = (0.25/6), a = 0.1
    if (oxygen_device == 'Simple mask'):
        return 0.1+ 0.4167*oxygen_flow_rate

    # 25% at 1.5L, 90% at 15L
    # parameters derived 
    if (oxygen_device == "Oxymask"): 
            return (0.17777+0.048*oxygen_flow_rate)

    # "Oxymizer Pendant Nasal Cannula"
    # this is a problem because the database string is not matching the code string 
    # Probably because of string encoding 
    if ("Pendant" in oxygen_device ):
            if (oxygen_flow_rate <= 0.5): return 0.26
            if (oxygen_flow_rate <= 1): return 0.32
            if (oxygen_flow_rate < 10): 
                return 0.32 + (oxygen_flow_rate - 1)*0.0445
    # 10.0 LPM = 72% FiO2
    # 11.0 LPM = 77% FiO2
    # 12.0 LPM = 82% FiO2
            if (oxygen_flow_rate > 10 and oxygen_flow_rate <= 12):
                return 0.72 + (oxygen_flow_rate-10)*0.02
        
            if (oxygen_flow_rate >  12):
                return 0.82 + (oxygen_flow_rate-10)*0.01
        
    if (oxygen_device == "Blow-by"):

        if (oxygen_flow_rate > 6): return 0.5
        if (oxygen_flow_rate > 2.9): return 0.3
        if (oxygen_flow_rate < 2.9): return 0.21

        # use https://www.nejm.org/doi/suppl/10.1056/NEJMoa2012410/suppl_file/nejmoa2012410_appendix.pdf
    if (oxygen_device == 'Non-rebreather' ): return 0.8

                           
    # Keep this simple, not common, patient is in deep trouble anyway
    #http://www.asaabstracts.com/strands/asaabstracts/abstract.htmyear=2012&index=15&absnum=5048#:~:text=Conclusion%3A,up%20to%2030%20breaths%2Fminute.
   # When using an AMBU, an oxygen flow rate of 25 l/minute 
   # will maintain an FIO2 of > 0.9 high 
   # for tidal volumes up to 800cc and respiratory rates up to 30 breaths/minute
    # 
    if (oxygen_device == "Ambu Bag"): return 0.9

    # "Oxymizer Pendant Nasal Cannula"
    # this is a problem because the database string is not matching the code string 
    # Probably because of string encoding 
   
     # print(f"Warning: {oxygen_device} not found in list of devices, using defaults")
    if (oxygen_flow_rate <= 0.5): return 0.26
    if (oxygen_flow_rate <= 1): return 0.32
    if (oxygen_flow_rate < 10): return 0.32 + (oxygen_flow_rate - 1)*0.0445
    if (oxygen_flow_rate > 10 and oxygen_flow_rate <= 12):
        return 0.72 + (oxygen_flow_rate-10)*0.02
    
    if (oxygen_flow_rate >  12):
                return 0.82 + (oxygen_flow_rate-10)*0.01
            
    # raise ValueError(f"{oxygen_device} not found in list of devices")

## Taken from https://www.intensive.org/epic2/Documents/Estimation%20of%20PO2%20and%20FiO2.pdf
def estimate_pao2(sao2):
    if sao2 < 75:
        return sao2/2
    if sao2 > 99: #  an error
        return 145
    
    # SO2:(%):PaO2:(mmHg)
    table={
            75:40, 76:41, 77:41.5, 78:42, 79:43,
            80:44, 81:45, 82:46, 83:47, 84:49,
            85:50, 86:52, 87:53, 88:55, 89:57,
            90:60, 91:62, 92:65, 93:69, 94:73,
            95:79, 96:86, 97:96, 98:112, 99:145
    }
    return table[sao2]

def estimate_pf_ratio(sao2,fio2):
    '''Estimates the pf_ratio from  pulse oximetry data and fio2.
    Parameters: sao2 measured by pulse oximetry; typical values are in the 80 to 100 range. 
    fio2: the fraction of oxygen in inspired air. In devices such as ventilators, this is set by the  therapists. For devices like nasal cannulas, the commonest device, we have to estimate the fio2'''
   
    if (fio2 == 0): fio2 = 0.21
    if (fio2 is None): fio2 = 0.21

    return  estimate_pao2(sao2)/fio2


def calculate_ratio_z_score(raw_data):
    #array to hold pf ratio later to be added to dataframe due to how I itterate through dataframe I have to append data at once at th end
    pf_ratio=[]
    for row in raw_data.itertuples():
        fio2=estimate_fio2(row[4],float(row[5]))
        pf_ratio.append(estimate_pf_ratio(float(row[3]),fio2))
    raw_data['pf_ratio']=pf_ratio
    return raw_data

# Modifies the incoming dataframe
def add_estimated_pf_ratio(df):
    """ Adds the estimated_pf_ratio column to the incoming dataframe
    Expects: A dataframe with columns named 'R OXYGEN DEVICE', 'R OXYGEN FLOW RATE' and 'PULSE OXIMETRY '
    """
    df["estimated_fio2"] = df.apply(estimate_fio2_row,axis=1)
    df["estimated_pao2"] = df.apply(estimate_pao2_row,axis=1)
    df["estimated_pf_ratio"] = df["estimated_pao2"]/df["estimated_fio2"]

def estimate_fio2_row(row):
    return estimate_fio2(row['R OXYGEN DEVICE'], float(row['R OXYGEN FLOW RATE']))

def estimate_pao2_row(row):
    return estimate_pao2(int(row['PULSE OXIMETRY']))

get_raw_data="""SELECT TOP 2000 [pat_enc_csn_id]
                    ,[hours_since_admitted]
                    ,[PULSE OXIMETRY]
                    ,[R OXYGEN DEVICE]
                    ,[R OXYGEN FLOW RATE]
                    FROM [Datarepo_DB].[COVID].[v_PF_ratio]"""
raw_data=get_data(get_raw_data)

# z_scored_data=calculate_ratio_z_score(raw_data)
add_estimated_pf_ratio(raw_data)
print(raw_data.head())
