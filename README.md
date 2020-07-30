## Predicting Respiratory Failure and Mortality In Covid-19: A Deep Learning Approach

This package provides an implementation of the prediction and analysis of respiratory failure and mortality using deep learning.

### Setup
**Dependencies**
* Python                    3.7
* numpy                     1.18.1
* tensorboard               2.2.2
* pandas                    0.23.0
* pyodbc                    4.0.30
* SQLAlchemy                1.3.13
* scikit-image              0.17.2
* scikit-learn              0.23.1
* scikit-survival           0.13.0
* seaborn                   0.10.1
* matplotlib                3.1.2
* mlflow                    1.9.1

### Data

**Encounter Table**
pat_enc_csn_id | pat_laber
------------ | -------------
Patient encounter ID  | Patient discharge status


**Encounter Detail Table**
pat_enc_csn_id | component_name | z_score_ord_num_value | hours_since_admitted | label | ind_kt | length_of_stay
------------ | ------------- | ------------ | ------------ | ------------ | ------------ | ------------
Patient encounter ID  | Physiological Variables or Laboratory Variables | Z score value | Hours since admitte | Patient discharge status | Dense ranking on Hours since admitte | Length of the stay
