## Predicting Respiratory Failure and Mortality In Covid-19: A Deep Learning Approach

This package provides an implementation of the prediction and analysis of respiratory failure and mortality using deep learning.

### Setup
**Dependencies**
* Python                    3.7
* numpy                     1.18.1
* tensorflow                2.2.0
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

**List of features (component_name) that were used in the model**
* RESPIRATIONS
* TEMPERATURE
* MAP
* PULSE
* P/F RATIO
* BMI
* C-REACTIVE PROTEIN
* FERRITIN
* ABSOLUTE LYMPH CT
* R LH SEPSIS WATCH SCORE
* LD
* AGE
* POTASSIUM
* CREATININE
* ANION GAP
* HEMATOCRIT
* WBC
* PLATELET_COUNT
* POLYS
* ABSOLUTE_GRAN_CT
* LYMPHOCYTE
* MONOCYTE
* EOSINOPHIL
* BASOPHIL
* ABSOLUTE MONO CT
* ABSOLUTE EOS CT
* ABSOLUTE BASO CT
* ALBUMIN
* AST SGOT
* GFR NON AFRICAN-AMER (CKD-EPI)
* MCHC

### Evaluation
Please see Notebook Mortality_prediction.ipynb

### Prediction
Using falcon to define RESTful services that use Tensorflow models.

We have trained to try to predict patient outcomes.

### Disclaimer
This tool is for research purpose and not approved for clinical use.
