from Model_Building.logger import logging
import os,sys
from Model_Building.exception import MartException
from Model_Building.entity import config_entity
from Model_Building.entity import artifact_entity
from Model_Building import utils
import pandas as pd
import numpy as np
from Model_Building.config import TARGET_COLUMN
from typing import Optional
from scipy.stats import ks_2samp


class DataValidation:

    def __init__(self,data_validation_config:config_entity.DataValidationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise MartException(e,sys)

    def handaling_missing_value(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function is basically to drop the missing value more than the specified threshold
        df:Accept the pandas  dataframe threshold : percentage criteria to drop the columns

        """

        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]


            # Selecting the columns that contain the null value
            logging.info(f"Selecting the columns containg the null value above to {threshold}")
            drop_columns_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop :{list(drop_columns_names)}")
            self.validation_error[report_key_name] = list(drop_columns_names)

            df.drop(list(drop_columns_names),axis = 1 , inplace=True)

            # Return None no columns is left

            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise MartException(e,sys)
    
    def is_required_column_exist(self,base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str)->bool:

        """
        Checking the column were drop or not 
        """

        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []

            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"columns:[{base_column}is not avaliable")
                    missing_columns.append(base_column)

            if len(missing_columns > 0):
                self.validation_error[report_key_name] = missing_columns
                return False
            return True

        except Exception as e:
            raise MartException(e,sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):

        try:
            drift_report:dict()
            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]

            # Null hypothesis is that both the column data drawn from the same distribution
            logging.info(f"Hypothesis{base_column}:{base_data.dtypes},{current_data.dtypes}")
            same_distribution = ks_2samp(base_data,current_data)

            if same_distribution.pvalue>0.05:
                # We are accepring the null hypothesisi
                drift_report[base_column] = {"p.value":float(same_distribution.pvalue),"same_distribution":True}
            else:
                drift_report[base_column] = {"p.value":float(same_distribution.pvalue),"same_distribution":False}

            # Different distribution
            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise MartException(e,sys)

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN},inplace=True)
            logging.info(f"Replace na value in base df")
            #base_df has na as null
            logging.info(f"Drop null values colums from base df")
            base_df=self.handaling_missing_value(df=base_df,report_key_name="missing_values_within_base_dataset")
         
            logging.info(f"Reading the train data")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            
            logging.info(f"Reading the test data")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Dropping the missing value from the train_df and test_df
            logging.info(f"Drop the null value from train_df")
            train_df = self.handaling_missing_value(df=train_df,report_key_name="missing value in the train dataset")

            logging.info(f"Drop the null value from the test_df")
            test_df = self.handaling_missing_value(df=test_df,report_key_name="missing value in the test dataset")
            
            # Is all the required is present in the train_df
            logging.info(f"Is all the required column is present in the train_df")
            train_df_columns_status = self.is_required_column_exist(base_df=base_df,current_df=train_df,report_key_name="missing column in the train dataset")

            logging.info(f"Is all the required column is present in the test df")
            test_df_columns_status = self.is_required_column_exist(base_df=base_df,current_df=test_df,report_key_name="missing column in the test dataset")

            if train_df_columns_status:
                logging.info(f"As all the column is present in the train_df hence detecting the data drift")
                self.data_drift(base_df=base_df,current_df=train_df,report_key_name="data drift within the train dataset")

            if test_df_columns_status:
                self.data_drift(base_df=base_df,current_df=test_df,report_key_name="data drift within the test dataset")

            #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MartException(e, sys)
