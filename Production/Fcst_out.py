import os
import configparser
import pandas as pd

config = configparser.ConfigParser()
config.read('/mnt/trade05_data/wind_v5_new_code/Production/config.ini')
out_path = config['paths']['out_path']

class Output:
    
    def __init__(self, dts,model_type, model_name, df):
        self.dts = dts
        self.model_type = model_type
        self.model_name = model_name
        self.df = df
        
    def create_path(self):    
        """
        Create output path for a new file.
        Args:
            sdt (datetime): Datetime object.
        Returns:
            str: The generated file name based on the input datetime.
        """
        file_name = f"ewind_pow.{self.dts}_{self.model_type}_wnd.NAM"
        return file_name

    def Create(self):
        new_path = f"{out_path}/{self.model_type}/" + self.dts
        os.makedirs(new_path, exist_ok=True)
        self.df.to_csv(new_path + '/' + self.dts + '.csv', index = False)
        fil = self.create_path()
        new_path = new_path + '/' + fil
        
        print(new_path)
        f = open(new_path, "w")
        f.write(f"{self.model_type}\n")
        f.write("\n")
        f.write(f"{self.model_type} Forecast Model Number: {self.model_name}\n")
        f.write("\n")
        f.write('================================================================================ \n')
        f.write("MWD \n")
        f.write('================================================================================ \n')
        f.write("\n")
        f.write(self.df.to_string(index = False))
        f.write("\n")
        f.write("\n")
        f.close()
