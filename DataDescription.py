# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 23:30:06 2021

@author: dfshi
"""

import pandas as pd
import numpy as np

data = pd.read_csv("Formatted_Data.csv")
description = pd.DataFrame(0, np.arange(DataSet.shape[1]), columns=["Name","Type","Values/Range", "Independent/Dependent", "Description"])


names = DataSet.columns
description["Name"][:] = names
description["Type"][:n-1] = "String value"
description["Values/Range"][:n-1] = "Positive Integers"
description["Independent/Dependent"][:n-1] = "Independent"
description["Description"][:n-1] = "Count of each time this unique value was seen in an article"

description["Type"][n-1] = "Boolean Value"
description["Values/Range"][n-1] = "0 or 1"
description["Independent/Dependent"][n-1] = "Dependent"
description["Description"][n-1] = "Target variable, whether or not the article was true or false (1 == true)"

description.to_csv("Description.csv")

n = len(names)


for j in range(DataSet.shape[1] - 1 ):
    description["Name"].iloc[j] = names[j]