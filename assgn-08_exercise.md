# Assignment 8

__Assignment Overview__

In this assignment we were tasked with using machine learning to optimize regression that predicts housing prices given a multitude of factors. We had a training and testing dataset to build our model, then a holdout dataset that we predicted the prices for.

I winsorized my data to reduce the effects of outliers then used the ridge function in a pipeline with GridSearchCV to optimize my model for alpha and k values.

My overall results were an R^2 of .883 when predicting the prices on my test dataset


```python
import pandas as pd
# from pandas_profiling import ProfileReport # now use ydata-profiling
from pandas_profiling import ProfileReport
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import argsort
from sklearn.pipeline import make_pipeline 
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

```

    C:\Users\AlexanderRomanowski\AppData\Local\Temp\ipykernel_21548\1978820628.py:3: DeprecationWarning: `import pandas_profiling` is going to be deprecated by April 1st. Please use `import ydata_profiling` instead.
      from pandas_profiling import ProfileReport
    

## Load data and cleaning

After looking at the key stats for the housing data, it's clear that there are significant outliers given the high max/min/std dev values. Winsorizing the data to the 1% and 99% tails greatly reduced the outliers and the std dev. After doing this, I noticed that my model became a lot more stable in its R^2 values.


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.linear_model import RidgeCV
from winsorizer_with_missing import winsorizer_with_missing
import warnings
```


```python
# put code here
housing = pd.read_csv('input_data2/housing_train.csv')
y = np.log(housing.v_SalePrice)

```


```python
housing.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>v_MS_SubClass</th>
      <td>1941.0</td>
      <td>58.088614</td>
      <td>42.946015</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>v_Lot_Frontage</th>
      <td>1620.0</td>
      <td>69.301235</td>
      <td>23.978101</td>
      <td>21.0</td>
      <td>58.00</td>
      <td>68.0</td>
      <td>80.00</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>v_Lot_Area</th>
      <td>1941.0</td>
      <td>10284.770222</td>
      <td>7832.295527</td>
      <td>1470.0</td>
      <td>7420.00</td>
      <td>9450.0</td>
      <td>11631.00</td>
      <td>164660.0</td>
    </tr>
    <tr>
      <th>v_Overall_Qual</th>
      <td>1941.0</td>
      <td>6.113344</td>
      <td>1.401594</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>v_Overall_Cond</th>
      <td>1941.0</td>
      <td>5.568264</td>
      <td>1.087465</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>v_Year_Built</th>
      <td>1941.0</td>
      <td>1971.321999</td>
      <td>30.209933</td>
      <td>1872.0</td>
      <td>1953.00</td>
      <td>1973.0</td>
      <td>2001.00</td>
      <td>2008.0</td>
    </tr>
    <tr>
      <th>v_Year_Remod/Add</th>
      <td>1941.0</td>
      <td>1984.073158</td>
      <td>20.837338</td>
      <td>1950.0</td>
      <td>1965.00</td>
      <td>1993.0</td>
      <td>2004.00</td>
      <td>2009.0</td>
    </tr>
    <tr>
      <th>v_Mas_Vnr_Area</th>
      <td>1923.0</td>
      <td>104.846074</td>
      <td>184.982611</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>v_BsmtFin_SF_1</th>
      <td>1940.0</td>
      <td>436.986598</td>
      <td>457.815715</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>361.5</td>
      <td>735.25</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>v_BsmtFin_SF_2</th>
      <td>1940.0</td>
      <td>49.247938</td>
      <td>169.555232</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <th>v_Bsmt_Unf_SF</th>
      <td>1940.0</td>
      <td>567.437629</td>
      <td>439.600535</td>
      <td>0.0</td>
      <td>225.75</td>
      <td>474.0</td>
      <td>815.00</td>
      <td>2153.0</td>
    </tr>
    <tr>
      <th>v_Total_Bsmt_SF</th>
      <td>1940.0</td>
      <td>1053.672165</td>
      <td>438.662147</td>
      <td>0.0</td>
      <td>796.75</td>
      <td>989.5</td>
      <td>1295.25</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>v_1st_Flr_SF</th>
      <td>1941.0</td>
      <td>1161.071613</td>
      <td>396.945408</td>
      <td>334.0</td>
      <td>886.00</td>
      <td>1085.0</td>
      <td>1383.00</td>
      <td>5095.0</td>
    </tr>
    <tr>
      <th>v_2nd_Flr_SF</th>
      <td>1941.0</td>
      <td>340.955178</td>
      <td>434.242152</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>717.00</td>
      <td>2065.0</td>
    </tr>
    <tr>
      <th>v_Low_Qual_Fin_SF</th>
      <td>1941.0</td>
      <td>4.282329</td>
      <td>42.943917</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>697.0</td>
    </tr>
    <tr>
      <th>v_Gr_Liv_Area</th>
      <td>1941.0</td>
      <td>1506.309119</td>
      <td>524.765289</td>
      <td>334.0</td>
      <td>1118.00</td>
      <td>1436.0</td>
      <td>1755.00</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>v_Bsmt_Full_Bath</th>
      <td>1939.0</td>
      <td>0.415162</td>
      <td>0.515395</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>v_Bsmt_Half_Bath</th>
      <td>1939.0</td>
      <td>0.064982</td>
      <td>0.254791</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>v_Full_Bath</th>
      <td>1941.0</td>
      <td>1.566718</td>
      <td>0.552693</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>v_Half_Bath</th>
      <td>1941.0</td>
      <td>0.378156</td>
      <td>0.498675</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>v_Bedroom_AbvGr</th>
      <td>1941.0</td>
      <td>2.866048</td>
      <td>0.827732</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>v_Kitchen_AbvGr</th>
      <td>1941.0</td>
      <td>1.039155</td>
      <td>0.201827</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>v_TotRms_AbvGrd</th>
      <td>1941.0</td>
      <td>6.465224</td>
      <td>1.577696</td>
      <td>2.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>v_Fireplaces</th>
      <td>1941.0</td>
      <td>0.595569</td>
      <td>0.641969</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>v_Garage_Yr_Blt</th>
      <td>1834.0</td>
      <td>1978.188113</td>
      <td>25.729319</td>
      <td>1895.0</td>
      <td>1960.00</td>
      <td>1980.0</td>
      <td>2002.00</td>
      <td>2207.0</td>
    </tr>
    <tr>
      <th>v_Garage_Cars</th>
      <td>1940.0</td>
      <td>1.769588</td>
      <td>0.763399</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>v_Garage_Area</th>
      <td>1940.0</td>
      <td>472.766495</td>
      <td>217.089624</td>
      <td>0.0</td>
      <td>318.75</td>
      <td>478.0</td>
      <td>576.00</td>
      <td>1488.0</td>
    </tr>
    <tr>
      <th>v_Wood_Deck_SF</th>
      <td>1941.0</td>
      <td>92.458011</td>
      <td>127.020523</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>1424.0</td>
    </tr>
    <tr>
      <th>v_Open_Porch_SF</th>
      <td>1941.0</td>
      <td>49.157135</td>
      <td>70.296277</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>28.0</td>
      <td>72.00</td>
      <td>742.0</td>
    </tr>
    <tr>
      <th>v_Enclosed_Porch</th>
      <td>1941.0</td>
      <td>22.947965</td>
      <td>65.249307</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1012.0</td>
    </tr>
    <tr>
      <th>v_3Ssn_Porch</th>
      <td>1941.0</td>
      <td>2.249871</td>
      <td>22.416832</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>407.0</td>
    </tr>
    <tr>
      <th>v_Screen_Porch</th>
      <td>1941.0</td>
      <td>16.249871</td>
      <td>56.748086</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>576.0</td>
    </tr>
    <tr>
      <th>v_Pool_Area</th>
      <td>1941.0</td>
      <td>3.386399</td>
      <td>43.695267</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>v_Misc_Val</th>
      <td>1941.0</td>
      <td>52.553838</td>
      <td>616.064459</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>v_Mo_Sold</th>
      <td>1941.0</td>
      <td>6.431221</td>
      <td>2.745199</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>v_Yr_Sold</th>
      <td>1941.0</td>
      <td>2006.998454</td>
      <td>0.801736</td>
      <td>2006.0</td>
      <td>2006.00</td>
      <td>2007.0</td>
      <td>2008.00</td>
      <td>2008.0</td>
    </tr>
    <tr>
      <th>v_SalePrice</th>
      <td>1941.0</td>
      <td>182033.238022</td>
      <td>80407.100395</td>
      <td>13100.0</td>
      <td>130000.00</td>
      <td>161900.0</td>
      <td>215000.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#pulling the column names to winsorize
housingwin = housing.select_dtypes(include=['int64', 'float64']).columns
```


```python
housingwin = housingwin.tolist()
```


```python
#winsorizing
new_house = winsorizer_with_missing(housing, cols= housingwin)
new_house = new_house.drop('v_SalePrice',axis=1)
```

    C:\Users\AlexanderRomanowski\Documents\HW\fin 377\asgn-08-ajr423\winsorizer_with_missing.py:45: FutureWarning: Downcasting integer-dtype results in .where is deprecated and will change in a future version. To retain the old behavior, explicitly cast the results to the desired dtype.
      df[cols] = df[cols].clip(lower=df[cols].quantile(low_),
    


```python
new_house.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>v_MS_SubClass</th>
      <td>1941.0</td>
      <td>58.088614</td>
      <td>42.946015</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.000</td>
    </tr>
    <tr>
      <th>v_Lot_Frontage</th>
      <td>1620.0</td>
      <td>68.772222</td>
      <td>21.368657</td>
      <td>21.0</td>
      <td>58.00</td>
      <td>68.0</td>
      <td>80.00</td>
      <td>130.000</td>
    </tr>
    <tr>
      <th>v_Lot_Area</th>
      <td>1941.0</td>
      <td>9936.527048</td>
      <td>4663.839454</td>
      <td>1896.6</td>
      <td>7420.00</td>
      <td>9450.0</td>
      <td>11631.00</td>
      <td>30867.700</td>
    </tr>
    <tr>
      <th>v_Overall_Qual</th>
      <td>1941.0</td>
      <td>6.107161</td>
      <td>1.356487</td>
      <td>3.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>9.000</td>
    </tr>
    <tr>
      <th>v_Overall_Cond</th>
      <td>1941.0</td>
      <td>5.561051</td>
      <td>1.033465</td>
      <td>3.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>8.000</td>
    </tr>
    <tr>
      <th>v_Year_Built</th>
      <td>1941.0</td>
      <td>1971.415250</td>
      <td>29.936818</td>
      <td>1900.0</td>
      <td>1953.00</td>
      <td>1973.0</td>
      <td>2001.00</td>
      <td>2007.000</td>
    </tr>
    <tr>
      <th>v_Year_Remod/Add</th>
      <td>1941.0</td>
      <td>1984.072643</td>
      <td>20.836733</td>
      <td>1950.0</td>
      <td>1965.00</td>
      <td>1993.0</td>
      <td>2004.00</td>
      <td>2008.000</td>
    </tr>
    <tr>
      <th>v_Mas_Vnr_Area</th>
      <td>1923.0</td>
      <td>100.885450</td>
      <td>166.196584</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>710.680</td>
    </tr>
    <tr>
      <th>v_BsmtFin_SF_1</th>
      <td>1940.0</td>
      <td>431.173093</td>
      <td>428.688416</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>361.5</td>
      <td>735.25</td>
      <td>1561.660</td>
    </tr>
    <tr>
      <th>v_BsmtFin_SF_2</th>
      <td>1940.0</td>
      <td>46.804820</td>
      <td>155.508430</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>828.745</td>
    </tr>
    <tr>
      <th>v_Bsmt_Unf_SF</th>
      <td>1940.0</td>
      <td>564.993428</td>
      <td>432.532077</td>
      <td>0.0</td>
      <td>225.75</td>
      <td>474.0</td>
      <td>815.00</td>
      <td>1693.575</td>
    </tr>
    <tr>
      <th>v_Total_Bsmt_SF</th>
      <td>1940.0</td>
      <td>1045.872912</td>
      <td>401.483213</td>
      <td>0.0</td>
      <td>796.75</td>
      <td>989.5</td>
      <td>1295.25</td>
      <td>2035.915</td>
    </tr>
    <tr>
      <th>v_1st_Flr_SF</th>
      <td>1941.0</td>
      <td>1153.883565</td>
      <td>360.236856</td>
      <td>525.1</td>
      <td>886.00</td>
      <td>1085.0</td>
      <td>1383.00</td>
      <td>2135.300</td>
    </tr>
    <tr>
      <th>v_2nd_Flr_SF</th>
      <td>1941.0</td>
      <td>338.057702</td>
      <td>426.044845</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>717.00</td>
      <td>1359.800</td>
    </tr>
    <tr>
      <th>v_Low_Qual_Fin_SF</th>
      <td>1941.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>v_Gr_Liv_Area</th>
      <td>1941.0</td>
      <td>1498.284905</td>
      <td>482.248402</td>
      <td>747.7</td>
      <td>1118.00</td>
      <td>1436.0</td>
      <td>1755.00</td>
      <td>2827.800</td>
    </tr>
    <tr>
      <th>v_Bsmt_Full_Bath</th>
      <td>1939.0</td>
      <td>0.403816</td>
      <td>0.490788</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>v_Bsmt_Half_Bath</th>
      <td>1939.0</td>
      <td>0.062919</td>
      <td>0.242880</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>v_Full_Bath</th>
      <td>1941.0</td>
      <td>1.570325</td>
      <td>0.545672</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>v_Half_Bath</th>
      <td>1941.0</td>
      <td>0.371458</td>
      <td>0.483319</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>v_Bedroom_AbvGr</th>
      <td>1941.0</td>
      <td>2.862442</td>
      <td>0.795047</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>v_Kitchen_AbvGr</th>
      <td>1941.0</td>
      <td>1.040701</td>
      <td>0.197647</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>v_TotRms_AbvGrd</th>
      <td>1941.0</td>
      <td>6.468315</td>
      <td>1.531340</td>
      <td>4.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>v_Fireplaces</th>
      <td>1941.0</td>
      <td>0.590417</td>
      <td>0.625647</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>v_Garage_Yr_Blt</th>
      <td>1834.0</td>
      <td>1978.200654</td>
      <td>24.813501</td>
      <td>1920.0</td>
      <td>1960.00</td>
      <td>1980.0</td>
      <td>2002.00</td>
      <td>2007.000</td>
    </tr>
    <tr>
      <th>v_Garage_Cars</th>
      <td>1940.0</td>
      <td>1.764433</td>
      <td>0.751600</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>v_Garage_Area</th>
      <td>1940.0</td>
      <td>470.119149</td>
      <td>209.319506</td>
      <td>0.0</td>
      <td>318.75</td>
      <td>478.0</td>
      <td>576.00</td>
      <td>953.405</td>
    </tr>
    <tr>
      <th>v_Wood_Deck_SF</th>
      <td>1941.0</td>
      <td>90.223596</td>
      <td>116.616823</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>465.400</td>
    </tr>
    <tr>
      <th>v_Open_Porch_SF</th>
      <td>1941.0</td>
      <td>47.604328</td>
      <td>62.672975</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>28.0</td>
      <td>72.00</td>
      <td>265.900</td>
    </tr>
    <tr>
      <th>v_Enclosed_Porch</th>
      <td>1941.0</td>
      <td>21.710974</td>
      <td>57.088499</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>243.600</td>
    </tr>
    <tr>
      <th>v_3Ssn_Porch</th>
      <td>1941.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>v_Screen_Porch</th>
      <td>1941.0</td>
      <td>15.062339</td>
      <td>50.325873</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>227.000</td>
    </tr>
    <tr>
      <th>v_Pool_Area</th>
      <td>1941.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>v_Misc_Val</th>
      <td>1941.0</td>
      <td>16.524987</td>
      <td>92.916618</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>600.000</td>
    </tr>
    <tr>
      <th>v_Mo_Sold</th>
      <td>1941.0</td>
      <td>6.431221</td>
      <td>2.745199</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.000</td>
    </tr>
    <tr>
      <th>v_Yr_Sold</th>
      <td>1941.0</td>
      <td>2006.998454</td>
      <td>0.801736</td>
      <td>2006.0</td>
      <td>2006.00</td>
      <td>2007.0</td>
      <td>2008.00</td>
      <td>2008.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_house.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1941 entries, 0 to 1940
    Data columns (total 80 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   parcel             1941 non-null   object 
     1   v_MS_SubClass      1941 non-null   int64  
     2   v_MS_Zoning        1941 non-null   object 
     3   v_Lot_Frontage     1620 non-null   float64
     4   v_Lot_Area         1941 non-null   float64
     5   v_Street           1941 non-null   object 
     6   v_Alley            136 non-null    object 
     7   v_Lot_Shape        1941 non-null   object 
     8   v_Land_Contour     1941 non-null   object 
     9   v_Utilities        1941 non-null   object 
     10  v_Lot_Config       1941 non-null   object 
     11  v_Land_Slope       1941 non-null   object 
     12  v_Neighborhood     1941 non-null   object 
     13  v_Condition_1      1941 non-null   object 
     14  v_Condition_2      1941 non-null   object 
     15  v_Bldg_Type        1941 non-null   object 
     16  v_House_Style      1941 non-null   object 
     17  v_Overall_Qual     1941 non-null   int64  
     18  v_Overall_Cond     1941 non-null   int64  
     19  v_Year_Built       1941 non-null   int64  
     20  v_Year_Remod/Add   1941 non-null   int64  
     21  v_Roof_Style       1941 non-null   object 
     22  v_Roof_Matl        1941 non-null   object 
     23  v_Exterior_1st     1941 non-null   object 
     24  v_Exterior_2nd     1941 non-null   object 
     25  v_Mas_Vnr_Type     1923 non-null   object 
     26  v_Mas_Vnr_Area     1923 non-null   float64
     27  v_Exter_Qual       1941 non-null   object 
     28  v_Exter_Cond       1941 non-null   object 
     29  v_Foundation       1941 non-null   object 
     30  v_Bsmt_Qual        1891 non-null   object 
     31  v_Bsmt_Cond        1891 non-null   object 
     32  v_Bsmt_Exposure    1889 non-null   object 
     33  v_BsmtFin_Type_1   1891 non-null   object 
     34  v_BsmtFin_SF_1     1940 non-null   float64
     35  v_BsmtFin_Type_2   1891 non-null   object 
     36  v_BsmtFin_SF_2     1940 non-null   float64
     37  v_Bsmt_Unf_SF      1940 non-null   float64
     38  v_Total_Bsmt_SF    1940 non-null   float64
     39  v_Heating          1941 non-null   object 
     40  v_Heating_QC       1941 non-null   object 
     41  v_Central_Air      1941 non-null   object 
     42  v_Electrical       1940 non-null   object 
     43  v_1st_Flr_SF       1941 non-null   float64
     44  v_2nd_Flr_SF       1941 non-null   float64
     45  v_Low_Qual_Fin_SF  1941 non-null   int64  
     46  v_Gr_Liv_Area      1941 non-null   float64
     47  v_Bsmt_Full_Bath   1939 non-null   float64
     48  v_Bsmt_Half_Bath   1939 non-null   float64
     49  v_Full_Bath        1941 non-null   int64  
     50  v_Half_Bath        1941 non-null   int64  
     51  v_Bedroom_AbvGr    1941 non-null   int64  
     52  v_Kitchen_AbvGr    1941 non-null   int64  
     53  v_Kitchen_Qual     1941 non-null   object 
     54  v_TotRms_AbvGrd    1941 non-null   int64  
     55  v_Functional       1941 non-null   object 
     56  v_Fireplaces       1941 non-null   int64  
     57  v_Fireplace_Qu     1001 non-null   object 
     58  v_Garage_Type      1836 non-null   object 
     59  v_Garage_Yr_Blt    1834 non-null   float64
     60  v_Garage_Finish    1834 non-null   object 
     61  v_Garage_Cars      1940 non-null   float64
     62  v_Garage_Area      1940 non-null   float64
     63  v_Garage_Qual      1834 non-null   object 
     64  v_Garage_Cond      1834 non-null   object 
     65  v_Paved_Drive      1941 non-null   object 
     66  v_Wood_Deck_SF     1941 non-null   float64
     67  v_Open_Porch_SF    1941 non-null   float64
     68  v_Enclosed_Porch   1941 non-null   float64
     69  v_3Ssn_Porch       1941 non-null   int64  
     70  v_Screen_Porch     1941 non-null   int64  
     71  v_Pool_Area        1941 non-null   int64  
     72  v_Pool_QC          13 non-null     object 
     73  v_Fence            365 non-null    object 
     74  v_Misc_Feature     63 non-null     object 
     75  v_Misc_Val         1941 non-null   int64  
     76  v_Mo_Sold          1941 non-null   int64  
     77  v_Yr_Sold          1941 non-null   int64  
     78  v_Sale_Type        1941 non-null   object 
     79  v_Sale_Condition   1941 non-null   object 
    dtypes: float64(18), int64(18), object(44)
    memory usage: 1.2+ MB
    


```python
# splitting the data up
rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(new_house, y, random_state=rng)
```


```python
#this pipe replaces empty variables and drops almost all of the non-numeric variables

numer_pipe = make_pipeline(
    SimpleImputer(),
    StandardScaler()
)

cat_pipe   = make_pipeline(OneHotEncoder()) #handle_unknown='ignore'

preproc_pipe = ColumnTransformer(
    [ 
    # numerical vars
    ("num_impute", numer_pipe, make_column_selector(dtype_include=np.number)),
    # categorical vars  
    ("cat_trans", cat_pipe, ['v_Lot_Config'])
    ]
    , remainder = 'drop'
)
```


```python
#using grid search cv to select the optimal alpha and k value through ridge estimation
pipe = Pipeline([('columntransformer',preproc_pipe),
                 ('feature_create',SelectKBest(f_classif, k=5)), 
                 ('feature_select','passthrough'), 
                 ('clf', Ridge())
                ])

param_grid = [    {'feature_create__k': [25,26,27,28,29,30,31,32,33,34,35],
                   'clf__alpha': [17,18,19,20,21,22,23,24,25,26]
                  },
]

grid_search = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2'
                          )
warnings.filterwarnings("ignore")
grid_search.fit(X_train, y_train)
results = grid_search.fit(X_train, y_train)
results_df = pd.DataFrame(results.cv_results_)
```


```python

#pulling in variables to the results df
results_df = results_df[['params','mean_test_score','std_test_score']]

#adding alpha and k variables into the results df for graphing later
results_df['alpha'] = results_df['params'].apply(lambda x: x.get('clf__alpha'))
results_df['k'] = results_df['params'].apply(lambda x: x.get('feature_select__k'))

#grabbing optimal values
print(results_df['mean_test_score'].max())
print(results_df['std_test_score'].max())
results_df.sort_values('mean_test_score').tail(15)
```

    0.8873861426042128
    0.021320530443543578
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>params</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>alpha</th>
      <th>k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63</th>
      <td>{'clf__alpha': 22, 'feature_create__k': 33}</td>
      <td>0.887215</td>
      <td>0.018216</td>
      <td>22</td>
      <td>None</td>
    </tr>
    <tr>
      <th>74</th>
      <td>{'clf__alpha': 23, 'feature_create__k': 33}</td>
      <td>0.887226</td>
      <td>0.018223</td>
      <td>23</td>
      <td>None</td>
    </tr>
    <tr>
      <th>85</th>
      <td>{'clf__alpha': 24, 'feature_create__k': 33}</td>
      <td>0.887234</td>
      <td>0.018231</td>
      <td>24</td>
      <td>None</td>
    </tr>
    <tr>
      <th>96</th>
      <td>{'clf__alpha': 25, 'feature_create__k': 33}</td>
      <td>0.887242</td>
      <td>0.018238</td>
      <td>25</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>{'clf__alpha': 17, 'feature_create__k': 35}</td>
      <td>0.887242</td>
      <td>0.017957</td>
      <td>17</td>
      <td>None</td>
    </tr>
    <tr>
      <th>107</th>
      <td>{'clf__alpha': 26, 'feature_create__k': 33}</td>
      <td>0.887248</td>
      <td>0.018246</td>
      <td>26</td>
      <td>None</td>
    </tr>
    <tr>
      <th>21</th>
      <td>{'clf__alpha': 18, 'feature_create__k': 35}</td>
      <td>0.887266</td>
      <td>0.017966</td>
      <td>18</td>
      <td>None</td>
    </tr>
    <tr>
      <th>32</th>
      <td>{'clf__alpha': 19, 'feature_create__k': 35}</td>
      <td>0.887288</td>
      <td>0.017974</td>
      <td>19</td>
      <td>None</td>
    </tr>
    <tr>
      <th>43</th>
      <td>{'clf__alpha': 20, 'feature_create__k': 35}</td>
      <td>0.887307</td>
      <td>0.017983</td>
      <td>20</td>
      <td>None</td>
    </tr>
    <tr>
      <th>54</th>
      <td>{'clf__alpha': 21, 'feature_create__k': 35}</td>
      <td>0.887324</td>
      <td>0.017993</td>
      <td>21</td>
      <td>None</td>
    </tr>
    <tr>
      <th>65</th>
      <td>{'clf__alpha': 22, 'feature_create__k': 35}</td>
      <td>0.887340</td>
      <td>0.018003</td>
      <td>22</td>
      <td>None</td>
    </tr>
    <tr>
      <th>76</th>
      <td>{'clf__alpha': 23, 'feature_create__k': 35}</td>
      <td>0.887354</td>
      <td>0.018013</td>
      <td>23</td>
      <td>None</td>
    </tr>
    <tr>
      <th>87</th>
      <td>{'clf__alpha': 24, 'feature_create__k': 35}</td>
      <td>0.887366</td>
      <td>0.018023</td>
      <td>24</td>
      <td>None</td>
    </tr>
    <tr>
      <th>98</th>
      <td>{'clf__alpha': 25, 'feature_create__k': 35}</td>
      <td>0.887377</td>
      <td>0.018034</td>
      <td>25</td>
      <td>None</td>
    </tr>
    <tr>
      <th>109</th>
      <td>{'clf__alpha': 26, 'feature_create__k': 35}</td>
      <td>0.887386</td>
      <td>0.018044</td>
      <td>26</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
#graphic for visualization
sns.scatterplot(data=results_df, x='std_test_score', y='mean_test_score')
plt.xlabel('Std Dev Test Scores')
plt.ylabel('Mean R2 Test Scores')
plt.title('Optimizing R2 & Std Dev')

```




    Text(0.5, 1.0, 'Optimizing R2 & Std Dev')




    
![png](output_16_1.png)
    



```python
#finding the best estimator and training it over the data
best_pipe = grid_search.best_estimator_ 
best_pipe.fit(X_train,y_train)


#using the trained model to predict and checking the R2
y_test_predict = best_pipe.predict(X_test)
test_score = r2_score(y_test,y_test_predict)
print(test_score)
```

    0.8830394937046963
    

Ok, I'm happy with that R^2, let's test this puppy out!


```python
#predict values on our holdout set, store them in a df, convert to csv for grading
best_pipe.fit(new_house,y)

holdout = pd.read_csv('input_data2/housing_holdout.csv')
holdout_x_vals = holdout.drop('parcel',axis=1)

y_pred = best_pipe.predict(holdout_x_vals)

df_out = pd.DataFrame({'parcel':holdout['parcel'],
                       'prediction':y_pred})

df_out.to_csv('submission/MY_PREDICTIONS.csv',index=False)
```
