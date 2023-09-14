# Ex-02_DS_Outlier_Detection_and_Removal

# AIM
To read the given data and detect outliers and remove them. 

# EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set.
Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.
Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.
Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# ALGORITHM
### Step 1
Read the given Data
### Step 2
Perform box plot to find the presence of outlier
### Step 3
Use IQR method and Z Score method to remove the outlier
### Step 4
Perform box plot on the new dataset to ensure the absence of outlier

# CODE
```python
import numpy as np
import pandas as pd
import seaborn as sb
from scipy import stats

# bhv.csv file
df1=pd.read_csv('bhp.csv')
df1.describe()
# IRQ Method
Q1 = df1['price_per_sqft'].quantile(0.25)
Q3 = df1['price_per_sqft'].quantile(0.75)
IQR = Q3-Q1
low=np.abs(Q1-1.5*IQR)
high=Q3+1.5*IQR
print(low,high)
df2=df1[((df1['price_per_sqft']>=low)&(df1['price_per_sqft']<=high))]
df2.describe()
# Z Score Method
z=np.abs(stats.zscore(df1['price_per_sqft']))
df13=df1[z<1]
df13.describe()

# height_weight.csv
df2=pd.read_csv('height_weight.csv')
sb.boxplot(data=df2)
# IRQ Method
Q1 = df2['height'].quantile(0.25)
Q3 = df2['height'].quantile(0.75)
IQR = Q3-Q1
low=Q1-1.5*IQR
high=Q3+1.5*IQR
print(low,high)
df22=df2[((df2['height']>=low)&(df2['height']<=high))]
sb.boxplot(data=df22['height'])
# Z Score Method
z=np.abs(stats.zscore(df2['height']))
df23=df2[z<2]
sb.boxplot(data=df23['height'])
# IRQ Method
Q1 = df2['weight'].quantile(0.25)
Q3 = df2['weight'].quantile(0.75)
IQR = Q3-Q1
low=Q1-1.5*IQR
high=Q3+1.5*IQR
print(low,high)
df24=df2[((df2['weight']>=low)&(df2['weight']<=high))]
sb.boxplot(data=df24['weight'])
# Z Score Method
z=np.abs(stats.zscore(df2['weight']))
df25=df2[z<3]
sb.boxplot(data=df25['weight'])

# heights.csv file
df31=pd.read_csv('heights.csv')
sb.boxplot(data=df31)
# IRQ Method
Q1 = df31['height'].quantile(0.25)
Q3 = df31['height'].quantile(0.75)
IQR = Q3-Q1
low=Q1-1.5*IQR
high=Q3+1.5*IQR
print(low,high)
df32=df31[((df31['height']>=low)&(df31['height']<=high))]
sb.boxplot(data=df32)
# Z Score Method
z=np.abs(stats.zscore(df31['height']))
df33=df31[z<1]
sb.boxplot(data=df33)
```

# OUTPUT
### bhp.csv describe
![image](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/b5c86409-0803-4de2-8a09-9782061037a5)

### bhv.csv after removing outlier using IQR
![image](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/a1d5ab2e-c634-46bd-afe3-f524bb2d1fac)

### bhv.csv after removing outlier using Z SCORE
![image](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/b2693743-a2a3-45f7-bf3b-c29bc5e9dfd0)

### height_weigth.csv boxplot
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/8c24d016-cebc-403b-b313-35d06dae6374)

### height_weigth.csv after removing outlier using IQR
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/e293af64-3bca-4df5-9ac2-eff47862747b)
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/3be2fe26-effb-42d0-81cf-bfea75e50337)

### height_weigth.csv after removing outlier using Z SCORE
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/833727f8-ee04-4366-8134-a00a8cc3ebd6)
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/b32ac805-bb90-418f-8ab5-3638eb62db84)

### heights.csv box plot
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/ca71a11f-75bd-43ed-a60c-8260bd00e3b2)

### heights.csv after removing outlier using IQR
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/925b1297-693e-43f9-8678-264bc4247850)

### heights.csv after removing outlier using Z SCORE
![download](https://github.com/yasin-sharif-SEC/ODD2023---Datascience---Ex-02/assets/142985837/44b5389b-1f59-4733-81ef-cb989ba0dad7)

# RESULT
Thus, the given data is read and the outliers are removed from it.
