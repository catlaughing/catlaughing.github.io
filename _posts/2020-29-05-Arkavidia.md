---
title: "Arkavidia Notebook"
date: 2020-05-29
tags: [data analysis, data science, competition]
header:
  image: "/images/arkav/arkav.svg"
excerpt: "Data Analysis, Data Science, Competition"
mathjax: "true"
---


![alt text](https://www.arkavidia.id/_nuxt/img/a10a78c.svg)

This notebook belongs to **Sour Soup Team**

---

# Cross Sell Transactions

Pada *notebook* ini, kami akan memprediksi apakah akan terjadi *cross sell* pada transaksi penerbangan pelanggan [tiket.com](https://www.tiket.com), proses ini dikategorikan sebagai klasifikasi. *Cross sell* adalah ketika pelanggan membuat booking hotel bersamaan dengan membeli tiket pesawat.

> [Tiket.com](https://www.tiket.com) adalah situs web yang menyediakan layanan pemesanan hotel, tiket pesawat, tiket kereta api, penyewaan mobil, tiket konser, tiket atraksi, tiket hiburan, dan tiket event yang berbasis di Jakarta, Indonesia.

Untuk melakukan proses klasifikasi, kami menggunakan alur *data science* yang umum digunakan pada gambar dibawah ini:

![](https://d3ansictanv2wj.cloudfront.net/Figure3-85d4b2a80132afe42305cb3b4b6c3aa8.png)


## Import Library

Tools yang akan digunakan pada seluruh proses.


```python
# helper libraries
from __future__ import print_function

import os
import gc
import ast
import csv
import datetime
import random
import numpy as np
import pandas as pd
import pandas_profiling as pp
from scipy.special import expit

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = [8,8]

# Model libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier

# Helper
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import SCORERS, f1_score, make_scorer, classification_report, f1_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

# Imblearn libraries
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# add path to directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    Using TensorFlow backend.
    

    /kaggle/input/datavidia2019/flight.csv
    /kaggle/input/datavidia2019/sample_submission.csv
    /kaggle/input/datavidia2019/Data Dictionary.pdf
    /kaggle/input/datavidia2019/hotel.csv
    /kaggle/input/datavidia2019/test.csv
    

## Dataset

Berikut ini data yang tersedia dan akan digunakan untuk proses klasifikasi:

- `flight.csv` - Data *training* yang berisi berbagai *flight 
transactions* beserta atribut-atributnya
- `hotel.csv` - Berisi data hotel dan atribut-atributnya
- `test.csv` - Data *test* yang berisi *flight transactions* yang harus diprediksi apakah terjadi *cross selling* atau tidak
- `sample_submission.csv` - Berisi format submisi ke Kaggle


```python
# directory kernel
dirname = '/kaggle/input/datavidia2019/'

# dataframes for each files
df_hotel = pd.read_csv(dirname+'hotel.csv')
df_flight = pd.read_csv(dirname+'flight.csv')
df_test = pd.read_csv(dirname+'test.csv')
sample = pd.read_csv(dirname+'sample_submission.csv')
```

### Data Penerbangan

Data yang terdapat pada file `flight.csv` memiliki fitur sebagai berikut:

- `account_id` : *unique key* dari pelanggan [tiket.com](https://www.tiket.com)
- `order_id` : *unique key* dari pesanan pelanggan
- `member_duration_days` : durasi member pelanggan sejak awal mendaftar
- `gender` : jenis kelamin
- `trip` : tipe perjalanan
- `service_class` : tipe layanan maskapai
- `price` : harga penerbangan
- `is_tx_promo` : penggunaan promosi saat transaksi penerbangan
- `no_of_seats` : jumlah kursi yang dipesan pelanggan
- `airlines_name` : nama maskapai
- `route` : rute penerbangan
- `hotel_id` : *unique key* dari hotel yang dipesan
- `visited_city` : daftar kota yang telah dikunjungi pelanggan
- `log_transaction` : daftar pengeluaran yang telah dihabiskan pelanggan

adapun tampilan 5 data teratas pada file `flight.csv` sebagai berikut:


```python
df_flight.head()
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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>service_class</th>
      <th>price</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>route</th>
      <th>hotel_id</th>
      <th>visited_city</th>
      <th>log_transaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>912aa410a02cd7e1bab414214a7005c0</td>
      <td>5c6f39c690f23650d3cde28e5b51c908</td>
      <td>566.0</td>
      <td>M</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>885898.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Semarang', 'Jakarta', 'Medan', 'Bali']'</td>
      <td>'[545203.03, 918492.11, 1774241.4, 885898.0]'</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d64a90a618202a5e8b25d8539377f3ca</td>
      <td>5cbef2b87f51c18bf399d11bfe495a46</td>
      <td>607.0</td>
      <td>M</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>2139751.25</td>
      <td>NO</td>
      <td>2.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Jakarta', 'Medan', 'Bali']'</td>
      <td>'[555476.36, 2422826.84, 7398697.64, 7930866.8...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1a42ac02bcb4a902973123323f84da55</td>
      <td>38fc35a1e62384012a358ab1fbd5ad03</td>
      <td>648.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>2695550.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Semarang', 'Jakarta', 'Medan', 'Bali']'</td>
      <td>'[7328957.45, 7027662.34, 1933360.88, 3461836....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92cddd64d4be4dec6dfbcc0c50e902f4</td>
      <td>c7f54cb748828b4413e02dea2758faf6</td>
      <td>418.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>1146665.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Jogjakarta', 'Bali', 'Jakarta', 'Medan']'</td>
      <td>'[5243631.69, 2474344.48, 1146665.0]'</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bf637abc47ea93bad22264f4956d67f6</td>
      <td>dec228e4d2b6023c9f1fe9cfe9c451bf</td>
      <td>537.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>1131032.50</td>
      <td>NO</td>
      <td>1.0</td>
      <td>6c483c0812c96f8ec43bb0ff76eaf716</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Jakarta', 'Bali', 'Medan', 'Jogjakarta', 'S...</td>
      <td>'[9808972.98, 9628619.79, 6712680.0, 5034510.0...</td>
    </tr>
  </tbody>
</table>
</div>



### Data Hotel

Data yang terdapat pada file `hotel.csv` memiliki fitur sebagai berikut:

- `hotel_id` : *unique key* dari hotel yang dipesan
- `starRating` : rating hotel
- `city` : kota hotel berada
- `free_wifi` : fasilitas wifi gratis
- `pool_access` : fasilitas kolam renang
- `free_breakfast` : fasilitas sarapan gratis

adapun tampilan 5 data teratas pada file `hotel.csv` sebagai berikut:


```python
df_hotel.head()
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
      <th>hotel_id</th>
      <th>starRating</th>
      <th>city</th>
      <th>free_wifi</th>
      <th>pool_access</th>
      <th>free_breakfast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e2733e84102226acf6b53bffd2e60cf8</td>
      <td>0.0</td>
      <td>bali</td>
      <td>YES</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9f9de5df06d64ada1026e930687a87e4</td>
      <td>0.0</td>
      <td>bali</td>
      <td>YES</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3cf6774fb4dc331bb49e7a959b74a67e</td>
      <td>0.0</td>
      <td>bali</td>
      <td>YES</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>eca261898220478834072b0c753a5229</td>
      <td>0.0</td>
      <td>bali</td>
      <td>YES</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c21f400013fa4f244a7168a3c155b8b5</td>
      <td>0.0</td>
      <td>bali</td>
      <td>YES</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing

Pada bagian data *preprocessing*, kami melakukan dua proses. Pertama, kami membuat dataframe baru yaitu `df_train` dan fitur baru pada dataframe tersebut sebagai variabel target yaitu `is_cross_sell`. Fitur `is_cross_sell` berdasarkan pada fitur `hotel_id` pada `df_train`. Jika `hotel_id` bernilai `None` maka tidak terjadi cross selling pada transaksi tersebut (`yes`) dan sebaliknya (`no`).

Kedua, kami mengubah tipe data fitur `log_transaction` dan `visited_city` masing-masing diubah dari string menjadi **list** dan **tuple**.


```python
df_train = df_flight.copy()
df_train['is_cross_sell'] = ~(df_flight['hotel_id']=='None')
```

*Helper functions* dibawah ini adalah untuk membantu mengubah data object **string** menjadi **list**, sehingga fitur dapat diolah dengan baik.


```python
def string_to_list(x):
    x = ast.literal_eval(x)
    x =  ast.literal_eval(x)
    return x
    
def string_to_list2(x):
    x = x.strip()
    x = ast.literal_eval(x)
    return x
```

*Helper function* dibawah ini berfungsi untuk mengekstrak nilai yang terdapat pada fitur `log_transaction`. Pada fitur tersebut akan kami ubah tipe datanya menjadi list sehingga kami dapat diambil nilai **max**, **min**, **mean**, **sum**, **std**, dan **panjang dari list** masing-masing baris.


```python
def log_tx(df):
    df['log_transaction'] = df['log_transaction'].apply(string_to_list)    
    df['max_log_transaction'] = df['log_transaction'].apply(max)
    df['min_log_transaction'] = df['log_transaction'].apply(min)
    df['mean_log_transaction'] = df['log_transaction'].apply(np.mean)
    df['len_log_transaction'] = df['log_transaction'].apply(len)
    df['sum_log_transaction'] = df['log_transaction'].apply(np.sum)
    df['sum_log_transaction'] = df['log_transaction'].apply(np.std)

    df.drop(['log_transaction'],inplace=True,axis=1)

log_tx(df_train)
log_tx(df_test)
```


```python
gc.collect()
```




    648



data *train* yang telah melalui *preprocessing* menggunakan *helper functions* diatas adalah sebagai berikut:


```python
df_train.head()
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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>service_class</th>
      <th>price</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>route</th>
      <th>hotel_id</th>
      <th>visited_city</th>
      <th>is_cross_sell</th>
      <th>max_log_transaction</th>
      <th>min_log_transaction</th>
      <th>mean_log_transaction</th>
      <th>len_log_transaction</th>
      <th>sum_log_transaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>912aa410a02cd7e1bab414214a7005c0</td>
      <td>5c6f39c690f23650d3cde28e5b51c908</td>
      <td>566.0</td>
      <td>M</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>885898.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Semarang', 'Jakarta', 'Medan', 'Bali']'</td>
      <td>False</td>
      <td>1774241.40</td>
      <td>545203.03</td>
      <td>1.030959e+06</td>
      <td>4</td>
      <td>4.533539e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d64a90a618202a5e8b25d8539377f3ca</td>
      <td>5cbef2b87f51c18bf399d11bfe495a46</td>
      <td>607.0</td>
      <td>M</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>2139751.25</td>
      <td>NO</td>
      <td>2.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Jakarta', 'Medan', 'Bali']'</td>
      <td>False</td>
      <td>18685958.20</td>
      <td>555476.36</td>
      <td>2.646397e+06</td>
      <td>1086</td>
      <td>2.624008e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1a42ac02bcb4a902973123323f84da55</td>
      <td>38fc35a1e62384012a358ab1fbd5ad03</td>
      <td>648.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>2695550.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Semarang', 'Jakarta', 'Medan', 'Bali']'</td>
      <td>False</td>
      <td>7328957.45</td>
      <td>1933360.88</td>
      <td>4.489474e+06</td>
      <td>5</td>
      <td>2.250021e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92cddd64d4be4dec6dfbcc0c50e902f4</td>
      <td>c7f54cb748828b4413e02dea2758faf6</td>
      <td>418.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>1146665.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Jogjakarta', 'Bali', 'Jakarta', 'Medan']'</td>
      <td>False</td>
      <td>5243631.69</td>
      <td>1146665.00</td>
      <td>2.954880e+06</td>
      <td>3</td>
      <td>1.706745e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bf637abc47ea93bad22264f4956d67f6</td>
      <td>dec228e4d2b6023c9f1fe9cfe9c451bf</td>
      <td>537.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>1131032.50</td>
      <td>NO</td>
      <td>1.0</td>
      <td>6c483c0812c96f8ec43bb0ff76eaf716</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>'['Jakarta', 'Bali', 'Medan', 'Jogjakarta', 'S...</td>
      <td>False</td>
      <td>13563940.00</td>
      <td>951639.00</td>
      <td>4.362199e+06</td>
      <td>157</td>
      <td>3.006539e+06</td>
    </tr>
  </tbody>
</table>
</div>



*Helper function* dibawah ini berfungsi untuk mengubah tipe data fitur `visited_city`. Pada fitur tersebut akan kami ubah tipe datanya menjadi dari **string** menjadi **list** tapi karena list ada tipe data yang ``unhashable`` maka kami mengubah lagi menjadi **tuple** agar dapat di modifikasi.


```python
def visited_pre(df):
    df['visited_city'] = df['visited_city'].apply(lambda x: x[1:-1])
    df['visited_city'] = df['visited_city'].apply(string_to_list2)
    df['visited_city'] = df['visited_city'].apply(tuple)

visited_pre(df_train)
visited_pre(df_test)
```

data *train* yang telah melalui *preprocessing* menggunakan *helper function* diatas adalah sebagai berikut:


```python
df_train.head()
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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>service_class</th>
      <th>price</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>route</th>
      <th>hotel_id</th>
      <th>visited_city</th>
      <th>is_cross_sell</th>
      <th>max_log_transaction</th>
      <th>min_log_transaction</th>
      <th>mean_log_transaction</th>
      <th>len_log_transaction</th>
      <th>sum_log_transaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>912aa410a02cd7e1bab414214a7005c0</td>
      <td>5c6f39c690f23650d3cde28e5b51c908</td>
      <td>566.0</td>
      <td>M</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>885898.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>(Semarang, Jakarta, Medan, Bali)</td>
      <td>False</td>
      <td>1774241.40</td>
      <td>545203.03</td>
      <td>1.030959e+06</td>
      <td>4</td>
      <td>4.533539e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d64a90a618202a5e8b25d8539377f3ca</td>
      <td>5cbef2b87f51c18bf399d11bfe495a46</td>
      <td>607.0</td>
      <td>M</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>2139751.25</td>
      <td>NO</td>
      <td>2.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>(Jakarta, Medan, Bali)</td>
      <td>False</td>
      <td>18685958.20</td>
      <td>555476.36</td>
      <td>2.646397e+06</td>
      <td>1086</td>
      <td>2.624008e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1a42ac02bcb4a902973123323f84da55</td>
      <td>38fc35a1e62384012a358ab1fbd5ad03</td>
      <td>648.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>2695550.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>(Semarang, Jakarta, Medan, Bali)</td>
      <td>False</td>
      <td>7328957.45</td>
      <td>1933360.88</td>
      <td>4.489474e+06</td>
      <td>5</td>
      <td>2.250021e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92cddd64d4be4dec6dfbcc0c50e902f4</td>
      <td>c7f54cb748828b4413e02dea2758faf6</td>
      <td>418.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>1146665.00</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>(Jogjakarta, Bali, Jakarta, Medan)</td>
      <td>False</td>
      <td>5243631.69</td>
      <td>1146665.00</td>
      <td>2.954880e+06</td>
      <td>3</td>
      <td>1.706745e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bf637abc47ea93bad22264f4956d67f6</td>
      <td>dec228e4d2b6023c9f1fe9cfe9c451bf</td>
      <td>537.0</td>
      <td>F</td>
      <td>trip</td>
      <td>ECONOMY</td>
      <td>1131032.50</td>
      <td>NO</td>
      <td>1.0</td>
      <td>6c483c0812c96f8ec43bb0ff76eaf716</td>
      <td>CGK - DPS</td>
      <td>None</td>
      <td>(Jakarta, Bali, Medan, Jogjakarta, Semarang)</td>
      <td>False</td>
      <td>13563940.00</td>
      <td>951639.00</td>
      <td>4.362199e+06</td>
      <td>157</td>
      <td>3.006539e+06</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis (EDA)

### Visualisasi

Agar data dapat dipahami dengan baik, visualisasi keterkaitan antara fitur-fitur yang ada dengan `is_cross_sell` perlu dilakukan. Selain untuk lebih memahami karakter data, kami dapat menganalisis lebih jauh hubungan `is_cross_sell` dengan fitur lainnya. Visualisasi menggunakan **scatter plot** dan **bar plot**


```python
sns.scatterplot(data=df_train,x='price',y='member_duration_days',hue='is_cross_sell')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f209cead048>




![png](images/sour-soup_files/sour-soup_24_1.png)


Dapat dilihat dari *scatter plot* diatas, persebaran dari harga dan durasi member terhadap *cross sell* menyebar sehingga harga dan durasi member tidak cukup untuk menentukan apakah suatu transaksi terjadi *cross sell* atau tidak. 


```python
sns.countplot(data=df_train,x='is_tx_promo',hue='is_cross_sell')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2090123be0>




![png](images/sour-soup_files/sour-soup_26_1.png)


Dari *bar plot* diatas dapat disimpulkan bahwa baik ada promo maupun tidak, transaksi tanpa *cross sell* banyak terjadi dengan lebih dari 45000 transaksi.  


```python
sns.countplot(data=df_train,x='no_of_seats',hue='is_cross_sell')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20a23d85f8>




![png](images/sour-soup_files/sour-soup_28_1.png)


Dapat dilihat bahwa pelanggan yang membeli lebih dari 5 tiket dalam sekali order hampir tidak pernah memesan hotel, asumsi kami bahwa pelanggan yang memesan lebih dari 5 tiket adalah keluarga yang mudik atau berkunjung ke rumah sanak saudara.


```python
sns.countplot(data=df_train,x='trip',hue='is_cross_sell')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2090eb42b0>




![png](images/sour-soup_files/sour-soup_30_1.png)


Pada fitur `trip`, lebih banyak bernilai `trip` yang melakukan transaksi *cross sell*. kami menganggap nilai `roundtrip` dan `round` merupakan nilai yang sama, karena memiliki definisi yang cukup mirip dan banyak nilai `round` yang terlalu sedikit.


```python
# nilai unik dari visited_city`
print(df_train['visited_city'].unique())

sns.countplot(data=df_train,x='visited_city',hue='is_cross_sell',orient='v')
```

    [('Semarang', 'Jakarta', 'Medan', 'Bali') ('Jakarta', 'Medan', 'Bali')
     ('Jogjakarta', 'Bali', 'Jakarta', 'Medan')
     ('Jakarta', 'Bali', 'Medan', 'Jogjakarta', 'Semarang')
     ('Bali', 'Jakarta', 'Medan') ('Medan', 'Bali', 'Jakarta')
     ('Manado', 'Medan', 'Bali', 'Jakarta')
     ('Surabaya', 'Medan', 'Bali', 'Jakarta', 'Aceh')]
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2093580a58>




![png](images/sour-soup_files/sour-soup_32_2.png)


Pelanggan yang pernah mengunjungi `('Jakarta', 'Medan', 'Bali')` paling banyak melakukan transaksi *cross sell* maupun tidak.


```python
sns.countplot(data=df_train,x='service_class',hue='is_cross_sell',orient='v')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f209031e198>




![png](images/sour-soup_files/sour-soup_34_1.png)


Pada fitur `service_class` jauh lebih banyak row yang bernilai `economy` daripada `businesss` sehingga kami rasa fitur ini tidak layak untuk dianalisa lebih lanjut.

### Processing EDA
Dari EDA diatas kami memutuskan untuk melakukan proses berikut:

1. Menghapus fitur:
  - `service_class` : lebih banyak bernilai **economy**.
  - `route` : hanya memiliki **satu** nilai,
  - `log_transaction` : sudah mengambil **nilai min, max, mean, median, std, sum dan count** (Proses ini telah dilakukan bersamaan dengan pengambilan nilai-nilai tersebut).


```python
df_train.drop(['service_class','route'],axis=1,inplace=True)
df_test.drop(['service_class','route'],axis=1,inplace=True)
```

2. Mengkombinasikan nilai menjadi satu pada suatu fitur :
  - `trip` : **roundtrip** dan **round** menjadi **round** saja.


```python
df_train['trip'] = np.where(df_train['trip']=='roundtrip', 'round', df_train['trip'])
df_test['trip'] = np.where(df_test['trip']=='roundtrip', 'round', df_test['trip'])
```

3. Mentransformasi nilai pada fitur `member_duration_days`, `price`, `max_log_transaction`, `min_log_transaction`, `mean_log_transaction`, `len_log_transaction`, dan `sum_log_transaction` yang bertipe numerik menggunakan logaritma natural.

Sebelum melakukan *normalisasi*, kami terlebih dahulu mengubah nilai negatif pada fitur `min_log_transaction` menjadi nilai **mean** fitur tersebut.


```python
df_train['min_log_transaction'] = np.where(df_train['min_log_transaction'] < 0 ,  df_train['min_log_transaction'].mean(), df_train['min_log_transaction'])
df_test['min_log_transaction'] = np.where(df_test['min_log_transaction'] < 0, df_test['min_log_transaction'].mean(), df_test['min_log_transaction'])
```


```python
cols = df_train.drop('no_of_seats',axis=1).select_dtypes([np.number]).columns
for col in cols:
    df_train[col] = np.log1p(df_train[col])
    df_test[col] = np.log1p(df_test[col])    
```

## Feature Engineering

### New Features
Proses ini merupakan pembuatan fitur baru menggunakan fitur yang sudah ada dan *insights* yang telah dianalisis pada proses EDA sebelumnya. Pada proses *feature engineering* ini, kami membuat beberapa fitur baru, yaitu:

- `count_use_hotel` : berapa kali pelanggan bersangkutan memesan hotel?
- `mode_facility_hotel` : *mode* dari fasilitas suatu hotel.
- `mode_rate_hotel` : *mode* rating yang diberikan oleh pelanggan.
- `count_user_id` : berapa kali customer melakukan transaksi.
- `count_user_id_with_air` : Berapa kali customer memesan dengan maskapai yang sama.
- `how_many_city` : berapa kota yang telah pelanggan kunjungi.
- `mean_price_air` : rata-rata harga setiap maskapai.
- `price_per_seat` : harga untuk tiap kursi yang dipesan.
- `order_rate` : jumlah berapa kali customer melakukan transaksi.

Untuk membuat fitur `count_use_hotel`, kami menggunakan aggregasi `account_id` dan `hotel_id` dan menggunakan *helper function* `count_hotel(x)` untuk menentukan berapa kali pelanggan [tiket.com](https://www.tiket.com) memesan tiket bersamaan dengan hotel.

Dan untuk fitur yang membutuhkan hotel id terdapat 2 kemungkinan nilai NaN dimana sebuah order memesan hotel yang tidak terdapat pada dataframe hotel atau tidak memesan hotel sama sekali, untuk kasus pertama kami menggunakan -999 untuk mengganti nilai NaN nya dan -1 untuk kasus kedua agar model dapat membedakan kedua kasus tersebut.


```python
# list of account that ever use hotel
account_with_hotel = df_train[df_train['hotel_id']!='None'].groupby('account_id').agg('count')['order_id']

def count_hotel(x):
    if x in account_with_hotel:
        return account_with_hotel[x]
    return 0

# add ever_use_hotel into dataframe
df_train['count_use_hotel'] = df_train['account_id'].apply(count_hotel)
df_test['count_use_hotel'] = df_test['account_id'].apply(count_hotel)
```

Sebelum membuat fitur `mode_facility_hotel`, kami tambahkan fitur `fasilitas` di `df_hotel` yang berisikan *banyak* fasilitas yang tersedia pada suatu hotel.


```python
df_hotel["free_wifi_bool"] = np.where(df_hotel["free_wifi"] == 'YES', 1, 0)
df_hotel["pool_access_bool"] = np.where(df_hotel['pool_access'] == 'YES', 1, 0)
df_hotel["free_breakfast_bool"] = np.where(df_hotel['free_breakfast'] == 'YES', 1, 0)
df_hotel['fasilitas'] = df_hotel['free_wifi_bool'] + df_hotel['pool_access_bool'] + df_hotel['free_breakfast_bool']
```

Untuk membuat fitur `mode_facility_hotel`, kami menggunakan aggregasi `account_id` dan `fasilitas` dan menggunakan *helper function* `mode_facility_hotel(x)` untuk menentukan ***mode*** fasilitas dari suatu hotel.


```python
# list of mode of facility in a hotel
mode_fasilitas = pd.merge(df_train[df_train['hotel_id']!='None'],df_hotel,on='hotel_id',how='left').fillna(-999).groupby('account_id')['fasilitas'].agg(lambda x:x.value_counts().index[0])

def mode_facilty_hotel(x):
    if x in mode_fasilitas:
        return mode_fasilitas[x]
    return -1

# add mode_facility_hotel into dataframe
df_train['mode_facilty_hotel'] = df_train['account_id'].apply(mode_facilty_hotel)
df_test['mode_facilty_hotel'] = df_test['account_id'].apply(mode_facilty_hotel)
```

![](http://)Untuk membuat fitur `mode_star_rating`, kami menggunakan mengkombinasikan `account_id` dan `hotel_id` sehingga  menggunakan *helper function* `mode_rate_hotel(x)` untuk menentukan **modus dari rating** hotel yang digunakan oleh pelanggan.


```python
# list of mode star rating in a hotel
mode_star_rating = pd.merge(df_train[df_train['hotel_id']!='None'],df_hotel,on='hotel_id',how='left').fillna(-999).groupby('account_id')['starRating'].agg(lambda x:x.value_counts().index[0])

def mode_rate_hotel(x):
    if x in mode_star_rating:
        return mode_star_rating[x]
    return -1

# add avg_rate_hotel into dataframe
df_train['mode_rate_hotel'] = df_train['account_id'].apply(mode_rate_hotel)
df_test['mode_rate_hotel'] = df_test['account_id'].apply(mode_rate_hotel)
```

Untuk membuat fitur `count_user_id`, kami aggregasi fitur `account_id` dan `order_id` untuk melihat jumlah transaksi pada seorang pelanggan.


```python
# Count how many he/she already done an order
df_train['count_user_id'] = df_train.groupby(['account_id'])['order_id'].transform('count')
df_test['count_user_id'] = df_test.groupby(['account_id'])['order_id'].transform('count')
```

Untuk membuat fitur `count_user_id_with_air`, kami aggregasi fitur `account_id`, `airlines_name` dan `order_id` untuk melihat jumlah transaksi pada suatu maskapai.


```python
# Count how many he/she already done an order within specified airlines
df_train['count_user_id_with_air'] = df_train.groupby(['account_id','airlines_name'])['order_id'].transform('count')
df_test['count_user_id_with_air'] = df_test.groupby(['account_id','airlines_name'])['order_id'].transform('count')
```


```python
sns.scatterplot(data=df_train,x='price',y='count_user_id',hue='is_cross_sell')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2090004358>




![png](images/sour-soup_files/sour-soup_57_1.png)


*Scatter plot* diatas memperlihatkan banyaknya transaksi dan harga penerbangan serta apakah terjadi *cross sell* atau tidak.

Untuk membuat fitur `how_many_city`, kami menggunakan fitur `visited_city` dengn kode dibawah ini sehingga didapatkan jumlah kota yang dikunjungi oleh pelanggan sebelum melakukan transaksi.


```python
# count how many cities was visited
df_train['how_many_city'] = df_train['visited_city'].apply(lambda x: len(x))
df_test['how_many_city'] = df_test['visited_city'].apply(lambda x: len(x))
```

Kemudian untuk membuat `mean_price_air`, kami menggunakan aggregasi fitur `airlines_name` dan `price` untuk menghitung **mean** sehingga didapatkan rata-rata harga pada setiap maskapai yang ada.


```python
df_prices = df_train.groupby(['airlines_name'])['price'].mean().reset_index()
df_train = pd.merge(df_train,df_prices,on='airlines_name',how='left')
df_train = df_train.rename(columns={'price_y':'mean_price_air'})

df_prices = df_test.groupby(['airlines_name'])['price'].mean().reset_index()
df_test = pd.merge(df_test,df_prices,on='airlines_name',how='left')
df_test = df_test.rename(columns={'price_y':'mean_price_air'})
```

Fitur `price_per_seat` didapatkan dari fitur `price_x` / `no_of_seats`, yakni harga pada order dibagi dengan jumlah kursi yang dipesan. Proses tersebut menghasilkan rata-rata **harga kursi** untuk maskapai terkait.


```python
# Price for each seat
df_train['price_per_seat'] = df_train['price_x']/df_train['no_of_seats']
df_test['price_per_seat'] = df_test['price_x']/df_test['no_of_seats']
```

Fitur `order_rate` didapatkan dari fitur `len_log_transaction` / `member_duration_days`, yakni banyaknya transaksi yang dilakukan oleh pelanggan dibagi dengan lamanya pelanggan menjadi member [tiket.com](https://www.tiket.com). Fitur ini mengindikasikan **seberapa sering** pelanggan melakukan transaksi sejak pelanggan terdaftar menjadi member [tiket.com](https://www.tiket.com).


```python
# Order Rate of Customer
df_train['order_rate'] = df_train['len_log_transaction']/df_train['member_duration_days']
df_test['order_rate'] = df_test['len_log_transaction']/df_test['member_duration_days']
```

Berikut ini hasil *feature engineering* yang telah kami lakukan pada `df_train` dan `df_test`:


```python
display(df_train.head())
display(df_test.head())
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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>price_x</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>hotel_id</th>
      <th>...</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>count_user_id</th>
      <th>count_user_id_with_air</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>912aa410a02cd7e1bab414214a7005c0</td>
      <td>5c6f39c690f23650d3cde28e5b51c908</td>
      <td>6.340359</td>
      <td>M</td>
      <td>trip</td>
      <td>13.694358</td>
      <td>NO</td>
      <td>1.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>None</td>
      <td>...</td>
      <td>13.024431</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>14.435097</td>
      <td>13.694358</td>
      <td>0.253840</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d64a90a618202a5e8b25d8539377f3ca</td>
      <td>5cbef2b87f51c18bf399d11bfe495a46</td>
      <td>6.410175</td>
      <td>M</td>
      <td>trip</td>
      <td>14.576201</td>
      <td>NO</td>
      <td>2.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>None</td>
      <td>...</td>
      <td>14.780214</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>3311</td>
      <td>575</td>
      <td>3</td>
      <td>14.640493</td>
      <td>7.288100</td>
      <td>1.090637</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1a42ac02bcb4a902973123323f84da55</td>
      <td>38fc35a1e62384012a358ab1fbd5ad03</td>
      <td>6.475433</td>
      <td>F</td>
      <td>trip</td>
      <td>14.807113</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>None</td>
      <td>...</td>
      <td>14.626451</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>14.640493</td>
      <td>14.807113</td>
      <td>0.276701</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92cddd64d4be4dec6dfbcc0c50e902f4</td>
      <td>c7f54cb748828b4413e02dea2758faf6</td>
      <td>6.037871</td>
      <td>F</td>
      <td>trip</td>
      <td>13.952369</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>None</td>
      <td>...</td>
      <td>14.350099</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>14.640493</td>
      <td>13.952369</td>
      <td>0.229600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bf637abc47ea93bad22264f4956d67f6</td>
      <td>dec228e4d2b6023c9f1fe9cfe9c451bf</td>
      <td>6.287859</td>
      <td>F</td>
      <td>trip</td>
      <td>13.938642</td>
      <td>NO</td>
      <td>1.0</td>
      <td>6c483c0812c96f8ec43bb0ff76eaf716</td>
      <td>None</td>
      <td>...</td>
      <td>14.916300</td>
      <td>10</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>161</td>
      <td>103</td>
      <td>5</td>
      <td>14.531102</td>
      <td>13.938642</td>
      <td>0.805138</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>price_x</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>visited_city</th>
      <th>...</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>count_user_id</th>
      <th>count_user_id_with_air</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>89a5fadd4d596610ff56044b9a0b1f4f</td>
      <td>5ca64fd80a069208e3c0aa05dd580fb8</td>
      <td>7.470224</td>
      <td>M</td>
      <td>trip</td>
      <td>14.960816</td>
      <td>YES</td>
      <td>3</td>
      <td>e35de6a36d385711a660c72c0286154a</td>
      <td>(Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>14.909172</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>14.751176</td>
      <td>4.986939</td>
      <td>0.260489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86b28323bec6d938d47cee887e509b28</td>
      <td>aca60904549a8a5958fe7a642efcb534</td>
      <td>6.989335</td>
      <td>F</td>
      <td>trip</td>
      <td>14.588673</td>
      <td>NO</td>
      <td>2</td>
      <td>e35de6a36d385711a660c72c0286154a</td>
      <td>(Medan, Bali, Jakarta)</td>
      <td>...</td>
      <td>14.881040</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>14.751176</td>
      <td>7.294337</td>
      <td>0.230271</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36ef956ac3ef963c48e67327a4b6cc78</td>
      <td>1771011e3adec5db9f30d15b3d439711</td>
      <td>7.774436</td>
      <td>M</td>
      <td>round</td>
      <td>14.030312</td>
      <td>NO</td>
      <td>1</td>
      <td>ad5bef60d81ea077018f4d50b813153a</td>
      <td>(Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>14.707013</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>14.475678</td>
      <td>14.030312</td>
      <td>0.250296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f7821289404d44db50eb2edd4f82ea5b</td>
      <td>6fc1b7d590c2a8c539ce56397403194d</td>
      <td>6.357842</td>
      <td>F</td>
      <td>trip</td>
      <td>14.500656</td>
      <td>YES</td>
      <td>2</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>(Jakarta, Bali, Medan, Jogjakarta, Semarang)</td>
      <td>...</td>
      <td>14.746227</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>14.431748</td>
      <td>7.250328</td>
      <td>0.218045</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f62f33d1de5aabc919b69b1b5697f27a</td>
      <td>c1f4712f60cd758e773555690d148764</td>
      <td>6.760415</td>
      <td>F</td>
      <td>trip</td>
      <td>14.910993</td>
      <td>YES</td>
      <td>1</td>
      <td>74c5549aa99d55280a896ea50068a211</td>
      <td>(Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>14.816352</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>14.933932</td>
      <td>14.910993</td>
      <td>0.205061</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


Setelah dianalisis, penggunaan fitur `hotel_id` tidak diperlukan kembali sehingga perlu dihapus.


```python
df_train.drop(['hotel_id'],axis=1,inplace=True)
```

### Feature Scaling

Pada proses ini bertujuan untuk menyamakan skala nilai dari semua variable dimana terdapat fitur `member_duration_days` yang skala yang kecil dan fitur lain, yaitu `price` yang memiliki nilai jutaan.

#### Standard Scale
> *Standard Scale* merupakan proses untuk menstandarkan fitur dengan mengurangi mean data dan kemudian menskalakannya ke varians unit. Varians unit berarti membagi semua nilai data dengan standar deviasi. *Standard Scale* menghasilkan distribusi dengan standar deviasi sama dengan 1. Selain itu, *Standard Scale* membuat rata-rata dari distribusi data menjadi 0.

Pada proses ini mengambil seluruh fitur pada `df_train` dan `df_test` yang digabungkan menjadi `df_all` yang bertipe numerik kecuali feature dengan kardinalitas yang kecil seperti `no_of_seats`, `mode_rate_hotel` dan `mode_facilty_hotel` untuk dilakukan proses *scaling*. Hasil proses tersebut kemudian disimpan pada dataframe `df_train_sc` dan `df_test_sc`.


```python
cols = df_train.drop(['no_of_seats','mode_rate_hotel','mode_facilty_hotel'],axis=1).select_dtypes([np.number]).columns
df_all = pd.concat([df_train.drop('is_cross_sell',axis=1),df_test])
sc = StandardScaler()
for i in cols:
    df_all[i] = sc.fit_transform(df_all[i].values.reshape(-1,1))   
df_all.set_index('order_id',inplace=True)

df_train_sc = df_all.loc[df_train['order_id'].values].reset_index()
df_test_sc = df_all.loc[df_test['order_id'].values].reset_index()
```


```python
df_train_sc = pd.merge(df_train_sc,df_train[['order_id','is_cross_sell']],on='order_id')
```

Saat ini kami memiliki data *train* dan data *test* yang di*scale* dan yang tidak di*scale*. Hasil *Standar Scale* pada `df_train_sc` dan `df_test_sc`:


```python
display(df_train_sc.head())
display(df_test_sc.head())
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
      <th>order_id</th>
      <th>account_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>price_x</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>visited_city</th>
      <th>...</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>count_user_id</th>
      <th>count_user_id_with_air</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>is_cross_sell</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5c6f39c690f23650d3cde28e5b51c908</td>
      <td>912aa410a02cd7e1bab414214a7005c0</td>
      <td>-0.661036</td>
      <td>M</td>
      <td>trip</td>
      <td>-1.387447</td>
      <td>NO</td>
      <td>1.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>(Semarang, Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.170508</td>
      <td>-0.165551</td>
      <td>0.543227</td>
      <td>-0.868480</td>
      <td>0.651444</td>
      <td>-0.223712</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5cbef2b87f51c18bf399d11bfe495a46</td>
      <td>d64a90a618202a5e8b25d8539377f3ca</td>
      <td>-0.534578</td>
      <td>M</td>
      <td>trip</td>
      <td>0.015859</td>
      <td>NO</td>
      <td>2.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>(Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>6.130028</td>
      <td>3.879247</td>
      <td>-0.661594</td>
      <td>0.491785</td>
      <td>-0.917180</td>
      <td>4.937157</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38fc35a1e62384012a358ab1fbd5ad03</td>
      <td>1a42ac02bcb4a902973123323f84da55</td>
      <td>-0.416376</td>
      <td>F</td>
      <td>trip</td>
      <td>0.383319</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>(Semarang, Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.168604</td>
      <td>-0.165551</td>
      <td>0.543227</td>
      <td>0.491785</td>
      <td>0.923911</td>
      <td>-0.082719</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c7f54cb748828b4413e02dea2758faf6</td>
      <td>92cddd64d4be4dec6dfbcc0c50e902f4</td>
      <td>-1.208935</td>
      <td>F</td>
      <td>trip</td>
      <td>-0.976865</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>(Jogjakarta, Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.172412</td>
      <td>-0.165551</td>
      <td>0.543227</td>
      <td>0.491785</td>
      <td>0.714620</td>
      <td>-0.373212</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dec228e4d2b6023c9f1fe9cfe9c451bf</td>
      <td>bf637abc47ea93bad22264f4956d67f6</td>
      <td>-0.756131</td>
      <td>F</td>
      <td>trip</td>
      <td>-0.998709</td>
      <td>NO</td>
      <td>1.0</td>
      <td>6c483c0812c96f8ec43bb0ff76eaf716</td>
      <td>(Jakarta, Bali, Medan, Jogjakarta, Semarang)</td>
      <td>...</td>
      <td>2.448484</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.132238</td>
      <td>0.553211</td>
      <td>1.748048</td>
      <td>-0.232671</td>
      <td>0.711258</td>
      <td>3.176367</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



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
      <th>order_id</th>
      <th>account_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>price_x</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>visited_city</th>
      <th>...</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>count_user_id</th>
      <th>count_user_id_with_air</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5ca64fd80a069208e3c0aa05dd580fb8</td>
      <td>89a5fadd4d596610ff56044b9a0b1f4f</td>
      <td>1.385497</td>
      <td>M</td>
      <td>trip</td>
      <td>0.627911</td>
      <td>YES</td>
      <td>3.0</td>
      <td>e35de6a36d385711a660c72c0286154a</td>
      <td>(Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>0.610797</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.172412</td>
      <td>-0.165551</td>
      <td>-0.661594</td>
      <td>1.224800</td>
      <td>-1.480638</td>
      <td>-0.182707</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aca60904549a8a5958fe7a642efcb534</td>
      <td>86b28323bec6d938d47cee887e509b28</td>
      <td>0.514459</td>
      <td>F</td>
      <td>trip</td>
      <td>0.035708</td>
      <td>NO</td>
      <td>2.0</td>
      <td>e35de6a36d385711a660c72c0286154a</td>
      <td>(Medan, Bali, Jakarta)</td>
      <td>...</td>
      <td>0.567207</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.172412</td>
      <td>-0.165551</td>
      <td>-0.661594</td>
      <td>1.224800</td>
      <td>-0.915653</td>
      <td>-0.369076</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1771011e3adec5db9f30d15b3d439711</td>
      <td>36ef956ac3ef963c48e67327a4b6cc78</td>
      <td>1.936517</td>
      <td>M</td>
      <td>round</td>
      <td>-0.852832</td>
      <td>NO</td>
      <td>1.0</td>
      <td>ad5bef60d81ea077018f4d50b813153a</td>
      <td>(Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>0.297556</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.172412</td>
      <td>-0.165551</td>
      <td>-0.661594</td>
      <td>-0.599725</td>
      <td>0.733705</td>
      <td>-0.245571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6fc1b7d590c2a8c539ce56397403194d</td>
      <td>f7821289404d44db50eb2edd4f82ea5b</td>
      <td>-0.629369</td>
      <td>F</td>
      <td>trip</td>
      <td>-0.104357</td>
      <td>YES</td>
      <td>2.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>(Jakarta, Bali, Medan, Jogjakarta, Semarang)</td>
      <td>...</td>
      <td>0.358317</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.172412</td>
      <td>-0.165551</td>
      <td>1.748048</td>
      <td>-0.890659</td>
      <td>-0.926429</td>
      <td>-0.444477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c1f4712f60cd758e773555690d148764</td>
      <td>f62f33d1de5aabc919b69b1b5697f27a</td>
      <td>0.099814</td>
      <td>F</td>
      <td>trip</td>
      <td>0.548626</td>
      <td>YES</td>
      <td>1.0</td>
      <td>74c5549aa99d55280a896ea50068a211</td>
      <td>(Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>0.466974</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-0.172412</td>
      <td>-0.165551</td>
      <td>-0.661594</td>
      <td>2.435134</td>
      <td>0.949346</td>
      <td>-0.524556</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


mengecek banyaknya data `df_train` yang terjadi *cross sell*.


```python
df_train['is_cross_sell'].sum()
```




    6748



### Encoding

Encoding yang kami lakukan adalah menggunakan *mean encoding* dengan *smoothing*, kami menggunakan ini untuk memberi gambaran distribusi target pada data train untuk model karena kami berasumsi bahwa distribusi nya akan mirip dengan banyak order tidak **cross sell** jauh mengalahkan banyak order yang **cross sell**.

#### Mean Encoding
> Salah satu teknik umum dalam *feature engineering* adalah mengubah kategorikal data menjadi numerik. *Mean encoding* memperhitungkan jumlah label beserta variabel target untuk diencode labelnya ke dalam nilai yang dapat dipahami model (numerik). Ilustrasi *mean encoding* terdapat pada gambar dibawah ini.

![](http://www.renom.jp/notebooks/tutorial/preprocessing/category_encoding/renom_cat_target.png)

Untuk Melakukan *mean encoding*, kami menggunakan *helper function* `calc_smooth_mean()` yang akan membantu untuk melakukan *mean encoding* pada fitur tertentu. Fitur yang akan dilakukan proses *mean encoding* adalah `gender`, `trip`, `is_tx_promo`, `no_of_seats`, `airlines_name`, dan `visited_city`.


```python
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth),means
```

1. Mean encoding pada fitur `gender`


```python
df_train['gender_enc'] = calc_smooth_mean(df_train, 'gender', 'is_cross_sell', m=300)[0]
df_train_sc['gender_enc'] = calc_smooth_mean(df_train_sc, 'gender', 'is_cross_sell', m=300)[0]

gender_enc = calc_smooth_mean(df_train, 'gender', 'is_cross_sell', m=300)[1]
df_test['gender_enc'] = df_test['gender'].apply(lambda x:gender_enc[x])
df_test_sc['gender_enc'] = df_test_sc['gender'].apply(lambda x:gender_enc[x])
```

2. Mean encoding pada fitur `trip`


```python
df_train['trip_enc'] = calc_smooth_mean(df_train, 'trip', 'is_cross_sell', m=300)[0]
df_train_sc['trip_enc'] = calc_smooth_mean(df_train_sc, 'trip', 'is_cross_sell', m=300)[0]

trip_enc = calc_smooth_mean(df_train, 'trip', 'is_cross_sell', m=300)[1]
df_test['trip_enc'] = df_test['trip'].apply(lambda x:trip_enc[x])
df_test_sc['trip_enc'] = df_test_sc['trip'].apply(lambda x:trip_enc[x])
```

3. Mean encoding pada fitur `is_tx_promo`


```python
df_train['promo_enc'] = calc_smooth_mean(df_train, 'is_tx_promo', 'is_cross_sell', m=300)[0]
df_train_sc['promo_enc'] = calc_smooth_mean(df_train_sc, 'is_tx_promo', 'is_cross_sell', m=300)[0]

promo_enc = calc_smooth_mean(df_train, 'is_tx_promo', 'is_cross_sell', m=300)[1]
df_test['promo_enc'] = df_test['is_tx_promo'].apply(lambda x:promo_enc[x])
df_test_sc['promo_enc'] = df_test_sc['is_tx_promo'].apply(lambda x:promo_enc[x])
```

4. Mean encoding pada fitur `no_of_seats`


```python
df_train['seats_enc'] = calc_smooth_mean(df_train, 'no_of_seats', 'is_cross_sell', m=300)[0]
df_train_sc['seats_enc'] = calc_smooth_mean(df_train_sc, 'no_of_seats', 'is_cross_sell', m=300)[0]

seats_enc = calc_smooth_mean(df_train, 'no_of_seats', 'is_cross_sell', m=300)[1]
df_test['seats_enc'] = df_test['no_of_seats'].apply(lambda x:seats_enc[x])
df_test_sc['seats_enc'] = df_test_sc['no_of_seats'].apply(lambda x:seats_enc[x])
```

5. Mean encoding pada fitur `airlines_name`


```python
df_train['air_enc'] = calc_smooth_mean(df_train, 'airlines_name', 'is_cross_sell', m=300)[0]
df_train_sc['air_enc'] = calc_smooth_mean(df_train_sc, 'airlines_name', 'is_cross_sell', m=300)[0]

air_enc = calc_smooth_mean(df_train, 'airlines_name', 'is_cross_sell', m=300)[1]
df_test['air_enc'] = df_test['airlines_name'].apply(lambda x:air_enc[x])
df_test_sc['air_enc'] = df_test_sc['airlines_name'].apply(lambda x:air_enc[x])
```

6. Mean encoding pada fitur `visited_city`


```python
df_train['visited_city_enc'] = calc_smooth_mean(df_train, 'visited_city', 'is_cross_sell', m=300)[0]
df_train_sc['visited_city_enc'] = calc_smooth_mean(df_train_sc, 'visited_city', 'is_cross_sell', m=300)[0]

visited_city_enc = calc_smooth_mean(df_train, 'visited_city', 'is_cross_sell', m=300)[1]
df_test['visited_city_enc'] = df_test['visited_city'].apply(lambda x:visited_city_enc[x])
df_test_sc['visited_city_enc'] = df_test_sc['visited_city'].apply(lambda x:visited_city_enc[x])
```

Hasil *mean encoding* pada `df_train` dan `df_test`:


```python
display(df_train.head())
display(df_test.head())
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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>price_x</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>visited_city</th>
      <th>...</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>gender_enc</th>
      <th>trip_enc</th>
      <th>promo_enc</th>
      <th>seats_enc</th>
      <th>air_enc</th>
      <th>visited_city_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>912aa410a02cd7e1bab414214a7005c0</td>
      <td>5c6f39c690f23650d3cde28e5b51c908</td>
      <td>6.340359</td>
      <td>M</td>
      <td>trip</td>
      <td>13.694358</td>
      <td>NO</td>
      <td>1.0</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>(Semarang, Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>4</td>
      <td>14.435097</td>
      <td>13.694358</td>
      <td>0.253840</td>
      <td>0.054131</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.046482</td>
      <td>0.060278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d64a90a618202a5e8b25d8539377f3ca</td>
      <td>5cbef2b87f51c18bf399d11bfe495a46</td>
      <td>6.410175</td>
      <td>M</td>
      <td>trip</td>
      <td>14.576201</td>
      <td>NO</td>
      <td>2.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>(Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>3</td>
      <td>14.640493</td>
      <td>7.288100</td>
      <td>1.090637</td>
      <td>0.054131</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.070097</td>
      <td>0.059034</td>
      <td>0.050315</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1a42ac02bcb4a902973123323f84da55</td>
      <td>38fc35a1e62384012a358ab1fbd5ad03</td>
      <td>6.475433</td>
      <td>F</td>
      <td>trip</td>
      <td>14.807113</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>(Semarang, Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>4</td>
      <td>14.640493</td>
      <td>14.807113</td>
      <td>0.276701</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059034</td>
      <td>0.060278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92cddd64d4be4dec6dfbcc0c50e902f4</td>
      <td>c7f54cb748828b4413e02dea2758faf6</td>
      <td>6.037871</td>
      <td>F</td>
      <td>trip</td>
      <td>13.952369</td>
      <td>NO</td>
      <td>1.0</td>
      <td>0a102015e48c1f68e121acc99fca9a05</td>
      <td>(Jogjakarta, Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>4</td>
      <td>14.640493</td>
      <td>13.952369</td>
      <td>0.229600</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059034</td>
      <td>0.057819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bf637abc47ea93bad22264f4956d67f6</td>
      <td>dec228e4d2b6023c9f1fe9cfe9c451bf</td>
      <td>6.287859</td>
      <td>F</td>
      <td>trip</td>
      <td>13.938642</td>
      <td>NO</td>
      <td>1.0</td>
      <td>6c483c0812c96f8ec43bb0ff76eaf716</td>
      <td>(Jakarta, Bali, Medan, Jogjakarta, Semarang)</td>
      <td>...</td>
      <td>5</td>
      <td>14.531102</td>
      <td>13.938642</td>
      <td>0.805138</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059379</td>
      <td>0.062092</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



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
      <th>account_id</th>
      <th>order_id</th>
      <th>member_duration_days</th>
      <th>gender</th>
      <th>trip</th>
      <th>price_x</th>
      <th>is_tx_promo</th>
      <th>no_of_seats</th>
      <th>airlines_name</th>
      <th>visited_city</th>
      <th>...</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>gender_enc</th>
      <th>trip_enc</th>
      <th>promo_enc</th>
      <th>seats_enc</th>
      <th>air_enc</th>
      <th>visited_city_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>89a5fadd4d596610ff56044b9a0b1f4f</td>
      <td>5ca64fd80a069208e3c0aa05dd580fb8</td>
      <td>7.470224</td>
      <td>M</td>
      <td>trip</td>
      <td>14.960816</td>
      <td>YES</td>
      <td>3</td>
      <td>e35de6a36d385711a660c72c0286154a</td>
      <td>(Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>3</td>
      <td>14.751176</td>
      <td>4.986939</td>
      <td>0.260489</td>
      <td>0.054116</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.074429</td>
      <td>0.062465</td>
      <td>0.058189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86b28323bec6d938d47cee887e509b28</td>
      <td>aca60904549a8a5958fe7a642efcb534</td>
      <td>6.989335</td>
      <td>F</td>
      <td>trip</td>
      <td>14.588673</td>
      <td>NO</td>
      <td>2</td>
      <td>e35de6a36d385711a660c72c0286154a</td>
      <td>(Medan, Bali, Jakarta)</td>
      <td>...</td>
      <td>3</td>
      <td>14.751176</td>
      <td>7.294337</td>
      <td>0.230271</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.037074</td>
      <td>0.070228</td>
      <td>0.062465</td>
      <td>0.066370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36ef956ac3ef963c48e67327a4b6cc78</td>
      <td>1771011e3adec5db9f30d15b3d439711</td>
      <td>7.774436</td>
      <td>M</td>
      <td>round</td>
      <td>14.030312</td>
      <td>NO</td>
      <td>1</td>
      <td>ad5bef60d81ea077018f4d50b813153a</td>
      <td>(Jakarta, Medan, Bali)</td>
      <td>...</td>
      <td>3</td>
      <td>14.475678</td>
      <td>14.030312</td>
      <td>0.250296</td>
      <td>0.054116</td>
      <td>0.065113</td>
      <td>0.037074</td>
      <td>0.049269</td>
      <td>0.074769</td>
      <td>0.050261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f7821289404d44db50eb2edd4f82ea5b</td>
      <td>6fc1b7d590c2a8c539ce56397403194d</td>
      <td>6.357842</td>
      <td>F</td>
      <td>trip</td>
      <td>14.500656</td>
      <td>YES</td>
      <td>2</td>
      <td>33199710eb822fbcfd0dc793f4788d30</td>
      <td>(Jakarta, Bali, Medan, Jogjakarta, Semarang)</td>
      <td>...</td>
      <td>5</td>
      <td>14.431748</td>
      <td>7.250328</td>
      <td>0.218045</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.070228</td>
      <td>0.046377</td>
      <td>0.062168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f62f33d1de5aabc919b69b1b5697f27a</td>
      <td>c1f4712f60cd758e773555690d148764</td>
      <td>6.760415</td>
      <td>F</td>
      <td>trip</td>
      <td>14.910993</td>
      <td>YES</td>
      <td>1</td>
      <td>74c5549aa99d55280a896ea50068a211</td>
      <td>(Bali, Jakarta, Medan)</td>
      <td>...</td>
      <td>3</td>
      <td>14.933932</td>
      <td>14.910993</td>
      <td>0.205061</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.049269</td>
      <td>0.054228</td>
      <td>0.058189</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>


### Drop Some Features

Proses terakhir dalam *feature engineering* kami adalah menghapus fitur-fitur category karena telah digantikan dengan counterpart mereka yang telah di encoding.


```python
def drop_ever(df):
    df.drop('gender',axis=1,inplace=True)
    df.drop('trip',axis=1,inplace=True)
    df.drop('is_tx_promo',axis=1,inplace=True)
    df.drop('no_of_seats',axis=1,inplace=True)
    df.drop('airlines_name',axis=1,inplace=True)
    df.drop('visited_city',axis=1,inplace=True)
    df.drop(['account_id','order_id'],axis=1,inplace=True)

drop_ever(df_train)
drop_ever(df_test)
drop_ever(df_train_sc)
drop_ever(df_test_sc)
```

Hasil dari proses-proses sebelumnya pada `df_train` dan `df_test`:


```python
display(df_train.head())
display(df_train.shape)
display(df_test.head())
display(df_test.shape)
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
      <th>member_duration_days</th>
      <th>price_x</th>
      <th>is_cross_sell</th>
      <th>max_log_transaction</th>
      <th>min_log_transaction</th>
      <th>mean_log_transaction</th>
      <th>len_log_transaction</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>...</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>gender_enc</th>
      <th>trip_enc</th>
      <th>promo_enc</th>
      <th>seats_enc</th>
      <th>air_enc</th>
      <th>visited_city_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.340359</td>
      <td>13.694358</td>
      <td>False</td>
      <td>14.388884</td>
      <td>13.208915</td>
      <td>13.846001</td>
      <td>1.609438</td>
      <td>13.024431</td>
      <td>0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>4</td>
      <td>14.435097</td>
      <td>13.694358</td>
      <td>0.253840</td>
      <td>0.054131</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.046482</td>
      <td>0.060278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.410175</td>
      <td>14.576201</td>
      <td>False</td>
      <td>16.743283</td>
      <td>13.227583</td>
      <td>14.788710</td>
      <td>6.991177</td>
      <td>14.780214</td>
      <td>0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>3</td>
      <td>14.640493</td>
      <td>7.288100</td>
      <td>1.090637</td>
      <td>0.054131</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.070097</td>
      <td>0.059034</td>
      <td>0.050315</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.475433</td>
      <td>14.807113</td>
      <td>False</td>
      <td>15.807344</td>
      <td>14.474771</td>
      <td>15.317246</td>
      <td>1.791759</td>
      <td>14.626451</td>
      <td>0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>4</td>
      <td>14.640493</td>
      <td>14.807113</td>
      <td>0.276701</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059034</td>
      <td>0.060278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.037871</td>
      <td>13.952369</td>
      <td>False</td>
      <td>15.472525</td>
      <td>13.952369</td>
      <td>14.898969</td>
      <td>1.386294</td>
      <td>14.350099</td>
      <td>0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>4</td>
      <td>14.640493</td>
      <td>13.952369</td>
      <td>0.229600</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059034</td>
      <td>0.057819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.287859</td>
      <td>13.938642</td>
      <td>False</td>
      <td>16.422925</td>
      <td>13.765942</td>
      <td>15.288487</td>
      <td>5.062595</td>
      <td>14.916300</td>
      <td>10</td>
      <td>1.0</td>
      <td>...</td>
      <td>5</td>
      <td>14.531102</td>
      <td>13.938642</td>
      <td>0.805138</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059379</td>
      <td>0.062092</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



    (117946, 23)



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
      <th>member_duration_days</th>
      <th>price_x</th>
      <th>max_log_transaction</th>
      <th>min_log_transaction</th>
      <th>mean_log_transaction</th>
      <th>len_log_transaction</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>...</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>gender_enc</th>
      <th>trip_enc</th>
      <th>promo_enc</th>
      <th>seats_enc</th>
      <th>air_enc</th>
      <th>visited_city_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.470224</td>
      <td>14.960816</td>
      <td>15.886192</td>
      <td>12.410176</td>
      <td>14.963379</td>
      <td>1.945910</td>
      <td>14.909172</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>3</td>
      <td>14.751176</td>
      <td>4.986939</td>
      <td>0.260489</td>
      <td>0.054116</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.074429</td>
      <td>0.062465</td>
      <td>0.058189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.989335</td>
      <td>14.588673</td>
      <td>16.109695</td>
      <td>14.588673</td>
      <td>15.614947</td>
      <td>1.609438</td>
      <td>14.881040</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>3</td>
      <td>14.751176</td>
      <td>7.294337</td>
      <td>0.230271</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.037074</td>
      <td>0.070228</td>
      <td>0.062465</td>
      <td>0.066370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.774436</td>
      <td>14.030312</td>
      <td>15.864208</td>
      <td>12.427091</td>
      <td>14.936979</td>
      <td>1.945910</td>
      <td>14.707013</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>3</td>
      <td>14.475678</td>
      <td>14.030312</td>
      <td>0.250296</td>
      <td>0.054116</td>
      <td>0.065113</td>
      <td>0.037074</td>
      <td>0.049269</td>
      <td>0.074769</td>
      <td>0.050261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.357842</td>
      <td>14.500656</td>
      <td>15.629852</td>
      <td>10.984279</td>
      <td>14.818595</td>
      <td>1.386294</td>
      <td>14.746227</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>5</td>
      <td>14.431748</td>
      <td>7.250328</td>
      <td>0.218045</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.070228</td>
      <td>0.046377</td>
      <td>0.062168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.760415</td>
      <td>14.910993</td>
      <td>16.082854</td>
      <td>14.910993</td>
      <td>15.664584</td>
      <td>1.386294</td>
      <td>14.816352</td>
      <td>0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>3</td>
      <td>14.933932</td>
      <td>14.910993</td>
      <td>0.205061</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.049269</td>
      <td>0.054228</td>
      <td>0.058189</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



    (10000, 22)



```python
display(df_train_sc.head())
display(df_train_sc.shape)
display(df_test_sc.head())
display(df_test_sc.shape)
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
      <th>member_duration_days</th>
      <th>price_x</th>
      <th>max_log_transaction</th>
      <th>min_log_transaction</th>
      <th>mean_log_transaction</th>
      <th>len_log_transaction</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>...</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>is_cross_sell</th>
      <th>gender_enc</th>
      <th>trip_enc</th>
      <th>promo_enc</th>
      <th>seats_enc</th>
      <th>air_enc</th>
      <th>visited_city_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.661036</td>
      <td>-1.387447</td>
      <td>-3.013092</td>
      <td>-0.701810</td>
      <td>-3.180068</td>
      <td>-0.310166</td>
      <td>-2.309567</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>-0.868480</td>
      <td>0.651444</td>
      <td>-0.223712</td>
      <td>False</td>
      <td>0.054131</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.046482</td>
      <td>0.060278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.534578</td>
      <td>0.015859</td>
      <td>2.040763</td>
      <td>-0.681693</td>
      <td>-0.850595</td>
      <td>4.832445</td>
      <td>0.410979</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>0.491785</td>
      <td>-0.917180</td>
      <td>4.937157</td>
      <td>False</td>
      <td>0.054131</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.070097</td>
      <td>0.059034</td>
      <td>0.050315</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.416376</td>
      <td>0.383319</td>
      <td>0.031715</td>
      <td>0.662335</td>
      <td>0.455440</td>
      <td>-0.135946</td>
      <td>0.172726</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>0.491785</td>
      <td>0.923911</td>
      <td>-0.082719</td>
      <td>False</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059034</td>
      <td>0.060278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.208935</td>
      <td>-0.976865</td>
      <td>-0.686993</td>
      <td>0.099370</td>
      <td>-0.578140</td>
      <td>-0.523395</td>
      <td>-0.255474</td>
      <td>-0.119540</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>0.491785</td>
      <td>0.714620</td>
      <td>-0.373212</td>
      <td>False</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059034</td>
      <td>0.057819</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.756131</td>
      <td>-0.998709</td>
      <td>1.353096</td>
      <td>-0.101532</td>
      <td>0.384374</td>
      <td>2.989556</td>
      <td>0.621841</td>
      <td>2.448484</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>-0.232671</td>
      <td>0.711258</td>
      <td>3.176367</td>
      <td>False</td>
      <td>0.060507</td>
      <td>0.055719</td>
      <td>0.037163</td>
      <td>0.049303</td>
      <td>0.059379</td>
      <td>0.062092</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



    (117946, 23)



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
      <th>member_duration_days</th>
      <th>price_x</th>
      <th>max_log_transaction</th>
      <th>min_log_transaction</th>
      <th>mean_log_transaction</th>
      <th>len_log_transaction</th>
      <th>sum_log_transaction</th>
      <th>count_use_hotel</th>
      <th>mode_facilty_hotel</th>
      <th>mode_rate_hotel</th>
      <th>...</th>
      <th>how_many_city</th>
      <th>mean_price_air</th>
      <th>price_per_seat</th>
      <th>order_rate</th>
      <th>gender_enc</th>
      <th>trip_enc</th>
      <th>promo_enc</th>
      <th>seats_enc</th>
      <th>air_enc</th>
      <th>visited_city_enc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.385497</td>
      <td>0.627911</td>
      <td>0.200968</td>
      <td>-1.562569</td>
      <td>-0.418982</td>
      <td>0.011355</td>
      <td>0.610797</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>-0.661594</td>
      <td>1.224800</td>
      <td>-1.480638</td>
      <td>-0.182707</td>
      <td>0.054116</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.074429</td>
      <td>0.062465</td>
      <td>0.058189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.514459</td>
      <td>0.035708</td>
      <td>0.680730</td>
      <td>0.785081</td>
      <td>1.191071</td>
      <td>-0.310166</td>
      <td>0.567207</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>-0.661594</td>
      <td>1.224800</td>
      <td>-0.915653</td>
      <td>-0.369076</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.037074</td>
      <td>0.070228</td>
      <td>0.062465</td>
      <td>0.066370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.936517</td>
      <td>-0.852832</td>
      <td>0.153778</td>
      <td>-1.544341</td>
      <td>-0.484216</td>
      <td>0.011355</td>
      <td>0.297556</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>-0.661594</td>
      <td>-0.599725</td>
      <td>0.733705</td>
      <td>-0.245571</td>
      <td>0.054116</td>
      <td>0.065113</td>
      <td>0.037074</td>
      <td>0.049269</td>
      <td>0.074769</td>
      <td>0.050261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.629369</td>
      <td>-0.104357</td>
      <td>-0.349281</td>
      <td>-3.099183</td>
      <td>-0.776748</td>
      <td>-0.523395</td>
      <td>0.358317</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>1.748048</td>
      <td>-0.890659</td>
      <td>-0.926429</td>
      <td>-0.444477</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.070228</td>
      <td>0.046377</td>
      <td>0.062168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.099814</td>
      <td>0.548626</td>
      <td>0.623114</td>
      <td>1.132428</td>
      <td>1.313726</td>
      <td>-0.523395</td>
      <td>0.466974</td>
      <td>-0.11954</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>...</td>
      <td>-0.661594</td>
      <td>2.435134</td>
      <td>0.949346</td>
      <td>-0.524556</td>
      <td>0.060525</td>
      <td>0.055715</td>
      <td>0.084272</td>
      <td>0.049269</td>
      <td>0.054228</td>
      <td>0.058189</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



    (10000, 22)


## Modelling

Untuk *modelling*, kami menggunakan beberapa algoritma klasik *machine learning* hingga beberapa teknik ensemble seperti *voting* dan *blending* untuk didapatkan nilai evaluasi yang terbaik. Selain itu, kami juga memproses data sehingga data seimbang antara transaksi yang terjadi *cross sell* dan tidak. Untuk tahap awal, kami menggunakan beberapa model dan menginisiasinya.


```python
# Inisiasi Semua Model
xgb_1 = XGBClassifier(n_estimators=100)
lgb_1 = LGBMClassifier(n_estimators=100)
et_1 = ExtraTreesClassifier(n_estimators=100)
rf_1 = RandomForestClassifier(n_estimators=100)
cb_1 = CatBoostClassifier(n_estimators=100, silent=True,loss_function='Logloss')
ada_1 = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=100))

models = {'xgb':xgb_1, 'lgb':lgb_1, 'cb':cb_1, 'et':et_1, 'rf':rf_1, 'ada':ada_1}
```

### Cross Validation

Agar hasil evaluasi menjadi lebih presisi, kami menggunakan *cross validation* untuk mengevaluasi model yang akan digunakan untuk memprediksi *cross sell*, adapun K yang digunakan adalah K=5 (ilustrasi dapat dilihat pada gambar dibawah). Evaluasi model dilakukan menggunakan F1-score sesuai dengan ketentuan.

![](https://miro.medium.com/max/3018/1*IjKy-Zc9zVOHFzMw2GXaQw.png)

Juga karena tujuan kami menggunakan teknik ensemble maka kami menggunakan Cross Validasi disini untuk menentukan model-model mana saja yang akan digunakan sebagai base estimator.


```python
kf = KFold(n_splits=5)

def cross_val(model, sampling):
    imba_pipeline = make_pipeline(sampling, model)
    res = cross_val_score(imba_pipeline, X_scaled.values, y_scaled.astype(int), scoring='f1', cv=kf)
    return res
```


```python
# Dataset untuk modelling yang tidak discale dan yang discale
X, y = df_train.drop(['is_cross_sell'],axis=1), df_train['is_cross_sell'].astype(bool)
X_scaled, y_scaled = df_train_sc.drop(['is_cross_sell'],axis=1), df_train_sc['is_cross_sell'].astype(bool)
```

Kami menggunakan *oversampling* dengan teknik SMOTE untuk mengatasi data yang *imbalance* kemudian mengevaluasi masing-masing model yang telah didefinisikan sebelumnya.

Ilustrasi teknik SMOTE sebagai berikut:

![](https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/smote.png)



```python
for model in models:
    res = cross_val(models[model],SMOTE(random_state=42))
    print(f'{model} : {np.mean(res)} +- {np.std(res)}')
```

    xgb : 0.8040062971727762 +- 0.00976663371514552
    lgb : 0.8531939808533864 +- 0.011945998072817885
    cb : 0.8462804413768813 +- 0.012424279811237702
    et : 0.8419926875957617 +- 0.009441331931789896
    rf : 0.8482162835667142 +- 0.009658381496305128
    ada : 0.8401772047144351 +- 0.00928256212150458
    

### Blending

> *Blending* memiliki pendekatan yang sama dengan *stacking*, akan tetapi untuk melakukan prediksi hanya menggunakan data validasi. Sebagai contoh, jika digunakan model A dan model B untuk melakukan prediksi, hal yang pertama dilakukan adalah memprediksi data validasi dan data *test* menggunakan kedua model tersebut. Kemudian hasil prediksi masing-masing model digunakan sebagai fitur baru pada data validasi dan data *test*. Sehingga model baru C akan menggunakan data validasi dan data *test* yang sudah terdapat fitur terbaru tersebut untuk difit pada model C dan kemudian memprediksi data *test*. Dibawah ini merupakan ilustrasi dari *blending*.

![](http://i.imgur.com/QBuDOjs.jpg)

Pada proses *blending* yang kami lakukan juga digunakan *cross validation* agar mengetahui performansi model dengan komprehensif.

### Arsitektur

Dari hasil diatas kami menggunakan 4 model terbaik yaitu Catboost, Lightgbm, Random Forest dan Extra Tree untuk menjadi base estimator untuk model blending.

![test](https://docs.google.com/uc?export=download&id=19ZrkfmIos-YOckCC401Bft_6_Oi7R3L3)


```python
n_folds = 10
verbose = True
shuffle = False

sm = SMOTE(random_state=42)
X_scaled_resampled, y_sc_resampled = sm.fit_sample(X_scaled,y_scaled.astype(int))

skf = StratifiedKFold(n_folds)

## Inisialisasi model untuk level 1
clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        LGBMClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]

print("Creating train and test sets for blending.")

## Inisialisasi dataset penampung hasil predict dari model level 1 untuk dijadikan fitur di model level 2
dataset_blend_train = np.zeros((X_scaled_resampled.shape[0], len(clfs)))
dataset_blend_test = np.zeros((df_test_sc.shape[0], len(clfs)))

## Untuk setiap model di level 1
for j, clf in enumerate(clfs):
    print(j, clf)
    dataset_blend_test_j = np.zeros((df_test_sc.shape[0], n_folds))
    i=0
    ## Untuk setiap fold model tersebut akan difit dengan fold tersebut dan dipakai untuk mempredict df_test_sc
    for train, test in skf.split(X_scaled_resampled,y_sc_resampled):
        print("Fold", i+1)
        X_train = X_scaled_resampled[train]
        y_train = y_sc_resampled[train]
        X_test = X_scaled_resampled[test]
        y_test = y_sc_resampled[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict(X_test)
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict(df_test_sc)
        i+=1
    ## untuk setiap model kami mengambil rata-rata dari hasil prediksi di setiap fold
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

## Level 2
print()
print("Blending.")
clf = CatBoostClassifier(silent=True) ## inisialisasi model Level 2
clf.fit(dataset_blend_train, y_sc_resampled.astype(int)) ##  Fit hasil dataset dari level 1 diatas ke level 2
y_submission = clf.predict(dataset_blend_test) ## predict menggunakan classifier level 2
```

    Creating train and test sets for blending.
    0 RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                           warm_start=False)
    Fold 1
    Fold 2
    Fold 3
    Fold 4
    Fold 5
    Fold 6
    Fold 7
    Fold 8
    Fold 9
    Fold 10
    1 RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                           warm_start=False)
    Fold 1
    Fold 2
    Fold 3
    Fold 4
    Fold 5
    Fold 6
    Fold 7
    Fold 8
    Fold 9
    Fold 10
    2 ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                         oob_score=False, random_state=None, verbose=0,
                         warm_start=False)
    Fold 1
    Fold 2
    Fold 3
    Fold 4
    Fold 5
    Fold 6
    Fold 7
    Fold 8
    Fold 9
    Fold 10
    3 ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                         oob_score=False, random_state=None, verbose=0,
                         warm_start=False)
    Fold 1
    Fold 2
    Fold 3
    Fold 4
    Fold 5
    Fold 6
    Fold 7
    Fold 8
    Fold 9
    Fold 10
    4 LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                   importance_type='split', learning_rate=0.05, max_depth=6,
                   min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                   n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                   random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                   subsample=0.5, subsample_for_bin=200000, subsample_freq=0)
    Fold 1
    Fold 2
    Fold 3
    Fold 4
    Fold 5
    Fold 6
    Fold 7
    Fold 8
    Fold 9
    Fold 10
    
    Blending.
    

## Submission

Setelah model yang dibuat menghasilkan evaluasi yang baik, kami submit file prediksi.


```python
sample['is_cross_sell'] = y_submission ## ganti value dengan hasil prediksi
sample['is_cross_sell'] = np.where(sample['is_cross_sell']==1.0, 'yes', 'no') ## menyamakan format submisi
```


```python
sample['is_cross_sell'].value_counts() ## Sanity check
```




    no     8852
    yes    1148
    Name: is_cross_sell, dtype: int64




```python
sample.to_csv('submission_blending_newfeat_fikhri.csv',index=False) ## Smpan file
```

---
Sour Soup &copy; 2019
