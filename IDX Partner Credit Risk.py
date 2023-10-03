#!/usr/bin/env python
# coding: utf-8

# ## Business Understanding
# Membuat model untuk prediksi loan_status/Loan_ending
# 
# Kami ingin memprediksi apakah suatu pinjaman berisiko atau tidak, jadi, perlu mengetahui akhir dari setiap pinjaman secara historis, apakah itu defaulted / charged off, or fully paid. Seperti yang bisa kita lihat, ada nilai-nilai seperti "Saat ini(Current)", "Dalam Masa Tenggang"("In Grace Period") yang ambigu. Pengakhiran pinjaman tersebut dapat dilunasi atau dilunasi, jadi kami tidak dapat menggunakan status tersebut. Terlambat("Late") juga agak ambigu, tetapi saya pribadi tidak ingin berinvestasi dalam pinjaman yang terlambat, jadi saya akan mengklasifikasikannya sebagai pinjaman macet.
# 

# 
#     

# ## Data Dictionary

# Column | Description
# :---|:---
# `id` | A unique LC assigned ID for the loan listing.
# `member_id ` | A unique LC assigned Id for the borrower member.
# `loan_amnt` | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# `funded_amnt` | The total amount committed to that loan at that point in time.
# `funded_amnt_inv` | The total amount committed to that loan at that point in time.
# `term` | The number of payments on the loan. Values are in months and can be either 36 or 60.
# `int_rate` | Interest Rate on the loan
# `installment` | The monthly payment owed by the borrower if the loan originates.
# `grade` | LC assigned loan grade
# `sub_grade` | LC assigned loan subgrade
# `emp_title` | The job title supplied by the Borrower when applying for the loan.
# `emp_length` | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
# `home_ownership` | The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
# `annual_inc` | The self-reported annual income provided by the borrower during registration.
# `verification_status` | Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
# `issue_d` | Last month payment was received
# `loan_status` | Loan payment status
# `pymnt_plan` | Loan payment plan
# `url` | URL for the LC page with listing data.
# `desc` | Loan description provided by the borrower
# `purpose` | A category provided by the borrower for the loan request. 
# `title` | The loan title provided by the borrower
# `zip_code` | The first 3 numbers of the zip code provided by the borrower in the loan application.
# `addr_state` | The state provided by the borrower in the loan application
# `dti` | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
# `delinq_2yrs` | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
# `earliest_cr_line` | The date the borrower's earliest reported credit line was opened
# `inq_last_6mths` | The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
# `mths_since_last_delinq` | The number of months since the borrower's last delinquency.
# `mths_since_last_record` | The number of months since the last public record.
# `open_acc` | Number of open trades
# `pub_rec` | Number of derogatory public records
# `revol_bal` | Total credit revolving balance
# `revol_util` | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
# `total_acc` | The total number of credit lines currently in the borrower's credit file
# `initial_list_status` | wholeloan platform expiration date
# `out_prncp` | 
# `out_prncp_inv` |
# `total_pymnt` |
# `total_pymnt_inv` |
# `total_rec_prncp` |
# `total_rec_int` |
# `total_rec_late_fee` |
# `recoveries` |
# `collection_recovery_fee` |
# `last_pymnt_d` | Last month payment was received 
# `last_pymnt_amnt` | Last total payment amount received
# `next_pymnt_d` | Next scheduled payment date
# `last_credit_pull_d` | The most recent month LC pulled credit for this loan
# `collections_12_mths_ex_med` | Number of collections in 12 months excluding medical collections
# `mths_since_last_major_derog` | Months since most recent 90-day or worse rating
# `policy_code` | publicly available policy_code=1; new products not publicly available policy_code=2
# `application_type` | Indicates whether the loan is an individual application or a joint application with two co-borrowers
# `annual_inc_joint` | The combined self-reported annual income provided by the co-borrowers during registration
# `dti_joint` | A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income
# `verification_status_joint` | Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
# `acc_now_delinq` | The number of accounts on which the borrower is now delinquent.
# `tot_coll_amt` | Total collection amounts ever owed
# `tot_cur_bal` | Total current balance of all accounts
# `open_acc_6m` | Number of open trades in last 6 months
# `open_il_6m` | Number of currently active installment trades
# `open_il_12m` | Number of installment accounts opened in past 12 months
# `open_il_24m` | Number of installment accounts opened in past 24 months
# `mths_since_rcnt_il` | Months since most recent installment accounts opened
# `total_bal_il` | Total current balance of all installment accounts
# `il_util` | Ratio of total current balance to high credit/credit limit on all install acct
# `open_rv_12m` | Number of revolving trades opened in past 12 months
# `open_rv_24m` | Number of revolving trades opened in past 24 months
# `max_bal_bc` | Maximum current balance owed on all revolving accounts
# `all_util` | 
# `total_rev_hi_lim` | Total revolving high credit/credit limit
# `inq_fi` | Number of personal finance inquiries
# `total_cu_tl` | Number of finance trades
# `inq_last_12m` | Number of credit inquiries in past 12 months

# ## Import Package

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


df = pd.read_csv("loan_data_2007_2014.csv")


# In[3]:


df.head()


# In[4]:


print('jumlah baris dari dataset ini %d dan jumlah kolomnya %d'%df.shape)


# In[5]:


df.columns


# In[6]:


df.info()


# Observation:
# - In columns `annual_inc_joint`, `dti_joint`, `verification_status_joint`, `open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `,`_bal _b `,`_bal total_cu_tl`, `inq_last_12m` all rows are null, will be dropped first
# - In columns `desc`, `mths_since_last_delinq`, `mths_since_last_record`, `next_payment_d`, and `mths_since_last_major_derog` more than half of the rows have null values, may be imputed during preprocessing
# - Columns `emp_title`, `emp_length`, `tot_coll_amt`, `tot_cur_bal`, and `total_rev_hi_lim` have not too many null values, will be imputed during pre-processing
# - In `title`, `earliest_cr_line`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_util`, `total_acc`, `last_pymnt_d`, `pulllasd_credit`, `null_credit` columns dropped during pre processing

# # Handling Missing Value and Duplicates

# In[7]:


#checing the missing values(%)
def chek_missing(df):
    sum_NAN = df.isnull().sum().reset_index()
    sum_NAN.columns = ['Columns', 'NaN_count']
    sum_NAN["Percentage"] = sum_NAN.NaN_count/len(df)*100
    return sum_NAN.sort_values("Percentage", ascending =  False)
chek_missing(df)


# In[8]:


missing = chek_missing(df)


# In[117]:


missing.head(28).groupby(["Columns"]).NaN_count.sum().sort_values(ascending=True).plot.bar()
plt.title("Amount of Missing Value")
plt.show()


# + Terdapat beberapa kolom yang memilki nilai missing value 100% sehingga akan dilakukan drop

# In[10]:


drop_cols=['mths_since_last_major_derog','mths_since_last_record','annual_inc_joint', 'dti_joint',
       'verification_status_joint', 'acc_now_delinq', 
       'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
           'inq_fi', 'total_cu_tl', 'inq_last_12m','desc','mths_since_last_delinq','next_pymnt_d','id', 'Unnamed: 0']
df2 = df.drop(drop_cols, axis=1)


# In[11]:


#checing the missing values(%)
def chek_missing(df):
    sum_NAN = df.isnull().sum().reset_index()
    sum_NAN.columns = ['Columns', 'NaN_count']
    sum_NAN["Percentage"] = sum_NAN.NaN_count/len(df)*100
    return sum_NAN.sort_values("Percentage", ascending =  False)
missing2 = chek_missing(df2)
missing2.head(15)


# In[120]:


missing2.head(20).groupby(["Columns"]).NaN_count.mean().sort_values(ascending=True).plot.bar()
plt.title("Amount of Missing Value")
plt.show()


# # Cleaning/Feature Engineering

# In[13]:


modus = df2['emp_title'].mode()[0]
df2['emp_title'] = df2['emp_title'].fillna(modus)
modus2 = df2['last_pymnt_d'].mode()[0]
df2['last_pymnt_d'] = df2['last_pymnt_d'].fillna(modus2)


# In[14]:


df2.emp_length.unique


# In[15]:


df2['emp_length'] = df2['emp_length'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)


# In[16]:


rata = df2['emp_length'].mode()[0]
df2['emp_length'] = df2['emp_length'].fillna(rata)
df2.emp_length


# In[17]:


df3 = df2.dropna()


# In[18]:


df3.isna().sum()


# Sudah tidak terdapat null value

# In[19]:


print(f'Duplicates in dataframe:{df3.iloc[:,1:].duplicated().sum()}, ({np.round(100*df3.iloc[:,1:].duplicated().sum()/len(df3),1)}%)')


# In[20]:


print('Jumlah baris dari dataset sekarang %d dan jumlah kolomnya %d'%df3.shape)


# In[21]:


df3.head()


# In[122]:


def viz(df,types):
    num = df
    f = pd.melt(num, value_vars=num)
    g = sns.FacetGrid(f, col="variable",  col_wrap=3, 
                      sharex=False, sharey=False, size = 5)
    g = g.map(types, "value")
    plt.show()
    return (g)
import warnings
warnings.filterwarnings('ignore')
#plots outliers
viz(df3[['loan_amnt', 'funded_amnt','funded_amnt_inv', 'annual_inc', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int','total_rec_late_fee']], sns.boxplot)


# # Defining Variable 

# In[23]:


df.loan_status.unique()


# In[24]:


#define values
ambiguous = ['Current', 'In Grace Period']
good =  ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']

#drop rows that contain ambiguous ending
df3 = df3[df3.loan_status.isin(ambiguous) == False]

#create new column to classify ending
df3['loan_ending'] = np.where(df3['loan_status'].isin(good), 'good', 'bad')


# In[25]:


# check the balance
plt.title('good vs risky loans balance')
sns.barplot(x=df3.loan_ending.value_counts().index,y=df3.loan_ending.value_counts().values)


# # EDA

# In[26]:


df3['earliest_cr_line'].head() #supposed to be date


# In[27]:


df3['earliest_cr_line'] = pd.to_datetime(df3['earliest_cr_line'], format='%b-%y')


# In[28]:


df3['earliest_cr_line'].head()


# In[29]:


(pd.to_datetime('2017-12-01') - df3['earliest_cr_line'])


# ubah menjadi bulan

# In[30]:


df3['mths_since_earliest_cr_line'] = round((pd.to_numeric((pd.to_datetime('2017-12-01') - df3['earliest_cr_line'])/np.timedelta64(1, 'M'))))


# In[31]:


df3['mths_since_earliest_cr_line']


# In[32]:


df3['term'].unique()


# In[33]:


df3['term'] = df3['term'].str.replace(' ', '')
df3['term'] = df3['term'].str.replace('months', '')


# In[34]:


df3['term'] = pd.to_numeric(df3['term'])


# In[35]:


df3['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%y')


# In[36]:


df3.issue_date.head()


# In[37]:


df3['mths_since_issue_date'] = round((pd.to_numeric((pd.to_datetime('2017-01-01')-df3['issue_date'])/np.timedelta64(1, 'M'))))


# In[38]:


feature = df3[['emp_length','loan_ending','mths_since_earliest_cr_line','issue_date','term']]
feature


# In[39]:


sns.countplot(df3.verification_status,hue=df3.grade.sort_values())
plt.title('Grade Count base on Verification Status ')
plt.show()


# berdasarkan keterangan grafik diatas dapat terlihat bahwa masih banyak grade B yang masih terbilang bagus namum tidak dapat diverifikasi Lending Club

# In[40]:


#Home own vs Loan Amount
df3.groupby(["home_ownership","loan_ending"]).loan_amnt.mean().sort_values(ascending=False).unstack().plot.barh()
plt.title('Home Ownership - loan Amount', loc='center',pad=30, fontsize=20, color='black')
plt.xlabel('Loan Amount', fontsize=15)
plt.ylabel('Type of Home Ownership', fontsize = 15)
plt.show()


# dapat dilihat berdasarkan grafik jumlah peminjam terbesar ada pada prang-orang yang memiliki hipotek.

# In[41]:


plt.clf()
df3.groupby(['issue_date','loan_status'])['loan_amnt'].sum().unstack().plot(cmap='Set1')
plt.title('Loan Amount in year  - Breakdown by Loan Status', loc='center',pad=30, fontsize=20, color='black')
plt.xlabel('Order Month', fontsize=15)
plt.ylabel('Total Amount (in Billions)', fontsize = 15)
plt.grid(color='darkgray', linestyle=':', linewidth = 0.5)
plt.ylim(ymin=0)
labels, locations = plt.yticks()
plt.yticks(labels, (labels/10000000).astype(int))
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=3, title='Loan Status', fontsize=9, title_fontsize=11)
plt.gcf().set_size_inches(10, 5)
plt.tight_layout()
plt.show()

dapat dilihat pada grafik diatat jumlah pinjaman tertinggi dengan status pembayaran penuh terjadi kenaikan dan tiba-tiba jatuh pada akhir tahun 2014
# In[42]:


states = {"AL":"Alabama", "AK":"Alaska", "AZ":"Arizona", "AR":"Arkansas", "CA":"California", "CO":"Colorado", "CT":"Connecticut", 
          "DC":"Washington DC", "DE":"Delaware", "FL":"Florida", "GA":"Georgia", "HI":"Hawaii", "ID":"Idaho", "IL":"Illinois", 
          "IN":"Indiana", "IA":"Iowa", "KS":"Kansas", "KY":"Kentucky", "LA":"Louisiana", "ME":"Maine", "MD":"Maryland",
          "MA":"Massachusetts", "MI":"Michigan", "MN":"Minnesota", "MS":"Mississippi", "MO":"Missouri", "MT":"Montana",
          "NE":"Nebraska", "NV":"Nevada", "NH":"New Hampshire", "NJ":"New Jersey", "NM":"New Mexico", "NY":"New York", 
          "NC":"North Carolina", "ND":"North Dakota", "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon", "PA":"Pennsylvania", 
          "RI":"Rhode Island", "SC":"South Carolina", "SD":"South Dakota", "TN":"Tennessee", "TX":"Texas", "UT":"Utah", "VT":"Vermont",
          "VA":"Virginia", "WA":"Washington", "WV":"West Virginia","WI":"Wisconsin", "WY":"Wyoming"}
 
df3["States_long"] = df3.addr_state.map(states)


# In[43]:


df3.States_long.unique()


# In[44]:


df3.verification_status.unique()


# In[45]:


us_data = df3[df3["verification_status"]=='Verified']['addr_state'].value_counts().reset_index()
us_data.columns=['State','No of people']
us_data.head()


# In[112]:


import plotly.express as px
fig = px.choropleth(us_data,
                    locations='State', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='No of people',
                    color_continuous_scale="Viridis_r", 
                    title="Peta Sebaran Lokasi Pinjaman dengan Status Terverifikasi ",labels=True,
                    hover_name= 'No of people'
                    )
fig.show()


# Jumlah peminjam terbanyak yang telah terverifikasi ada di CA=California 11.679K

# # Preprocessing

# # Variabel yang agak tidak jelas

# kolom menarik lainnya adalah tot_coll_amt, tot_cur_bal, total_rev_hi_lim. Mereka adalah tiga kolom terakhir, dan memiliki jumlah nilai bukan nol yang sama. Saya berasumsi bahwa itu adalah fitur baru dalam beberapa waktu antara 2007 - 2014, begitu banyak nilai yang masih nol, terutama yang lebih lama. Deskripsi kolom-kolom ini juga tidak jelas, jadi saya agak tidak yakin apakah datanya mengandung kebocoran atau tidak. Pertama-tama kita akan mencoba menjelajahi kolom-kolom ini lebih lanjut, sehingga kita dapat memutuskan apa yang harus dilakukan.

# In[47]:


cols = ['tot_coll_amt', 'total_rev_hi_lim']

# pivot table aggregated by mean
print(pd.pivot_table(df3, index = 'loan_ending', values = cols))

# pivot table aggregated by max value
print(pd.pivot_table(df3, index = 'loan_ending', values = cols, aggfunc = np.max))


# In[48]:


# berdasarkan tabel pivot, tot_coll_amt agak mencurigakan, mari kita periksa
df3[cols].describe()


# In[49]:


plt.figure(figsize=(10,6))

# Saya menggunakan "> 0" karena 75% datanya adalah 0 ... jadi plot di bawah ini hanya menggunakan < 25% dari datanya
sns.kdeplot(data = df3[(df3['tot_coll_amt'] < 100000) & (df3['tot_coll_amt'] > 0)], x='tot_coll_amt', hue='loan_ending')


# In[50]:


plt.figure(figsize=(10,6))
sns.kdeplot(data=df3[df3['total_rev_hi_lim'] < 250000], x='total_rev_hi_lim', hue='loan_ending')


# In[51]:


drop_col = ['tot_coll_amt', 'total_rev_hi_lim', 'States_long', 'zip_code', 'last_pymnt_d','last_credit_pull_d']
df3.drop(drop_col, inplace=True, axis = 1)


# # variabel dengan kategori sedikit

# In[52]:


# Filtering data with less than 8 unique values
df3.nunique()[df3.nunique() < 8].sort_values()


# In[53]:


drop_col = ['policy_code', 'application_type']
df3.drop(drop_col, inplace=True, axis = 1)


# Untuk data dengan nilai unik yang kecil, kita dapat menelusurinya secara visual dengan menggunakan bad loan untuk setiap kategori.

# In[54]:


print(df3.nunique()[df3.nunique() < 8].sort_values().index)


# In[55]:


def risk_pct_chart(x):
    ratio = (df3.groupby(x)['loan_ending'] # group by
         .value_counts(normalize=True) # calculate the ratio
         .mul(100) # multiply by 100 to be percent
         .rename('risky_pct') # rename column as percent
         .reset_index())

    sns.lineplot(data=ratio[ratio['loan_ending'] == 'bad'], x=x, y='risky_pct')
    plt.title(x)
    plt.show()


# In[56]:


small_unique = ['term', 'pymnt_plan', 'initial_list_status',
       'verification_status', 'home_ownership', 'grade',
       'collections_12_mths_ex_med']

for cols in small_unique:
    risk_pct_chart(cols)


# kesimpulan kolom dengan perubahan signifikan rasio 'good' vs 'bad' antara lain:
# 'term'
# 'pymnt_plan'
# 'initial_list_status'
# 'grade
# 
# kolom dengan sedikit perubahan rasio antara lain:
# 
# home ownership
# verification status
# initial_list_status
# collections_12_mths_ex_med
# Tapi semuanya bagus, dan bisa dipertahankan, karena setidaknya berkontribusi sesuatu, baik itu kecil atau besar.

# In[57]:


#buang data yang masih tidak dibutuhkan
drop_col = ['emp_title', 'url', 'title', 'member_id', 'issue_d','issue_date', 'earliest_cr_line' ]
df3.drop(drop_col, inplace=True, axis = 1)


# # Numerical vs categorical

# In[58]:


# numerical
num_data = df3.select_dtypes(exclude= 'object')
num_data.columns


# In[59]:


# categorical
cat_data = df3.select_dtypes(include= 'object')
cat_data.columns


# Yang Akan kita lakukan selanjutnya adalah untuk 
# 
# #numerikal data
# 
# 1. Melihat distribusi dengan membuat histogram
# 
# 2. Melihat Korrelasi
# 
# 3. Kemudian membuat pivot terhadap variabel target
# 
# #kategorikal data
# 
# 1. Melihat balance
# 
# 2. Membuat pivot

# # Numerical data

# In[60]:


# 1. distribution
for i in num_data.columns:
    plt.hist(num_data[i])
    plt.title(i)
    plt.show()


# In[61]:


#2. Pivot
pd.pivot_table(df3, index = 'loan_ending', values = num_data.columns)


# In[62]:


#3. Correlation
plt.figure(figsize=(15,7))
sns.heatmap(data=num_data.corr(), annot=True)


# Di sini, jika ada pasangan fitur-fitur yang memiliki korelasi tinggi maka akan diambil salah satu saja. Nilai korelasi yang dijadikan patokan sebagai korelasi tinggi tidak pasti, umumnya digunakan angka 0.7, namun karena banyaknya data berada diatas angka 0.8 maka dipilih angka 0.8 untuk tidak membuang terlalu banyak feature

# In[63]:


corr_matrix = num_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop_hicorr = [column for column in upper.columns if any(upper[column] > 0.8)]
to_drop_hicorr


# In[64]:


num_data.drop(to_drop_hicorr, axis=1, inplace=True)


# Kesimpulan:
# 
# 1. Hanya sejumlah kecil data numerik yang terdistribusi secara normal
# 2. Beberapa data mengandung outlier
# 3. Seperti yang diharapkan, jumlah angsuran & pinjaman berkorelasi, hampir sempurna. Karena cicilan = jumlah_pinjaman * tingkat_bunga. Meskipun jumlah pinjaman dapat bervariasi, tingkat bunga biasanya tidak terlalu bervariasi.
# 
# Berdasarkan tabel pivot, karakteristik pinjaman berisiko:
# 
# Berdasarkan catatan pribadi yang buruk:
# 
# 1. akun delinq yang lebih tinggi
# 2. kenakalan yang lebih tinggi dalam 2 tahun terakhir
# 3. pertanyaan yang lebih tinggi dalam 6 bulan terakhir -> pertanyaan sulit dapat memengaruhi penilaian kredit
# 4. tahun yang lebih rendah sejak pertanyaan terakhir -> lebih rendah = baru-baru ini memiliki pertanyaan kredit
# 5. Berdasarkan kesulitan pembayaran yang lebih sulit
# 6. Pendapatan tahunan lebih rendah
# 7. rasio utang terhadap pendapatan lebih tinggi (dti) -> dti = angsuran bulanan / pendapatan bulanan
# 8. angsuran & jumlah pinjaman yang lebih tinggi
# 9. tingkat bunga yang lebih tinggi (biasanya berkorelasi dengan peringkat pinjaman)

# # Categorical Data

# In[65]:


to_chart = ['grade', 'sub_grade', 'home_ownership',
       'verification_status', 'loan_status', 'pymnt_plan',
       'purpose', 'addr_state', 'initial_list_status']

for cols in to_chart:
    plt.figure(figsize=(14,4))
    risk_pct_chart(cols)


# Kesimpulan:
# 
# 1. Grade dan subgrade sesuai harapan, semakin rendah grade (A-G), semakin berisiko pinjaman
# 2. Untuk tujuan tertentu, 'mobil', 'pembelian_utama', dan 'pernikahan' memiliki risiko terendah, dan 'usaha_kecil' memiliki risiko tertinggi
# 3. ni juga menarik bagi negara bagian untuk memiliki persentase risiko yang bervariasi.
# 4. Bagaimanapun, kita akan mengubah semua data kategorikal menjadi data numerik, dan karena grade dan subgrade sama, saya hanya akan menghapus subgrade untuk mengurangi jumlah kolom.

# In[66]:


drop_col = ['sub_grade']
cat_data.drop(drop_col, inplace=True, axis = 1)


# In[67]:


df3.info()


# # Inilah yang akan kita lakukan untuk kategorikal data
# 
# 1. Untuk 'grade' kita akan menggunakan ordinal encoder atau map
# 2. dan one hot encoding  untuk:
# 
# home_ownership
# 
# verification status
# 
# purpose
# 
# addr_state
# 
# initial_list_status
# 
# initial_list_status -> tetapi hanya 1 yang cukup jadi kita akan menjatuhkan 1 kolom dummy

# In[68]:


cat_data.columns


# In[69]:


cat_data['grade'].unique()


# In[70]:


# 1. transforming grade
grade_map = {
    'A' : 1,
    'B' : 2,
    'C' : 3,
    'D' : 4,
    'E' : 5,
    'F' : 6,
    'G' : 7,
}

cat_data['grade'] = cat_data['grade'].map(grade_map)


# In[71]:


cat_data.grade.unique()


# In[72]:


# 3. one hot encode?
to_dummies = ['home_ownership', 'verification_status', 'loan_status',
       'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status']

dummies = pd.get_dummies(cat_data[to_dummies])
dummies.drop('initial_list_status_w', axis=1, inplace=True)


# In[73]:


dummies.info()


# In[74]:


dummies.columns


# In[75]:


# buang kolom yang sudah ada OHC
cat_data.drop(to_dummies, axis=1, inplace=True)


# In[76]:


# gabung data dengan data kategorikal
cat_data_final = pd.concat([cat_data, dummies], axis = 1)
cat_data_final.info()


# In[77]:


#gabung data numerik dan kategori
df4 = pd.concat([num_data, cat_data_final], axis = 1).dropna().reset_index().drop('index', axis = 1)
df4.head()


# In[78]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


# In[79]:


df4['loan_ending'] = labelencoder.fit_transform(df4['loan_ending'])


# In[80]:


#pisah variable dependent (y), dependent(X)
X = df4.drop('loan_ending', axis = 1)
y = df4['loan_ending']


# In[81]:


df3.loan_ending.unique()


# In[82]:


df4.loan_ending.unique()


# # Modelling

# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[85]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Random Forest

# In[86]:


from sklearn.ensemble import RandomForestClassifier
# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[87]:


rfc = RandomForestClassifier(max_depth=1)
rfc.fit(X_train, y_train)
pred_y = rfc.predict(X_test)
print(classification_report(y_test, pred_y))


# In[88]:


from sklearn.metrics import accuracy_score
#Evaluate Model Performance
print('Training Accuracy :', rfc.score(X_train, y_train))  
print('Testing Accuracy :', rfc.score(X_test, y_test))


# In[89]:


arr_feature_importances = rfc.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending= False,)
df_all_features.head(10)


# In[90]:


df_all_features.head(10).groupby(["feature"]).importance.mean().sort_values(ascending=True).plot.barh()
plt.show()


# In[91]:


y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index


# In[92]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# # Hyperparameter Turning Using RandomizedSearchCV

# In[93]:


from sklearn.model_selection import RandomizedSearchCV 


# In[94]:


n_estimators = [5,20,50,100] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap}


# In[95]:


rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)


# In[96]:


rfc_random.fit(X_train, y_train)


# In[97]:


print ('Random grid: ', random_grid, '\n')
# print the best parameters
print ('Best Parameters: ', rfc_random.best_params_, ' \n')


# In[98]:


rfbest = [{'n_estimators': 20,
 'min_samples_split': 10,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 10,
 'bootstrap': True}]


# In[99]:


rfbest = pd.DataFrame(rfbest)
rfbest.to_csv('rfbest_param.csv')


# Menggunakan parameter terbaik

# In[100]:


rfc_best = RandomForestClassifier(n_estimators = 20, min_samples_split = 10, min_samples_leaf= 4, max_features = 'sqrt', max_depth= 10, bootstrap=True) 
rfc_best.fit( X_train, y_train) 


# In[101]:


pred_y_best = rfc_best.predict(X_test)
print(classification_report(y_test, pred_y_best))


# In[102]:


from sklearn.metrics import accuracy_score

print('Training Accuracy :', rfc_best.score(X_train, y_train))  
print('Testing Accuracy :', rfc_best.score(X_test, y_test))


# In[103]:


y_pred = rfc_best.predict(X_test)
accuracy = accuracy_score(y_test, pred_y_best)
print('Model Accuracy on Test Data:', accuracy)
confusion_matrix(y_test, pred_y_best)

fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(confusion_matrix(y_true = y_test, y_pred = pred_y_best), fmt = 'g', annot = True)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Prediction', fontsize = 14, color = 'red')
ax.set_xticklabels(['Bad Loan','Good'])
ax.set_ylabel('Actual', fontsize = 14)
ax.set_yticklabels(['Bad Loan','Good'])
plt.show()


# # Kesimpulan 
# Dengan penggunaan Algoritma Random Forest:  
# Hasil yang didapatkan adalah jumlah Loan_ending/portofolia akan terdiri dari 76.3% pinjaman bagus dan 23.6% pinjaman berisiko, dan sebaiknya Anda akan berinvestasi dalam 100% dari pinjaman bagus yang tersedia.
