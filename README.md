# ML_on_Forex_USDJPY
### Study on dependence structure for forex exchange rate on USD/JPY by machine learning

This is a Machine-Learning study based on a traditional aspect that many fundamental analysis traders consider there is a dependence structure between oil price, gold price, Nikkei225 index on forex-exchange rate of USD/JPY. Japan yen as this investigation as it is a major oil-consuming and gold-holding country.Furthermore, Japanese yen is a major currency, bound test was widely used on by many traders in this topic.

### Dataset
The dataset - daily prices of the variables were obtained from investing.com & YAHOO finance.
<br>
Including the period from 9-Apr-1990 to 8-May-2020. (Prices of USD/JPY, gold price, WTI oil price and nikkei225 index)

### Data-preprocessing
The data obtained from the sources are clean. Except the products are traded in different markets different countries.
<br>
They have recorded the price only the dates being traded. However, the closing prices are remain unchange til the next trading day.
<br>
Thus,<br>
```df = df.fillna(method='bfill')```
<br>
I filled the N/A with the closing price in the previous date.

### EDA and data visualization
The daily closing prices for (from top left to bottom right) exchange rate for USD/JPY, gold price, WTI oil price and Nikkei225 index price.<br>
![Daily prices for variables](/images/0_Daily_prices.png)
<br>
![Daily prices comparison](/images/1_Prices_compare.png)
<br>
The % of changes in daily closing prices of gold price, WTI oil price and Nikkei225 index price
![% change of variables](/images/2_Percentage_changes_of_prices.png)
<br><br>
Drop the outliers of over 1% of price drop due to the impact of Covid-19.
<br>
```oil_outlier = df[df['oil_WTI_%']<-1].index```
<br>
```df = df.drop(oil_outlier, axis=0)```
![% change of variables after remove outliers](/images/3_Percentage_changes_of_prices(wo_outliers).png)
<br><br>
### Import TA-lib for financial indicators
[TA-lib](https://github.com/mrjbq7/ta-lib) for downloading TA-lib.
I used this Cpython based library for adding RSI, Moving average(MA) 5-day & 20-day as additional features for the dataset.
RSI and MA are common indicators that technical analysis users will be used for their reference on judging the entry or retreat point in buying financial instruments.
<br>
### Correlation
Furthermore, we need to investigate the correlation between each features.
![Heatmap_1](/images/4_heatmap_1.png)
Obviouly, the MA indicators are showing very strong collinearity to their corresponding products.
<br>
For the multicollinearity in the dataset, the prediction result may be affected by the features drastically.
<br>
Let's see how the results in this situation, but we need to further process the data before the prediction.
<br>
### Min-Max Scale the dataset
I used the MinMaxScaler from sklearn for scales and translates each feature individually between 0 and 1.
<br>
### Split the train and test sets of data in a time-series
```
x_train = x.loc['1990-05-04':'2015-05-03']
x_test = x.loc['2015-05-04':'2020-05-08']
y_train = y[:6491]
y_test = y[6491:]
```
The dataset was splited into train set and test set for training.
<br>
### Prediction result (with multicollinearity)
The following algorithms were used, and the results showed may be __Over-fitting__.
<br>
1. DecisionTree
   * explained_variance_score is 0.96043
   * mean_squared_error is 0.0018153
   * r2_score is 0.95622
1. RandomForest
   * explained_variance_score is 0.96534
   * mean_squared_error is 0.0015877
   * r2_score is 0.96170
1. XGboost
   * explained_variance_score is 0.98651
   * mean_squared_error is 0.0006251
   * r2_score is 0.98492

### Prediciton result (w/o multicollinearity)
Our objective is to find the dependence structure on the determination of exchange rate of USD/JPY.
<br>
Thus, we shall remove the collineared features - the MA lines from the dataset.
```
df = df.drop('usd/jpy_ema5', axis = 1)
df = df.drop('usd/jpy_ema20', axis = 1)
```
And the revised heatmap:
![Heatmap_2](/images/5_heatmap_2.png)
And the results are as followings:
1. DecisionTree
   * explained_variance_score is -0.40221
   * mean_squared_error is 0.060731
   * r2_score is -0.46466
1. RandomForest
   * explained_variance_score is -0.055485
   * mean_squared_error is 0.043972
   * r2_score is -0.060481
1. XGboost
   * explained_variance_score is 0.15647
   * mean_squared_error is 0.035267
   * r2_score is 0.14945
<br>
<p>The results for both DecisionTree and RandomForest gives <b>-ve</b> explained variance score and r2 score, which indicates that either the model do not capture any'real' underlying dependence or the data is fitting in an intercept term. In a simpler term, the prediction results are worst than taking a <b>mean</b> of predictor as a result.</p>
<br>
Therefore, is this mean there is no dependence structure between them?

### Prediction result (remove nikkei225 index)
Since the nikkei225 index maybe affected by traders investing decisions. Thus, i excluded it as a feature for the model. And try the algorithms.
<br>
With only oil price and gold price left, the results are as follows:
<br>
1. DecisionTree
   * explained_variance_score is 0.35602
   * mean_squared_error is 0.039068
   * r2_score is 0.057792
1. RandomForest
   * explained_variance_score is 0.39430
   * mean_squared_error is 0.038243
   * r2_score is 0.077691
1. XGboost
   * explained_variance_score is 0.44150
   * mean_squared_error is 0.031158
   * r2_score is 0.24856

Although the result not yet accurate, it shows there is a dependence structure between oil price, gold price and exchange rate of USD/JPY at some certain level.

### Improvements (To-be-continued)

1. Using minute-basis or hourly-basis data to implement real-time relationship may improve the results;
1. Explore more financial indicators since traders may depends on them for decision-making;
1. Explore more similar products that could possibly has a dependent relationship with oil and gold.
<br>

### Limitation:
1. The time used in this project was just 12 hours, more aspects can be considered if more time allowed.
1. The impact of COVID-19 on oil price starts on early months of 2020, making Japanese yen as safe haven of year, which having the following effects:
    1. Conditional dependence structure failed to exist, sharp decline on oil price has no effect on the inflation of japanese yen;
    1. Extreme low oil price was unexpectly having fewer impact on forex exchange level due to air-traffic lock-down.
1. Gold price surged at the late of 2019 due to the conflicts between China and US market, and after that COVID-19 making the two countries negotiation stopped and more tense atmosphere was expected.
1. The products are trade in different countries and markets. the time-series data would perform better in a real-time basis than daily basis.

#

### REFERENCE
1. Dynamic relationships between the price of oil, gold and financial variables in Japan: a bounds testing approach Le, Thai-Ha and Chang, Youngho Nanyang Technological University ,19 August 2011, Online at https://mpra.ub.uni-muenchen.de/33030/ MPRA Paper No. 33030, posted 28 Aug 2011 15:32 UTC;
1. On the conditional dependence structure between oil, gold and USD exchange rates: Nested copula based GJR-GARCH model Rihab Bedouia, b, Sana Braieka, Khaled Guesmi c, Julien Chevallier Article history: Received 10 April 2018, Received in revised form 14 November 2018, Available online 21 February 2019;
1. On the study of conditional dependence structure between oil, gold and USD exchange rates, Authors: Rihab Bedoui, Sana Braeik, StÃ©phane Goutte, Khaled Guesmi <br> https://www.sciencedirect.com/science/article/abs/pii/S1057521918302369;
