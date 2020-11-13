# Project Premise
The project explores the extent to which future stock prices can be predicted using supervised machine learning models with fundamental and technical analysis data as inputs. 

The project aims to determine which supervised machine learning model, from Time Series Multi-Linear Regression (TS-MLR), Recurrent Neural Networks (RNN), to Long Short-Term Memory (LSTM), can predict future stock prices with the lowest Root Mean Square Error (RMSE). In doing so, we conducted dimensionality reduction as well as feature selection, providing an insight into the categories of fundamental and technical analysis data that are particularly significant in predicting future stock prices. This insight may be integrated into stock-picking strategies and provide a benchmark on the ideal timing to buy or sell stocks.

The project singles out LSTM as the best performing machine learning model, with an average RMSE of 8.03 in predicting the closing price one month ahead, and an average RMSE of 13.45 in predicting the closing price six months ahead.

# Motivations

Investing in the stock market tends to be the most volatile type of investment. As such, our project explores one of the ways to minimize such volatilities - analyzing company data to uncover possible trends in price changes of stocks. In doing so, our project hopes that these trends will be able to help increase certainty for investors. Idealistically, the best (least RMSE) model will allow investors to profit from investments and “beat the market”. 

The significance of our project is two-fold. Firstly, it provides insight about the more significant factors that affect the prices of stocks, which in turn allows investors to narrow down the scope of research and analysis. Secondly, it provides evidence demonstrating the presence of market inefficiencies, which is one that does not incorporate all available information into a true reflection of an asset’s fair price successfully.   

The problem statement our project intends to solve entails the question of the extent to which it is possible to observe patterns or relationships in stock price changes. On a micro level, we aim to determine which predictor(s) would serve as the most relevant ones in determining stock price changes as well as the model(s) that can enable investors to best minimize risks involved.

# Findings

The LSTM model performed better than the traditional RNN and time series MLR in predicting stocks due to the ability of LSTM to memorize time related information. In contrast, time series MLR does not retain time related information and RNN has a short- term memory problem due to vanishing gradients. However, in predicting prices of stocks 6 times ahead, the RMSE was always higher for all models due to the lack of training data used in training the 6 months model. Since we require companies to have 180 priors data as one point, we have less data points for each company, resulting in more companies being filtered out.


# Summary
This repository contains the source code used for model training and testing. It focuses on the <b>RNN model</b>, a commonly used machine learning model that uses stored internal state to process input layers.

# Libraries
* matplotlib, numpy, pandas, sklearn, keras, tensorflow

