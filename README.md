# TDS Project: Price Forecasting for Airbnb Listings in New York City

This project aims to build a regression model to predict the price of Airbnb apartments in New York City using a variety of factors such as neighborhood, room type, and availability. The project is part of a university assignment in the Tabular Data Science (TDS) course.

## **Dataset**
The dataset used for this project is the New York City Airbnb Open Data available on Kaggle. The data includes various attributes for each listing, such as:

* **Price:** The price of the apartment per night
* **Number of Reviews:** The number of reviews the listing has received
* **Room Type:** Type of room (e.g., Entire home/apt, Private room, Shared room)
* **Neighbourhood Group:** The neighborhood the apartment is located in
* **Minimum Nights:** The minimum number of nights required to book the apartment
* **Availability:** Number of available days in the year

Data Source:
[New York City Airbnb Open Data on Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

Problem Description
The goal of this project is to predict the price of an apartment based on various features, such as the number of reviews, minimum nights, room type, availability, and the neighborhood group. By building a regression model, we can forecast apartment prices and understand the factors that influence them.

Key Steps:
1. Dataset Selection: Choosing the relevant dataset and performing basic data preprocessing.
2. Exploratory Data Analysis (EDA): Visualizing and understanding the nature of the data.
3. Modeling: Implementing a regression model (XGBoost) to predict the price of an apartment.
4. Error Analysis: Evaluating the model and identifying its weaknesses.


Requirements:
* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* jupyter notebook
