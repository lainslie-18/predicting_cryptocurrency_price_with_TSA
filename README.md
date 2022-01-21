# Using Time Series Analysis To Predict Cryptocurrency Price Changes

## Table of Contents

1. [Project Planning / About The Project](link)

2. [Acquisition](link)

3. [Preparation](link)

4. [Exploration](link)

5. [Modeling](link)

6. [Delivery / Conclusion](link)

## Project Planning / About The Project

### Project Goal

* To use time series analysis to analyze cryptocurrency prices and develop a model that can predict changes in prices.

### Project Description

Cryptocurrency is a massive and growing industry that provides incredible opportunity for profit but also comes with risk of loss due to its volatility. 
With models that can identify trends and predict upcoming price changes, these risks can be mitigated.

The cryptocurrency used in this analysis are HOLO (HOT1-USD) and NEM (XEM-USD). Both have a little over $1B market capitalization at the time of this report. The close price is what I will be trying to predict.

### Initial Hypotheses/Questions

* What trends can we see in the data?

* Does cryptocurrency have a seasonality component?

* Are there any months we can identify as having significantly higher or lower closing prices? 

* Are there any days of the week we can identify as having significantly higher or lower closing prices? 


## Acquisition

* Use YFinance to import data (may have to pip install yfinance)
* Save data to csv
* Create function for pulling in and caching the data
* Create wrangle.py to save these functions for importing
* Test functions
* Get familiar with data
* Document takeaways & things to address during cleaning 

### Data dictionary

|   Column_Name   | Description | Type      |
|   -----------   | ----------- | ---------- |
| col |  desc | float |
| col   |  desc | int64  |
| col      |  desc   | int64 |
| col      | desc | object |
| col      |  desc| int64 |

## Preparation

* Create function that cleans data
  * Handle missing values
  * Convert data types
  * Rename columns 
* Create function that splits data into train, validate, and test samples
  * Split 20% (test), 24% (validate), and 56% (train)
* Test functions
* Add functions to wrangle.py to save for importing

## Exploration 

* Ask questions/form hypotheses
  * What trends can we see in the data?
  * Does cryptocurrency have a seasonality component?
  * Are there any months we can identify as having significantly higher or lower closing prices? 
  * Are there any days of the week we can identify as having significantly higher or lower closing prices?
* Create visualizations to help identify patterns
* Use statistical tests to test hypotheses
* Document answers to questions and takeaways

## Modeling

* Establish a baseline
* Build, fit and use models to make predictions
* Compute evaluation metrics to evaluate models' performance
* Select best model and use on test dataset

## Delivery / Conclusion

### Recommendation
* Recommendation

### Next Steps
 * With more time and resources ...

### To Recreate This Project:
* You will need to download the data from Yahoo Finance
* Download the wrangle.py file to your working directory
* Download the ??? notebook to your working directory
* Read this README.md
* Run the ??? notebook
