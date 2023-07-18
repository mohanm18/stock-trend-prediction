# Stock Trend Prediction
- A stock trend prediction web app made using Python.
- Data is fetched using _yfinance package_, that allows us to fetch financial data for various companies from Yahoo Finance.
- And further graph is plotted displaying Original and Predicted values.
- Used _Streamlit_, an open-source Python library, that makes it easy to build beautiful custom web apps for Machine Learning and Data Science.

## Working
- Input the name of the company whose stocks needs to be predicted and hit enter.
- Further we'll get 4 graphs depicting
  1. Closing Price vs Time Chart
  2. Closing Price vs Time Chart with 100ma (100 days moving average)
  3. Closing Price vs Time Chart with 100ma & 200ma(200 days moving average)
  4. Predicted vs Original
 
## Video demonstrating the working of the project. 
  [DemoVideo](https://youtu.be/dHHKJOmddxg)

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- streamlit
- yfinance (yfinance is a package that allows you to fetch financial data for various companies from Yahoo Finance.)
- keras
- pandas_datareader

## Installation of Packages
- pip install numpy
- pip install pandas
- pip install matplotlib
- pip install streamlit
- pip install yfinance
- pip install keras
- pip install pandas_datareader

## Setting up the project
Steps for running this project on local host: 

- Initially download the zip file or clone the repo.
  Once the whole source code has been downloaded, open the command prompt within the project's directory and type
  
  ```bash
  streamlit run main.py
  ```
  and hit enter. The project will be ready on the localhost 3000(which is the default port number in most cases), if the browser is not opened by default,then paste the url given below in the browsers tab.<p>
  **_NOTE: Check the port number on which the project is ready and mention that specific port number._**</p>
  ```bash
  http://localhost:3000/
  ```
  
  
 
  
  
   

   


