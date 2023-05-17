# Country-wise Mental Disorders forecaster using machine learning
This project aims to develop a machine learning model for yearly mental disorder forecasting. The model predicts the prevalence of various mental disorders based on the country and year provided as input. The predictions can help identify potential mental health trends and guide resource allocation for mental health support.
## Installation

1. Clone the repository from GitHub:

```bash
  git clone https://github.com/Omkar-Rajkumar-Khade/Mental_disorder_prediction.git
```


2. Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## How to Use
To get started with this project, follow these steps:
* Clone this repository
* Install the required dependencies using pip: pip install -r requirements.txt
* Start the streamlit server: streamlit run app.py
* Access the application in your browser at http://localhost:8501.

### Folder Structure 

`dataset` folder contains the engineering.csv dataset used in the project.

`Mental disorder prediction.ipynb` notebook that was used to develop and train the model.

`app.py` is the streamlit application file that defines the API endpoints and loads the saved model.

`model.pkl` is file that contain the serialized machine learning model that is used for prediction.

`README.md` is the project documentation file.

`requirements.txt` lists the Python dependencies required to run the project.

### Usage

1. Enter the country and year in the input fields provided.
2. Click the "Predict" button to obtain predictions for mental disorder prevalence.
3. The predictions will be displayed in a table, showing the prevalence for each mental disorder category.

### Prerequisites

To run the project, you must have the following installed on your system:

* Python 3.6+
* streamlit
* Pandas
* Scikit-learn
* matplotlib

## Machine Learning Model Training

Data Preprocessing: The first step in this process is to gather and preprocess the data. The data used in this project is from a dataset containing countrywise mental disorder prevalence rates from 1990 to 2016. The data is then grouped by country and year, and the mean values of the mental disorder columns are calculated for each group. This step helps to aggregate the data and make it suitable for training the model.

Feature and Target Split: The features (X) and target (y) are separated from the preprocessed data. The features include the country and year columns, while the target consists of the mental disorder columns. This split allows us to train the model using the features and predict the target values.

Train-Test Split: The dataset is split into training and testing sets using the train_test_split function from the scikit-learn library. This division allows us to assess the performance of the trained model on unseen data.The resulting dataset is split into training and testing sets, with 80% of the data used for training and 20% used for testing

Pipeline Creation: A pipeline is created to streamline the training process. The pipeline consists of a column transformer, which applies one-hot encoding to the country column, and a regression model for prediction. In this project, three different regression models are used: Random Forest, Decision Tree, and Linear Regression.

Model Training: The pipeline is fitted to the training data using the fit method. This step involves transforming the features using the column transformer and training the regression model on the transformed data.

Model Evaluation: The trained models are evaluated on the testing data using two metrics: R-squared score and Root Mean Squared Error (RMSE). The R-squared score measures the goodness of fit of the model, while RMSE provides an estimation of the prediction error.The R-squared score measures how well the model fits the data, with a score of 1 indicating a perfect fit and a score of 0 indicating no correlation

Results: The R-squared scores and RMSE values are printed to assess the performance of each model. Additionally, line and bar graphs are generated to visualize the R-squared scores and RMSE scores for comparison between the different regression models.

## Results

The Results section of this project provides an overview of the performance of the trained machine learning models for yearly mental disorder forecasting. Three different regression models were evaluated: Random Forest Regression, Decision Tree Regression, and Linear Regression.

Random Forest Regression:

R-squared score: 0.9996331772476117
R2 Score: 0.9996331772476117
RMSE: 0.011423657211499086
The Random Forest Regression model achieved an outstanding R-squared score of 0.9996331772476117. This indicates that the model explains approximately 99.96% of the variance in the target variable, capturing the patterns and trends in the data exceptionally well. The high R2 Score further confirms the model's performance. Additionally, the Root Mean Squared Error (RMSE) of 0.011423657211499086 suggests that the average difference between the predicted and actual values is very low, indicating the model's accuracy.

Decision Tree Regression:

R-squared score: 0.9993736878978943
R2 Score: 0.9993736878978943
RMSE: 0.014737179665872322
The Decision Tree Regression model also performed remarkably well, with an R-squared score of 0.9993736878978943, indicating an excellent fit to the data. The R2 Score confirms this performance level. The RMSE value of 0.014737179665872322 suggests a slightly higher average difference between the predicted and actual values compared to the Random Forest Regression model, but it still reflects a good level of accuracy.

Linear Regression:

R-squared score: 0.9869736705560125
R2 Score: 0.9869736705560125
RMSE: 0.06222799210678069
The Linear Regression model achieved a respectable R-squared score of 0.9869736705560125. This indicates that the model explains approximately 98.70% of the variance in the target variable, capturing a significant portion of the patterns in the data. The R2 Score confirms this performance. The RMSE value of 0.06222799210678069 suggests a higher average difference between the predicted and actual values compared to the other models. However, it still provides valuable insights into mental disorder prevalence forecasting.

## Model Integration

The Model Integration part of this project focuses on integrating the trained machine learning model into a Streamlit application for easy interaction and prediction of mental disorder prevalence.

The Streamlit app begins by loading the saved model from the 'model.pkl' file using the pickle library. The model is then stored in the 'model' variable for later use.

Next, a list of disorder names is defined, representing the different mental disorders the model predicts. This list will be used to label the columns in the prediction output.

The Streamlit application is titled "Mental Disorder Forecaster" using the st.title() function.

For user input, a dropdown select box is provided for choosing the country from a list of country options. The options are stored in the country_options list, which contains the names of various countries.

The user can also input the year using a number input field, with a default value of 2022. The minimum and maximum values are set to 1900 and 2100, respectively, to ensure a valid year range.

When the user clicks the "Predict" button, the selected country and year are used to create a feature input for prediction. Any necessary preprocessing, such as scaling or encoding, can be applied to the features at this stage.

The loaded model is then used to make predictions based on the input features. The predictions are stored in the 'predictions' variable.

A dataframe is created using the predictions, with the disorder names as column headers. This dataframe, called 'predictions_df', organizes the predictions in a tabular format.

Finally, the predictions are displayed in a table using the st.table() function, providing an easy-to-read output of the predicted mental disorder prevalence for each disorder category.

By integrating the model into a Streamlit application, users can easily input their desired country and year to obtain predictions for mental disorder prevalence. This user-friendly interface enhances the accessibility and usability of the machine learning model for forecasting mental disorders.

## Conclusion
In conclusion, this project aimed to develop a machine learning model for yearly mental disorder forecasting. The model was trained using a dataset containing information about various mental disorders across different countries and years.

Several steps were undertaken during the project. The dataset was loaded and preprocessed by grouping the data by country and year and taking the mean of the disorder rates for each group. The data was then split into training and testing sets.

Three different regression models, namely Random Forest Regression, Decision Tree Regression, and Linear Regression, were trained using a pipeline and column transformer to handle categorical features. Each model was evaluated using the R-squared score and Root Mean Squared Error (RMSE) on the testing data.

The results of the model evaluation showed that all three models performed exceptionally well in predicting the prevalence of mental disorders. The Random Forest Regression and Decision Tree Regression models achieved high R-squared scores close to 1, indicating a strong correlation between the predicted and actual values. Additionally, the RMSE scores were relatively low, indicating small errors in the predictions.

Comparing the models, the Random Forest Regression model exhibited the highest R-squared score and the lowest RMSE, suggesting that it performed the best among the three models for this particular task.

The integration of the trained model into a Streamlit web application provided a user-friendly interface for users to interact with the model. Users can input the country and year of interest and obtain predictions for the prevalence rates of different mental disorders.

Overall, this project demonstrates the effectiveness of machine learning models in forecasting mental disorder prevalence. The high accuracy and low error rates observed in the models' predictions highlight their potential in assisting with mental health trend analysis and resource allocation.