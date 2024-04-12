PROBLEM:  
The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection.  
Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer .  
PROCEDURE:  
1)Data Understanding: Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null.  
2)Data Preprocessing:   
Handle missing values with mean/median/mode.  
Treat Outliers using IQR or Isolation Forest from sklearn library.  
Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation.  
Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding.  
3)EDA: Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot.  
4)Model Building and Evaluation:  
Split the dataset into training and testing/validation sets.  
Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve.   
Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.  
Interpret the model results and assess its performance based on the defined problem statement. Same steps for Regression modelling. 
5)STREAMLIT:  
In streamlit webpage, enter the input values to get predicted selling price and the status of product.  
LIBRARIES:  
Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.



