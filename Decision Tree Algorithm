  import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score

   file_path = '/content/drug200.csv'
   df = pd.read_csv(file_path)


   le_sex = LabelEncoder()
   le_bp = LabelEncoder()
   le_cholesterol = LabelEncoder()
   le_drug = LabelEncoder()

  df['Sex'] = le_sex.fit_transform(df['Sex'])
  df['BP'] = le_bp.fit_transform(df['BP'])
  df['Cholesterol'] = le_cholesterol.fit_transform(df['Cholesterol'])
  df['Drug'] = le_drug.fit_transform(df['Drug'])

  #Separate features (X) and target (y)
  X = df.drop('Drug', axis=1)
  y = df['Drug']

  #Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #Create and train the Decision Tree classifier
  clf = DecisionTreeClassifier()
  clf.fit(X_train, y_train)

  #Make predictions on the test set
  y_pred = clf.predict(X_test)

  #Convert the numerical predictions back to drug names
  y_pred_names = le_drug.inverse_transform(y_pred)
  y_test_names = le_drug.inverse_transform(y_test)

  #Calculate accuracy
  accuracy = accuracy_score(y_test_names, y_pred_names)
  print("Accuracy: {:.2f}%".format(accuracy * 100))

 #Assuming you have the new data in a dictionary format, you can convert it to a DataFrame
 new_data = {
    'Age': [30, 50],
    'Sex': ['F', 'M'],
    'BP': ['NORMAL', 'LOW'],
    'Cholesterol': ['NORMAL', 'HIGH'],
    'Na_to_K': [10.0, 8.5]
 }

 new_df = pd.DataFrame(new_data)

 #Preprocess the new data (convert categorical features to numerical using the same LabelEncoders)
 new_df['Sex'] = le_sex.transform(new_df['Sex'])
 new_df['BP'] = le_bp.transform(new_df['BP'])
 new_df['Cholesterol'] = le_cholesterol.transform(new_df['Cholesterol'])

 #Make predictions on the new data using the trained Decision Tree classifier
 new_predictions = clf.predict(new_df)

 #Convert the numerical predictions back to drug names
 new_predictions_names = le_drug.inverse_transform(new_predictions)

 #Print the results
 print("New Data Predictions:")
 for i, pred in enumerate(new_predictions_names):
  print(f"Data Point {i+1} - Predicted Drug: {pred}")
