Training a model based on a kaggle dataset which contains a lot of categorical data which needs to be preprocessed

**Key Takeaways:**
- Preprocessing data:
    - Drop product name because there are 618 unique names and one-hot-encoding them is no bueno
    - Create new features for "Company" and "TypeName" and set values as 1 or 0 (1 if its that company and 0 for everything else) using the <u>get_dummies</u> function from pandas
        - Don't want to ordinal encode them because their values are not based on a scale (eg: if 2 numbers are closer to each other, the model might think they are more similar than if the numbers are further apart)
    - Extract only the width and height from the ScreenResolution column and turn them into new features
    - Extract only the brand name and the frequency from the Cpu column and turn them into new features
    - Type cast the values to int/float (tip: use df.hist() to check because it only displays numerical values)
- Using seaborn and matplotlib to generate a heatmap for the correlation between colunmns and price
- Use df.corr() to find the most relevant features and select the most relevant 20
- Using random forest regressor model because it is optimal for structured data
    - First split the data from limited_df (one with the most relevant features)
    - Then use StandardScaler() to scale X_train and X_test
- Plotting the prediction to visualize how accurate the model is (the closer the points are to the line the more accurate it is)

**Result:** score of 0.8312756168295186