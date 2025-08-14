# Titanic Decision Tree Prediction - Clean Python Script

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/kaggle/input/titanic-decision-tree/titanic.csv')

# Encode 'Sex'
labeled = LabelEncoder()
df['Sex'] = labeled.fit_transform(df['Sex'])

# Fill missing 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Predictions on test set
pred = model.predict(X_test)

# Print accuracy
print("Accuracy on test set:", accuracy_score(y_test, pred))

# Cross-validation score
cv_score = cross_val_score(model, X, y, cv=5)
print("Cross-validation mean score:", cv_score.mean())

# --- Interactive prediction ---
try:
    user_pclass = int(input("Enter Pclass (1/2/3): "))
    user_sex = input("Enter Sex (male/female): ").strip().lower()
    user_age = input("Enter Age (or leave blank for median): ").strip()
    user_fare = float(input("Enter Fare: "))

    # Handle missing age input
    if user_age == "":
        user_age = df['Age'].median()
    else:
        user_age = float(user_age)

    # Encode sex
    user_sex_enc = labeled.transform([user_sex])[0]

    # Prepare input DataFrame
    user_features = pd.DataFrame([[user_pclass, user_sex_enc, user_age, user_fare]],
                                 columns=['Pclass', 'Sex', 'Age', 'Fare'])

    # Predict
    prediction = model.predict(user_features)
    print("Predicted:", "Survived" if prediction[0] == 1 else "Not Survived")

except Exception as e:
    print("Error:", e)
