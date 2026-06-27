import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

df = pd.read_csv(url, header=None, names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])

print("Shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDiabetic vs Not:")
print(df['Outcome'].value_counts())

print("\nBasic statistics:")
print(df.describe())
# Replace fake zeros with column average
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, df[cols_with_zeros].mean())

print("\nAfter cleaning - min values:")
print(df[cols_with_zeros].min())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data into inputs and output
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test it
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
import matplotlib.pyplot as plt

# Show which features matter most
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='steelblue')
plt.xlabel('Importance')
plt.title('Which factors predict diabetes most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nChart saved as feature_importance.png")