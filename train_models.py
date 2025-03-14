import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("SD.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Train SLR Model (YearsExperience -> Salary)
slr_model = LinearRegression()
slr_model.fit(df[['YearsExperience']], df['Salary'])
pickle.dump(slr_model, open("slr_model.pkl", "wb"))

# Train MLR Model (Experience, Education, Job Role, Location -> Salary)
le_edu = LabelEncoder()
le_job = LabelEncoder()
le_loc = LabelEncoder()

df['Education Level'] = le_edu.fit_transform(df['Education Level'])
df['Job Role'] = le_job.fit_transform(df['Job Role'])
df['Location'] = le_loc.fit_transform(df['Location'])

X_mlr = df[['YearsExperience', 'Education Level', 'Job Role', 'Location']]
y_mlr = df['Salary']

mlr_model = LinearRegression()
mlr_model.fit(X_mlr, y_mlr)

# Save MLR Model and Encoders
pickle.dump(mlr_model, open("mlr_model.pkl", "wb"))
pickle.dump((le_edu, le_job, le_loc), open("encoders.pkl", "wb"))

print("Models trained and saved successfully!")
