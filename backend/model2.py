from sklearn.preprocessing import LabelEncoder
import pandas as pd


df = pd.read_csv("./depression_data.csv")
print(df.head())
print(df.describe())

df.drop("Name", axis=1, inplace=True)
print(df.describe())

df.rename(
    columns={
        "Name": "name",
        "Age": "age",
        "Marital Status": "marital_status",
        "Education Level": "education",
        "Smoking Status": "smoking",
        "Employment Status": "employment",
        "History of Mental Illness": "mental_illness",
        "Physical Activity Level": "physical_activity",
        "History of Substance Abuse": "substance_abuse",
        "Alcohol Consumption": "alcohol",
        "Dietary Habits": "diet",
        "Sleep Patterns": "sleep",
        "Family History of Depression": "family_depression",
        "Chronic Medical Conditions": "chronic_conditions",
    },
    inplace=True,
)

le_employment = LabelEncoder()
df["employment"] = le_employment.fit(df["employment"])

le_smoking = LabelEncoder()
df["smoking"] = le_smoking.fit(df["smoking"])

le_marriage = LabelEncoder()
df["marital_status"] = le_marriage.fit(df["marital_status"])

le_education = LabelEncoder()
df["education"] = le_education.fit(df["education"])

le_physical = LabelEncoder()
df["physical_activity"] = le_education.fit(df["physical_activity"])

le_alcohol = LabelEncoder()
df["alcohol"] = le_alcohol.fit(df["alcohol"])

le_diet = LabelEncoder()
df["diet"] = le_diet.fit(df["diet"])

le_sleep = LabelEncoder()
df["sleep"] = le_sleep.fit(df["sleep"])

le_ilness = LabelEncoder()
df["mental_illness"] = le_ilness.fit(df["mental_illness"])

le_substance = LabelEncoder()
df["substance_abuse"] = le_substance.fit(df["substance_abuse"])

le_depression = LabelEncoder()
df["family_depression"] = le_depression.fit(df["family_depression"])

le_medical = LabelEncoder()
df["chronic_conditions"] = le_medical.fit(df["chronic_conditions"])
