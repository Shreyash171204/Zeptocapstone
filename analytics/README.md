
# Analytics Pipeline

## Dataset
The Titanic dataset was loaded once using `sns.load_dataset("titanic")`.
Immediately after loading, the raw DataFrame was saved to `titanic.csv`
as the offline fallback.

## Missing Values

Missing percentages were measured before cleaning.

age            19.865320
embarked        0.224467
deck           77.216611
embark_town     0.224467

Cleaning decisions:
- `age` was median-imputed because its missing percentage falls between 5% and 30%.
- `embarked` and `embark_town` were handled by dropping rows because their missing percentages are below 5%.
- `deck` was dropped because its missing percentage is too high for reliable imputation.

## Outliers and Fare Distribution

Age and fare outliers were identified using the IQR rule.

Fare:
- Mean = 32.10
- Median = 14.45
- Mode = 8.05

Fare is right-skewed.

## Data Story

1. **Survival by sex:** The survival-rate chart shows a substantial difference
   between female and male passengers. This indicates that sex was strongly
   associated with survival in this dataset.

2. **Survival by passenger class:** Survival rates differ across passenger
   classes. Passenger class therefore provides useful information about
   survival outcomes.

3. **Age by passenger class:** The box plot shows differences in the age
   distributions across passenger classes, helping explain how passenger
   characteristics varied between classes.

4. **Age, fare and survival:** The scatter plot combines age, fare and survival.
   It shows that survival was related to passenger characteristics and fare,
   although the relationship is not perfectly separated.

5. **Correlation heatmap:** The required six numeric columns were examined.
   The two strongest absolute correlations were:
   - pclass vs fare = -0.5482
   - sibsp vs parch = 0.4145

## Standardization

Age and fare were standardized using:

`z = (x - mean) / standard deviation`

The resulting means are approximately zero and standard deviations are
approximately one. This was an exploratory check only and was not used as
the modeling pipeline's preprocessing.

## Charts

The `charts` directory contains the generated EDA charts.
