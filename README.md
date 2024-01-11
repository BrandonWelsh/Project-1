## Program Goals
"For Project 1, you will work with your group to find and analyze a dataset of your choice. For this project, you can focus your efforts within a specific industry." (BootcampSpot Project 1 Overview)

## Data Source
https://www.kaggle.com/datasets/meirnizri/covid19-dataset

This dataset was downloaded to each of our local machines before we began to analyze it.

## Dependencies
Prophet Library, Pandas Library, Datetime Library, Numpy Library, matplotlib.pyplot Library

## Setup Instructions
Run pip install Prophet, Pandas, Datetime, Numpy, and matplotlib.pyplot to acquire the required libraries

## How to use
Run each cell of the individual .ipynb files to generate the visuals used in this project. They should have already been run when uploaded to github and are ready to be viewed.

## Team Member Responsibilities

Brandon Welsh: I set out to determine whether obesity, cardiovascular disease, or tobacco use had any effect on ICU admission, Intubation, or Death as a result of Covid-19. I also wanted to determine if there was any correlation between these variables and the severity of Covid-19 infection.

Omar Hassanein: In the COVID data project, I focused on identifying occurrences of values 97 and 99 in data frame. I conducted analyses on hypertension, renal chronic conditions, medical units used and explored correlations with medical units, symptoms, and age. Additionally, I performed exploratory data analyses (EDAs) including descriptive statistics, missing data visualization, histograms for data distributions, and a correlation heatmap. My role also involved creating visually informative bar charts to highlight key data distributions.

Marnie Brannon:



## Team Member Analysis
This section is dedicated to each member's interpretations of the raw dataset, their own process for cleaning the data, what visuals they chose to pursue, and their interpretation of the data based upon their visuals:

Brandon Welsh: To begin this project, I first had to load in the dataset and clean it. I did this by first dropping any column I didn't have any interest in analyzing (or that I knew another group member would be analyzing). I then had to recode all of the data (as much of it was boolean, labeled with 0 or 1) as "yes" or "no", altering my approach depending on the column I needed to recode (for instance, the date died column listed date died, or 9999-99-99 if they did not die. I was able to make a separate "DEAD" column from this based on true/false logic). While cleaning the dataset, I realized that there was an issue with one of the columns. The CLASIFFICATION_FINAL column, according to the dataset documentation, was labeled as 1 thru 7, with 4 or higher being a negative covid result. However, we had no way of knowing which of Covid-1, Covid-2, or Covid-3 was the most severe. As such, I kept these columns distinct, but did not assign them definite "severity" levels. This was going to affect my analysis of the covid severity, as I had no definite way of knowing which side of the spectrum was the most severe cases. I cannot make assumptions, so I will perform my analysis and write-up with these column names as they are.

Once I had the data cleaned, I had to drop any null values. I ran a short script to count and print the number of null values per column, and found very few overall, with the unfortunate exception of ICU and Intubation. These had nearly 80% of their values as null. Dropping these values would have wiped out 80% of the dataset, and much of my otherwise good data. So my solution was to make two dataframes, one which was cleaned but kept the nulls in those two columns, and one with dropped all null values from the entire dataset. I would only use the latter for visuals which required ICU and Intubation as factors.

I chose to first create a correlation diagram. It would seem that the areas of highest correlation is Age and Death, Intubation and Death, and ICU and Intubation, while the areas of lowest correlation is Classification and Cardiovascular Disease, and Tobacco Use and any other column.

Next, I created an age distribution boxplot, looking at the average age distributions across ICU, Intubation, and Death among Covid-19 patients. I found the average age increased from ICU to Intubation to Death, which is also the same order of severity of each of those patient conditons. As such, I can conclude that there is a correlation between age and Covid-19 severity, which confirms part of what I found from the correlation diagram.

Next, I got into the meat of my analysis. As stated earlier, I wanted to look at Obesity, Cardiovascular Disease, and Tobacco use, and try to determine their effects on covid severity, ICU admission, Hospitalization, and Death. Annoyingly, much of this data was boolean, with a simple 'Yes' or 'No' for each of these variables (with the exception of covid severity). For the severity calculation, using ClASIFFICATION_FINAL column, I separated each of the severity levels (covid-1 thru covid-3) and then compared obesity ratios for each of these. I utilized crosstab within the pandas library to do this comparison, and came up with a series of pie charts. According to my visuals for obesity, 16% of Covid-1 patients were obese. This number jumps to 19.2% for Covid-2, and 23.38% for Covid-3. As a result of these findings, I can conclude that obesity affects covid-19 severity. However, due to the limitations of the dataset, I am unable to confidently determine which Covid level (Covid-1 thru Covid-3) is the most severe, and I cannot make assumptions. For the next visual, I wanted to look at Obesity and its affects on ICU Admission, Intubation, and Death. I chose a stacked bar graph so I could simutaneously view the distribution of whether or not patients were admitted to the ICU, Intubated, or Died of Covid. My findings show that the ratio of obese covid patients within the ICU, Intubated, or Died was slightly higher than those who were not in the ICU, Intubated, or had died. In layman's terms, a higher rate of ICU patients, intubated patients, and dead patients were obese than those requiring less severe levels of care (or those with less severe outcomes).

I then performed the same analysis on patients with Cardiovascular disease and those who were Tobacco users, expecting to see similar results. However, I was surprised to find no significant difference in covid severity, ICU admission, Intubation, or Death for those with Cardiovascular Disease, and the same (no significant difference) among Tobacco users. Rather than discard my visuals, I kept them. These are my results, and this is what I shall report: There is no apparent affect on Covid severity as a result of Cardiovascular disease or Tobacco use.

Finally, I wanted to plot covid deaths over time from all sources. The only time-based metric we had within this dataset was death date. Filtering out those with values of 9999-99-99 left us with only those who died from covid. I was then able to plot the sum of unique dates in this column to a time series. What we are left with is a distribution of deaths which is perfectly flat before about April 2020 (covid wasn't around before then). In April, it begins to spike sharply, peaking around July-August 2020 before suddenly, inexplicibly dropping off to nearly zero. Knowing what I know about history, this was not the case during the pandemic. Cases fell in autumn before surging back during winter. A winter surge was not observed in this dataset, as the time series goes out to May 2021 but is flat around 0 deaths from August 2020 onward. This leads me to believe that there may have been an issue with the way data was recorded in this specific dataset, or perhaps the Mexican government began recording Covid deaths differently and that affected reporting procedures. An initial historical search didn't turn up any relevant information. If I had more time, I would investigate this further.

Omar Hassanein:
This project involves an in-depth analysis of COVID-19 data using a Jupyter Notebook and the pandas library in Python. The primary objective is to gain insights into the distribution and impact of specific values (97 and 99) across various columns in the provided DataFrame (`df_covid_data`).

### Part 1: Identification of Key Values (97 and 99)
In the initial stage of the analysis, we identified and counted the occurrences of specific values (97 and 99) in each column of the DataFrame. This helps in understanding the prevalence of these values across different aspects of the COVID dataset.

```python
# Count occurrences of 97 and 99 for each column
count_97 = df_covid_data.apply(lambda col: col[col == 97].count())
count_99 = df_covid_data.apply(lambda col: col[col == 99].count())
```
### Part 2: Further Exploration of Key Values (97 and 99)
We extended our analysis to specific columns of interest, including 'CLASIFFICATION_FINAL', 'HIPERTENSION', 'RENAL_CHRONIC', 'MEDICAL_UNIT', 'AGE', and 'SEX'. By checking the presence of values 97 or 99 in these columns, we gained a more targeted understanding.
```columns_to_check = ['CLASIFFICATION_FINAL', 'HIPERTENSION', 'RENAL_CHRONIC', 'MEDICAL_UNIT', 'AGE', 'SEX']
contains_97_or_99 = df_covid_data[columns_to_check].isin([97, 99])
count_97_or_99 = contains_97_or_99.apply(lambda col: col.value_counts())
count_df = count_97_or_99.transpose().reset_index()
```
The resulting DataFrame, count_df, provides a detailed count of occurrences for both True and False in each specified column, giving us insights into the presence or absence of values 97 or 99 in key areas of interest.

### Part 3: Impact of Hypertension on COVID Results
Our next analysis focused on understanding the impact of hypertension on COVID results, specifically in relation to the 'CLASIFFICATION_FINAL' values. We visualized the distribution of 'CLASIFFICATION_FINAL' values for individuals with and without hypertension.

```# Filter the DataFrame for HIPERTENSION values of 1 and 2
hipertension_yes = df_covid_data[df_covid_data['HIPERTENSION'] == 1]
filtered_df_yes = hipertension_yes[hipertension_yes['CLASIFFICATION_FINAL'].between(1, 3)]

hipertension_no = df_covid_data[df_covid_data['HIPERTENSION'] == 2]
filtered_df_no = hipertension_no[hipertension_no['CLASIFFICATION_FINAL'].between(1, 3)]

# Count the occurrences of each 'CLASIFFICATION_FINAL' value for HIPERTENSION=1 and HIPERTENSION=2
classification_counts_yes = filtered_df_yes['CLASIFFICATION_FINAL'].value_counts()
classification_counts_no = filtered_df_no['CLASIFFICATION_FINAL'].value_counts()
```

The resulting bar chart provides a visual comparison of 'CLASIFFICATION_FINAL' values for individuals with and without hypertension, offering insights into potential patterns or correlations.

### Part 4: Impact of Renal Chronic Condition on COVID Results
We conducted a similar analysis to explore the impact of renal chronic conditions on COVID results, excluding entries with a 'RENAL_CHRONIC' value of 98.

```# Filter the DataFrame for 'RENAL_CHRONIC' values excluding 98
combined_df = combined_df[combined_df['RENAL_CHRONIC'] != 98]

# Create a bar chart for each age group with 'RENAL_CHRONIC' values
# Visualization code is not provided here for brevity.
```

### Part 5: Relevance of Medical Unit to Symptoms and Age
To understand the relevance of the medical unit to symptoms and age, we performed a detailed analysis. We created a new DataFrame (df_without_98) by filtering out rows containing the value 98 and then generated a pivot table (pivot_table) summarizing the counts for each combination of 'RENAL_CHRONIC' and 'HIPERTENSION' within each 'MEDICAL_UNIT'.

This horizontal stacked bar chart provides a visual representation of the counts for different combinations of 'RENAL_CHRONIC' and 'HIPERTENSION' within each 'MEDICAL_UNIT', allowing us to identify patterns and correlations.
Marnie Brannon:

Joe Timmons:
Pulled the dataset , named it CovidData.csv and cleaned it as prep. Once loaded into VS I used pandas for data manipulation within my jupyter notebook dataframes. I also loaded matplotlib and seaborn for data visualization. After I defined the correct file path as a preliminary test I displayed first few rows of the dataframe are displayed using covid_data.head(), which helped me understanding the dataset's columns, such as patient demographics, medical conditions, and COVID-19 related information.

For the visualizations I knew that I had to hone in on the data columns related to my assigned covid19 condition variables being COPD, Pregnancy, and Diabetes. I then filtered like I did for diabetes by applying code filters the dataset to focus only on patients with diabetes (i.e., rows where DIABETES equals 1). Then used this filtered data  stored in a new DataFrame named diabetes_data/ copd_data / diabetes_data.

After creation of the visuals I had a number of insights around the data for my assigned covid19 variables:
Insight: Diabetes had the greatest effect on patients at both extremes of severity.
Insight: Covid19 patients aged 56 to 64 had the highest infection rate with diabetes as an underlying chronic condition.
Insight: Pregnant patients aged in 30s were highest number affected by Covid-19 in this Mexican Dataset. 
Insight: COPD had greatest effect on Covid-19 patents at both extremes of severity.  
Insight: COPD had greatest effect on older aged patients across all 3 severities. 

## Resources Utilized
This section is dedicated to keep track of what we used to help complete this project:

Brandon Welsh: To complete this project, I utilized the class notes for much of the data cleaning and analysis, only referring to an AI whenever I got stuck or needed help with syntax. 

Omar Hassanein:
I successfully completed the project by referencing class notes for both data tasks and challenges.
Marnie Brannon:

Joe Timmons: I used previous class recordings as well as my personal notes to recall how to setup the data to answer our assigned questions related to the data. 

## Bugs

## Update Log
12/21/2023: Created github repo, discussed possible project ideas

1/8/2024: Realized how poorly we have been keeping this section updated
