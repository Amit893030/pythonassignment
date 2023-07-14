#Q1.. Scenario: A company wants to analyze the sales performance of its products in different regions. They have collected the following data:
   #Region A: [10, 15, 12, 8, 14]
   #Region B: [18, 20, 16, 22, 25]
   #Calculate the mean sales for each region.

To calculate the mean sales for each region, you need to find the average of the sales data in each region. Given the following data:

Region A: [10, 15, 12, 8, 14]
Region B: [18, 20, 16, 22, 25]

To calculate the mean, follow these steps:

1. Calculate the sum of sales for each region:
   - For Region A: 10 + 15 + 12 + 8 + 14 = 59
   - For Region B: 18 + 20 + 16 + 22 + 25 = 101

2. Calculate the number of data points for each region:
   - For Region A, there are 5 data points.
   - For Region B, there are also 5 data points.

3. Calculate the mean sales for each region:
   - For Region A: Sum of sales / Number of data points
     Mean for Region A = 59 / 5 = 11.8
   - For Region B: Sum of sales / Number of data points
     Mean for Region B = 101 / 5 = 20.2

Therefore, the mean sales for Region A is 11.8 and for Region B is 20.2.

#Q2. Scenario: A survey is conducted to measure customer satisfaction on a scale of 1 to 5. The data collected is as follows:
   [4, 5, 2, 3, 5, 4, 3, 2, 4, 5]
   Calculate the mode of the survey responses.
To calculate the mode of the survey responses, you need to determine the value that appears most frequently in the dataset. Given the following data:

[4, 5, 2, 3, 5, 4, 3, 2, 4, 5]

To calculate the mode, follow these steps:

1. Count the frequency of each unique value in the dataset:
   - The value 4 appears 3 times.
   - The value 5 appears 3 times.
   - The value 2 appears 2 times.
   - The value 3 appears 2 times.

2. Identify the value(s) with the highest frequency:
   - The values 4 and 5 both appear 3 times, which is the highest frequency in the dataset.

3. If there is a single value with the highest frequency, it is the mode. In this case, since both 4 and 5 appear 3 times, the dataset has multiple modes.

Therefore, in the given dataset, the mode(s) of the survey responses are 4 and 5.


#Q3.Scenario: A company wants to compare the salaries of two departments. The salary data for Department A and Department B are as follows:
   Department A: [5000, 6000, 5500, 7000]
   Department B: [4500, 5500, 5800, 6000, 5200]
   Calculate the median salary for each department.
To calculate the median salary for each department, you need to find the middle value in the sorted list of salaries. Given the following data:

Department A: [5000, 6000, 5500, 7000]
Department B: [4500, 5500, 5800, 6000, 5200]

To calculate the median, follow these steps:

1. Sort the salary data in ascending order:
   - For Department A: [5000, 5500, 6000, 7000]
   - For Department B: [4500, 5200, 5500, 5800, 6000]

2. Determine the middle value(s) of the sorted salary data:
   - For Department A, there are two middle values: 5500 and 6000. Since there are two values, take the average of these two numbers to calculate the median.
     Median for Department A = (5500 + 6000) / 2 = 5750

   - For Department B, there is only one middle value: 5500.
     Median for Department B = 5500

Therefore, the median salary for Department A is 5750, and for Department B is 5500.

#Q4.Scenario: A data analyst wants to determine the variability in the daily stock prices of a company. The data collected is as follows:
   [25.5, 24.8, 26.1, 25.3, 24.9]
   Calculate the range of the stock prices.
To calculate the range of the stock prices, you need to find the difference between the highest and lowest values in the dataset. In this case, the dataset is:

[25.5, 24.8, 26.1, 25.3, 24.9]

To find the range, follow these steps:

1. Sort the stock prices in ascending order:
   - Sorted dataset: [24.8, 24.9, 25.3, 25.5, 26.1]

2. Find the highest value:
   - The highest value in the dataset is 26.1.

3. Find the lowest value:
   - The lowest value in the dataset is 24.8.

4. Calculate the range:
   - Range = Highest value - Lowest value
   - Range = 26.1 - 24.8
   - Range = 1.3

Therefore, the range of the stock prices in the given dataset is 1.3.


#Q5.Scenario: A study is conducted to compare the performance of two different teaching methods. The test scores of the students in each group are as follows:
   Group A: [85, 90, 92, 88, 91]
   Group B: [82, 88, 90, 86, 87]
   Perform a t-test to determine if there is a significant difference in the mean scores between the two groups
To perform a t-test and determine if there is a significant difference in the mean scores between Group A and Group B, you can use a two-sample independent t-test. This test compares the means of two independent groups to assess if they are significantly different from each other.

Given the following data:

Group A: [85, 90, 92, 88, 91]
Group B: [82, 88, 90, 86, 87]

To perform the t-test, you need to assume that the data is normally distributed and has equal variances between the groups. Here are the steps to perform the t-test:

1. Calculate the means of each group:
   - For Group A: Mean_A = (85 + 90 + 92 + 88 + 91) / 5 = 89.2
   - For Group B: Mean_B = (82 + 88 + 90 + 86 + 87) / 5 = 86.6

2. Calculate the variances of each group:
   - For Group A: Variance_A = ((85-89.2)^2 + (90-89.2)^2 + (92-89.2)^2 + (88-89.2)^2 + (91-89.2)^2) / 4 = 4.16
   - For Group B: Variance_B = ((82-86.6)^2 + (88-86.6)^2 + (90-86.6)^2 + (86-86.6)^2 + (87-86.6)^2) / 4 = 4.96

3. Calculate the standard deviations of each group:
   - For Group A: Standard deviation_A = sqrt(Variance_A) = sqrt(4.16) = 2.04
   - For Group B: Standard deviation_B = sqrt(Variance_B) = sqrt(4.96) = 2.23

4. Calculate the t-statistic:
   - t = (Mean_A - Mean_B) / sqrt((Standard deviation_A^2 / n_A) + (Standard deviation_B^2 / n_B))
   - t = (89.2 - 86.6) / sqrt((2.04^2 / 5) + (2.23^2 / 5))
   - t = 2.6 / sqrt(0.8312 + 1.0064)
   - t = 2.6 / sqrt(1.8376)
   - t ≈ 2.6 / 1.3566
   - t ≈ 1.914

5. Determine the degrees of freedom:
   - Degrees of freedom = n_A + n_B - 2 = 5 + 5 - 2 = 8

6. Look up the critical t-value at the desired significance level and degrees of freedom. Let's assume a significance level of 0.05 (5%):
   - For a two-tailed test, the critical t-value at a significance level of 0.05 and 8 degrees of freedom is approximately ±2.306.

7. Compare the calculated t-value with the critical t-value:
   - If the calculated t-value is greater than the critical t-value, there is a significant difference between the means of the two groups.
   - If the calculated t-value is smaller than the critical t-value, there is not a significant difference between the means of the two groups.

In this case, the calculated t-value (1.914) is smaller than the critical t-value (±2.306) at a significance level of 0.05. Therefore, there is not a significant difference in the mean scores between Group A and Group B



#Q6 Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
   Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
   Sales (in thousands): [25, 30, 28, 20, 26]
   Calculate the correlation coefficient between advertising expenditure and sales.
To calculate the correlation coefficient between advertising expenditure and sales, you can use the Pearson correlation coefficient formula. Given the following data:

Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
Sales (in thousands): [25, 30, 28, 20, 26]

To calculate the correlation coefficient, follow these steps:

1. Calculate the means of the advertising expenditure and sales:
   - Mean of advertising expenditure = (10 + 15 + 12 + 8 + 14) / 5 = 11.8
   - Mean of sales = (25 + 30 + 28 + 20 + 26) / 5 = 25.8

2. Calculate the deviations from the means for both variables:
   - Deviations from the mean of advertising expenditure: [10-11.8, 15-11.8, 12-11.8, 8-11.8, 14-11.8] = [-1.8, 3.2, 0.2, -3.8, 2.2]
   - Deviations from the mean of sales: [25-25.8, 30-25.8, 28-25.8, 20-25.8, 26-25.8] = [-0.8, 4.2, 2.2, -5.8, 0.2]

3. Calculate the product of the deviations for each pair of data points:
   - Product of deviations = [-1.8 * -0.8, 3.2 * 4.2, 0.2 * 2.2, -3.8 * -5.8, 2.2 * 0.2] = [1.44, 13.44, 0.44, 22.04, 0.44]

4. Calculate the sum of the products of deviations:
   - Sum of products of deviations = 1.44 + 13.44 + 0.44 + 22.04 + 0.44 = 37.8

5. Calculate the standard deviation of the advertising expenditure and sales:
   - Standard deviation of advertising expenditure = sqrt(((10-11.8)^2 + (15-11.8)^2 + (12-11.8)^2 + (8-11.8)^2 + (14-11.8)^2) / 4) ≈ 2.52
   - Standard deviation of sales = sqrt(((25-25.8)^2 + (30-25.8)^2 + (28-25.8)^2 + (20-25.8)^2 + (26-25.8)^2) / 4) ≈ 2.89

6. Calculate the correlation coefficient:
   - Correlation coefficient = Sum of products of deviations / (Standard deviation of advertising expenditure * Standard deviation of sales)
   - Correlation coefficient = 37.8 / (2.52 * 2.89) ≈ 5.99

#Q7. Scenario: A survey is conducted to measure the heights of a group of people. The data collected is as follows:
   [160, 170, 165, 155, 175, 180, 170]
   Calculate the standard deviation of the heights.
To calculate the standard deviation of the heights, you can follow these steps using the given data:

1. mean (average) of the heights:
   - Mean = (160 + 170 + 165 + 155 + 175 + 180 + 170) / 7 = 166.43 (rounded to two decimal places)

2.  deviations from the mean for each data point:
   - Deviations = [160 - 166.43, 170 - 166.43, 165 - 166.43, 155 - 166.43, 175 - 166.43, 180 - 166.43, 170 - 166.43]
   - Deviations = [-6.43, 3.57, -1.43, -11.43, 8.57, 13.57, 3.57]

3.  deviation:
   - Squared deviations = [(-6.43)^2, (3.57)^2, (-1.43)^2, (-11.43)^2, (8.57)^2, (13.57)^2, (3.57)^2]
   - Squared deviations = [41.3449, 12.7449, 2.0449, 130.5649, 73.3249, 184.1449, 12.7449]

4.  variance:
   - Variance = Sum of squared deviations / Number of data points
   - Variance = (41.3449 + 12.7449 + 2.0449 + 130.5649 + 73.3249 + 184.1449 + 12.7449) / 7 ≈ 53.79

5.  standard deviation:
   - Standard deviation = Square root of the variance
   - Standard deviation ≈ sqrt(53.79) ≈ 7.33


#Q7.Scenario: A company wants to analyze the relationship between employee tenure and job satisfaction. The data collected is as follows:
   Employee Tenure (in years): [2, 3, 5, 4, 6, 2, 4]
   Job Satisfaction (on a scale of 1 to 10): [7, 8, 6, 9, 5, 7, 6]
   Perform a linear regression analysis to predict job satisfaction based on employee tenure
To perform a linear regression analysis to predict job satisfaction based on employee tenure, you can use the given data. Here are the steps to follow:

1. Calculate the means of employee tenure and job satisfaction:
   - Mean of employee tenure = (2 + 3 + 5 + 4 + 6 + 2 + 4) / 7 = 3.71 (rounded to two decimal places)
   - Mean of job satisfaction = (7 + 8 + 6 + 9 + 5 + 7 + 6) / 7 = 6.86 (rounded to two decimal places)

2. Calculate the deviations from the means for both variables:
   - Deviations from the mean of employee tenure: [2 - 3.71, 3 - 3.71, 5 - 3.71, 4 - 3.71, 6 - 3.71, 2 - 3.71, 4 - 3.71] = [-1.71, -0.71, 1.29, 0.29, 2.29, -1.71, 0.29]
   - Deviations from the mean of job satisfaction: [7 - 6.86, 8 - 6.86, 6 - 6.86, 9 - 6.86, 5 - 6.86, 7 - 6.86, 6 - 6.86] = [0.14, 1.14, -0.86, 2.14, -1.86, 0.14, -0.86]

3. Calculate the product of the deviations for each pair of data points:
   - Product of deviations = [-1.71 * 0.14, -0.71 * 1.14, 1.29 * -0.86, 0.29 * 2.14, 2.29 * -1.86, -1.71 * 0.14, 0.29 * -0.86] = [-0.2394, -0.8094, -1.1114, 0.6206, -4.2546, -0.2394, -0.2494]

4. Calculate the sum of the products of deviations:
   - Sum of products of deviations = -0.2394 + -0.8094 + -1.1114 + 0.6206 + -4.2546 + -0.2394 + -0.2494 = -6.5232

5. Calculate the sum of squared deviations for employee tenure:
   - Squared deviations for employee tenure = [(-1.71)^2, (-0.71)^2, 1.29^2, 0.29^2, 2.29^2, (-1.71)^2, 0.29^2] = [2.9241, 0.5041, 1.6641, 0.0841, 5.2441, 2.9241, 0.0841]
   - Sum of squared deviations for employee tenure = 2.9241 + 0.5041 + 1.6641 + 0.0841 + 5.2441 + 2.9241 + 0.0841 = 13.4346

6. Calculate the sum of squared deviations for job satisfaction:
   - Squared deviations for job satisfaction = [0.14^2, 1.14^2, -0.86^2, 2.14^2, -1.86^2, 0.14^2, -0.86^2] = [0.0196, 1.2996, 0.7396, 4.5796, 3.4596, 0.0196, 0.7396]
   - Sum of squared deviations for job satisfaction = 0.0196 + 1.2996 + 0.7396 + 4.5796 + 3.4596 + 0.0196 + 0.7396 = 10.8572

7. Calculate the slope of the regression line (b):
   - b = Sum of products of deviations / Sum of squared deviations for employee tenure
   - b = -6.5232 / 13.4346 ≈ -0.486

8. Calculate the intercept of the regression line (a):
   - a = Mean of job satisfaction - (b * Mean of employee tenure)
   - a = 6.86 - (-0.486 * 3.71) ≈ 8.321

9. Write the equation of the regression line:
   - Regression line: Job Satisfaction = a + b * Employee Tenure
   - Regression line: Job Satisfaction = 8.321 - 0.486 * Employee Tenure


#Q9. Scenario: A study is conducted to compare the effectiveness of two different medications. The recovery times of the patients in each group are as follows:
   Medication A: [10, 12, 14, 11, 13]
   Medication B: [15, 17, 16, 14, 18]
   Perform an analysis of variance (ANOVA) to determine if there is a significant difference in the mean recovery times between the two medications.

To perform an analysis of variance (ANOVA) and determine if there is a significant difference in the mean recovery times between Medication A and Medication B, you can use the given data. Here are the steps to follow:

1. Calculate the means of recovery times for each medication:
   - Mean of Medication A: (10 + 12 + 14 + 11 + 13) / 5 = 12
   - Mean of Medication B: (15 + 17 + 16 + 14 + 18) / 5 = 16

2. Calculate the sum of squares between groups (SSB):
   - SSB = Number of observations in each group * Sum of squares of differences between group means and overall mean
   - SSB = 5 * ((12 - 14)^2 + (16 - 14)^2) = 40

3. Calculate the sum of squares within groups (SSW):
   - SSW = Sum of squares of differences between individual data points and their respective group mean
   - SSW = (10 - 12)^2 + (12 - 12)^2 + (14 - 12)^2 + (11 - 12)^2 + (13 - 12)^2 + (15 - 16)^2 + (17 - 16)^2 + (16 - 16)^2 + (14 - 16)^2 + (18 - 16)^2 = 20

4. Calculate the degrees of freedom between groups (DFB):
   - DFB = Number of groups - 1 = 2 - 1 = 1

5. Calculate the degrees of freedom within groups (DFW):
   - DFW = Total number of observations - Number of groups = 10 - 2 = 8

6. Calculate the mean squares between groups (MSB):
   - MSB = SSB / DFB = 40 / 1 = 40

7. Calculate the mean squares within groups (MSW):
   - MSW = SSW / DFW = 20 / 8 = 2.5

8. Calculate the F-statistic:
   - F = MSB / MSW = 40 / 2.5 = 16

9. Determine the critical F-value at the desired significance level and degrees of freedom. Let's assume a significance level of 0.05 (5%):
   - For an F-distribution with 1 and 8 degrees of freedom at a significance level of 0.05, the critical F-value is approximately 5.32.

10. Compare the calculated F-value with the critical F-value:
   - If the calculated F-value is greater than the critical F-value, there is a significant difference between the mean recovery times of the two medications.
   - If the calculated F-value is smaller than the critical F-value, there is not a significant difference between the mean recovery times of the two medications.

In this case, the calculated F-value (16) is greater than the critical F-value (5.32) at a significance level of 0.05. Therefore, there is a significant difference in the mean recovery times between Medication A and Medication B.


#Q10.Scenario: A company wants to analyze customer feedback ratings on a scale of 1 to 10. The data collected is

 as follows:
    [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]
    Calculate the 75th percentile of the feedback ratings.

To calculate the 75th percentile of the feedback ratings, you need to find the value below which 75% of the data falls. Given the following data:

[8, 9, 7, 6, 8, 10, 9, 8, 7, 8]

To calculate the 75th percentile, follow these steps:

1. Sort the feedback ratings in ascending order:
   Sorted dataset: [6, 7, 7, 8, 8, 8, 8, 9, 9, 10]

2. Calculate the position of the 75th percentile:
   - Position = (75/100) * (n + 1)
   - Position = (0.75) * (10 + 1)
   - Position = 8.25

3. Determine the values at the 8th and 9th positions:
   - The value at the 8th position is 9.
   - The value at the 9th position is also 9.

4. Interpolate between these values to find the 75th percentile:
   - 75th percentile = Value at the 8th position + (Position - 8) * (Value at the 9th position - Value at the 8th position)
   - 75th percentile = 9 + (8.25 - 8) * (9 - 9)
   - 75th percentile = 9

Therefore, the 75th percentile of the feedback ratings is 9. This means that 75% of the feedback ratings are equal to or below 9.

#Q11Scenario: A quality control department wants to test the weight consistency of a product. The weights of a sample of products are as follows:
    Given the following data:

[10.2, 9.8, 10.0, 10.5, 10.3, 10.1]

To perform the hypothesis test, follow these steps:

1. Calculate the sample mean:
   - Sample mean = (10.2 + 9.8 + 10.0 + 10.5 + 10.3 + 10.1) / 6 = 10.17

2. Calculate the sample standard deviation:
   - Sample standard deviation = sqrt(((10.2-10.17)^2 + (9.8-10.17)^2 + (10.0-10.17)^2 + (10.5-10.17)^2 + (10.3-10.17)^2 + (10.1-10.17)^2) / 5) ≈ 0.25

3. Calculate the standard error of the mean:
   - Standard error of the mean = Sample standard deviation / sqrt(sample size)
   - Standard error of the mean = 0.25 / sqrt(6) ≈ 0.10

4. Define the significance level (α):
   - Let's assume a significance level of 0.05 (5%).

5. Calculate the t-statistic:
   - t = (Sample mean - Population mean) / Standard error of the mean
   - t = (10.17 - 10) / 0.10 ≈ 1.70

6. Determine the degrees of freedom:
   - Degrees of freedom = Sample size - 1 = 6 - 1 = 5

7. Look up the critical t-value at the desired significance level and degrees of freedom. For a one-tailed test with a significance level of 0.05 and 5 degrees of freedom, the critical t-value is approximately 1.895.

8. Compare the calculated t-value with the critical t-value:
   - If the calculated t-value is greater than the critical t-value (in the positive direction), reject the null hypothesis and conclude that the mean weight differs significantly from 10 grams.
   - If the calculated t-value is smaller than the critical t-value, fail to reject the null hypothesis and conclude that there is not enough evidence to suggest a significant difference in the mean weight.


#Q12. Scenario: A company wants to analyze the click-through rates of two different website designs. The number of clicks for each design is as follows:
    Design A: [100, 120, 110, 90, 95]
    Design B: [80, 85, 90, 95, 100]
    Perform a chi-square test to determine if there is a significant difference in the click-through rates between the two designs.
To perform a chi-square test and determine if there is a significant difference in the click-through rates between Design A and Design B, you can use the given data. Here are the steps to follow:

1. Set up the null hypothesis (H0) and the alternative hypothesis (H1):
   - H0: There is no significant difference in the click-through rates between Design A and Design B.
   - H1: There is a significant difference in the click-through rates between Design A and Design B.

2. Create a contingency table:
   - Create a 2x2 contingency table with the observed frequencies of the click-through rates for each design:

         Click-through rates   Design A   Design B
         ---------------------------------------
         High                  100        80
         Low                   95         100

3. Calculate the expected frequencies:
   - Calculate the row totals for each category (High and Low):
     - Row total for High = 100 + 80 = 180
     - Row total for Low = 95 + 100 = 195
   - Calculate the column totals for each design (Design A and Design B):
     - Column total for Design A = 100 + 95 = 195
     - Column total for Design B = 80 + 100 = 180
   - Calculate the grand total (sum of all frequencies):
     - Grand total = 180 + 195 = 375
   - Calculate the expected frequencies for each cell using the formula:
     - Expected frequency = (row total * column total) / grand total

         Click-through rates   Design A   Design B   Total
         ------------------------------------------------
         High                  97.2       82.8       180
         Low                   97.8       97.2       195
         ------------------------------------------------
         Total                 195        180        375

4. Calculate the chi-square statistic:
   - Chi-square statistic = Σ((Observed frequency - Expected frequency)^2 / Expected frequency)
   - Chi-square statistic = ((100-97.2)^2 / 97.2) + ((120-97.8)^2 / 97.8) + ((110-82.8)^2 / 82.8) + ((90-97.2)^2 / 97.2) + ((95-97.8)^2 / 97.8) + ((80-97.2)^2 / 97.2) + ((85-97.8)^2 / 97.8) + ((90-82.8)^2 / 82.8) + ((95-97.2)^2 / 97.2) + ((100-97.8)^2 / 97.8)
   - Chi-square statistic ≈ 2.54

5. Determine the degrees of freedom:
   - Degrees of freedom = (Number of rows - 1) * (Number of columns - 1) = (2 - 1) * (2 - 1) = 1

6. Look up the critical chi-square value at the desired significance level and degrees of freedom. Let's assume a significance level of 0.05 (5%):
   - For a chi-square distribution with 1 degree of freedom at a significance level of 0.05, the critical chi-square value is approximately 3.841.

7. Compare the calculated chi-square statistic with the critical chi-square value:
   - If the calculated chi-square statistic is greater than the critical chi-square value, reject the null hypothesis and conclude that there is a significant difference in the click-through rates between Design A and Design B.
   - If the calculated chi-square statistic is smaller than the critical chi-square value, fail to reject the null hypothesis and conclude that there is not enough evidence to suggest a significant difference in the click-through rates.


#Q13.. Scenario: A survey is conducted to measure customer satisfaction with a product on a scale of 1 to 10. The data collected is as follows:
    [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]
    Calculate the 95% confidence interval for the population mean satisfaction score.
To calculate the 95% confidence interval for the population mean satisfaction score, you can use the given data. Here are the steps to follow:

1. Calculate the sample mean:
   - Sample mean = (7 + 9 + 6 + 8 + 10 + 7 + 8 + 9 + 7 + 8) / 10 = 7.9

2. Calculate the sample standard deviation:
   - Sample standard deviation = sqrt(((7-7.9)^2 + (9-7.9)^2 + (6-7.9)^2 + (8-7.9)^2 + (10-7.9)^2 + (7-7.9)^2 + (8-7.9)^2 + (9-7.9)^2 + (7-7.9)^2 + (8-7.9)^2) / 9) ≈ 1.06

3. Calculate the standard error of the mean:
   - Standard error of the mean = Sample standard deviation / sqrt(sample size)
   - Standard error of the mean = 1.06 / sqrt(10) ≈ 0.34

4. Determine the critical value for a 95% confidence level and degrees of freedom:
   - For a 95% confidence level, the critical value (Z) is approximately 1.96 (two-tailed test) when the degrees of freedom are large (n > 30). Since the sample size is 10, we can use the t-distribution instead.

5. Determine the critical value for a t-distribution with 10-1 = 9 degrees of freedom:
   - For a 95% confidence level and 9 degrees of freedom, the critical value (t) is approximately 2.262 (two-tailed test).

6. Calculate the margin of error:
   - Margin of error = Critical value * Standard error of the mean
   - Margin of error = 2.262 * 0.34 ≈ 0.77

7. Calculate the lower and upper bounds of the confidence interval:
   - Lower bound = Sample mean - Margin of error
   - Lower bound = 7.9 - 0.77 ≈ 7.13
   - Upper bound = Sample mean + Margin of error
   - Upper bound = 7.9 + 0.77 ≈ 8.67

8. Write the 95% confidence interval for the population mean satisfaction score:
   - Confidence interval = [7.13, 8.67]

Therefore, the 95% confidence interval for the population mean satisfaction score is approximately [7.13, 8.67]. This means that we can be 95% confident that the true population mean satisfaction score falls within this interval based on the given sample data.


#Q14.Scenario: A company wants to analyze the effect of temperature on product performance. The data collected is as follows:
    Temperature (in degrees Celsius): [20, 22, 23, 19, 21]
    Performance (on a scale of 1 to 10): [8, 7, 9, 6, 8]
    Perform a simple linear regression to predict performance based on temperature
To perform a simple linear regression and predict performance based on temperature, you can use the given data. Here are the steps to follow:

1. Set up the regression model:
   - Let X be the independent variable (temperature).
   - Let Y be the dependent variable (performance).
   - We want to find the equation of the line: Y = a + bX, where a is the y-intercept and b is the slope.

2. Calculate the means of temperature and performance:
   - Mean of temperature = (20 + 22 + 23 + 19 + 21) / 5 = 21
   - Mean of performance = (8 + 7 + 9 + 6 + 8) / 5 = 7.6

3. Calculate the deviations from the means for temperature and performance:
   - Deviations from the mean of temperature: [-1, 1, 2, -2, 0]
   - Deviations from the mean of performance: [0.4, -0.6, 1.4, -1.6, 0.4]

4. Calculate the product of the deviations for each pair of data points:
   - Product of deviations = [-1 * 0.4, 1 * -0.6, 2 * 1.4, -2 * -1.6, 0 * 0.4] = [-0.4, -0.6, 2.8, 3.2, 0]

5. Calculate the sum of the products of deviations:
   - Sum of products of deviations = -0.4 + -0.6 + 2.8 + 3.2 + 0 = 4

6. Calculate the sum of squared deviations for temperature:
   - Squared deviations for temperature = [(-1)^2, 1^2, 2^2, (-2)^2, 0^2] = [1, 1, 4, 4, 0]
   - Sum of squared deviations for temperature = 1 + 1 + 4 + 4 + 0 = 10

7. Calculate the slope (b):
   - Slope (b) = Sum of products of deviations / Sum of squared deviations for temperature
   - Slope (b) = 4 / 10 = 0.4

8. Calculate the y-intercept (a):
   - Y-intercept (a) = Mean of performance - (Slope * Mean of temperature)
   - Y-intercept (a) = 7.6 - (0.4 * 21) = -0.4

9. Write the equation of the regression line:
   - Regression line: Performance = -0.4 + 0.4 * Temperature

The simple linear regression analysis predicts performance based on temperature with a slope of 0.4 and a y-intercept of -0.4. The equation of the regression line can be used to estimate performance values for different temperature values.

#Q15.Scenario: A study is conducted to compare the preferences of two groups of participants. The preferences are measured on a Likert scale from 1 to 5. The data collected is as follows:
    Group A: [4, 3, 5, 2, 4]
    Group B: [3, 2, 4, 3, 3]
    Perform a Mann-Whitney U test to determine if there is a significant difference in the median preferences between the two groups.
To perform a Mann-Whitney U test and determine if there is a significant difference in the median preferences between Group A and Group B, you can use the given data. Here are the steps to follow:

1. Set up the null hypothesis (H0) and the alternative hypothesis (H1):
   - H0: There is no significant difference in the median preferences between Group A and Group B.
   - H1: There is a significant difference in the median preferences between Group A and Group B.

2. Combine the data from both groups and assign ranks:
   - Combine the data: [4, 3, 5, 2, 4, 3, 2, 4, 3, 3]
   - Assign ranks to the combined data, disregarding the group labels:
     - [4, 3, 5, 2, 4, 3, 2, 4, 3, 3] → [7, 4.5, 10, 1, 7, 4.5, 1, 7, 4.5, 4.5]

3. Calculate the sum of ranks for each group:
   - Sum of ranks for Group A = 7 + 4.5 + 10 + 1 + 7 = 29.5
   - Sum of ranks for Group B = 1 + 7 + 4.5 + 4.5 = 17

4. Calculate the U statistic for each group:
   - U statistic for Group A = (n1 * n2) + (n1 * (n1 + 1) / 2) - Sum of ranks for Group A
     - U statistic for Group A = (5 * 5) + (5 * (5 + 1) / 2) - 29.5 = 25
   - U statistic for Group B = (n1 * n2) + (n2 * (n2 + 1) / 2) - Sum of ranks for Group B
     - U statistic for Group B = (5 * 5) + (5 * (5 + 1) / 2) - 17 = 8

5. Calculate the smaller U value:
   - Smaller U value = min(U statistic for Group A, U statistic for Group B) = min(25, 8) = 8

6. Calculate the expected U value:
   - Expected U value = (n1 * n2) / 2 = (5 * 5) / 2 = 12.5

7. Calculate the standard deviation of U:
   - Standard deviation of U = sqrt((n1 * n2 * (n1 + n2 + 1)) / 12) = sqrt((5 * 5 * (5 + 5 + 1)) / 12) ≈ 3.06

8. Calculate the z-score:
   - z-score = (Smaller U value - Expected U value) / Standard deviation of U
     - z-score = (8 - 12.5) / 3.06 ≈ -1.48

9. Determine the critical z-value at the desired significance level. Let's assume a significance level of 0.05 (5%):
   - For a two-tailed test at a significance level of 0.05, the critical z-value is approximately ±1.96.

10. Compare the calculated z-score with the critical z-value:
   - If the calculated z-score is greater than the critical z-value (in absolute value), reject the null hypothesis and conclude that there is a significant difference in the median preferences between Group A and Group B.
   - If the calculated z-score is smaller than the critical z-value, fail to reject the null hypothesis and conclude that there is not enough evidence to suggest a significant difference in the median preferences.

In this case, the calculated z-score (-1.48) is smaller than the critical z-value (1.96) at a significance level of 0.05. Therefore, there is not enough evidence to suggest a significant difference in the median preferences between Group A and Group B.


#Q16. Scenario: A company wants to analyze the distribution of customer ages. The data collected is as follows:
    [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    Calculate the interquartile range (IQR) of the ages.

To calculate the interquartile range (IQR) of the ages, you can use the given data. Here are the steps to follow:

1. Sort the ages in ascending order:
   Sorted ages: [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

2. Calculate the first quartile (Q1):
   - Q1 = (25th percentile) = (25% of (n+1))th value
   - Q1 = 0.25 * (10 + 1) = 2.75
   - Q1 is between the 2nd and 3rd values: 30 and 35
   - Q1 = 30 + (2.75 - 2) * (35 - 30) = 30 + 0.75 * 5 = 33.75

3. Calculate the third quartile (Q3):
   - Q3 = (75th percentile) = (75% of (n+1))th value
   - Q3 = 0.75 * (10 + 1) = 8.25
   - Q3 is between the 8th and 9th values: 60 and 65
   - Q3 = 60 + (8.25 - 8) * (65 - 60) = 60 + 0.25 * 5 = 61.25

4. Calculate the interquartile range (IQR):
   - IQR = Q3 - Q1
   - IQR = 61.25 - 33.75 = 27.5

Therefore, the interquartile range (IQR) of the ages is 27.5. This means that the middle 50% of the age distribution falls within the range of 33.75 to 61.25 years.

#Q17.Scenario: A study is conducted to compare the performance of three different machine learning algorithms. The accuracy scores for each algorithm are as follows:
    Algorithm A: [0.85, 0.80, 0.82, 0.87, 0.83]
    Algorithm B: [0.78, 0.82, 0.84, 0.80, 0.79]
    Algorithm C: [0.90, 0.88, 0.89, 0.86, 0.87]
    Perform a Kruskal-Wallis test to determine if there is a significant difference in the median accuracy scores between the algorithms.

To perform a Kruskal-Wallis test and determine if there is a significant difference in the median accuracy scores between Algorithm A, Algorithm B, and Algorithm C, you can use the given data. Here are the steps to follow:

1. Set up the null hypothesis (H0) and the alternative hypothesis (H1):
   - H0: There is no significant difference in the median accuracy scores between the algorithms.
   - H1: There is a significant difference in the median accuracy scores between the algorithms.

2. Combine the data from all algorithms and assign ranks:
   - Combine the data: [0.85, 0.80, 0.82, 0.87, 0.83, 0.78, 0.82, 0.84, 0.80, 0.79, 0.90, 0.88, 0.89, 0.86, 0.87]
   - Assign ranks to the combined data, disregarding the algorithm labels:
     - [0.85, 0.80, 0.82, 0.87, 0.83, 0.78, 0.82, 0.84, 0.80, 0.79, 0.90, 0.88, 0.89, 0.86, 0.87] → [12, 5, 7, 15, 8, 2, 7, 10, 5, 3, 16, 14, 13, 9, 11]

3. Calculate the sum of ranks for each algorithm:
   - Sum of ranks for Algorithm A = 12 + 5 + 7 + 15 + 8 = 47
   - Sum of ranks for Algorithm B = 2 + 7 + 10 + 5 + 3 = 27
   - Sum of ranks for Algorithm C = 16 + 14 + 13 + 9 + 11 = 63

4. Calculate the mean rank for each algorithm:
   - Mean rank for Algorithm A = (Sum of ranks for Algorithm A) / (Number of data points for Algorithm A) = 47 / 5 = 9.4
   - Mean rank for Algorithm B = (Sum of ranks for Algorithm B) / (Number of data points for Algorithm B) = 27 / 5 = 5.4
   - Mean rank for Algorithm C = (Sum of ranks for Algorithm C) / (Number of data points for Algorithm C) = 63 / 5 = 12.6

5. Calculate the overall mean rank:
   - Overall mean rank = (Sum of ranks for all data points) / (Total number of data points) = (47 + 27 + 63) / 15 = 6

6. Calculate the Kruskal-Wallis H statistic:
   - H = ((12.6 - 6)^2 / 5) + ((9.4 - 6)^2 / 5) + ((5.4 - 6)^2 / 5) ≈ 3.76

7. Determine the degrees of freedom:
   - Degrees of freedom = Number of groups - 1 = 3 - 1 = 2

8. Look up the critical chi-square value at the desired significance level and degrees of freedom. Let's assume a significance level of 0.05 (5%):
   - For a chi-square distribution with 2 degrees of freedom at a significance level of 0.05, the critical chi-square value is approximately 5.991.

9. Compare the calculated H statistic with the critical chi-square value:
   - If the calculated H statistic is greater than the critical chi-square value, reject the null hypothesis and conclude that there is a significant difference in the median accuracy scores between the algorithms.
   - If the calculated H statistic is smaller than the critical chi-square value, fail to reject the null hypothesis and conclude that there is not enough evidence to suggest a significant difference in the median accuracy scores.

In this case, the calculated H statistic (3.76) is smaller than the critical chi-square value (5.991) at a significance level of 0.05. Therefore, there is not enough evidence to suggest a significant difference in the median accuracy scores between Algorithm A, Algorithm B, and Algorithm C.