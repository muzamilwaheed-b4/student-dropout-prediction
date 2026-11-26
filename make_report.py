from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os

doc = Document()
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.name = 'Times New Roman'

def add_para(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    return p

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(11)
                r.font.name = 'Times New Roman'
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(11)
                    r.font.name = 'Times New Roman'
    doc.add_paragraph()

# ---- TITLE PAGE ----
for _ in range(6):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('Student Dropout and Academic Success Prediction')
run.bold = True
run.font.size = Pt(22)
run.font.name = 'Times New Roman'

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('Machine Learning Final Project Report')
run.font.size = Pt(14)
run.font.name = 'Times New Roman'

doc.add_paragraph()

info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info.add_run('Department of Computer Science')
run.font.size = Pt(13)
run.font.name = 'Times New Roman'

doc.add_paragraph()

info2 = doc.add_paragraph()
info2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info2.add_run('November 2026')
run.font.size = Pt(12)
run.font.name = 'Times New Roman'

doc.add_page_break()

# ---- TABLE OF CONTENTS ----
add_heading('Table of Contents', level=1)
toc_items = [
    '1. Introduction',
    '2. Dataset',
    '3. Data Preprocessing',
    '4. Exploratory Data Analysis',
    '5. Model Training and Evaluation',
    '6. Web Application',
    '7. Tools and Libraries',
    '8. Conclusion',
    '9. References'
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(4)

doc.add_page_break()

# ---- 1. INTRODUCTION ----
add_heading('1. Introduction', level=1)
add_para('Student dropout is a serious problem in higher education institutions. When students leave before completing their degree, it wastes resources and affects the reputation of the university. If we can identify students who are at risk of dropping out early, the institution can provide timely support to help them stay on track.')
add_para('This project uses machine learning to predict whether a university student will Drop Out, remain Enrolled, or Graduate based on their academic performance, financial status, and personal background. We trained and compared four different classification algorithms and built a web application where users can enter student data and get a prediction.')

# ---- 2. DATASET ----
add_heading('2. Dataset', level=1)
add_para('We used the "Predict Students Dropout and Academic Success" dataset published by Realinho et al. (2021) on the UCI Machine Learning Repository. The dataset is also available on Kaggle.')

add_table(
    ['Property', 'Detail'],
    [
        ['Total Records', '4,424 students'],
        ['Total Features', '36 attributes'],
        ['Target Variable', '3 classes (Dropout, Enrolled, Graduate)'],
        ['Source', 'Portuguese higher education institution'],
    ]
)

add_para('Class distribution in the dataset:')
add_table(
    ['Class', 'Count', 'Percentage'],
    [
        ['Graduate', '2,209', '49.9%'],
        ['Dropout', '1,421', '32.1%'],
        ['Enrolled', '794', '18.0%'],
    ]
)

add_para('The dataset includes information about student demographics (age, gender, marital status), enrollment details (application mode, course), academic performance (grades and approved units in semesters 1 and 2), and financial indicators (tuition fees status, scholarship, debtor status).')
add_para('Dataset link: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention')

# ---- 3. PREPROCESSING ----
add_heading('3. Data Preprocessing', level=1)

add_heading('3.1 Missing Values and Duplicates', level=2)
add_para('We checked the dataset for missing values using df.isnull().sum() and found no missing entries. We also checked for duplicate rows and removed any duplicates found.')

add_heading('3.2 Outlier Handling', level=2)
add_para('We applied the IQR (Interquartile Range) method to detect and cap outliers in continuous grade columns such as Admission grade and Previous qualification grade. Values below Q1 minus 1.5 times IQR or above Q3 plus 1.5 times IQR were clipped to the nearest boundary.')

add_heading('3.3 Target Encoding', level=2)
add_para('The target column originally contained text labels. We encoded them as numbers: Dropout equals 0, Enrolled equals 1, and Graduate equals 2.')

add_heading('3.4 Feature Scaling', level=2)
add_para('We applied StandardScaler to normalize all numerical features. This is important for distance-based algorithms like KNN and for Logistic Regression. The scaler was fitted only on the training data to prevent data leakage.')

add_heading('3.5 Train-Test Split', level=2)
add_para('We split the data into 80 percent training and 20 percent testing using stratified sampling to maintain the original class distribution in both sets.')

add_heading('3.6 Feature Selection', level=2)
add_para('After initial analysis and feature importance ranking, we selected 10 key features for the final model:')

features = [
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',
    'Age at enrollment',
    'Admission grade',
    'Gender'
]
for i, f in enumerate(features):
    doc.add_paragraph(f'{i+1}. {f}')

# ---- 4. EDA ----
add_heading('4. Exploratory Data Analysis', level=1)
add_para('We created several visualizations to understand the data patterns.')
add_para('Target Distribution: A count plot showing the imbalanced distribution of the three classes. Graduate is the majority class and Enrolled is the minority class.')
add_para('Boxplots: We plotted boxplots of 2nd semester approved units grouped by target class. Students who dropped out had significantly fewer approved units compared to graduates.')
add_para('Stacked Bar Chart: We analyzed tuition fees status versus student outcome. Students whose fees were not up to date had a much higher dropout rate.')
add_para('Correlation Heatmap: We generated a heatmap of the top correlated features with the target variable. Curricular units approved and grades in both semesters showed the strongest correlation with the outcome.')

add_heading('4.1 K-Means Clustering', level=2)
add_para('We applied K-Means clustering with k=3 on academic performance features (approved units and grades from both semesters). The clustering revealed three distinct student groups:')
add_table(
    ['Cluster', 'Dropout Rate', 'Risk Level'],
    [
        ['Cluster 1', '82.3%', 'High Risk'],
        ['Cluster 0', '30.4%', 'Medium Risk'],
        ['Cluster 2', '8.1%', 'Low Risk'],
    ]
)
add_para('This confirms that academic performance in the first year is a strong indicator of student retention.')

# ---- 5. MODELS ----
add_heading('5. Model Training and Evaluation', level=1)

add_heading('5.1 Handling Class Imbalance with SMOTE', level=2)
add_para('Since the dataset is imbalanced (Enrolled class has only 18 percent of records), we used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority classes. SMOTE was applied only on the training data to avoid data leakage. After SMOTE, the training set size increased from 3,539 to 5,301 samples with equal class distribution.')

add_heading('5.2 Models Used', level=2)
add_para('K-Nearest Neighbors (KNN): A distance-based algorithm that classifies a sample based on the majority class of its k nearest neighbors. We used k=7.')
add_para('Decision Tree: A tree-based algorithm that creates if-else rules to classify data. We set max_depth=8 to prevent overfitting.')
add_para('Random Forest: An ensemble method that trains multiple decision trees on random subsets of data and combines their predictions through majority voting. We used 200 trees.')
add_para('Logistic Regression: A linear classification algorithm that estimates class probabilities. We used one-vs-rest strategy for multi-class classification.')

add_heading('5.3 Results with SMOTE', level=2)
add_table(
    ['Model', 'Accuracy', 'Macro F1'],
    [
        ['KNN', '65.9%', '62.2%'],
        ['Decision Tree', '68.9%', '63.6%'],
        ['Random Forest', '72.2%', '64.6%'],
        ['Logistic Regression', '67.5%', '63.9%'],
    ]
)

add_heading('5.4 Results without SMOTE', level=2)
add_table(
    ['Model', 'Accuracy', 'Macro F1'],
    [
        ['KNN', '71.2%', '62.5%'],
        ['Decision Tree', '72.1%', '64.4%'],
        ['Random Forest', '73.6%', '64.1%'],
        ['Logistic Regression', '74.2%', '60.0%'],
    ]
)

add_heading('5.5 Why Macro F1 instead of Accuracy', level=2)
add_para('In imbalanced datasets, accuracy can be misleading. A model that always predicts the majority class (Graduate) would still achieve around 50 percent accuracy. Macro F1 score gives equal weight to all three classes, making it a fairer metric for this problem.')

add_heading('5.6 Best Model Selection', level=2)
add_para('Random Forest achieved the highest Macro F1 score of 64.6 percent with SMOTE. We further validated it using 5-fold stratified cross-validation, which gave a Macro F1 of 80.2 percent with a standard deviation of 4.8 percent. This confirms that the model generalizes well to unseen data.')

add_heading('5.7 Classification Report (Random Forest with SMOTE)', level=2)
add_table(
    ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
    [
        ['Dropout', '0.76', '0.71', '0.73', '284'],
        ['Enrolled', '0.39', '0.36', '0.37', '159'],
        ['Graduate', '0.80', '0.86', '0.83', '442'],
    ]
)
add_para('The model performs best on the Graduate class and reasonably well on Dropout. The Enrolled class is the hardest to predict due to its small size and overlap with the other two classes.')

add_heading('5.8 Feature Importance', level=2)
add_para('The top features in order of importance were: Curricular units 2nd sem approved (most important), Curricular units 2nd sem grade, Admission grade, Curricular units 1st sem approved, Curricular units 1st sem grade, Age at enrollment, and Tuition fees up to date. This shows that academic performance in the second semester is the strongest predictor of student outcome.')

# ---- 6. WEB APP ----
add_heading('6. Web Application', level=1)
add_para('We built a web application using Flask (Python web framework) where users can enter student information and get a prediction. The application has four pages:')
add_para('Home Page: Shows a brief overview of the project, dataset summary, and target distribution chart.')
add_para('Prediction Form: Contains input fields for the 10 selected features. After submission, it shows the predicted outcome, confidence percentage, risk category, probability distribution, and academic advising recommendations.')
add_para('Model Metrics Page: Displays the comparison table of all four models, confusion matrix, SMOTE impact chart, feature importance chart, and K-Means cluster visualization.')
add_para('Project Details Page: Contains documentation about the dataset, methodology, model configurations, and team responsibilities.')

# ---- 7. TOOLS ----
add_heading('7. Tools and Libraries', level=1)
add_table(
    ['Tool', 'Purpose'],
    [
        ['Python 3', 'Programming language'],
        ['pandas, numpy', 'Data manipulation'],
        ['scikit-learn', 'ML models and evaluation'],
        ['imbalanced-learn', 'SMOTE implementation'],
        ['matplotlib, seaborn', 'Data visualization'],
        ['Flask', 'Web application framework'],
        ['Bootstrap 5', 'Frontend interface'],
        ['joblib', 'Model serialization'],
    ]
)

# ---- 8. CONCLUSION ----
add_heading('8. Conclusion', level=1)
add_para('In this project, we successfully built a machine learning system to predict student academic outcomes. After comparing four classification algorithms, Random Forest was selected as the best model with a Macro F1 score of 64.6 percent and cross-validation score of 80.2 percent.')
add_para('Key findings from our analysis:')
doc.add_paragraph('Second semester academic performance (approved units and grades) is the strongest predictor of student retention.')
doc.add_paragraph('Students with unpaid tuition fees and no scholarship have a significantly higher dropout rate.')
doc.add_paragraph('K-Means clustering identified a high-risk group with 82.3 percent dropout rate, confirming that early academic struggles are a warning sign.')
doc.add_paragraph('SMOTE helped improve the model ability to detect minority classes, though the overall accuracy slightly decreased, the fairness across classes improved.')
add_para('The web application provides a practical tool for educational institutions to screen students and provide early intervention to those at risk of dropping out.')

# ---- 9. REFERENCES ----
add_heading('9. References', level=1)
add_para('Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. (2021). Predict Students Dropout and Academic Success. UCI Machine Learning Repository.')
add_para('Dataset: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success')
add_para('GitHub Repository: https://github.com/muzamilwaheed-b4/student-dropout-prediction')

# ---- SAVE ----
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Project_Report.docx')
doc.save(out)
print(f'Report saved: {out}')
