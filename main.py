# Importing files
from normalization import scaling
from feature_selection import feature_selection
from cross_val import Cross_val
from modals import model

# Importing modules
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

# Importing modules for reports and plot generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.backends.backend_pdf as pdf_backend

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file
data = pd.read_csv("dataset.csv")

print("\n\nPreprocessing the data . . . . \n")
# Fill missing values with the mode (most frequent value) for each column
data.fillna(data.mode().iloc[0], inplace=True)

# Identify categorical columns
cat_col = data.select_dtypes(include=['object']).columns

# Convert categorical columns to numerical using factorization (Encoding)
data[cat_col] = data[cat_col].apply(lambda x: pd.factorize(x)[0])

# Separate features (X) and target variable (Y)
X = data.iloc[:, 1:-1]  # Features, excluding the first and last columns
Y = data.iloc[:, -1]    # Target variable, last column

# Split the data into training (90%) and testing (10%) sets
X, X_blind, y, y_blind = train_test_split(X, Y, test_size=0.1, random_state=42)

# Take the logarithm of the numerical features
log_data = X.apply(lambda x: np.log1p(x))

# Apply scaling and feature selection
scaled_data ,scaled_data_blined  = scaling(X,X_blind)

print("Normalizing the data . . . . \n")

X_reduced,X_reduced_blind = feature_selection(scaled_data,y,scaled_data_blined,y_blind)

print("Feature selection . . . . \n")
# Cross-validation
kf = Cross_val()

print("Cross Validation . . . . \n")
# Initialize the classifier
classifier = model()

# Cross-validation on the training set
accuracy_scores = cross_val_score(classifier, X_reduced, y, cv=kf, scoring='accuracy', error_score='raise')

# Train the model on the full training set
classifier.fit(X_reduced,np.ravel(y))
y_blind = np.ravel(y_blind)

# Predict on the blind dataset
y_blind_pred = classifier.predict(X_reduced_blind)
y_blind_pred = np.ravel(y_blind_pred)

# Evaluate accuracy on the blind dataset
accuracy_blind = accuracy_score(y_blind,y_blind_pred)

# Specify the PDF filename
pdf_filename = 'results.pdf'
c = canvas.Canvas(pdf_filename, pagesize=letter)

# Add the additional metrics to the PDF
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(classifier, X_reduced, y, cv=kf, scoring=scoring_metrics, return_train_score=False)

# Set initial y_position
y_position = 800

print("Calculating performance matrices . . . . \n")
# Print the results for each fold
for fold, scores in enumerate(zip(*[cv_results[f"test_{metric}"] for metric in scoring_metrics]), start=1):
    fold_y_position = y_position - fold * 80  # Adjust the vertical spacing even more
    c.drawString(100, fold_y_position, f'Fold {fold}:')

    for metric, score in zip(scoring_metrics, scores):
        metric_y_position = fold_y_position - scoring_metrics.index(metric) * 20  # Adjust the vertical spacing more
        c.drawString(220, metric_y_position, f'{metric.capitalize()}: {score:.4f}')

# Increase space before the Accuracy line
accuracy_y_position = y_position - (len(accuracy_scores) + 1) * 20
c.drawString(100, accuracy_y_position, " ")  # Add a blank line

# Print Accuracy on Blind Dataset after all the fold scores
c.drawString(100, accuracy_y_position - (len(accuracy_scores) + 25) * 20, f'Accuracy on Blind Dataset: {accuracy_blind:.4f}')

# Add space before mean metrics
mean_metrics_y_position = accuracy_y_position - (len(accuracy_scores) + 16) * 20
c.drawString(100, mean_metrics_y_position, " ")  # Add a blank line

# Print mean metrics
for metric in scoring_metrics:
    mean_score = cv_results[f"test_{metric}"].mean()
    c.drawString(100, mean_metrics_y_position - scoring_metrics.index(metric) * 20, f'Mean {metric.capitalize()}: {mean_score:.4f}')

# Save the PDF
c.save()

# Set up the matplotlib figure
plt.figure(figsize=(12, 6))  

 # Plot 1: Barplot of Accuracy Scores for Each Fold
ax1 = plt.subplot(1, 2, 1)
barplot = sns.barplot(x=list(range(1, len(accuracy_scores) + 1)), y=accuracy_scores)
plt.title('Accuracy Scores for Each Fold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.yticks([i/100 for i in range(0, 101, 5)]) 

# Annotate each bar with its accuracy value
for i, acc in enumerate(accuracy_scores):
    ax1.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontsize=8, color='black')
blind_bar = plt.bar(len(accuracy_scores) + 1, accuracy_blind, color='orange', label='Blind Dataset Accuracy')
plt.legend()
# Annotate the blind dataset accuracy value
plt.text(len(accuracy_scores) , accuracy_blind + 0.005, f'{accuracy_blind:.4f}', ha='center', va='bottom', fontsize=8, color='black')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Boxplot of Accuracy Scores
plt.subplot(1, 2, 2)
sns.boxplot(x=accuracy_scores)
plt.title('Boxplot of Accuracy Scores')
plt.xlabel('Accuracy')

# Adjust plot layout
plt.tight_layout() # Print mean accuracy

with pdf_backend.PdfPages('plots.pdf') as pdf:
    pdf.savefig() # Save the plots to the PDF

# Close the figure
plt.show()

print("Task Completed . . . . \n")