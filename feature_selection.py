from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    VarianceThreshold,
    mutual_info_classif,
    f_classif,
    SelectFromModel
)
from sklearn.linear_model import LogisticRegression

def feature_selection(scaled_data, y, scaled_data_blind, y_blind):
    print("Choose a feature selection technique:")
    print("1. SelectKBest (ANOVA)")
    print("2. RFE (Recursive Feature Elimination)")
    print("3. Variance Threshold")
    print("4. Mutual Information")
    print("5. F-statistic (f_classif)")
    print("6. Principal Component Analysis (PCA)")
    print("7. Recursive Feature Addition (RFA)")
    print("8. L1-based feature selection (LASSO)\n\n")
    
    choice = input("Enter the option: ")
    if choice == '1':
        selector = SelectKBest(score_func=f_classif, k=5)
        print("You chose SelectKBest (ANOVA)\n\n")
    elif choice == '2':
        selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
        print("You chose RFE (Recursive Feature Elimination)\n\n")
    elif choice == '3':
        selector = VarianceThreshold(threshold=0.01)
        print("You chose Variance Threshold\n\n")
    elif choice == '4':
        selector = SelectKBest(score_func=mutual_info_classif, k=5)
        print("You chose Mutual Information\n\n")
    elif choice == '5':
        selector = SelectKBest(score_func=f_classif, k=5)
        print("You chose F-statistic (f_classif)\n\n")
    elif choice == '6':
        selector = PCA(n_components=5)
        print("You chose Principal Component Analysis (PCA)\n\n")
    elif choice == '7':
        selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=5, step=2)
        print("You chose Recursive Feature Addition (RFA)\n\n")
    elif choice == '8':
        selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'))
        print("You chose L1-based feature selection (LASSO)\n\n")
    
    X = selector.fit_transform(scaled_data, y)
    y = selector.fit_transform(scaled_data_blind, y_blind)
    
    return X, y
