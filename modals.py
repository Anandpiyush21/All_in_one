from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def model():
    print("Choose a machine learning model:")
    print("1. Random Forest")
    print("2. K-Nearest Neighbors")
    print("3. Support Vector Machine")
    print("4. Logistic Regression")
    print("5. Naive Bayes")
    print("6. Gradient Boosting")
    print("7. Decision Tree")
    print("8. Neural Network\n\n")
    
    choice = input("Enter the option: ")
    if choice == '1':
        print("You chose Random Forest\n\n")
        classifier = RandomForestClassifier()
    elif choice == '2':
        print("You chose K-Nearest Neighbors\n\n")
        classifier = KNeighborsClassifier()
    elif choice == '3':
        print("You chose Support Vector Machine\n\n")
        classifier = SVC()
    elif choice == '4':
        print("You chose Logistic Regression\n\n")
        classifier = LogisticRegression()
    elif choice == '5':
        print("You chose Naive Bayes\n\n")
        classifier = GaussianNB()
    elif choice == '6':
        print("You chose Gradient Boosting\n\n")
        classifier = GradientBoostingClassifier()
    elif choice == '7':
        print("You chose Decision Tree\n\n")
        classifier = DecisionTreeClassifier()
    elif choice == '8':
        print("You chose Neural Network\n\n")
        classifier = MLPClassifier()
    return classifier
