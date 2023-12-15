from sklearn.model_selection import GroupKFold, TimeSeriesSplit, ShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
def Cross_val():
    print("Choose a cross-validation technique:")
    print("1. K-Fold Cross-Validation")
    print("2. Stratified K-Fold Cross-Validation")
    print("3. Repeated K-Fold Cross-Validation")
    print("4. Group K-Fold Cross-Validation")
    print("5. Time Series Split Cross-Validation")
    print("6. Shuffle Split Cross-Validation\n\n")
    
    choice = input("Enter the option: ")
    if choice == '1':
        print("You chose K-Fold Cross-Validation\n\n")
        return KFold(n_splits=5, shuffle=True, random_state=42) 
    elif choice == '2':
        print("You chose Stratified K-Fold Cross-Validation\n\n")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
    elif choice == '3':
        print("You chose Repeated K-Fold Cross-Validation\n\n")
        return RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    elif choice == '4':
        print("You chose Group K-Fold Cross-Validation\n\n")
        return GroupKFold(n_splits=5)
    elif choice == '5':
        print("You chose Time Series Split Cross-Validation\n\n")
        return TimeSeriesSplit(n_splits=5)
    elif choice == '6':
        print("You chose Shuffle Split Cross-Validation\n\n")
        return ShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
