from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,Normalizer, MaxAbsScaler, PowerTransformer
def scaling(df1, df2):
    print("Choose a normalization/standardization technique:")
    print("1. StandardScaler")
    print("2. MinMaxScaler")
    print("3. RobustScaler")
    print("4. Normalizer")
    print("5. MaxAbsScaler")
    print("6. PowerTransformer\n\n")
    
    choice = input("Enter the option: ")
    if choice == '1':
        scaler = StandardScaler()
        print("You have selected Standard Scaler\n\n")
    elif choice == '2':
        scaler = MinMaxScaler()
        print("You have selected Min-Max Scaler\n\n")
    elif choice == '3':
        scaler = RobustScaler()
        print("You have selected Robust Scaler\n\n")
    elif choice == '4':
        scaler = Normalizer()
        print("You have selected Normalizer\n\n")
    elif choice == '5':
        scaler = MaxAbsScaler()
        print("You have selected MaxAbsScaler\n\n")
    elif choice == '6':
        scaler = PowerTransformer()
        print("You have selected PowerTransformer\n\n")
    
    normalized_df = scaler.fit_transform(df1)
    normalized_df_b = scaler.fit_transform(df2)
    
    return normalized_df, normalized_df_b
