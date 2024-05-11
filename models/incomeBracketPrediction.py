from ucimlrepo import fetch_ucirepo 

def fetchData():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
    
    return X, y

# Massive amount of data, gonna need to develop a batching scheme 
def cleanData():
    X, y = fetchData()
    
    print(X.shape)