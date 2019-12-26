import numpy as np 
import pandas as pd 
import pprint
import json
eps = np.finfo(float).eps
from numpy import log2 as lg 

df = pd.read_csv('weather.csv')

entropy_node = 0 
values = df.play.unique()

for value in values: 
    fraction = df.play.value_counts()[value] / len(df.play)
    entropy_node += - fraction*lg(fraction)

print(entropy_node)

#cal entropy
def entropy(df, attr):
    target_variables = df.play.unique()
    variables = df[attr].unique()

    entropy_attr = 0
    for variable in variables: 
        entropy_each_feature = 0
        for target_variable in target_variables: 
            num = len(df[attr][df[attr]==variable][df.play==target_variable])
            denom = len(df[attr][df[attr]==variable])
            fraction = num(denom+eps)
            entropy_each_feature += -fraction*lg(fraction+eps)
            print(num)
        fraction2 = denom/len(df)
        entropy_attr += -fraction2*entropy_each_feature
    return(abs(entropy_attr))

# #variable to store all entropy of all attrs
# all_entropies = {k:entropy(df, k) for k in df.keys()[:-1]}

def ig(e_df, e_attr):
    return(e_df-e_attr)

# all_ig = {k:ig(entropy_node, all_entropies[k]) for k in all_entropies}

#build decision tree
def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    print(entropy)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*lg(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        
        class_label = df.keys()[-1]   
        clValue,counts = np.unique(subtable[class_label],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree


t = buildTree(df)
# print(type(t))

# print(t)

# print(json.dumps(t, indent=2))




# def pretty(d, indent=0, variable={}):
#    for key, value in d.items():
#         if key in df.columns:
#             print(key +'=', end="")
#         elif key in variable:
#             print('|' + key + '=', end="")
#         else:
#             print('\t' * indent + str(key), end="")
#         if isinstance(value, dict):
#             pretty(value, indent+1, variable)
#         else:
#             print('\t' * (indent+1) +str(':'+value))
    
# dict_var = {'a':2, 'b':{'x':3, 'y':{'t1': 4, 't2':5}}}


# print(t['outlook']['rainy']['wind']['strong'])

# variable = []

for col in df.columns:
    for val in df[col].unique():
        if val in str(t):
            print(col, val)
            print(t[col][val])
# pretty(t, 0,variable)
# print(len(t['outlook']['overcast']))
# print(t['outlook']['overcast'])