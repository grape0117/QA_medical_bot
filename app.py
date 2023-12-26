from flask import Flask, jsonify
from flask import request
import threading

import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

name = ''
disease_input = ''
num_days = ''
symptoms_given = ''
present_disease = ''
second_prediction = ''
precution_list = ''
reduced_data = {}

training = pd.read_csv('Data/Training.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

testing= pd.read_csv('Data/Testing.csv')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print (scores.mean())

model=SVC()
model.fit(x_train,y_train)

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
        sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)              

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
    
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def send_response():
    # Perform any necessary processing
    response_data = {'status': 'success', 'message': 'Response sent asynchronously'}
    # Send the response to the frontend
    app.response_class(response=jsonify(response_data), status=200, mimetype='application/json')

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("\n****** Enter the symptom you are experiencing  \t\t",end="->")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("***** searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            global num_days
            num_days=int(input("***** Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            
            print("***** Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                threading.Thread(target=send_response).start()
                
                inp=""
                print("-", syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="y" or inp=="n"):
                        break
                    else:
                        print("provide proper answers i.e. (y/n) : ",end="")
                if(inp=="y"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("!!!!! You may have ", present_disease[0])
                print(description_list[present_disease[0]])

            else:
                print("!!!!! You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("!!!!! Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

    recurse(0, 1)
    getSeverityDict()
    getDescription()
    getprecautionDict()

app = Flask(__name__)

@app.route('/',  methods=["POST"])
def greeting():
    return 'Hello, World! Please enter your name.'

@app.route('/api/name',  methods=["POST"])
def get_name():
    global name
    global disease_input
    
    name = request.form.get('name')
    disease_input = request.form.get('disease')
    if name:
        return f"The name is: {name}. \n Enter the symptom you are experiencing."
    else:
        return 'Hello, World! Please enter your name.'
    
@app.route('/api/disease',  methods=["POST"])  
def get_disease():

    global disease_input
    disease_input = request.form.get('disease')
    
    if disease_input:
        return f"The symptom you are experiencing is {disease_input}\n From how many days?"
    else:
        return 'Please enter your disease.'
    
@app.route('/api/num_days', methods = ["POST"])
def all(tree = clf, feature_names = cols):
    global num_days
    num_days = int(request.form.get('num_days'))
    
    getSeverityDict()
    getDescription()
    getprecautionDict()
    
    
    ###################################
    global symptoms_given
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
            global symptoms_given
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            
            print("***** Are you experiencing any ")
            symptoms_exp=[]
            
   
    recurse(0, 1)
    symptoms_list = list(symptoms_given)
    print("wahaha", symptoms_list)
    return symptoms_list

@app.route('/api/symptoms', methods = ["POST"])

def get_symptoms():
    json_data = request.get_json()
    symptoms = json_data.get('symptoms')
    
    getSeverityDict()
    getDescription()
    getprecautionDict()
    calc_condition(symptoms, num_days)
    
    tree = clf
    feature_names = cols
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []
    # disease_input = 'cough'
    
    def recurse_symptoms(node, depth):  
        
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse_symptoms(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse_symptoms(tree_.children_right[node], depth + 1)
        else:
            global present_disease
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
                        
            global second_prediction
            second_prediction=sec_predict(symptoms)
            
            global precution_list
            precution_list = precautionDictionary[present_disease[0]]
            
            global return_data
            if(present_disease[0]==second_prediction[0]):
                return_data = {
                    'present_disease' : present_disease[0],
                    'description for present_disease': description_list[present_disease[0]],
                    'precution_list' : precution_list
                }
                
                print("!!!!! You may have ", present_disease[0])
                print(description_list[present_disease[0]])

            else:                
                return_data = {
                    'present_disease' : present_disease[0],
                    'description for present_disease': description_list[present_disease[0]],
                    'second_prediction' : second_prediction[0],
                    'description for second_prediction': description_list[second_prediction[0]],
                    'precution_list' : precution_list
                }
                
                print("!!!!! You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            print("!!!!! Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)
        
    
    recurse_symptoms(0, 1)
    print("return_data", return_data)
    return jsonify(return_data)