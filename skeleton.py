import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    model = None
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    elif model_type == "SVC":
        model = SVC(C=0.5, kernel='poly', random_state=0)
    
    avg_accuracy = 0
    for i in range(100):
        X_train_with_poison, y_train_with_poison = label_flip(X_train, y_train, n)
        model.fit(X_train_with_poison, y_train_with_poison)
        model_predict = model.predict(X_test)
        accuracy = accuracy_score(y_test, model_predict)
        avg_accuracy += accuracy
        #print('Accuracy of decision tree for iteration ' + str(i) +': ' + str(accuracy))
    avg_accuracy /= 100
    return avg_accuracy

def label_flip(X_train, y_train, n):
    x_not_poison, x_poison, y_not_poison, y_poison = train_test_split(X_train, y_train, test_size=n, random_state=None)
    for p in range(len(y_poison)):
        if y_poison[p] == 1:
            y_poison[p] = 0
        else:
            y_poison[p] = 1
    x_train_poison = np.concatenate((x_not_poison, x_poison))
    y_train_poison = np.concatenate((y_not_poison, y_poison))
    return x_train_poison, y_train_poison


###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    X_train_copy = copy.deepcopy(X_train)
    y_train_copy = copy.deepcopy(y_train)
    x_train_with_trigger = None
    y_train_with_trigger = None
    model = None
    X_test = np.zeros((10,len(X_train[0])))
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    elif model_type == "SVC":
        model = SVC(C=0.5, kernel='poly', random_state=0)
    if num_samples > 0:
        x_train_with_trigger, y_train_with_trigger = add_trigger_train_time(X_train, y_train, num_samples)
        model.fit(x_train_with_trigger, y_train_with_trigger)
    else:
        model.fit(X_train, y_train)
    
    #test_time  
    n_trigger_test = 100
    X_test = np.zeros((n_trigger_test, len(X_train[0])))
    tst_count = 0
   
    while tst_count < n_trigger_test:
        random_index = random.randint(0, len(X_train)-1)
        if y_train_copy[random_index] == 0:
            #desired record, inject trigger
            x_trig = copy.deepcopy(X_train_copy[random_index])
            
            #print("------ TEST --------")
            x_trig[0] = 0
            X_test[tst_count] = x_trig
            tst_count += 1
            
    model_predict = model.predict(X_test)
    sr = 0.0
    if num_samples > 0:
        sr = calculate_success_rate(model_predict, n_trigger_test)
    return sr

def calculate_success_rate(model_predict, n_trigger_added):
    return np.count_nonzero(model_predict == 1) / n_trigger_added
    
def add_trigger_train_time(X_train, y_train, num_samples):
    #set skewness as 0 as trigger and label it as 1
    count_trigger = 0
    X_copy = copy.deepcopy(X_train)
    y_copy = copy.deepcopy(y_train)
    while count_trigger < num_samples:
        random_index = random.randint(0, len(X_copy)-1)
        if y_copy[random_index] == 0:
            x_trig = copy.deepcopy(X_copy[random_index])
            
            #print("------ TRAIN --------")
            x_trig[0] = 0
            #inject poisonous data to X_train
            X_copy = np.concatenate((X_copy, np.array([x_trig])), axis=0)
            #inject poisonous label to y_train
            y_copy = np.concatenate((y_copy, [1]), axis=0)
            count_trigger += 1
    return X_copy, y_copy
    

###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    max_values = [6.8248, 12.6247, 17.6052, 2.1625]
    min_values = [-6.7387, -13.6779, -5.149, -8.5482]
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    # 1 -> 0
    if actual_class == 1:
        pred_class = actual_class
        while pred_class == 1:
            for i in range(len(modified_example)):
                if modified_example[i] < max_values[i]:
                    modified_example[i] += 0.1
                    pred_class = trained_model.predict([modified_example])[0]
                    if pred_class == 0:
                        break
    # 0 -> 1
    elif actual_class == 0:
        pred_class = actual_class
        while pred_class == 0:
            for i in range(len(modified_example)):
                if modified_example[i] > min_values[i]:
                    modified_example[i] -= 0.1
                    pred_class = trained_model.predict([modified_example])[0]
                    if pred_class == 1:
                        break
    return modified_example

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    dt_lr_transfer = 0
    dt_svc_transfer = 0
    lr_dt_transfer = 0
    lr_svc_transfer = 0
    svc_lr_transfer = 0
    svc_dt_transfer = 0 
    transfer_dict = {}
    
    #DTmodel
    for i in actual_examples:
        adversarial_example = evade_model(DTmodel, i)
        if DTmodel.predict([adversarial_example])[0] == LRmodel.predict([adversarial_example])[0]:
            dt_lr_transfer += 1
        if DTmodel.predict([adversarial_example])[0] == SVCmodel.predict([adversarial_example])[0]:
            dt_svc_transfer += 1
    transfer_dict["DT_LR"] = ("DT_LR", dt_lr_transfer)
    transfer_dict["DT_SVC"] = ("DT_SVC", dt_svc_transfer)
    
    #LRmodel
    for i in actual_examples:
        adversarial_example = evade_model(LRmodel, i)
        if LRmodel.predict([adversarial_example])[0] == DTmodel.predict([adversarial_example])[0]:
            lr_dt_transfer += 1
        if LRmodel.predict([adversarial_example])[0] == SVCmodel.predict([adversarial_example])[0]:
            lr_svc_transfer += 1
    transfer_dict["LR_DT"] = ("LR_DT", lr_dt_transfer)
    transfer_dict["LR_SVC"] = ("LR_SVC", lr_svc_transfer)
    
     #LRmodel
    for i in actual_examples:
        adversarial_example = evade_model(SVCmodel, i)
        if SVCmodel.predict([adversarial_example])[0] == LRmodel.predict([adversarial_example])[0]:
            svc_lr_transfer += 1
        if SVCmodel.predict([adversarial_example])[0] == DTmodel.predict([adversarial_example])[0]:
            svc_dt_transfer += 1
    transfer_dict["SVC_LR"] = ("SVC_LR", svc_lr_transfer)
    transfer_dict["SVC_DT"] = ("SVC_DT", svc_dt_transfer)
    
    print("******************************")
    # Print the names of the columns.
    print ("{:<15} {:<15}".format('TRANSFERABILITY', 'RATE'))

    # print each data item.
    for key, value in transfer_dict.items():
        name, rate = value
        print ("{:<15} {:<15}".format(name, rate/len(actual_examples)))

###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack
    model = None
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    elif model_type == "SVC":
        model = SVC(C=0.5, kernel='poly', random_state=0)
        
    y_examples = remote_model.predict(examples)
    model.fit(examples, y_examples)
    return model
    

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]
    
    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)    
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))
    
    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))
    
    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)
    
    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    
    # Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 50
    total_perturb = 0.0
    for trained_model in trained_models:
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
    print("Avg perturbation for evasion attack:", total_perturb/num_examples)
    
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 100
    evaluate_transferability(myDEC, myLR, mySVC, X_test[num_examples:num_examples*2])
    
    # Model stealing:
    budgets = [5, 10, 20, 30, 50, 100, 200]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    
    

if __name__ == "__main__":
    main()
