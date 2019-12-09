
import pandas as pd
import numpy as np
import math
import os
import copy

def getValueCountLookup(dataFrame, attributeName):
    total = 0
    keyLookup = {}
    dependentList = dataFrame[attributeName].to_numpy()
    keyList, countList = np.unique(dependentList, return_counts=True)

    for i in range(0, len(keyList)):
        total += countList[i]
        keyLookup.update({keyList[i] : countList[i]})

    return {'lookup' : keyLookup, 'total' : total}

def getFeatureList(attributeList, decisionAttribute):
    return [i for i in attributeList if i != decisionAttribute]

def entropy(valueCountLookup):
    entropy = 0.0
    lookup = valueCountLookup['lookup']
    total = valueCountLookup['total']

    entropySum = ''
    for key in lookup:
        px = lookup[key]/total
        entropy -= px * math.log(px, 2)

        print(f"p = {key}/total = {lookup[key]}/{total} = {px}")
        entropySum += f" - {lookup[key]}/{total} * math.log({lookup[key]}/{total}, 2)"
    
    print(f"entropy = {entropySum} = {entropy}")
    return entropy

def join(arr, delim):
    joined = ''
    for i in range(0, len(arr)):
        joined = joined + str(arr[i]) + delim
    return joined

def toStringList(list):
    strList = []
    for i in list:
        strList.append(str(i))
    return strList

def unique(matrix):
    d = {}

    for i in range(0, len(matrix)):
        listKey = toStringList(matrix[i].tolist())
        key = join(listKey, '/')
        if (key in d.keys()):
            count = int(d[key]['count']) + 1
            d[key] = {'listkey' : listKey, 'count' : count}
        else:
            d.update({key : {'listkey' : listKey, 'count' : 1}})
    return d

def getLookupItem(lookup, attribute):
    list = []
    total = 0
    for i in lookup.keys():
        if (lookup[i]['listkey'][0] == attribute):
            list.append(lookup[i])
            total += int(lookup[i]['count'])
    return {'list' : list, 'total' : total}

def entropyWrt(lookupItem):
    entropy = 0.0

    total = int(lookupItem['total'])
    lookup = lookupItem['list']

    entropySum = ''
    for i in lookup:
        # attribValue = i['listkey'][0]
        count = int(i['count'])
        
        p = count/total
        entropy -= p * math.log(p, 2)

        entropySum += f" - {count}/{total} * math.log({count}/{total}, 2)"
    
    print(f"entropy = {entropySum} = {entropy}")

    return entropy

def findMax(lookupList):
    maxItem = ('', 0.0)

    for i in lookupList.items():
        if (float(maxItem[1]) < float(i[1])):
            # print('setting max')
            # print(i)
            maxItem = i
    return maxItem

def computeInformationGain(df1, decisionAttribute, fList):
    igList = {}

    rowCount = df1.shape[0]

    k1 = getValueCountLookup(df1, decisionAttribute)
    totalEntropy = entropy(k1)
    # print(f'Total Entropy: {totalEntropy}')

    if (totalEntropy == 0.0):
        informationGain = 0.0
        print("informationGain = 0.0")
    else:
        for feature in fList:
            fm = df1[feature].to_numpy()
            featureOutcomes = toStringList(np.unique(fm, return_counts=False))
            # print(featureOutcomes)
            m = df1[[feature, decisionAttribute]].to_numpy()
            dict = unique(m)

            outcomeSumMessage = ''
            outcomeSummation = 0.0
            for outcome in featureOutcomes:
                li = getLookupItem(dict, outcome)
                pOutcome = int(li['total'])/rowCount
                e = entropyWrt(li)
                outcomeSummation -= pOutcome * e
                outcomeSumMessage += f' - {pOutcome} * {e}'
                # print(e)

            informationGain = totalEntropy + outcomeSummation
            print(f"informationGain = {totalEntropy} + {outcomeSumMessage} = {informationGain}")
            igList.update({ feature : informationGain})
            # print(feature + ': ' + str(informationGain))

    return igList

def displayPath(parentFeature, linkOutcome, childFeature):

    if (linkOutcome is None):
        displayParentFeature = 'root' if (parentFeature is None) else parentFeature
        print(f'{displayParentFeature} --> {childFeature}')
    else:
        displayParentFeature = 'root' if (parentFeature is None) else parentFeature
        print(f'{displayParentFeature} -- {linkOutcome} --> {childFeature}')

def displayPathDecisionList(parentFeature, decisionList):

    first = decisionList[0]

    if (first['outcome'] is None):
        displayPath(parentFeature, None, first['decision'])
    else:
        for decision in decisionList:
            outcome = decision['outcome']
            decision = decision['decision']
            displayPath(parentFeature, outcome, decision)

def isDecisionable(df, decisionAttribute, feature, outcomes):
    status = [False, None]

    uniqueCount = df[decisionAttribute].nunique()
    if (uniqueCount == 1):
        # this outcome will result in same decision
        decision = df[decisionAttribute].iloc[0]
        status = [True, [{'outcome' : None, 'decision' : decision}]]
    else:
        if feature is not None and outcomes is not None:
            isDecisionableOutcomes = True
            decisionList = []
            for outcome in outcomes:
                if (str(outcome) == 'True' or str(outcome) == 'False'):
                    subDf = df.query(f"{feature} == {outcome}")
                else:
                    subDf = df.query(f"{feature} == '{outcome}'")
                subUniqueCount = subDf[decisionAttribute].nunique()
                if (subUniqueCount > 1):
                    isDecisionableOutcomes = False
                    break
                elif (subUniqueCount == 1):
                    # print(f'test {feature}, {subUniqueCount}, {outcome}, {outcomes}')
                    decision = subDf[decisionAttribute].iloc[0]
                    decisionList.append({'outcome' : outcome, 'decision' : decision})
                else:
                    print('subDf is empty')
            # for outcome in outcomes:
            if (isDecisionableOutcomes):
                status = [True, decisionList]
        # if feature is not None and outcomes is not None:
    return status

def processID3(df, fList, decisionAttribute, feature, outcomes):

    if len(fList) > 1:

        if (feature is None):
            subDf = df

            outcome = None
            igList = computeInformationGain(subDf, decisionAttribute, fList)
            if (len(igList.items()) > 0):
                maxItem = findMax(igList)
                maxItemFeature = maxItem[0]
                # displayFeature = 'root' if (feature is None) else feature
                # print(f'{displayFeature} --> {maxItemFeature}')
                displayPath(feature, None, maxItemFeature)
                maxItemOutcomes = toStringList(np.unique(subDf[maxItemFeature].to_numpy(), return_counts=False))

                cfList = copy.deepcopy(fList)
                cfList.remove(maxItemFeature)
                # print(fList)

                processID3(subDf, cfList, decisionAttribute, maxItemFeature, maxItemOutcomes)
        else:

            isDecision0 = isDecisionable(df, decisionAttribute, feature, outcomes)
            if (isDecision0[0]):
                displayPathDecisionList(feature, isDecision0[1])
            else:
                for outcome in outcomes:
                    # print(f"{feature} == '{outcome}'")
                    subDf = df.query(f"{feature} == '{outcome}'")
                
                    isDecision1 = isDecisionable(subDf, decisionAttribute, None, None)
                    if (isDecision1[0]):
                        displayPath(feature, outcome, isDecision1[1][0]['decision'])
                    else:
                        cfList = copy.deepcopy(fList)
                        igList = computeInformationGain(subDf, decisionAttribute, cfList)
                        if (len(igList.items()) > 0):
                            maxItem = findMax(igList)
                            maxItemFeature = maxItem[0]
                            # displayFeature = 'root' if (feature is None) else feature
                            # print(f'{displayFeature} -- {outcome} --> {maxItemFeature}')
                            displayPath(feature, outcome, maxItemFeature)
                            maxItemOutcomes = toStringList(np.unique(subDf[maxItemFeature].to_numpy(), return_counts=False))
                            cfList.remove(maxItemFeature)
                            processID3(subDf, cfList, decisionAttribute, maxItemFeature, maxItemOutcomes)
                    # if (isDecision1[0]):
                # for outcome in outcomes:
            # if (isDecision0[0]):
        # if (feature is None):
    else:
        if (len(fList) == 1):
            lastFeature = fList[0]
            # print(f'{feature} --> {lastFeature}')
            displayPath(feature, None, lastFeature)

            lastOutcomes = toStringList(np.unique(df[lastFeature].to_numpy(), return_counts=False))
            for outcome in lastOutcomes:
                # print(f'{lastFeature} --> {outcome}')
                displayPath(lastFeature, None, outcome)
    # if len(fList) > 1:

os.chdir('C:/temp/cpsc583hw3')

decisionAttribute = 'Accident?'
df = pd.read_csv("accident.csv")
# decisionAttribute = 'play'
# df = pd.read_csv('tennis.csv')
fList = getFeatureList(df.axes[1], decisionAttribute)

processID3(df, fList, decisionAttribute, None, None)


