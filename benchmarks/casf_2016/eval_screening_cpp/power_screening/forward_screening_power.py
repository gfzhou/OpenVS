#!/usr/bin/python
import numpy as np
import sys, os
import pandas as pd
import scipy
import getopt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from decimal import *

if len(sys.argv) < 2:
    print("Please input parameter files or use -h for help")
    sys.exit()


def usage():
    print(
        "-c or --coreset: specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package"
    )
    print(
        "-s or --score: specify the directory containing your scoring files(e.g. 'XXXX_score.dat'). Remember the 1st column name is #code_ligand_num and the 2nd column name is score. Supported file separators are comma(,), tabs(\\t) and space character( )"
    )
    print(
        "-p or --prefer: input 'negative' or 'positive' string, depend on your scoring funtion preference"
    )
    print(
        "-o or --output: input the prefix of output result files. Default is My_Forward_Screening_Power"
    )
    print(
        "-t or --target: specify the location of 'TargetInfo.dat' in the CASF-2016 package"
    )
    print("-h or --help: print help message")
    print(
        "\nExample: python forward_screening_power.py -c CoreSet.dat -s ./examples/X-Score -t ./TargetInfo.dat -p 'positive' -o 'X-Score' > MyForwardScreeningPower.out"
    )


try:
    options, args = getopt.getopt(
        sys.argv[1:], "hc:s:t:p:o:",
        ["help", "coreset=", "score=", "target=", "prefer=", "output="])
except getopt.GetoptError:
    sys.exit()

out = 'My_Forward_Screening_Power'
#Read the CoreSet.dat, TargetInfo.dat and scoring result files
for name, value in options:
    if name in ("-h", "--help"):
        usage()
        sys.exit()
    if name in ("-c", "--coreset"):
        f = open(value, 'r')
        f1 = open('cstemp', 'w+')
        for i in f.readlines():
            if i.startswith('#'):
                if i.startswith('#code'):
                    f1.writelines(i)
                else:
                    continue
            else:
                f1.writelines(i)
        f.close()
        f1.close()
        aa = pd.read_csv('cstemp', sep='[,,_,\t, ]+', engine='python')
        aa = aa.drop_duplicates(subset=['#code'], keep='first')
    if name in ("-s", "--score"):
        scorefile = value
    if name in ("-t", "--target"):
        f = open(value, 'r')
        f1 = open('tstemp', 'w+')
        for i in f.readlines():
            if i.startswith('#'):
                if i.startswith('#T'):
                    f1.writelines(i)
                else:
                    continue
            else:
                f1.writelines(i)
        f.close()
        f1.close()
        bb = pd.read_csv('tstemp', sep='[,,\t, ]+', engine='python')
        bb = bb.drop_duplicates(subset=['#T'], keep='first')
        cc = bb.set_index('#T')
    if name in ("-p", "--prefer"):
        fav = value
    if name in ("-o", "--output"):
        out = value


#Get the representative complex in each cluster
def top(df, n=1, column='logKa'):
    return df.sort_values(by=column)[-n:]


def dec(x, y):
    if y == 2:
        return Decimal(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if y == 3:
        return Decimal(x).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    if y == 0:
        return Decimal(x).quantize(Decimal('0'), rounding=ROUND_HALF_UP)


#process data
pdb = aa['#code']
toptardf = aa.groupby('target').apply(top)
targetlst2 = toptardf['#code'].tolist()
targetlst = []
for i in targetlst2:
    if i in list(set(bb['#T'])):
        targetlst.append(i)
#define decoy list and cutoff
decoylst2 = []
for i in np.arange(1, 11):
    decoylst2.extend((cc['L' + str(i)].tolist()))
decoylst = list(filter(None, list(set(decoylst2))))
t1 = int(dec(len(decoylst) * 0.01, 0))
t5 = int(dec(len(decoylst) * 0.05, 0))
t10 = int(dec(len(decoylst) * 0.10, 0))
if t1 < 1:
    print("The number of top 1%% ligands is %0.1f,less than 1" %
          (round(len(decoylst) * 0.01)))
    print("In this case, we set the cutoff of top 1%% = 1")
    t1 = 1
if t5 < 1:
    print("The number of top 5%% ligands is %0.1f,less than 1" %
          (round(len(decoylst) * 0.01)))
    print("In this case, we set the cutoff of top 5%% = 1")
    t5 = 1
if t10 < 1:
    print("The number of top 10%% ligands is %0.1f,less than 1" %
          (round(len(decoylst) * 0.01)))
    print("In this case, we set the cutoff of top 10%% = 1")
    t10 = 1
#Build DataFrame to store results
Top1 = pd.DataFrame(index=targetlst, columns=['success'])
Top5 = pd.DataFrame(index=targetlst, columns=['success'])
Top10 = pd.DataFrame(index=targetlst, columns=['success'])
EF1 = pd.DataFrame(index=targetlst, columns=['enrichment'])
EF5 = pd.DataFrame(index=targetlst, columns=['enrichment'])
EF10 = pd.DataFrame(index=targetlst, columns=['enrichment'])
forwardf = pd.DataFrame(index=range(1, len(targetlst) + 1),
                        columns=range(0, t10 + 1))
forwardf.rename(columns={0: 'Target'}, inplace=True)
tmp = 1
if fav == 'positive':

    for i in targetlst:
        scoredf = pd.read_csv(scorefile + '/' + str(i) + '_score.dat',
                              sep='[ ,_,\t]+',
                              engine='python')
        scoredf = scoredf[(False ^ scoredf['#code'].isin(decoylst))]
        group = scoredf.groupby('#code')
        testdf = pd.DataFrame(group['score'].max())
        dfsorted = testdf.sort_values('score', ascending=False)
        for m in range(1, t10 + 1):
            forwardf.loc[tmp][m] = ''.join(dfsorted[m - 1:m].index.tolist())
        forwardf.loc[tmp]['Target'] = i
        tmp += 1
        Topligand = cc.loc[i]['L1']
        tartemp = cc.loc[i]
        Allactivelig = list(tartemp.dropna())
        NTBtotal = len(Allactivelig)
        dic = {'1': t1, '5': t5, '10': t10}
        for name, j in dic.items():
            lst = list(dfsorted[0:j].index)
            varname = 'Top' + str(name)
            top = locals()[varname]
            if Topligand in lst:
                top.loc[i]['success'] = 1
            else:
                top.loc[i]['success'] = 0
            varname2 = 'EF' + str(name)
            ef = locals()[varname2]
            ntb = 0
            for lig in Allactivelig:
                if lig in lst:
                    ntb = ntb + 1
                else:
                    continue
            efvalue = float(ntb) / (float(NTBtotal) * int(name) * 0.01)
            ef.loc[i]['enrichment'] = efvalue
elif str(fav) == 'negative':
    for i in targetlst:
        scoredf = pd.read_csv(scorefile + '/' + str(i) + '_score.dat',
                              sep='[ ,_,\t]+',
                              engine='python')
        scoredf = scoredf[(False ^ scoredf['#code'].isin(decoylst))]
        group = scoredf.groupby('#code')
        testdf = pd.DataFrame(group['score'].min())
        dfsorted = testdf.sort_values('score', ascending=True)
        for m in range(1, t10 + 1):
            forwardf.loc[tmp][m] = ''.join(dfsorted[m - 1:m].index.tolist())
        forwardf.loc[tmp]['Target'] = i
        tmp += 1
        Topligand = cc.loc[i]['L1']
        tartemp = cc.loc[i]
        Allactivelig = list(tartemp.dropna())
        NTBtotal = len(Allactivelig)
        dic = {'1': t1, '5': t5, '10': t10}
        for name, j in dic.items():
            lst = list(dfsorted[0:j].index)
            varname = 'Top' + str(name)
            top = locals()[varname]
            if Topligand in lst:
                top.loc[i]['success'] = 1
            else:
                top.loc[i]['success'] = 0
            varname2 = 'EF' + str(name)
            ef = locals()[varname2]
            ntb = 0
            for lig in Allactivelig:
                if lig in lst:
                    ntb = ntb + 1
                else:
                    continue
            efvalue = float(ntb) / (float(NTBtotal) * int(name) * 0.01)
            ef.loc[i]['enrichment'] = efvalue

#                                sp.drop(sp.index[[i]],inplace=True)
else:
    print('please input negative or positive')
    sys.exit()

#Calculate success rates  and enrichment factors
top1success = dec(float(Top1['success'].sum()) / float(Top1.shape[0]), 3) * 100
top5success = dec(float(Top5['success'].sum()) / float(Top5.shape[0]), 3) * 100
top10success = dec(float(Top10['success'].sum()) / float(Top10.shape[0]),
                   3) * 100
ef1factor = dec(float(EF1['enrichment'].sum()) / float(EF1.shape[0]), 2)
ef5factor = dec(float(EF5['enrichment'].sum()) / float(EF5.shape[0]), 2)
ef10factor = dec(float(EF10['enrichment'].sum()) / float(EF10.shape[0]), 2)

if os.path.exists('cstemp'):
    os.remove('cstemp')
if os.path.exists('tstemp'):
    os.remove('tstemp')
#Print the output of forward screening power evluation
tmplen = len((Top1))
forwardf.style.set_properties(align="right")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(forwardf)
print(
    "\nSummary of the forward screening power: ========================================================="
)
print("Average enrichment factor among top 1%% = %0.2f" % (ef1factor))
print("Average enrichment factor among top 5%% = %0.2f" % (ef5factor))
print("Average enrichment factor among top 10%% = %0.2f" % (ef10factor))
print(
    "The best ligand is found among top 1%% candidates for %2d cluster(s); success rate = %0.1f%%"
    % (Top1['success'].sum(), top1success))
print(
    "The best ligand is found among top 5%% candidates for %2d cluster(s); success rate = %0.1f%%"
    % (Top5['success'].sum(), top5success))
print(
    "The best ligand is found among top 10%% candidates for %2d cluster(s); success rate = %0.1f%%"
    % (Top10['success'].sum(), top10success))
print(
    "================================================================================================"
)
print(
    "\nTemplate command for running the bootstrap in R program=========================================\n"
)
print(
    "rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_Top1.results\",header=TRUE);\ndata<-as.matrix(data_all[,2]);"
    % (out))
print("mymean<-function(x,indices) sum(x[indices])/%d;" % (tmplen))
print(
    "data.boot<-boot(aa,mymean,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s_Top1-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"
    % (out))
print(
    "===============================================================================================\n"
)

Top1.to_csv(out + '_Top1.dat', sep='\t', index_label='#Target')
Top5.to_csv(out + '_Top5.dat', sep='\t', index_label='#Target')
Top10.to_csv(out + '_Top10.dat', sep='\t', index_label='#Target')
EF1.to_csv(out + '_EF1.dat', sep='\t', index_label='#Target')
EF5.to_csv(out + '_EF5.dat', sep='\t', index_label='#Target')
EF10.to_csv(out + '_EF10.dat', sep='\t', index_label='#Target')
