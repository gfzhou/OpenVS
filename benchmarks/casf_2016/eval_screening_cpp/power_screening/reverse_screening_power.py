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
        "-l or --ligandinfo: specify the location of 'LigandInfo.dat' in the CASF-2016 package"
    )
    print(
        "-o or --output: input the prefix of output result files. Default is My_Reverse_Screening_Power"
    )
    print("-h or --help: print help message")
    print(
        "\nExample: python reverse_screening_power.py -c CoreSet.dat -s ./examples/X-Score -l ./LigandInfo.dat -p 'positive' -o 'X-Score' > MyReverseScreeningPower.out"
    )


try:
    options, args = getopt.getopt(
        sys.argv[1:], "hc:s:l:p:o:",
        ["help", "coreset=", "score=", "ligandinfo=", "prefer=", "output="])
except getopt.GetoptError:
    sys.exit()

out = 'My_Reverse_Screening_Power'
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
        aa = pd.read_csv('cstemp', sep='[,,\t, ]+', engine='python')
        aa = aa.drop_duplicates(subset=['#code'], keep='first')
    if name in ("-s", "--score"):
        scorefile = value
    if name in ("-l", "--ligandinfo"):
        f = open(value, 'r')
        f1 = open('tstemp', 'w+')
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
        bb = pd.read_csv('tstemp', sep='[,,\t, ]+', engine='python')
        bb = bb.drop_duplicates(subset=['#code'], keep='first')
        cc = bb.set_index('#code')
    if name in ("-p", "--prefer"):
        fav = value
    if name in ("-o", "--ouput"):
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
#pdb=aa['code']
pdb = []
pdbtmp = aa['#code']
codelst = []
targettmp = cc.groupby('group')
targetlst = []
for n, m in targettmp.__iter__():
    if len(targettmp.get_group(n)) == 5:
        targetlst.extend(targettmp.get_group(n)['T1'].tolist())
        codelst.extend(targettmp.get_group(n).index.tolist())
targetlst = list(set(targetlst))
for c in pdbtmp:
    if c in codelst:
        pdb.append(c)
t1 = int(dec(len(targetlst) * 0.01, 0))
t5 = int(dec(len(targetlst) * 0.05, 0))
t10 = int(dec(len(targetlst) * 0.10, 0))
if t1 < 1:
    print("The number of top 1%% targets is %0.1f,less than 1" %
          (round(len(decoylst) * 0.01)))
    print("In this case, we set the cutoff of top 1%% = 1")
    t1 = 1
if t5 < 1:
    print("The number of top 5%% targets is %0.1f,less than 1" %
          (round(len(decoylst) * 0.05)))
    print("In this case, we set the cutoff of top 5%% = 1")
    t5 = 1
if t10 < 1:
    print("The number of top 10%% targets is %0.1f,less than 1" %
          (round(len(decoylst) * 0.10)))
    print("In this case, we set the cutoff of top 10%% = 1")
    t10 = 1

#toptardf=aa.groupby('target').apply(top)
#targetlst=toptardf['code'].tolist()
Top1 = pd.DataFrame(index=pdb, columns=['success'])
Top5 = pd.DataFrame(index=pdb, columns=['success'])
Top10 = pd.DataFrame(index=pdb, columns=['success'])
Ligdf = pd.DataFrame()
reversedf = pd.DataFrame(index=range(1,
                                     len(pdb) + 1),
                         columns=range(0, t10 + 1))
reversedf.rename(columns={0: 'code'}, inplace=True)
tmp = 1
dic = {'1': t1, '5': t5, '10': t10}
if fav == 'positive':
    for i in targetlst:
        scoredf = pd.read_csv(scorefile + '/' + str(i) + '_score.dat',
                              sep='[ ,_,\t]+',
                              engine='python')
        group = scoredf.groupby('#code')
        testdf = pd.DataFrame(group['score'].max())
        testdf['T1'] = i
        Ligdf = pd.concat([Ligdf, testdf])
    Ligdf['ligname'] = list(Ligdf.index)
    grouplig = Ligdf.groupby('ligname')
    for l in pdb:
        grouptemp = grouplig.get_group(l)
        dfsorted = grouptemp.sort_values('score', ascending=False)
        for m in range(1, t10 + 1):
            reversedf.loc[tmp][m] = ''.join(dfsorted[m - 1:m]['T1'].tolist())
        reversedf.loc[tmp]['code'] = l
        tmp += 1
        Toptar = cc.loc[l]['T1']
        for name, j in dic.items():
            lst = list(dfsorted[0:j]['T1'])
            varname = 'Top' + str(name)
            top = locals()[varname]
            if Toptar in lst:
                top.loc[l]['success'] = 1
            else:
                top.loc[l]['success'] = 0
elif str(fav) == 'negative':
    for i in targetlst:
        scoredf = pd.read_csv(scorefile + '/' + str(i) + '_score.dat',
                              sep='[ ,_,\t]+',
                              engine='python')
        group = scoredf.groupby('#code')
        testdf = pd.DataFrame(group['score'].min())
        testdf['T1'] = i
        Ligdf = pd.concat([Ligdf, testdf])
    Ligdf['ligname'] = list(Ligdf.index)
    grouplig = Ligdf.groupby('ligname')
    for l in pdb:
        grouptemp = grouplig.get_group(l)
        dfsorted = grouptemp.sort_values('score', ascending=True)
        for m in range(1, t10 + 1):
            reversedf.loc[tmp][m] = ''.join(dfsorted[m - 1:m]['T1'].tolist())
        reversedf.loc[tmp]['code'] = l
        tmp += 1
        Toptar = cc.loc[l]['T1']
        for name, j in dic.items():
            lst = list(dfsorted[0:j]['T1'])
            varname = 'Top' + str(name)
            top = locals()[varname]
            if Toptar in lst:
                top.loc[l]['success'] = 1
            else:
                top.loc[l]['success'] = 0

#                                sp.drop(sp.index[[i]],inplace=True)
else:
    print('please input negative or positive')
    sys.exit()
if os.path.exists('cstemp'):
    os.remove('cstemp')
if os.path.exists('tstemp'):
    os.remove('tstemp')

#Calculate success rates
reversedf.style.set_properties(align="right")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(reversedf)
top1success = dec(float(Top1['success'].sum()) / float(Top1.shape[0]), 3) * 100
top5success = dec(float(Top5['success'].sum()) / float(Top5.shape[0]), 3) * 100
top10success = dec(float(Top10['success'].sum()) / float(Top10.shape[0]),
                   3) * 100
tmplen = len(Top1)

print(
    "\nSummary of the reverse screening power: ========================================================="
)
print(
    "The best target is found among top 1%% candidates for %2d ligand(s); success rate = %0.1f%%"
    % (Top1['success'].sum(), top1success))
print(
    "The best target is found among top 5%% candidates for %2d ligand(s); success rate = %0.1f%%"
    % (Top5['success'].sum(), top5success))
print(
    "The best target is found among top 10%% candidates for %2d ligand(s); success rate = %0.1f%%"
    % (Top10['success'].sum(), top10success))
print(
    "================================================================================================="
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

Top1.to_csv(out + '_Top1.dat', sep='\t', index_label='#code')
Top5.to_csv(out + '_Top5.dat', sep='\t', index_label='#code')
Top10.to_csv(out + '_Top10.dat', sep='\t', index_label='#code')
