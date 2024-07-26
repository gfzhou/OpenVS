#!/usr/bin/python
import numpy as np
import sys, os
import pandas as pd
import scipy
import getopt
import string
from decimal import *

if len(sys.argv) < 2:
    print("Please input parameter files or use -h for help")
    sys.exit()


def usage():
    print( "-c or --coreset: specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package")
    print( "-s or --score: specify the directory containing your scoring files(e.g. 'XXXX_score.dat'). Remember the 1st column name is #code and the 2nd column name is score. Supported file separators are comma(,), tabs(\\t) and space character( )")
    print( "-p or --prefer: input 'negative' or 'positive' string, depend on your scoring funtion preference")
    print( "-r or --rmsd: specify the directory containing the RMSD data files(e.g. 'XXXX_rmsd.dat')")
    print( "-o or --output: input the prefix of output result files. Default is My_Docking_Power")
    print( "-h or --help: print help message")
    print( "-l or --limit: set the RMSD cutoff (in angstrom) to define near-native docking pose")
    print( "\nExample: python docking_power.py -c CoreSet.dat -s ./examples/X-Score -r ../decoys_docking/ -p 'positive' -l 2 -o 'X-Score' > MyDockingPower.out")


try:
    options, args = getopt.getopt(
        sys.argv[1:], "hc:s:r:p:l:o:",
        ["help", "coreset=", "score=", "rmsd=", "prefer=", "limit=", "output"])
except getopt.GetoptError:
    sys.exit()

#Read the CoreSet.dat, scoring result files and rmsd result files
out = 'My_Docking_Power'
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
    if name in ("-r", "--rmsd"):
        rmsdfile = value
    if name in ("-p", "--prefer"):
        fav = value
    if name in ("-l", "--limit"):
        cut = float(value)
    if name in ("-o", "--output"):
        out = value


def dec(x, y):
    if y == 2:
        return Decimal(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if y == 3:
        return Decimal(x).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    if y == 4:
        return Decimal(x).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)


#process data
pdb = aa['#code']
Top1 = pd.DataFrame(index=pdb, columns=['success'])
Top2 = pd.DataFrame(index=pdb, columns=['success'])
Top3 = pd.DataFrame(index=pdb, columns=['success'])
SP2 = pd.DataFrame(index=pdb, columns=['spearman'])
SP3 = pd.DataFrame(index=pdb, columns=['spearman'])
SP4 = pd.DataFrame(index=pdb, columns=['spearman'])
SP5 = pd.DataFrame(index=pdb, columns=['spearman'])
SP6 = pd.DataFrame(index=pdb, columns=['spearman'])
SP7 = pd.DataFrame(index=pdb, columns=['spearman'])
SP8 = pd.DataFrame(index=pdb, columns=['spearman'])
SP9 = pd.DataFrame(index=pdb, columns=['spearman'])
SP10 = pd.DataFrame(index=pdb, columns=['spearman'])
dockresults = pd.DataFrame(
    index=range(1,
                len(pdb) + 1),
    columns=['code', 'Rank1', 'RMSD1', 'Rank2', 'RMSD2', 'Rank3', 'RMSD3'])
tmp = 1
if fav == 'positive':
    for i in pdb:
        rmsddf = pd.read_csv(rmsdfile + '/' + str(i) + '_rmsd.dat',
                             sep='[,, ,\t]+',
                             engine='python')
        scoredf = pd.read_csv(scorefile + '/' + str(i) + '_score.dat',
                              sep='[,, ,\t]+',
                              engine='python')
        testdf = pd.merge(rmsddf, scoredf, on='#code')
        #	 	dfsorted=testdf.sort_values(by=['score','rmsd'],ascending=[False,True])
        dfsorted = testdf.sort_values(by=['score'], ascending=[False])
        dockresults.loc[tmp]['Rank1'] = ''.join(dfsorted[0:1]['#code'])
        dockresults.loc[tmp]['RMSD1'] = float(dfsorted[0:1]['rmsd'])
        dockresults.loc[tmp]['Rank2'] = ''.join(dfsorted[1:2]['#code'])
        dockresults.loc[tmp]['RMSD2'] = float(dfsorted[1:2]['rmsd'])
        dockresults.loc[tmp]['Rank3'] = ''.join(dfsorted[2:3]['#code'])
        dockresults.loc[tmp]['RMSD3'] = float(dfsorted[2:3]['rmsd'])
        dockresults.loc[tmp]['code'] = i
        tmp += 1
        lst = list(dfsorted.index)
        for j in np.arange(1, 4):
            minrmsd = dfsorted[0:j]['rmsd'].min()
            varname = 'Top' + str(j)
            top = locals()[varname]
            if minrmsd <= cut:
                top.loc[i]['success'] = 1
            else:
                top.loc[i]['success'] = 0
        for s in np.arange(2, 11):
            sptemp = testdf[testdf.rmsd <= s]
            varname2 = 'SP' + str(s)
            sp = locals()[varname2]
            if float(sptemp.shape[0]) >= 5:
                sp.loc[i]['spearman'] = np.negative(
                    sptemp['rmsd'].corr(sptemp['score'], method='spearman'))
            else:
                continue
elif str(fav) == 'negative':
    for i in pdb:
        rmsddf = pd.read_csv(rmsdfile + '/' + str(i) + '_rmsd.dat',
                             sep='[,, ,\t]+',
                             engine='python')
        scorefn = os.path.join(scorefile, f"{i}_score.dat")
        if not os.path.exists(scorefn): continue
        scoredf = pd.read_csv(scorefn,
                              sep='[,, ,\t]+',
                              engine='python')
        testdf = pd.merge(rmsddf, scoredf, on='#code')
        #                dfsorted=testdf.sort_values(['score','rmsd'],ascending=[True,True])
        dfsorted = testdf.sort_values(by=['score'], ascending=[True])
        dockresults.loc[tmp]['Rank1'] = ''.join(dfsorted[0:1]['#code'])
        dockresults.loc[tmp]['RMSD1'] = float(dfsorted[0:1]['rmsd'])
        dockresults.loc[tmp]['Rank2'] = ''.join(dfsorted[1:2]['#code'])
        dockresults.loc[tmp]['RMSD2'] = float(dfsorted[1:2]['rmsd'])
        dockresults.loc[tmp]['Rank3'] = ''.join(dfsorted[2:3]['#code'])
        dockresults.loc[tmp]['RMSD3'] = float(dfsorted[2:3]['rmsd'])
        dockresults.loc[tmp]['code'] = i
        tmp += 1
        lst = list(dfsorted.index)
        for j in np.arange(1, 4):
            minrmsd = dfsorted[0:j]['rmsd'].min()
            varname = 'Top' + str(j)
            top = locals()[varname]
            if minrmsd <= cut:
                top.loc[i]['success'] = 1
            else:
                top.loc[i]['success'] = 0
        for s in np.arange(2, 11):
            sptemp = testdf[testdf.rmsd <= s]
            varname2 = 'SP' + str(s)
            sp = locals()[varname2]
            if float(sptemp.shape[0]) >= 5:
                sp.loc[i, 'spearman'] = sptemp['rmsd'].corr(sptemp['score'], method='spearman')
            else:
                continue
#                                sp.drop(sp.index[[i]],inplace=True)
else:
    print('please input negative or positive')
    sys.exit()
#Calculate success rates and spearman correlation coefficient
SP2 = SP2.dropna(subset=['spearman'])
SP3 = SP3.dropna(subset=['spearman'])
SP4 = SP4.dropna(subset=['spearman'])
SP5 = SP5.dropna(subset=['spearman'])
SP6 = SP6.dropna(subset=['spearman'])
SP7 = SP7.dropna(subset=['spearman'])
SP8 = SP8.dropna(subset=['spearman'])
SP9 = SP9.dropna(subset=['spearman'])
SP10 = SP10.dropna(subset=['spearman'])
top1success = dec(float(Top1['success'].sum()) / float(Top1.shape[0]), 3) * 100
top2success = dec(float(Top2['success'].sum()) / float(Top2.shape[0]), 3) * 100
top3success = dec(float(Top3['success'].sum()) / float(Top3.shape[0]), 3) * 100
sp2 = dec(float(SP2['spearman'].sum()) / float(SP2.shape[0]), 3)
sp3 = dec(float(SP3['spearman'].sum()) / float(SP3.shape[0]), 3)
sp4 = dec(float(SP4['spearman'].sum()) / float(SP4.shape[0]), 3)
sp5 = dec(float(SP5['spearman'].sum()) / float(SP5.shape[0]), 3)
sp6 = dec(float(SP6['spearman'].sum()) / float(SP6.shape[0]), 3)
sp7 = dec(float(SP7['spearman'].sum()) / float(SP7.shape[0]), 3)
sp8 = dec(float(SP8['spearman'].sum()) / float(SP8.shape[0]), 3)
sp9 = dec(float(SP9['spearman'].sum()) / float(SP9.shape[0]), 3)
sp10 = dec(float(SP10['spearman'].sum()) / float(SP10.shape[0]), 3)
tmplen = len(Top1)
if os.path.exists('cstemp'):
    os.remove('cstemp')

#Print the output of docking power evluation
dockresults['RMSD1'] = dockresults['RMSD1'].map(lambda x: ('%.2f') % x)
dockresults['RMSD2'] = dockresults['RMSD2'].map(lambda x: ('%.2f') % x)
dockresults['RMSD3'] = dockresults['RMSD3'].map(lambda x: ('%.2f') % x)
dockresults.style.set_properties(align="right")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(dockresults)
print(
    "\nSummary of the docking power: ========================================")
print("Among the top1 binding pose ranked by the given scoring function:")
print("Number of correct binding poses = %d, success rate = %0.1f%%" %
      (Top1['success'].sum(), top1success))
print("Among the top2 binding pose ranked by the given scoring function:")
print("Number of correct binding poses = %d, success rate = %0.1f%%" %
      (Top2['success'].sum(), top2success))
print("Among the top3 binding pose ranked by the given scoring function:")
print("Number of correct binding poses = %d, success rate = %0.1f%%" %
      (Top3['success'].sum(), top3success))
print("Spearman correlation coefficient in rmsd range [0-2]: %0.3f" %
      (dec(sp2, 3)))
print("Spearman correlation coefficient in rmsd range [0-3]: %0.3f" %
      (dec(sp3, 3)))
print("Spearman correlation coefficient in rmsd range [0-4]: %0.3f" %
      (dec(sp4, 3)))
print("Spearman correlation coefficient in rmsd range [0-5]: %0.3f" %
      (dec(sp5, 3)))
print("Spearman correlation coefficient in rmsd range [0-6]: %0.3f" %
      (dec(sp6, 3)))
print("Spearman correlation coefficient in rmsd range [0-7]: %0.3f" %
      (dec(sp7, 3)))
print("Spearman correlation coefficient in rmsd range [0-8]: %0.3f" %
      (dec(sp8, 3)))
print("Spearman correlation coefficient in rmsd range [0-9]: %0.3f" %
      (dec(sp9, 3)))
print("Spearman correlation coefficient in rmsd range [0-10]: %0.3f" %
      (dec(sp10, 3)))
print(
    "======================================================================\n")
print(
    "\nTemplate command for running the bootstrap in R program===============\n"
)
print(
    "rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_Top1.results\",header=TRUE);\ndata<-as.matrix(data_all[,2]);"
    % (out))
print("mymean<-function(x,indices) sum(x[indices])/%d;" % (tmplen))
print(
    "data.boot<-boot(aa,mymean,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s_Top1-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"
    % (out))
print(
    "========================================================================\n"
)

Top1.to_csv(out + '_Top1.dat', sep='\t')
Top2.to_csv(out + '_Top2.dat', sep='\t')
Top3.to_csv(out + '_Top3.dat', sep='\t')
