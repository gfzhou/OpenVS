#rm(list=ls());
#require(boot);
#data_all<-read.table("GALigandDock-Reverse-Score-Simple-RelaxLig_Top1.dat",header=TRUE);
#data<-as.matrix(data_all[,2]);
#mymean<-function(x,indices) sum(x[indices])/57;
#data.boot<-boot(data,mymean,R=10000,stype="i",sim="ordinary");
#sink("GALigandDock-Reverse-Score-Simple-RelaxLig_Top1-ci.dat");
#a<-boot.ci(data.boot,conf=0.9,type=c("bca"));
#print(a);
#sink();

#rm(list=ls());
#require(boot);
#data_all<-read.table("./GALigandDock-Score-Simple-RelaxLig_EF1.dat",header=TRUE);
#data<-as.matrix(data_all[,2]);
#mymean<-function(x,indices) sum(x[indices])/57;
#data.boot<-boot(data,mymean,R=10000,stype="i",sim="ordinary");
#sink("GALigandDock-Score-Simple-RelaxLig_EF1-ci.results");
#a<-boot.ci(data.boot,conf=0.9,type=c("bca"));
#print(a);
#sink();

#rm(list=ls());
#require(boot);
#data_all<-read.table("./GALigandDock-Score-Simple-RelaxLig-no-cst_EF1.dat",header=TRUE);
#data<-as.matrix(data_all[,2]);
#mymean<-function(x,indices) sum(x[indices])/57;
#data.boot<-boot(data,mymean,R=10000,stype="i",sim="ordinary");
#sink("GALigandDock-Score-Simple-RelaxLig-fix-cst_EF1-ci.results");
#a<-boot.ci(data.boot,conf=0.9,type=c("bca"));
#print(a);
#sink();

rm(list=ls());
require(boot);
data_all<-read.table("./GALigandDock-Score-MCEntropy-RelaxLig-no-cst_EF1.dat",header=TRUE);
data<-as.matrix(data_all[,2]);
mymean<-function(x,indices) sum(x[indices])/57;
data.boot<-boot(data,mymean,R=10000,stype="i",sim="ordinary");
sink("GALigandDock-Score-MCEntropy-RelaxLig-no-cst_EF1-ci.results");
a<-boot.ci(data.boot,conf=0.9,type=c("bca"));
print(a);
sink();

