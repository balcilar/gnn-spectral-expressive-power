close all
clear all
load coraA
A=double(A);
n=size(A,1);

d = sum(A,2);
% normalized laplacien
dis=1./sqrt(d);
dis(isinf(dis))=0;
dis(isnan(dis))=0;
D=diag(dis);
nL=eye(n)-(A*D)'*D;
[u v]=eig(nL);
% make eignevalue as vector
v=diag(v);

%A0=(A+eye(n))*(A+eye(n));
A0=A+eye(n);
%b=zeros(2708,2708,250);
bb=zeros(2708,250);
for i=1:250
    i
    W=randn(1433,8);
    w1=randn(8,1);
    w2=randn(8,1);
    f=F*W;
    f1=f*w1;
    f2=f*w2;
    ff=f1+f2';
    
    fff=max(0,ff)-0.2*max(0,-ff);
    
    ff=double(exp(-fff)); 
    ff=double((ff)); 
    ff=ff.*(A0);   
    qq=ff./sum(ff')';    
    bb(:,i)=diag(u'*qq*u);
end

gat=[];
ov=0:0.01:2;
for i=1:size(bb,2)
    gat(i,:)=interp1(v+0.0001*randn(size(v)),bb(:,i),ov);    
end

figure;plot(ov,abs(gat'))
title('Gat empirical freq response of each simulation on cora');
xlabel('Eigenvalue');
ylabel('Magnitude');



