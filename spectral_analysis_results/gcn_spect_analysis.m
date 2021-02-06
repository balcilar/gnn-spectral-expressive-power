clear all
clc


load coraA;
W=double(A);
n=size(W,1);

% normalized laplacien
d = sum(W,2);
dis=1./sqrt(d);
dis(isinf(dis))=0;
dis(isnan(dis))=0;
D=diag(dis);
L=eye(n)-(W*D)'*D;
[u v]=eig(L);
% make eignevalue as vector
v=diag(v);

% calculate GCN support
nA=W+1*eye(n);
rowsum=sum(nA,1);
rowsum(isinf(rowsum))=0;
rowsum(isnan(rowsum))=0;
Dm=diag(rowsum.^(-0.5));
gcn1=Dm*nA*Dm;
% freq response
B=u'*gcn1*u;
figure;mesh(v,v,abs(B))
xlim([0 2]);ylim([0 2]);
view(-21,27)
title('Full empirical freq response of GCN on Cora');
xlabel('eigenvalue');
ylabel('eigenvalue');
zlabel('magnitude');

% theoretical freq response
figure;
hold on;plot(v,abs(diag(B)),'b-')
p=mean(sum(A));
tfr=1-v*p/(p+1);
plot(v,abs(tfr),'r-')
legend({'empirical freq response','theoretical freq response'});
title('empirical and theoretical freq response of GCN on Cora');


