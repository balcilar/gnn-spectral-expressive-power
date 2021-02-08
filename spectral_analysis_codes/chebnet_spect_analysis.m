clear all
clc

% number of convolution support
S=5;


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

vmax=max(v);

% empirical chebnet kernels on Cora
C{1}=eye(n);
nL=2/vmax*L-eye(n);
C{2}=nL;
for i=3:S
    C{i}=2*nL*C{i-1}-C{i-2};
end

b=[];
l=[];
for i=1:S
    B=u'*C{i}*u;    
    b(:,i)=diag(B);
end

figure;hold on;
for i=1:S    
    l{i}=num2str(i);
    stem3(v,0-0.4*i*ones(length(v),1),abs(b(:,i)));
end

grid on;
axis equal
view(46,41)
title('Chebnet empirical freq response on Cora')
xlabel('eigenvalues');
ylabel('Convolution Supports');
zlabel('Magnitude');


% theoretical chebnet kernels on Cora
v=0:0.001:2;
b=zeros(length(v),5);
b(:,1)=1;
b(:,2)=2*v/vmax-1;
for i=3:S
    b(:,i)=2*b(:,2).*b(:,i-1)-b(:,i-2);
end
figure;hold on;
for i=1:S    
    l{i}=num2str(i);
    stem3(v,0-0.4*i*ones(length(v),1),abs(b(:,i)));
end

grid on;
axis equal
view(46,41)
title('Chebnet theoretical freq response on Cora')
xlabel('eigenvalues');
ylabel('Convolution Supports');
zlabel('Magnitude');



