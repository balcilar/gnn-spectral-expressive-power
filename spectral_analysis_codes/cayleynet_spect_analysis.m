clear all
clc
close all

% set zoom parameter
h=1;
% set vmax for theoretical freq response
vmax=3;

% empricial frequency response of CayleyNet on Cora graph
load coraA;
W=double(A);
n=size(W,1);
d = sum(W,2);
dis=1./sqrt(d);
dis(isinf(dis))=0;
dis(isnan(dis))=0;
D=diag(dis);
L=eye(n)-(W*D)'*D;
[u v]=eig(L);
% make eignevalue as vector
vv=diag(v);


R=3;
v=vv;

B=ones(n,1);
for r=1:R    
    tmp1=(h*L-i*eye(n))^r;
    tmp2=(h*L+i*eye(n))^r; 
    tmp=(tmp1/tmp2);
    c1=real((1+0)*tmp);   
    c2=real((0+i)*tmp);
    c1(isnan(c1))=0;
    c2(isnan(c2))=0;
    B1=diag(u'*c1*u); 
    B2=diag(u'*c2*u); 
    B=[B B1 B2];
end
figure;hold on;
for i=1:7    
    l{i}=num2str(i);
    stem3(v,0-0.4*i*ones(length(v),1),abs(B(:,i)));
end

grid on;
axis equal
view(46,41)
title('CayleyNet empirical freq response on Cora')
xlabel('eigenvalues');
ylabel('Convolution Supports');
zlabel('Magnitude');




% theoretical frequency response of CayleyNet in range of [0 vmax]

v=0:0.001:vmax;
clear Fr Fi
teta=atan2(-1,v)-atan2(1,v);
for j=1:3
    Fr(:,j)=cos(j*teta);
    Fi(:,j)=-sin(j*teta);
end
F=zeros(size(Fr,1),7);
F(:,2)=Fr(:,1);
F(:,3)=Fi(:,1);
F(:,4)=Fr(:,2);
F(:,5)=Fi(:,2);
F(:,6)=Fr(:,3);
F(:,7)=Fi(:,3);
F(:,1)=1;

b=F;
figure;hold on;
for i=1:7
    
    l{i}=num2str(i);
    stem3(v,0-0.4*i*ones(length(v),1),abs(b(:,i)));
end
grid on;
axis equal
view(46,41)

title('CayleyNet theoretical freq response')
xlabel('eigenvalues');
ylabel('Convolution Supports');
zlabel('Magnitude');


    

    



