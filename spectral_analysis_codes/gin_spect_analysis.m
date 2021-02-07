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


%gin-0
E=[-2 -1 0 1];

for i=1:4
    e=E(i);
    C=W+(1+e)*eye(size(W));
    B=u'*C*u;    
    b(:,i)=diag(B);    
end

figure;hold on;
for i=1:4 
    l{i}=num2str(i);   
    stem3(v,0-0.4*i*ones(length(v),1),abs(b(:,i)));
end
grid on;

view(46,38)
title(' empirical freq response of GIN on Cora');
xlabel('eigenvalue');
ylabel('epsilon');
zlabel('magnitude');

% theoretical freq response
p=mean(sum(A));
t1=p*((1+E(1))/p+1-v);
t2=p*((1+E(2))/p+1-v);
t3=p*((1+E(3))/p+1-v);
t4=p*((1+E(4))/p+1-v);

figure;
subplot(2,2,1);hold on;
plot(v,abs(b(:,1)),'b-');
plot(v,abs(t1),'r-');
legend({'empirical','theoretical'});
xlabel('Eigenvalue');
title('epsilon= -2');

subplot(2,2,2);hold on;
plot(v,abs(b(:,2)),'b-');
plot(v,abs(t2),'r-');
legend({'empirical','theoretical'});
xlabel('Eigenvalue');
title('epsilon= -1');

subplot(2,2,3);hold on;
plot(v,abs(b(:,3)),'b-');
plot(v,abs(t3),'r-');
legend({'empirical','theoretical'});
xlabel('Eigenvalue');
title('epsilon= 0');

subplot(2,2,4);hold on;
plot(v,abs(b(:,4)),'b-');
plot(v,abs(t4),'r-');
legend({'empirical','theoretical'});
xlabel('Eigenvalue');
title('epsilon= 1');


