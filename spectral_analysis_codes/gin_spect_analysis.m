clear all
clc


%epsilons
E=[-2 -1 0 1];


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




for i=1:length(E)
    e=E(i);
    C=W+(1+e)*eye(size(W));
    B=u'*C*u;    
    b(:,i)=diag(B);    
end

figure;hold on;
for i=1:length(E)
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
for i=1:length(E)
    t{i}=p*((1+E(i))/p+1-v);
end

figure;
for i=1:length(E)
    subplot(2,2,i);hold on;
    plot(v,abs(b(:,i)),'b-');
    plot(v,abs(t{i}),'r-');
    legend({'empirical','theoretical'});
    xlabel('Eigenvalue');
    title(['epsilon=' num2str(E(i))]);
end

