function [X,err,y] = l21(M,Iter,k,lambda,LI,LP)
% --Input  
%   --M is a [n,n] PMI matrix, n is node number, T is the total time
%   --k represents dimension we select default 30
%   --lambda is a parameter  default 1
%   --d is a embedding dimension  default 100
%   --Iter is a parameter represents the iteration times default 300
%   --LI 蕴含图的拉普拉斯矩阵
%   --LP惩罚图的拉普拉斯矩阵

% --Output
%   --X represent clustering index matrix 

if nargin>7
	error('parameter is too much,the max number of parameter is 6');
end
switch nargin
	case 1
		Iter=300;
		k=30;
		d=100;
		lambda=1;
	case 2
		k=30;
		d=100;
		lambda=1;
	case 3 
		d=100;
		lambda=1;
	case 4
		lambda=1;
end



M=full(M);
%%Initialization 
%%PCA initialize B
[a,aa]=pca(M);
B=aa;
[n,d]=size(B);
F=a';
%[n,~]=size(M);
%[a,aa,aaa]=svds(M,d);
%B=abs(a*sqrt(aa));
%F=abs(sqrt(aa)*aaa');


%%CAN initialize X
%[y, ~, ~] = CAN(B',k,10);

%%K-means initialize X
%[y,~]=kmeans(B,k,'Replicates',1000);
[y,~]=kmeans(B,k);
%%num*1 cluster indicator vector
X=zeros(n,k);
for o=1:n
    ll=y(o,1);
    X(o,ll)=1;
end
Y=zeros(k,d);
%[c,cc,ccc]=svds(B,k);
%X=abs(c*sqrt(cc));
%Y=abs(sqrt(cc)*ccc');

err=zeros(Iter,1);

%Initialize Lagrangian multiplier
LE1=zeros(n,n);
LE2=zeros(n,d);
LZ1=zeros(n,d);
mu=0.00001;
ro=1.1;



E1=zeros(n,n);
E2=zeros(n,d);
Z1=zeros(n,d);

%normalize
LIP=LI-LP;
ssu=sum(abs(LIP),2);
for o=1:n
    LIP(o,:)=LIP(o,:)/(ssu(o,1)*ssu(o,1));
end




% Updata Process
for  i=1:Iter
    %Update E1
    T1=B*F-mu-LE1/mu;
    for o1=1:n
        if norm(T1(:,o1),2)>1/mu
            E1(:,o1)=(1-1/(mu*norm(T1(:,o1),2)))*T1(:,o1);
        else
            E1(:,o1)=zeros(n,1);
        end
    end
    %Update E2
    T2=X*Y-B-LE2/mu;
    for o2=1:d
        if norm(T2(:,o2),2)>1/mu
            E2(:,o2)=(1-1/(mu*norm(T2(:,o2),2)))*T2(:,o2);
        else
            E2(:,o2)=0;
        end
    end   
     %Update Y
     Y=X'*(B-E2+LE2/mu);
     Y(find(Y<0))=0;
     %Update B
     B=1/3*((X*Y+E2-LE2/mu)+(Z1-LZ1/mu)+(M-E1+LE1/mu)*F'-lambda/mu*(LIP)*Z1);
     B(find(B<0))=0;
     %Update Z1
     Z1=-lambda/mu*(LIP)'*B+B+LZ1/mu;
    %Z1(find(Z1<0))=0;
     %Update F
     T3=M-E1+LE1/mu;
     [bb,~,bbb]=svd(B'*T3,'econ');
     F=bb*bbb';
     %Update X
     T4=B-E2+LE2/mu;
     [cc,~,ccc]=svd(T4*Y','econ');
     X=cc*ccc';
     X(find(X<0))=0;
     %Update other variables
     LE1=mu*(-M+B*F+E1)/4;
     LE2=mu*(-B+X*Y+E2)/4;
     LZ1=mu*(-B+Z1)/4;
     mu=ro*mu;
     if mu>10
         mu=10;
     end
     err(i,1)=norm(M-B*F,'fro')+norm(B-X*Y,'fro');
end
%kl=min(err);
%for o6=1:Iter
   % err(o6,1)=(err(o6,1)-kl)/(err(1,1)-kl);
%end


end

     
     
     
     
     
     
     
     
   