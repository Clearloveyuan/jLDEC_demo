function [X] = newcode(M,Iter,d,k,lambda,LI,LP)
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


ro=0.0001;
[n,~]=size(M);



[b,bb,f]=svds(M,d);
B=(b*sqrt(bb));
F=(sqrt(bb)*f');

%Initialization Lagrangian multiplier 
LE1 = zeros(n,n);
LE2 = zeros(n,d);
LZ= LE2;
LZ1= LE2';
LZ2 = zeros(k,d);


%Initialization X,Y,Z
[x,xx,y]=svds(B,k);
X=(x*sqrt(xx));
X(find(X<0))=0;
Y=(sqrt(xx)*y');
Y(find(Y<0))=0;
Z = B-LZ/ro-lambda/ro*(LI-LP)*B;
Z(find(Z<0))=0;
Z1 = F-LZ1/ro;
Z1(find(Z1<0))=0;
Z2 = Y-LZ2/ro;
Z2(find(Z2<0))=0;

i=1;
% Updata Process
for  i=1:Iter
    %Update E1
     E1=M-B*F-(LE1/ro);
     E1(find(abs(E1)<sqrt(ro/2)))=0;
     %Update E2
     E2=B-X*Y-(LE2/ro);
     E2(find(abs(E2)<sqrt(ro/2)))=0;
     %Update B 
     B=1/3*((E2+X*Y+LE2/ro)+(Z+LZ/ro)-(E1-M+LE1/ro)*F'-lambda/ro*(LI-LP)*Z);
     B(find(B<0))=0;
     %Update F
     N=Z1+LZ1/ro-B'*(E1-M+LE1/ro);
     [left,bb,right]=svds(N,d);
     F=left*bb*right';
     F(find(F<0))=0;
     %Update Y
     N2 = Z2+LZ2/ro-X'*(E2-B+LE2/ro);
     [left2,bbb,right2]=svds(N2,k);
     Y=left2*bbb*right2';
     Y(find(Y<0))=0;
     %Update X
     X=(B-E2-LE2/ro)*Y';
     X(find(X<0))=0;
     %Update Z
     Z = B-LZ/ro-lambda/ro*(LI-LP)*B;
     Z(find(Z<0))=0;
     % Update other variables
     Z1 = F-LZ1/ro;
     Z1(find(Z1<0))=0;
     Z2 = Y-LZ2/ro;
     Z2(find(Z2<0))=0;
     LE1 = LE1+ro*(E1-M+B*F);
     LE1(find(LE1<0))=0;
     LE2 = LE2 +ro*(E2-B+X*Y);
     LE2(find(LE2<0))=0;
     LZ=LZ+ro*(Z-B);
     LZ(find(LZ<0))=0;
     LZ1=LZ1+ro*(Z1-F);
     LZ1(find(LZ1<0))=0;
     LZ2=LZ2+ro*(Z2-Y);
     LZ2(find(LZ2<0))=0;
     ro=ro*1.1;
     if ro>1000
         ro=1000;
     end
end


end











