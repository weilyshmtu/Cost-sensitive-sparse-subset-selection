function [Z,  elapse] = cs4_v1(X, lambda1, lambda2, q, report)
% cost-sensitive subset selection
% min_{Z}||X - X*Z||_s + lambda_1||Z||_{1,q} + lambda_2||D.*Z||_1
% where D = 1/|preZ|, namely D_ij = 1/|Z_ij|, q = 1,2, inf
%                         ||
%                         \/
% min_{Z}||X - X*Z||_s + lambda_1||C1||_{1,q} + lambda_2||D.*C2||_1
% s.t. Z = C1, Z = C2, eZ = e, E = X - X*Z


if nargin < 5
    report = true;
end
t0 = cputime;

% parameters
tol = 1e-6;
maxIter = 1e3;
rho = 1.1;

mu = 1e-4;
max_mu = 1e30;
eps = 1e-6;

% parameters
tol = 1e-6;
maxIter = 1e3;
rho = 1.1;

mu = 1e-4;
max_mu = 1e30;
eps = 1e-6;

% Initializing optimization variables
[d, n] = size(X);
e = ones(1,n);
I = eye(n);

XtX = X'*X;
ete = e'*e;



Y1 = zeros(n,n);
Y2 = zeros(n,n);
Y3 = zeros(1,n);

Z = zeros(n,n);
C1 = Z;
C2 = Z;

iter = 0;
while iter < maxIter
    iter = iter + 1;
    % update Z
    A = 2*XtX + mu*(2*I + ete);
    B = 2*XtX + mu*(C1 + C2 + ete)- Y1 - Y2 - e'*Y3;
    Z = A\B;

    % update C1
    A = Z + Y1/mu;
    C1 = shrinkL1Lq(A,lambda1/mu,q);

    % update C2
    D = 1./(abs(Z) + eps);
    D = D ./ max(max(D));
    A = Z + Y2/mu;
    B = lambda2/mu*D;
    C2 = max(A - B, 0) + min(A + B, 0);



    leq1 = Z - C1;
    leq2 = Z - C2;
    leq3 = e*Z - e;

    stopC = max([max(max(abs(leq1))),max(max(abs(leq2))),max(max(abs(leq3)))]);

     if (iter==1 || mod(iter,50)==0 || stopC<tol) &&  report
          disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',leq1 =' num2str(max(max(abs(leq1))), '%2.1e')  ',leq2 =' num2str(max(max(abs(leq2))), '%2.1e')  ... 
            ',leq3 =' num2str(max(max(abs(leq3))), '%2.1e')]);

    end

    if stopC<tol 
        elapse = cputime - t0;
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
end
