function [C, ind] = CSSP(X, k)
addpath("C:\Users\admin\OneDrive\科研\程序\Matlab\Subset Selection\rrqr")
% initial 
n = size(X, 2);
[~,~,V] = svd(X);
Vk = V(:, 1:k);

XVVt = X*(Vk*Vk');
Delta = 2*(norm(X,"fro")^2 - norm(XVVt,"fro")^2);

p = zeros(1, n);
for i=1:n
    norm2_Vki = norm(Vk(i,:))^2;
    p(i) = norm2_Vki/(2*k) + (norm(X(:,i))^2 - norm(XVVt(:,i))^2)/Delta;
end

c = 0.8*k*log(k);

% randomize stage
ind = [];
rescale = [];
for i=1:n
    q = min([1, c*p(i)]);
    q = q(1);
    if q == 1
        ind = [ind i];
        rescale = [rescale sqrt(q)];
    elseif q > rand
        ind = [ind i];
        rescale = [rescale sqrt(q)];
    end
end

len = length(ind);
S1 = zeros(n, len);
D1 = zeros(len, len);
for i=1:len
    S1(ind(i), i) = 1;
    D1(i, i) = 1/ rescale(i);
end

% deterministic stage
VtSD = Vk'*S1*D1;
[Q,R,p,r] = rrqry(VtSD,0);

S2 = zeros(length(p), k);
for i=1:min(k,length(p))
    S2(p(i), i) = 1;
end

C = X*S1*S2;