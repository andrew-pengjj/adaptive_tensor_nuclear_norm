function out = COMT(ten,L)
[n1,n2,n3]=size(ten);
mat = reshape(ten,[n1*n2,n3]); % mat local smoothness
[r3,~]=size(L);
out = reshape(mat*L',[n1,n2,r3]);