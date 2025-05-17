function out = SliceProduct(ten1,ten2)
[n1,n2,n3]=size(ten1);
[n2,n4,n3]=size(ten2);
out = zeros(n1,n4,n3);
for i=1:n3
    out(:,:,i)=ten1(:,:,i)*ten2(:,:,i);
end