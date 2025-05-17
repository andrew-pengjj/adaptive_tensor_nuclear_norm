function result = GenerateLRT(sizen,R,smooth_flag)
n1=sizen(1);
n2=sizen(2);
n3=sizen(3);
c = max(R,n3)+1;
if smooth_flag ==1
    mask = generate_mask(n1,n2,c);
    mat = randn(c,n3);
    resmat = zeros(n1*n2,n3);
    for i=1:c
        index = find(mask==i);
        resmat(index,:)= repmat(mat(i,:),[length(index),1]);
    end
end
if smooth_flag ==1
    result = reshape(resmat,[n1,n2,n3]);
else
    result = randn(n1,n2,n3);
end
for i=1:n3
    tmp = result(:,:,i);
    [u,s,v]=svd(tmp);
    result(:,:,i) = u(:,1:R)*s(1:R,1:R)*v(:,1:R)';
end

