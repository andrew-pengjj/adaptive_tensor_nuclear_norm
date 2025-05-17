function [outer_matrix] =generate_TS(n1,n2,n3,k_num)

% Oct 2021
% written by Jiangjun Peng

outer_matrix = zeros(n1,n2,n3);
rand_index   = randperm(n1*n2*n3);
choose_index = rand_index(1:k_num);
outer_matrix(choose_index(1:round(k_num/2)))=1;
outer_matrix(choose_index(round(k_num/2)+1:k_num))=-1;
    
