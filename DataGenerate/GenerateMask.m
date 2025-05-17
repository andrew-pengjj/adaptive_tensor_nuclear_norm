% 创建一个示例矩阵
function mask = GenerateMask(n1,n2,r)
A = rand(n1,n2); % 这里假设原始矩阵为一个10x10的随机矩阵
n = r+2;
selected_points = zeros(n, 2);
for i = 1:n
    % 生成随机行和列索引
    random_row = randi(size(A, 1));
    random_col = randi(size(A, 2));
    selected_points(i, :) = [random_row, random_col];
end
% 计算每个点到所有选择的点的距离
distances = zeros(size(A, 1), size(A, 2), n);

for i = 1:n
    a1 = repmat(([1:size(A, 1)]' - selected_points(i, 1)).^2,[1,size(A, 2)]);
    a2 = repmat(([1:size(A, 2)] - selected_points(i, 2)).^2,[size(A, 1),1]);
    distances(:, :, i) = sqrt(a1+a2);
end
% 对每个点，确定其所在的最近的点的索引
[~, mask] = min(distances, [], 3);
end


