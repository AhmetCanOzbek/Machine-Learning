function [matrix] = deleteRows(matrix, rowsToBeDeleted)

rows = 1:size(matrix,1);

    for i=1:max(size(rowsToBeDeleted))
        rows = rows(rows ~= rowsToBeDeleted(i));
    end
    
    matrix = matrix(rows,:);

end