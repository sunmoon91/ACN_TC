%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main function of perform iterative
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function main_function(inputFileFullPath,inputFilename,num_iter,main_iter,res,w)

if main_iter==0
    FirstPreprocessing(inputFileFullPath,inputFilename,main_iter+1,res);
    ExtractPatch(inputFileFullPath,inputFilename,main_iter+1,w)        
else
    AssignLabel(inputFileFullPath,inputFilename,main_iter,res, w);
    PostProcessing(inputFileFullPath,inputFilename,main_iter,num_iter,res);
    if main_iter<num_iter
        Preprocessing(inputFileFullPath,inputFilename,main_iter+1,res);
        ExtractPatch(inputFileFullPath,inputFilename,main_iter+1,w)
    end   
end
exit();

