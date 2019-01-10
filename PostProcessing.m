%1 fill its remaining holes by using the topo-preserving levelset,
%2 clean the outlier clusters in the correct image
function PostProcessing(inputFileFullPath,inputFilename,main_iter,num_iter,res)

Result_path = [inputFileFullPath,inputFilename,'/iter',num2str(main_iter+1),'/'];
mkdir(Result_path);

% 1 output in the iter+1 fold
HalfResult_filename= [inputFilename,'_half'];%no .hdr or .img surfix
% HalfResult_filename= ['n',strNum,shpereFlag,'_half'];%no .hdr or .img surfix

% 2 processed with LS (no holes)
LSResult_filename=[inputFilename,'_halfLS'];
% LSResult_filename=['n',strNum,shpereFlag,'_halfLS'];

RawResult_fullfilename=[Result_path,HalfResult_filename];
source_filename = [Result_path,HalfResult_filename,'.hdr'];
target_filename = [Result_path,LSResult_filename,'.hdr'];
command=['TopologyCorrectionByLevelSet2 ' source_filename ' ' target_filename ' -d ' '3'];
system(command);

%% remove the outliers by simply remaining the largest connected regions, then save them to "*_half" files
%read the raw result volume
    RawResultVolume = analyze75read([RawResult_fullfilename,'.hdr']);
    BinaVolume=zeros(size(RawResultVolume));
    BinaVolume(find(RawResultVolume==250))=1;
    BinaVolume = logical(BinaVolume);
    [row,col,depth]=size(BinaVolume);

%read the levelset processed result volume
    LSResultVolume = analyze75read(target_filename);
    BinaLSVolume=zeros(size(LSResultVolume));
    BinaLSVolume(find(LSResultVolume~=0))=1;
    BinaLSVolume =logical(BinaLSVolume);

%  and
    AndVolume=and(BinaVolume,BinaLSVolume); %logical volume
%label those non-zero positions in AndVolume as 250
    RawResultVolume(find(RawResultVolume==250))=150; %all WM gone
    RawResultVolume(find(AndVolume==1))=250; % only AndVolume positions are turned back to WM

%
    HalfResultVolume_fullfilename = [Result_path,HalfResult_filename];

% if main_iter=num_iter, store it as the final result where the holes in
% WM are all filled

if (main_iter==num_iter)
    RawResultVolume(find(LSResultVolume==250))=250;
    %write file
    analyze75write(RawResultVolume, [HalfResultVolume_fullfilename,'.img'],'uchar');
    %write the .hdr file
    command = ['makeavwheader ' HalfResultVolume_fullfilename,'.hdr'  ' -d ' num2str(col) ,',', num2str(row),',', num2str(depth) ,' -r ',num2str(res) ,',',num2str(res) ,',',num2str(res), ' -t  CHAR '];
    system(command);
else
    % if main_iter is not equal to num_iter, just store the intermediate
    % result volume for the next iteration, where the holes in WM are not filled
    %write the    
    RawResultVolume = analyze75read([RawResult_fullfilename,'.hdr']);
    analyze75write(RawResultVolume, [HalfResultVolume_fullfilename,'.img'],'uchar');
    command = ['makeavwheader ' HalfResultVolume_fullfilename,'.hdr'  ' -d ' num2str(col) ,',', num2str(row),',', num2str(depth) ,' -r ',num2str(res) ,',',num2str(res) ,',',num2str(res), ' -t  CHAR '];
    system(command); 
end
end