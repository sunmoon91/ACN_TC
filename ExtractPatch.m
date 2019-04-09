% for the (iter)th iteration
%1 read the image from the /iter folder as the input
%2 extract the patches with topological errors
function ExtractPatch(inputFileFullPath,inputFilename,main_iter,w)
% sphere can be "lh" or "rh"

%% specify all kinds of filename and path


SRinput_path  = [inputFileFullPath,inputFilename,'/iter',num2str(main_iter),'/'];


HalfResult_filename= [inputFilename,'_half'];


ExtNEW_filename = [inputFilename,'_ExtNEW'];


%load testing volume

testVolume_fullpath = [SRinput_path,HalfResult_filename];
testVolume = analyze75read( [testVolume_fullpath,'.hdr'] );
img = double(testVolume);


%load candidate index volume

candidate_fullpath = [SRinput_path,ExtNEW_filename];
candidate = analyze75read( [candidate_fullpath,'.hdr'] );

%do the anatomical constrainted SR
batch_size=0;

idx=find(candidate>0);
[xx yy zz] = ind2sub(size(candidate),find(candidate==1));
for j=1:length(idx)
    x=xx(j);
    y=yy(j);
    z=zz(j);
    temp_img = img(x-w:x+w,y-w:y+w,z-w:z+w);
    batch_size=batch_size+1;
    pte(:,:,:,batch_size)=temp_img;    
end
eval(['save ',inputFileFullPath,inputFilename,'/patch_topological_errors.mat pte -v7.3'])
end