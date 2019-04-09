% for the (iter)th iteration
%1 read the image from the /iter folder as the input
%2 do the infer new labels of voxles
%3 write the image (same name)as the output in the /iter+1 folder
function AssignLabel(inputFileFullPath,inputFilename,main_iter, res, w)


%% specify all kinds of filename and path

eval(['load ',inputFileFullPath,inputFilename,'/new_labels.mat']);

SRinput_path  = [inputFileFullPath,inputFilename,'/','iter',num2str(main_iter),'/'];
SRoutput_path = [inputFileFullPath,inputFilename,'/','iter',num2str(main_iter+1),'/'];

mkdir(SRoutput_path);

% 1 output in the iter+1 fold
HalfResult_filename= [inputFilename,'_half'];

% 5 ExtNEW volume filename
ExtNEW_filename = [inputFilename,'_ExtNEW'];


testVolume_fullpath = [SRinput_path,HalfResult_filename];
testVolume = analyze75read( [testVolume_fullpath,'.hdr'] );
testVolume = double(testVolume);
[row,col,depth] = size(testVolume);

%load candidate index volume

candidateVolume_fullpath = [SRinput_path,ExtNEW_filename];
candidateVolume = analyze75read( [candidateVolume_fullpath,'.hdr'] );

Img = testVolume;
resultVolume=Img;

idx=find(candidateVolume>0);
[xx yy zz] = ind2sub(size(candidateVolume),find(candidateVolume==1));
temp_bg = zeros(size(candidateVolume));
temp_wm = zeros(size(candidateVolume));

patchsize=2*w+1;

for j=1:length(idx)
    x=xx(j);
    y=yy(j);
    z=zz(j);
    temp_bg(x-w:x+w,y-w:y+w,z-w:z+w) = temp_bg(x-w:x+w,y-w:y+w,z-w:z+w) + tranxz(reshape(new_labels(j,:,:,:,1),[patchsize,patchsize,patchsize]));
	temp_csf(x-w:x+w,y-w:y+w,z-w:z+w) = temp_wm(x-w:x+w,y-w:y+w,z-w:z+w) + tranxz(reshape(new_labels(j,:,:,:,2),[patchsize,patchsize,patchsize])); 
	temp_gm(x-w:x+w,y-w:y+w,z-w:z+w) = temp_wm(x-w:x+w,y-w:y+w,z-w:z+w) + tranxz(reshape(new_labels(j,:,:,:,3),[patchsize,patchsize,patchsize])); 
    temp_wm(x-w:x+w,y-w:y+w,z-w:z+w) = temp_wm(x-w:x+w,y-w:y+w,z-w:z+w) + tranxz(reshape(new_labels(j,:,:,:,4),[patchsize,patchsize,patchsize])); 
end

for j=1:length(idx)
    x=xx(j);
    y=yy(j);
    z=zz(j);
    t_bg = temp_bg(x,y,z);
	t_csf = temp_csf(x,y,z);
	t_gm = temp_gm(x,y,z);
    t_wm = temp_wm(x,y,z);
    [~,idx]=max([t_bg,t_csf,t_gm,t_wm]);
	labels_intensity=[0,10,150,250];
    
    resultVolume(x,y,z)=labels_intensity(idx);
    end  
end




resultVolume=uint8(resultVolume);

gene_fullfilename=[SRoutput_path,HalfResult_filename];
analyze75write(resultVolume, [gene_fullfilename,'.img'],'uchar');
%write the .hdr file
command = ['makeavwheader ' gene_fullfilename,'.hdr'  ' -d ' num2str(col) ,',', num2str(row),',', num2str(depth) ,' -r ',num2str(res) ,',',num2str(res) ,',',num2str(res), ' -t  CHAR '];
system(command);

end
