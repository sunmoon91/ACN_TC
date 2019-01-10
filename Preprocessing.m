% used in iter>1 iterations
% 1 generate the diffVolume and store it
% 2 generate the ExtNEW volume as the candidate set and store it
% function Preprocessing(currVolIndex,w,template_num,main_iter,shpereFlag)
function Preprocessing(inputFileFullPath,inputFilename,main_iter,res)
%assume the processed file in iter(iter>1) folder is "test***" file, so no
%specific index is used anymore


%fold path
Result_path = [inputFileFullPath,inputFilename,'/iter',num2str(main_iter),'/'];


% 2 processed with LS (have hole) and excluding outliers
 HalfResult_filename= [inputFilename, '_half'];%no .hdr or .img surfix
% HalfResult_filename= ['n',strNum,shpereFlag,'_half'];%no .hdr or .img surfix

% 3 processed with LS (no holes)
LSResult_filename=[inputFilename,'_halfLS'];


% 4 diffVolume between _half and _halfLS volume
DiffVolume_filename=[inputFilename,'DiffLSandNEWUncorrected'];



% 5 ExtNEW volume filename
ExtNEW_filename = [inputFilename,'_ExtNEW'];



%% read the _half and _halfLS volume

HalfResult_fullfilename=[Result_path,HalfResult_filename];
HalfVolume = analyze75read( HalfResult_fullfilename );
HalfVolume = double(HalfVolume);
[row,col,depth] = size(HalfVolume); 

LSResult_fullfilename=[Result_path,LSResult_filename];
LSVolume = analyze75read( LSResult_fullfilename );
LSVolume = double(LSVolume);

%% generate the diff volume and store it
BinaHalfVolume=zeros([row,col,depth]); %only contain positions of WM matter of HalfVolume
BinaHalfVolume(find(HalfVolume==250))=1;

BinaLSVolume=zeros([row,col,depth]); %only contain positions of WM matter of LSVolume
BinaLSVolume(find(LSVolume~=0))=1;

DiffVolume = BinaLSVolume-BinaHalfVolume;
DiffVolume(find(DiffVolume~=0))=1;


DiffReult_fullfilename = [Result_path,DiffVolume_filename];

analyze75write(DiffVolume, [DiffReult_fullfilename,'.img'],'uchar');
%write the .hdr file
command = ['makeavwheader ' DiffReult_fullfilename,'.hdr'  ' -d ' num2str(col) ,',', num2str(row),',', num2str(depth) ,' -r ',num2str(res) ,',',num2str(res) ,',',num2str(res), ' -t  CHAR '];
system(command);


%% generate the Ext and ExtNEW file
connect_str = bwconncomp(DiffVolume,26);

bin=[];
% build a new structure,ignore those cluster whose number less than 6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
update_str.Connectivity=connect_str.Connectivity;
update_str.ImageSize=connect_str.ImageSize;
temp_counter=0;
temp_sum = 0;
%find the largest
num_cluster_array=[];
for i=1:connect_str.NumObjects
    curr_cluster = connect_str.PixelIdxList{i};
    num_cluster_array=[num_cluster_array,length(curr_cluster)];
end

for i=1:connect_str.NumObjects
    curr_cluster = connect_str.PixelIdxList{i};
    if((length(curr_cluster)>6)&&(length(curr_cluster)~=max(num_cluster_array)+1))
       temp_counter=temp_counter+1;
       update_str.PixelIdxList{temp_counter}=curr_cluster;
       temp_sum = temp_sum+length(curr_cluster); 
    end
end
update_str.NumObjects=temp_counter;
update_str.TotalPixel=temp_sum;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we should try to ignore the inner clusters! these cluster couldn't belong to handles
update_strNEW.Connectivity = connect_str.Connectivity;
update_strNEW.ImageSize=connect_str.ImageSize;
temp_counterNEW=0;
temp_sumNEW = 0;

for i=1:update_str.NumObjects
   
    curr_cluster = update_str.PixelIdxList{i};
    temp_volume=zeros(connect_str.ImageSize);
    temp_volume(curr_cluster)=1;
    dilate_mask = ones(3,3,3);
    dilate_volume=imdilate(temp_volume,dilate_mask);
    dilate_volume = dilate_volume-temp_volume;
    surround_index = find(dilate_volume~=0);
    num_surround=length(surround_index);
    surround_ori = HalfVolume(surround_index);
    num_surroundWM=length(find(surround_ori==250));
    WMratio=num_surroundWM/num_surround;
    if(WMratio<0.90)
       temp_counterNEW=temp_counterNEW+1;
       update_strNEW.PixelIdxList{temp_counterNEW}=curr_cluster;
       temp_sumNEW = temp_sumNEW+length(curr_cluster);   
    end
    
end
update_strNEW.NumObjects=temp_counterNEW;
update_strNEW.TotalPixel=temp_sumNEW;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%try to make a geometric aware dialation 
%%%%%%%%%%%%%%%%%%%%
update_str=update_strNEW;
dilate_size=13;
radius = floor(dilate_size/2);
template = zeros(dilate_size,dilate_size,dilate_size);
template(2:dilate_size-1,2:dilate_size-1,radius+1)=1;
template(2:dilate_size-1,2:dilate_size-1,radius)=1;
[temp_row,temp_col,temp_dep]=ind2sub(size(template),find(template~=0));
temp_position=[temp_row,temp_col,temp_dep]-radius-1;
for i=1:update_str.NumObjects
   
    curr_cluster = update_str.PixelIdxList{i};
    [row_cluster,col_cluster,dep_cluster]=ind2sub(update_str.ImageSize,curr_cluster);
    [coeff,score,latent]=pca([col_cluster,row_cluster,dep_cluster]);%send xyz into PCA
    
    templateX=temp_position(:,2);
    templateY=temp_position(:,1);
    templateZ=temp_position(:,3);
    templateXY=[templateX,templateY]*diag([1,latent(2)/latent(1)+eps]);
    templateXYZ=[templateXY,templateZ];
    new_positionXYZ = templateXYZ*inv(coeff)+radius+1;   
    new_positionXYZ(find(new_positionXYZ<1))=1;
    new_positionXYZ(find(new_positionXYZ>dilate_size))=dilate_size;
    new_positionXYZ = round(new_positionXYZ);
    %xyz to row col dep
    new_position = new_positionXYZ;
    new_position(:,1)=new_positionXYZ(:,2);
    new_position(:,2)=new_positionXYZ(:,1);
    
    new_template = zeros(dilate_size,dilate_size,dilate_size);
    newindex_array = sub2ind(size(new_template),new_position(:,1),new_position(:,2),new_position(:,3));
    new_template(newindex_array)=1;

    
    
    cluster_Volume=zeros(connect_str.ImageSize);
    cluster_Volume(curr_cluster)=1;
    dilate_Volume = imdilate(cluster_Volume,new_template);
    add_index = find((dilate_Volume-cluster_Volume)~=0);

    bin = [bin,add_index'];
end
index_array = unique(bin);

extVolume = zeros(size(DiffVolume));
extVolume(index_array)=1;


ExtNEW_fullfilename = [Result_path,ExtNEW_filename];

analyze75write(extVolume, [ExtNEW_fullfilename,'.img'],'uchar');

command = ['makeavwheader ' ExtNEW_fullfilename,'.hdr'  ' -d ' num2str(col) ,',', num2str(row),',', num2str(depth) ,' -r ',num2str(res) ,',',num2str(res) ,',',num2str(res), ' -t  CHAR '];
system(command);


  
    
end