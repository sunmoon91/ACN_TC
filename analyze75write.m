function [ output_args ] = analyze75write( data, filename, precision )
%ANALYZE75WRITE Summary of this function goes here
%   filename:	'*.img'
%   precision:	'uchar', 'short', 'float', 'double'
%   
%   created by wangqian, 10/09/2009, 5:20pm

% info = analyze75info('C:\Workspace\Atrophy\images\AC.hdr');
% data = analyze75read(info);
% 
% filename = '..\Data\output.img';
fid = fopen(filename, 'w', 'ieee-le');

temp = permute(data, [2 1 3]);
count = numel(data);
temp = reshape(temp, [1, count]);

fwrite(fid, temp, precision);

fclose(fid);

end
