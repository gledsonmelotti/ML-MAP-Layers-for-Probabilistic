clear
close all
clc
format long
%% files
pcd_path = "D:/KITTI_DATASET/training/velodyne/";
label_path = "D:/KITTI_DATASET/training/label_2";
calib_path = "D:/KITTI_DATASET/training/calib/";
img_path = "D:/KITTI_DATASET/training/image_2/";

%% Reading Labels from tx
bounding_box = [];
indice_folder = 0; % frame-KITTI
fileID = fopen(sprintf('%s/%06d.txt',label_path,indice_folder),'r'); % open txt
Label = textscan(fileID,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' '); % read the file in matrix format
fclose(fileID);
%% Per frame independent of the number of objects: calibration of velodyne points with the RGB image
local=sprintf('%s/%06d.txt',calib_path,indice_folder);
calib={};
fc = fopen(local);
while ~feof(fc)
    i=i+1;
    tline = fgetl(fc);
    calib=[calib;{tline}];
end
fclose(fc);
calib(length(calib))=[];

%% P0
p0 = calib{1,1};
posicao = find(calib{1,1} == ':')+2;
p0 = p0(1,posicao:length(p0));
P0 = reshape(str2num(p0),[4,3])';
P0(4,:) = [0 0 0 1];
%% P1
p1 = calib{2,1};
posicao = find(calib{2,1} == ':')+2;
p1 = p1(1,posicao:length(p1));
P1 = reshape(str2num(p1),[4,3])';
P1(4,:) = [0 0 0 1];
%% P2
p2 = calib{3,1};
posicao = find(calib{3,1} == ':')+2;
p2 = p2(1,posicao:length(p2));
P2 = reshape(str2num(p2),[4,3])';
P2(4,:) = [0 0 0 1];
%% P3
p3 = calib{4,1};
posicao = find(calib{4,1} == ':')+2;
p3 = p3(1,posicao:length(p3));
P3 = reshape(str2num(p3),[4,3])';
P3(4,:) = [0 0 0 1];
%% R0_rect
rect = calib{5,1};
posicao = find(calib{5,1} == ':')+2;
rect = rect(1,posicao:length(rect));
R0_rect = reshape(str2num(rect),[3,3])';
R0_rect(:,4) = 0;
R0_rect(4,:) = [0 0 0 1];
%% Tr_velo_to_cam
velo_to_cam = calib{6,1};
posicao = find(calib{6,1} == ':')+2;
velo_to_cam = velo_to_cam(1,posicao:length(velo_to_cam));
Tr_velo_to_cam = reshape(str2num(velo_to_cam),[4,3])';
Tr_velo_to_cam(4,:) = [0 0 0 1];
%% Tr_imu_to_velo
imu_to_cam = calib{7,1};
posicao = find(calib{7,1} == ':')+2;
imu_to_cam = imu_to_cam(1,posicao:length(imu_to_cam));
Tr_imu_to_cam = reshape(str2num(imu_to_cam),[4,3])';
Tr_imu_to_cam(4,:) = [0 0 0 1];

%% loading velodyne points
fd = fopen(sprintf('%s/%06d.bin',pcd_path,indice_folder),'rb');
if fd > 1
    velo = fread(fd,[4 inf],'single')'; % 3D point clouds x, y, z and reflectance
    fclose(fd);
end

xmin=min((velo(:,1)));
xmax=max((velo(:,1)));
ymin=min((velo(:,2)));
ymax=max((velo(:,2)));
zmin=min((velo(:,3)));
zmax=max((velo(:,3)));
player = pcplayer([xmin,xmax], [ymin,ymax], [zmin,zmax]);
view(player, velo(:,1:3))

% Defining which points belong to the image plane. (approximation)
velo_pc = velo;
idx = velo_pc(:,1)<5;
velo_pc(idx,:) = [];

xmin=min((velo_pc(:,1)));
xmax=max((velo_pc(:,1)));
ymin=min((velo_pc(:,2)));
ymax=max((velo_pc(:,2)));
zmin=min((velo_pc(:,3)));
zmax=max((velo_pc(:,3)));
player = pcplayer([xmin,xmax], [ymin,ymax], [zmin,zmax]);
view(player, velo_pc(:,1:3))

% projecting to 2D image plane
px = (P2 * R0_rect * Tr_velo_to_cam * velo_pc')';
px(:,1) = px(:,1)./px(:,3);
px(:,2) = px(:,2)./px(:,3);

%% Definindo apenas os pontos dentro do bounding box
objects = [];
frame = indice_folder;
for o = 1:numel(Label{1})
    % extract label, truncation, occlusion
    lbl = Label{1,1}(o); % for converting: cell -> string
    if strcmp(lbl,'Pedestrian')
        objects(o).type = lbl{1}; % 'Car', 'Pedestrian', ...
        % extract 2D bounding box in 0-based coordinates
        objects(o).x1 = Label{5}(o); % left
        objects(o).y1 = Label{6}(o); % top
        objects(o).x2 = Label{7}(o); % right
        objects(o).y2 = Label{8}(o); % bottom
        % calculos;
        x1=objects(o).x1;
        y1=objects(o).y1;
        x2=objects(o).x2;
        y2=objects(o).y2;
        x21=x2-x1+1;
        y21=y2-y1+1;
        ind = find((px(:,1) >= x1 & px(:,1) <= x1+x21) & ...
            (px(:,2) >= y1 & px(:,2) <= y1+y21));
        point_cloud=px(ind,:);
        velo_point=velo_pc(ind,:);
        xmin=min((velo_point(:,1)));
        xmax=max((velo_point(:,1)));
        ymin=min((velo_point(:,2)));
        ymax=max((velo_point(:,2)));
        zmin=min((velo_point(:,3)));
        zmax=max((velo_point(:,3)));
        player = pcplayer([xmin,xmax], [ymin,ymax], [zmin,zmax]);
        view(player, velo(:,1:3))
        primeiro = sprintf('%s%06d','',frame);
        segundo = num2str(o);
        juntar1=strcat('_',segundo);
        juntar2=strcat(primeiro,juntar1);
        save([juntar2 '.mat'],'velo_point')
        lbl=[];
    end
end
fclose('all');