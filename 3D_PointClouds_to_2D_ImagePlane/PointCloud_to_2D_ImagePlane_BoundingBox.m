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
indice_folder = 2222; % frame-KITTI
fileID = fopen(sprintf('%s/%06d.txt',label_path,indice_folder),'r'); % open txt
Label = textscan(fileID,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' '); % read the file in matrix format
fclose(fileID);
for o = 1:numel(Label{1})
    %% By object in the frame
    if Label{1,1}{o}=="Car" | Label{1,1}{o}=="Cyclist" | Label{1,1}{o}=="Pedestrian"
        car=Label{1,1}{o}=="Car";
        cyclist=Label{1,1}{o}=="Cyclist";
        pedestrian=Label{1,1}{o}=="Pedestrian";
        y_class = [car.*1, cyclist.*1, pedestrian.*1];
        y_labels = [(Label{1,9}(o)).*1 Label{1,10}(o).*1 Label{1,11}(o).*1 Label{1,12}(o).*1 Label{1,13}(o).*1 Label{1,14}(o).*1 Label{1,15}(o).*1 y_class(1) y_class(2) y_class(3)];
        bounding_box = [bounding_box;y_labels];
    end
end
if ~isempty(bounding_box)
    %% By object in the frame: Get the position data of each object in the frame
    places=bounding_box(:,4:6);
    size_=bounding_box(:,1:3);
    rotates=bounding_box(:,7);
    y_class=bounding_box(:,8:10);
    rotates = pi / 2 - rotates;
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
end

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

% Filter camera angles for KITTI dataset
idx = (velo(:,2)<(velo(:,1)-0.27) & -velo(:,2)<(velo(:,1)-0.27));
velo_point = velo(idx,:);
xmin=min((velo_point(:,1)));
xmax=max((velo_point(:,1)));
ymin=min((velo_point(:,2)));
ymax=max((velo_point(:,2)));
zmin=min((velo_point(:,3)));
zmax=max((velo_point(:,3)));
player = pcplayer([xmin,xmax], [ymin,ymax], [zmin,zmax]);
view(player, velo_point(:,1:3))

% Defining which points belong to the image plane. (approximation)
idx = velo_point(:,1)<5;
velo_point(idx,:) = [];
xmin=min((velo_point(:,1)));
xmax=max((velo_point(:,1)));
ymin=min((velo_point(:,2)));
ymax=max((velo_point(:,2)));
zmin=min((velo_point(:,3)));
zmax=max((velo_point(:,3)));
player = pcplayer([xmin,xmax], [ymin,ymax], [zmin,zmax]);
view(player, velo_point(:,1:3))

% Defining which points belong to the image plane. (approximation)
idx = velo(:,1)<5;
velo(idx,:) = [];

xmin=min((velo(:,1)));
xmax=max((velo(:,1)));
ymin=min((velo(:,2)));
ymax=max((velo(:,2)));
zmin=min((velo(:,3)));
zmax=max((velo(:,3)));
player = pcplayer([xmin,xmax], [ymin,ymax], [zmin,zmax]);
view(player, velo(:,1:3))

% projecting to 2D image plane
px = (P2 * R0_rect * Tr_velo_to_cam * velo')';
px(:,1) = px(:,1)./px(:,3);
px(:,2) = px(:,2)./px(:,3);

im = sprintf('%s%06d.png',img_path,indice_folder);
imshow(im);hold on
cols = jet;

for c=1:size(px,1)
    col_idx = round(256*px(c,3)/max(px(:,3)));
    if col_idx == 0
        color = [0 0 0];
    else
        color = cols(col_idx,:);
    end
    plot(px(c,1),px(c,2),'o','LineWidth',4,'MarkerSize',1,'Color',color);
end

%% Bounding Boxes

% Projection matrix to 3D axis for 3D Label
velo_to_cam = calib{6,1};
posicao_ = find(calib{6,1} == ':')+2;
velo_to_cam_ = velo_to_cam(1,posicao:length(velo_to_cam));
velo_cam = reshape(str2num(velo_to_cam_),[4,3]);
velo_cam_ = (velo_cam(1:3,1:3));

rect = calib{5,1};
posicao = find(calib{5,1} == ':')+2;
rect_ = rect(1,posicao:length(rect));
Rect_ = reshape(str2num(rect_),[3,3]);

proj_velo=(velo_cam_*Rect_)';
for pl=1:size(places,1)
places(pl,:) = places(pl,:)*proj_velo;
end

% Every 0.27m from the center of the object, the original label undergoes a change.
places(:,1) = places(:,1)+0.27;

corner_=[];
corners=[];
% Compute the corners
for cor=1:size(places,1)
    x = places(cor,1);
    y = places(cor,2);
    z = places(cor,3);
    h = size_(cor,1);
    w = size_(cor,2);
    l = size_(cor,3);
    rotate = rotates(cor,1);
    if l <= 10
        corner = [x - l / 2., y - w / 2., z;
            x + l / 2., y - w / 2., z;
            x - l / 2., y + w / 2., z;
            x - l / 2., y - w / 2., z + h;
            x - l / 2., y + w / 2., z + h;
            x + l / 2., y + w / 2., z;
            x + l / 2., y - w / 2., z + h;
            x + l / 2., y + w / 2., z + h];
        corner = corner - [x, y, z];
        rotate_matrix = [
            cos(rotate()), -sin(rotate), 0;
            sin(rotate), cos(rotate),  0;
            0             , 0,               1];
        % corner_ is 8x3
        corner_ = corner*(rotate_matrix)';
        corner_ = corner_+[x,y,z];
        corner_ =reshape(corner_,[1,size(corner_,1)*size(corner_,2)]);
        corners = [corners;corner_];
    else
        corners = [corners;corner_];
    end
end

% Project the 3D box points into the image plane
Corne3DFrame=[];
for cor=1:size(corners,1)
    point_ = corners(cor,:);
    point_ = reshape(point_,[8,3]);
    corner3D = []
    for pt=1:size(point_,1)
        point3D_to_2Dplane = [];
        point3D_to_2Dplane = point_(pt,:);
        point3D_to_2Dplane = [point3D_to_2Dplane 1];
        point3D_corner_2Dplane = (P2 * R0_rect * Tr_velo_to_cam *point3D_to_2Dplane')';
        point3D_corner_2Dplane = point3D_corner_2Dplane./point3D_corner_2Dplane(3);
        corner3D = [corner3D;point3D_corner_2Dplane];
    end
    corner3D(:,4)=[];
    Corne3DFrame=[Corne3DFrame;reshape(corner3D,[1,size(corner_,1)*size(corner_,2)])];
end