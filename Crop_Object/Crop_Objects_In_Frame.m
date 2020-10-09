clear;clc;close;
root_dir = 'D:\KITTI_DATASET\';
data_set = 'training'
% get sub-directories
cam = 2; % 2 = left color camera
image_dir = fullfile(root_dir,[data_set '/image_' num2str(cam)]);
label_dir = fullfile(root_dir,[data_set '/label_' num2str(cam)]);
calib_dir = fullfile(root_dir,[data_set '/calib']);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

frame=0;

for j=1:nimages
    img = imread(sprintf('%s/%06d.png',image_dir,frame)); % ler a imagem
    fid = fopen(sprintf('%s/%06d.txt',label_dir,frame),'r'); % abrir o arquivo
    C = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' '); % ler o arquivo em formato de matriz
    objects = [];
    for o = 1:numel(C{1})
        % extract label, truncation, occlusion
        lbl = C{1,1}(o); % for converting: cell -> string
      % if strcmp(lbl,'Car')
      % if strcmp(lbl,'Cyclist')
      % if strcmp(lbl,'Misc')
      % if strcmp(lbl,'Tram')
      % if strcmp(lbl,'Truck')
      % if strcmp(lbl,'Van')
      % if strcmp(lbl,'Person')
      % if strcmp(lbl,'Pedestrian')
        if strcmp(lbl,'Car') % Cyclist Pedestrian
            objects(o).type = lbl{1}; % 'Car', 'Pedestrian', ...
            % extract 2D bounding box in 0-based coordinates
            objects(o).x1 = C{5}(o); % left
            objects(o).y1 = C{6}(o); % top
            objects(o).x2 = C{7}(o); % right
            objects(o).y2 = C{8}(o); % bottom
            % calculos;
            x1=objects(o).x1;
            y1=objects(o).y1;
            x2=objects(o).x2;
            y2=objects(o).y2;
            x21=x2-x1+1;
            y21=y2-y1+1;
            rect=[x1 y1 x21 y21];
            I = imcrop(img,rect);
            %imshow(I)
            primeiro=sprintf('%s%06d','',frame);
            segundo=num2str(o);
            juntar1=strcat('_',segundo);
            juntar2=strcat(primeiro,juntar1);
            salvo_como='.png';
            juntar3=strcat(juntar2,salvo_como);
            imwrite(I,juntar3,'png');
        end
        lbl=[];
    end
    frame=frame+1;
    img=[];
    fid=[];
    C=[];
    fclose('all');   
end