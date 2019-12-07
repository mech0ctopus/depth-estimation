%Saves RGB and color images from nyu_depth_v2_labeled.mat to PNG files
%Assumes depths and images variables have been loaded

size_rgb_images=size(images);
num_rgb_images=size_rgb_images(4);

for i=1:num_rgb_images
    disp(strcat(num2str(i),'/',num2str(num_rgb_images)));
    rgb_filename=strcat('nyu_data/X_rgb/rgb_',num2str(i),'.png');
    imwrite(uint8(images(:,:,:,i)),rgb_filename);
    
    d_filename=strcat('nyu_data/y_depth/d_',num2str(i),'.png');
    imwrite(mat2gray(depths(:,:,i)),d_filename);
end