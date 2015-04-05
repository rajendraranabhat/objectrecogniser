function [tags] = getTags()
% setup MtConvNet in MATLAB
addpath matlab
run vl_compilenn
%run matlab/vl_setupnn

% download a pre-trained CNN from the web

net = load('imagenet-vgg-f.mat') ;

% obtain and preprocess an image
im = imread('temp.jpg') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore));
tags = net.classes.description{best};
dlmwrite('pred.txt', tags, '');
end
