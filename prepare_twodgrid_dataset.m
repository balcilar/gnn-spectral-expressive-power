clear all
I=imread('peppers.png');
I=I(4:383,66:66+379,:);
I=rgb2gray(I);
I=imresize(I,1/4);


% calculate centered frequcy of given image
F=fftshift(fft2(I));

[U V]=meshgrid(-fix(size(F,2)/2):fix(size(F,2)/2),-fix(size(F,1)/2):fix(size(F,1)/2));
U=0.5*U/max(U(:));
V=0.5*V/max(V(:));
p=sqrt(U.^2+V.^2);

% create band pass filter
f=exp(-1000*(p-0.25).^2);
% filter image in frequency domain
FD=F.*f;
% transform the filtered image in to spatial domain
fI=ifft2(ifftshift(FD));
% normalize it
fI1=(fI-min(fI(:)))/(max(fI(:))-min(fI(:)));

% create low-pass filter
f=exp(-100*(p-0).^2);
% filter image in frequency domain
FD=F.*f;
% transform the filtered image in to spatial domain and normalize it
fI2=ifft2(ifftshift(FD));
fI2=(fI2-min(fI2(:)))/(max(fI2(:))-min(fI2(:)));


% create high-pass filter
f=1-exp(-10*(p-0).^2);
FD=F.*f;
fI3=ifft2(ifftshift(FD));
fI3=(fI3-min(fI3(:)))/(max(fI3(:))-min(fI3(:)));

F=double(I)/255;
F=F(:);
Y=[fI1(:) fI2(:) fI3(:)];

% create 2D adjacency using gspbox library.
% basically A shows 4 neighborhood connection to the concerned node
% download gspbox and be sure the path is true, then run its gsp_start
% script in the library folder
run /home/mb/projects/Graph_Signal_Processing/gspbox/gsp_start
G=gsp_2dgrid(size(I,1));
A=full(G.W);
A=A+A';
A(A>0)=1;

% create mask to avoid inconsistend of nodes near to borders
mask=zeros(30,30);
mask(3:end-2,3:end-2)=1;
mask=mask(:);

% save dataset
save 2Dgrid A F Y mask

