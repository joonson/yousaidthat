clear all

run matconvnet_gen/matlab/vl_setupnn
addpath utils

% ===== PARAMS =====

inputfile   = 'data/transcript.mp3';
tmpwavfile  = 'tmp.wav';
writepath   = 'data/output';

asf = 1; % Audio start frame
opt.frame_rate = 25;
val = 640;

opt.audio.window   = [0 1];
opt.audio.fs       = 16000;
opt.audio.Tw       = 25;
opt.audio.Ts       = 10;            % analysis frame shift (ms)
opt.audio.alpha    = 0.97;          % preemphasis coefficient
opt.audio.R        = [ 300 3700 ];  % frequency range to consider
opt.audio.M        = 40;            % number of filterbank channels 
opt.audio.C        = 13;            % number of cepstral coefficients
opt.audio.L        = 22;            % cepstral sine lifter parameter


% ===== LOAD ORIGINAL NET =====

netStructv201 = load('model/v201.mat'); 

net = dagnn.DagNN.loadobj(netStructv201.net);
net.mode = 'test';
net.move('gpu')

names = {'loss1a','loss1b','loss2a','loss2b','loss1','loss2','loss_SR'} ;
for i = 1:numel(names)
  try
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end

% ===== LOAD DEBLUR NET =====

netStructv114 = load('model/v114.mat'); 

netR = dagnn.DagNN.loadobj(netStructv114.net);
netR.mode = 'test';
netR.move('gpu')

names = {'loss'} ;
for i = 1:numel(names)
  layer = netR.layers(netR.getLayerIndex(names{i})) ;
  netR.removeLayer(names{i}) ;
  netR.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end

fprintf('Net loaded.\n');


% ===== LOAD FACE IMAGES =====

load('data/faceimg.mat')
numface = numel(faceimg);
faceY   = gpuArray(cat(4,faceimg{:}));
faceYG  = gather(faceY)/255;
faceYG  = faceYG(2:110,2:110,1:3,:);
numcol  = 2;
numrow  = floor(numface/numcol);

%% ===== LOAD AUDIO =====

system(sprintf('ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s ', inputfile, tmpwavfile)); 

Zpad              = zeros(16000,1);
[Zo1,fs]          = audioread(tmpwavfile);

Z = [Zpad; Zo1; Zpad];
[ C, F, ~ ]     = runmfcc( Z, opt.audio );

%% ===== FORWARD PASS =====

padn    = 2;
Y       = cell(0);

for j = 1:4: size(C,2)-34

    mfcc = gpuArray(repmat(single(C (2:13,j:j+34)),1,1,1,numface));

    net.eval({'input_audio',mfcc,'input_face',faceY});

    im = gather(net.getVar('prediction').value);
    netR.eval({'input_lip',gpuArray(im)});
    imb         = gather(netR.vars(end).value);
    imR3        = imtile(imresize((im+imb),[109 109]),numcol,numrow);
    imO3        = imtile(imresize(im,[109 109]),numcol,numrow);

    netR.eval({'input_lip',gpuArray(im)});

    ims         = [zeros(109, padn, 3, numface) faceYG  (im+imb) zeros(109, padn, 3, numface)];

    Y       = [Y {imtile(ims,2,3)}];

    fprintf('Frame %d \n',numel(Y))

end

%% ===== SAVE VIDEO =====

af = asf+1: asf + numel(Y)* val ;

imgAudioToVideo (opt,Y,Z(af),[writepath '.avi']);
avi2mp4 ([writepath '.avi'],[writepath '.mp4'],5000);

delete([writepath '.avi'])
delete(tmpwavfile);


