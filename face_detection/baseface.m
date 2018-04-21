%% Params

function Y_out = baseface (image_path)

	
	load ../data/avglm.mat
	lmid = 28:48;
	U = avglm.landmarks(lmid,:);
	
	%% ========== ========== ==========
	
	preload 	= 'export LD_LIBRARY_PATH="~/.local/lib"';
	model_path   	= '../model/landmark_68.dat';
	output_path     = 'tmp.mat';

	tic
	[st, out] = system(sprintf('%s && python face_detector.py %s %s %s',preload,image_path,output_path,model_path)); %,'-echo'
	toc

	if st == 0
		load(output_path);
	else
		error(out)
	end

	%% ========== ========== ==========

	Y 	= imread(image_path);

	for i = 1: numel(facedet)

		V 	= double(facedet{i}.landmarks(lmid,:));

		[d,Z,transform] = procrustes(U,V,'scaling',true,'reflection',false); 

		% Make transformation
		T = [0 0 0; 0 0 0; transform.c(1,1) transform.c(1,2) 1];
		T(1:2,1:2) = transform.T * transform.b;
		tf = maketform('affine',T);

		% Apply transformation
		YT  = imtransform(Y,tf,'XData',[1 240],'YData',[1 240],'FillValues',110);

		YT  = imresize( imcrop(YT,[22 22 196 196]), [112 112]);

		Y_out{i} = YT;

	end

	delete(output_path);

end
