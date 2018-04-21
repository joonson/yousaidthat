function avi2mp4 (readpath,writepath,bitrate)

	[writeDir,~,~] = fileparts(writepath);
	system(sprintf('mkdir -p %s',writeDir));
	cmdStr= sprintf('ffmpeg -y -threads 1 -i %s -async 1 -vcodec mpeg4 -b:v %dk -strict -2 -flags +aic+mv4 -threads 1 %s &> /dev/null',readpath,bitrate,writepath);
	system(cmdStr); 

end