function imgAudioToVideo (opt,im,audio,writepath)
    
    [writeDir,~,~] = fileparts(writepath);
	system(sprintf('mkdir -p %s',writeDir));
    
    % It is assumed that audio is stored in "data" variable

    % Idea is simple: Just divide length of the audio sample by the number of frames to be written in the video frames. ( it is equivalent to saying that what audio you   % want to have with that particular frame)

    % First make AudioInputPort property true (by default this is false)

    writerObj = vision.VideoFileWriter(writepath,'AudioInputPort',true);

    % total number of frames
    numFrames   = numel(im);

    % assign FrameRate (by default it is 30)

    writerObj.FrameRate =  opt.frame_rate;

    % length of the audio to be put per frame

    val = floor(size(audio,1)/numFrames);

    % Read one frame at a time

    for k = 1 : numFrames
        % reading frames from a directory
        % adding the audio variable in the step function
        step(writerObj,im{k},audio(val*(k-1)+1:val*k,:)); % it is 2 channel that is why I have put (:)

    end

    % release the video

    release(writerObj)



end