classdef Conv3D < dagnn.Filter
  properties
    size = [0 0 0 0 0]
    hasBias = true
    exBackprop = false
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = mex_conv3d(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      
        if(~obj.exBackprop)
            [derInputs{1}, derParams{1}, derParams{2}] = mex_conv3d(...
                inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                'pad', obj.pad, ...
                'stride', obj.stride) ;
        else
            wPlus = max(0,params{1});
            b = params{2};
            b = gpuArray(zeros(size(b),'single'));
%             b = zeros(size(b),'single');
            
            Abot = inputs{1};
            Ptop = derOutputs{1};
            
            Atop = mex_conv3d(Abot, wPlus, b, ...
                                    'pad', obj.pad, ...
                                    'stride', obj.stride) ;
            
            Y = gpuArray(zeros(size(Atop),'single'));
%             Y = zeros(size(Atop),'single');
            nonZeroXIdx = logical(gather(Atop ~= 0));
            Y(nonZeroXIdx) = Ptop(nonZeroXIdx) ./ Atop(nonZeroXIdx);
            
            % Backward Pass            
            [Pbot, derParams{1}, derParams{2}] = mex_conv3d(Abot, wPlus, b, Y, ...
                                                                    'pad', obj.pad, ...
                                                                    'stride', obj.stride) ;
                                                                
            derInputs{1} = Abot .* Pbot;
        end
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:4))) ;
%       sc = sqrt(2 / prod(obj.size(1:3))) ;
%         sc = 0.75;
%     if(sc > 1)
%         sc = 0.2;
%     end
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(1,obj.size(5),'single') * sc ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 5 dimensions
      ksize = [ksize(:)' 1 1 1 1 1] ;
      obj.size = ksize(1:5) ;
    end

    function obj = Conv3D(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
