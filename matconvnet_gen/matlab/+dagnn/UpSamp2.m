% Wrapper for BilinearSampler block:

classdef UpSamp2 < dagnn.Layer
  methods
    function outputs = forward(obj, inputs, params)
        % Grid        
        nbatch = size(inputs{1},4);
        upsampFactor = 2;
        Ho = upsampFactor * size(inputs{1},2);
        Wo = upsampFactor * size(inputs{1},1);
        % generate the grid coordinates:
        xi = linspace(-1, 1, Ho);
        yi = linspace(-1, 1, Wo);
        [yy,xx] = meshgrid(yi,xi);
        xxyy = [xx(:), yy(:)]' ; % 2xM
        g = xxyy ;
        
        % transform the grid:
        g = repmat(g, [1 1 nbatch]);
        g = reshape(g, 2, Ho, Wo, nbatch);
        g = gpuArray(single(g));
        
        
      outputs = vl_nnbilinearsampler(inputs{1}, g);
      outputs = {outputs};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
        % Grid        
        nbatch = size(inputs{1},4);
        upsampFactor = 2;
        Ho = upsampFactor * size(inputs{1},2);
        Wo = upsampFactor * size(inputs{1},1);
        % generate the grid coordinates:
        xi = linspace(-1, 1, Ho);
        yi = linspace(-1, 1, Wo);
        [yy,xx] = meshgrid(yi,xi);
        xxyy = [xx(:), yy(:)]' ; % 2xM
        g = xxyy ;
        
        % transform the grid:
        g = repmat(g, [1 1 nbatch]);
        g = reshape(g, 2, Ho, Wo, nbatch);
        g = gpuArray(single(g));
        
        
      [dX,dG] = vl_nnbilinearsampler(inputs{1}, g, derOutputs{1});
      derInputs = {dX};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      xSize = inputSizes{1};
      outputSizes = {[xSize(1) * 2, xSize(2) * 2, xSize(3), xSize(4)]};
    end

    function obj = BilinearSampler(varargin)
      obj.load(varargin);
    end
  end
end
