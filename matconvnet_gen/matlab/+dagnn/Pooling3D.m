classdef Pooling3D < dagnn.Filter
  properties
    poolSize = [1 1]
  end

  methods
    function outputs = forward(self, inputs, params)
      [outputs{1}, ind] = mex_maxpool3d(inputs{1}, 'pool', self.poolSize, ...
                             'pad', self.pad, ...
                             'stride', self.stride) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)        
      [y, ind] = mex_maxpool3d(inputs{1}, 'pool', self.poolSize, ...
                             'pad', self.pad, ...
                             'stride', self.stride) ;
                         
      derInputs{1} = mex_maxpool3d(derOutputs{1}, ind, size(inputs{1}), 'pool', self.poolSize, ...
                               'pad', self.pad, ...
                               'stride', self.stride) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Pooling3D(varargin)
      obj.load(varargin) ;
    end
  end
end
