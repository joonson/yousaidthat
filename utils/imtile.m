function ima = image_tile(im,numcol,numrow)

    ima     = squeeze(num2cell(im,[1 2 3]));
    ima     = reshape(ima,numcol,numrow);
    for k = 1: numcol
        imc{k} = cat(1,ima{k,:});
    end
    ima     = cat(2,imc{:});

end