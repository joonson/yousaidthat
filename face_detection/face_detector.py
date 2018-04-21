# Face detection
# With Matlab Interface -- Joon Son Chung 2016.04.27

import sys
import dlib
import numpy
from skimage import io
import scipy.io as sio

detector = dlib.get_frontal_face_detector()

predictor_path = sys.argv[3]
predictor = dlib.shape_predictor(predictor_path)

img = io.imread(sys.argv[1])

dets, scores, idx = detector.run(img, 1)

f = []

for j, d in enumerate(dets):

	conf = scores[j]

	if conf >= 0.25 :

	    fleft   = d.left()
	    ftop    = d.top()
	    fwidth  = d.width()
	    fheight = d.height()

	    d = dlib.rectangle(left=fleft, top=ftop, right=fwidth+fleft, bottom=fheight+ftop)

	    shape = predictor(img, d)

	    lmark = [[shape.part(i).x,shape.part(i).y] for i in range(0,67)]

	    f.append({'conf': conf, 'left': fleft, 'top': ftop, 'width': fwidth, 'height': fheight, 'landmarks': lmark })

sio.savemat(sys.argv[2], {'facedet': f})