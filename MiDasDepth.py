import cv2
import torch

import matplotlib.pyplot as plt
def getDepth(transform,frame,model):




    #transform input for midas
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    imgbatch = None
    if torch.cuda.is_available():
         imgbatch = transform(img).to('cuda')
    else:
        imgbatch = transform(img).to('cpu')
    #prediction
    with torch.no_grad():
        prediction = model(imgbatch)
        #resize output (upscale)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners = False

        ).squeeze()

        #get numpy value back
        output = prediction.cpu().numpy()
        #print(output)
        return output
        #plt.imshow(output)
        #plt.pause(0.00001)
        #plt.show()