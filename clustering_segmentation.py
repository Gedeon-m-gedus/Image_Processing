import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib
import matplotlib.image as image
import matplotlib.pyplot as plt
from datetime import datetime


def segmentImage(image,k):
    (h1, w1) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h1, w1, 3))
    image = image.reshape((h1, w1, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    unique_name = f"file_{formatted_datetime}"

    cv2.imwrite(unique_name+'_classified.jpg', quant)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return quant

image = matplotlib.image.imread('./palm/image-2.png')
k = segmentImage(image,3)
plt.figure(figsize=(12,8))
ax = plt.subplot(1,2,1)
ax.imshow(k)
ax = plt.subplot(1,2,2)
ax.imshow(image)
