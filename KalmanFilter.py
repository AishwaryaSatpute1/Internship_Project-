{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7bd1f6db",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class KalmanFilter:\n",
    "    kf = cv2.KalmanFilter(4, 2)\n",
    "    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)\n",
    "    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)\n",
    "\n",
    "\n",
    "    def predict(self, coordX, coordY):\n",
    "        ''' This function estimates the position of the object'''\n",
    "        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])\n",
    "        self.kf.correct(measured)\n",
    "        predicted = self.kf.predict()\n",
    "        x, y = int(predicted[0]), int(predicted[1])\n",
    "        print(\"predicted[0]\",x,\"predicted[1]\",y)\n",
    "        predicted1 = self.kf.predict(predicted[0],predicted[1])\n",
    "        m, n = int(predicted1[0]), int(predicted1[1])\n",
    "        print(\"predicted1[0]\",m,\"predicted1[1]\",n)\n",
    "        return x, y, m, n\n",
    "#         return m, n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
