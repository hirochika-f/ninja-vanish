import cv2
import numpy as np


class Equirectangular:
    """
    Generate perspective image from equirectangular image.
    Retain values related to perspective(lat/lon, etc...).

    Attributes
    -----
    _img : ndarray(height, width)
        Input image matrix.
    _height : int
        Height of input image.
    _width : int
        Width of input image.
    _lat : ndarray(_height)
        Y Coordinates of perspective image on equirectangular.
    _lon : ndarray(_width)
        X Coordinates of perspective image on equirectangular.
    _
    """
    def __init__(self, im):
        """
        Parameters
        -----
        im : string
            The variable of input equirectangular image.
        """
        self._img = im
        [self._height, self._width, _] = self._img.shape

    def calc_perspective_positions(self, FOV, THETA, PHI, height, width, RADIUS=1024):
        """
        Generate perspective positions which the argumets designate. 
        
        Parameters
        -----
        FOV : float
            Field of view.
        THETA : float
            Left/Right angle(degree).
        PHI : float
            Up/Down angle(degree).
        height : int
            Height of perspective image.
        width : int
            Width of perspective image.
        """
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)

        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        self._lon = np.floor(lon).astype(np.int32)
        self._lat = np.floor(lat).astype(np.int32)

    def get_perspective_image(self, fov, theta, phi, height, width, RADIUS=1024):
        """
        Return perspective image has designated parameters.

        Parameters
        ----
        fov
            Same as above.
        theta
            Same as above.
        phi
            Same as above.
        height
            Same as above.
        width
            Same as above.
        """
        self.calc_perspective_positions(fov, theta, phi, height, width, RADIUS=RADIUS)
        perspective_image = cv2.remap(self._img, self._lon.astype(np.float32), self._lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return perspective_image, self._lat, self._lon

    def get_perspective_positions(self, fov, theta, phi, height, width, RADIUS=1024):
        """
        Return latitude/longtitude has designated parameters.

        Parameters
        ----
        fov
            Same as above.
        theta
            Same as above.
        phi
            Same as above.
        height
            Same as above.
        width
            Same as above.
        """ 
        self.calc_perspective_positions(fov, theta, phi, height, width)
        return self._lat, self._lon

    def back_perspective_image(self, pers, mask, lat=None, lon=None):
        """
        Return back projected equirectangular image.

        Parameters
        ----
        pers
            Perspective image for back projection.
        lat
            Positions of perspective image of y axis.
        lon
            Positions of perspective image of x axis.
        """ 
        equi = self._img
        if lat is None:
            lat = self._lat
        if lon is None:
            lon = self._lon
        equi[lat, lon] = pers[np.where(mask != 0)]
        return equi


if __name__ == '__main__':
    equ_name = '/Users/hirochika/opencv/world.jpg'
    im = cv2.imread(equ_name)
    equ = Equirectangular(im)
    fov = 60
    theta = 0
    phi = 0
    height = 224
    width = 224

    perspective_image, lat, lon = equ.get_perspective_image(fov, theta, phi, height, width)
    back_img = equ.back_perspective_image(perspective_image)

    equi = cv2.imread(equ_name)
    equi[lat[0], lon[0]] = [0, 255, 0]
    equi[lat[:, 0], lon[:, 0]] = [0, 255, 0]
    equi[lat[-1], lon[-1]] = [0, 255, 0]
    equi[lat[:, -1], lon[:, -1]] = [0, 255, 0]

    cv2.imshow('o', perspective_image)
    cv2.imshow('aa', equi)
    cv2.waitKey(0)
