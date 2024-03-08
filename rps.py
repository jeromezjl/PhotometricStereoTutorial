#!/usr/bin/env python
# -*- coding: utf-8 -*-


import psutil
import numpy as np
from sklearn.preprocessing import normalize


class PS(object):


    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.height = None  # image height
        self.width = None   # image width
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = psutil.load_lighttxt(filename)

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = psutil.load_lightnpy(filename)

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = psutil.load_images(foldername, ext)

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = psutil.load_npyimages(foldername)

    def load_mask(self, filename=None):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = psutil.load_image(filename=filename)
        mask = mask.reshape((-1, 1))
        print(f"mask shape is:{mask.shape}")

        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        psutil.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)

    def solve(self):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")
        if self.M.shape[1] != self.L.shape[1]:
            raise ValueError("Inconsistent dimensionality between M and L")

        # #############################################
        #
        # # Please write your code here to solve the surface normal N whose size is (p, 3) as discussed in the tutorial
        #
        # # Step 1: solve Ax = b
        # # Hint: You can use np.linalg.lstsq(A, b) to solve Ax = b
        #
        # # 根据前景掩码选择有效像素
        # I = self.M[self.foreground_ind, :].T  # 50行，20317列
        #
        # # 光照矩阵
        # L = self.L.T  # 50行，3列
        #
        # # 最小二乘法求解法向量 N
        # self.N = np.linalg.lstsq(L, I, rcond=None)[0].T  # 解出的 N 需要转置，以使其行对应于不同的像素点，3行，20317列
        #
        # # Step 2: We need to normalize the normal vectors as the norm of the normal vectors should be 1
        # # Hint: You can use function normalize from sklearn.preprocessing
        #
        # # 化为单位向量
        # self.N = normalize(self.N, axis=1)
        #
        # #############################################
        #
        # # self.N = self.N.T
        #
        # print(self.background_ind.shape)
        # print(self.foreground_ind.shape)
        # #
        # # if self.background_ind is not None:
        # #     for i in range(self.N.shape[1]):
        # #         self.N[self.background_ind, i] = 0

        # 根据前景掩码选择有效像素
        I = self.M[self.foreground_ind, :].T  # 取出前景像素的测量矩阵
        L = self.L.T
        # 最小二乘法求解法向量 N
        N_foreground = np.linalg.lstsq(L, I, rcond=None)[0].T  # 解出的 N 需要转置，使其行对应于不同的像素点

        # 归一化法向量
        N_foreground = normalize(N_foreground, axis=1)

        # 初始化包含所有像素法向量的矩阵，初始为零
        total_pixels = self.M.shape[0]  # 总像素数为 M 的行数
        N_all = np.zeros((total_pixels, 3))  # 初始化全零矩阵

        # 将前景像素的法向量填充到 N_all 中对应的位置
        N_all[self.foreground_ind, :] = N_foreground

        # 背景像素的法向量已经是零，无需额外操作

        # 更新 self.N 为包含所有像素的法向量矩阵
        self.N = N_all
