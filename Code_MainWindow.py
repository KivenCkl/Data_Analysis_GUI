#!/usr/bin/env python
# coding=UTF-8
'''
@Author: KivenChen
@Date: 2019-04-19
'''
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import pyqtSignal
from Ui_MainWindow import Ui_MainWindow
from qtpandas.models.DataFrameModel import DataFrameModel
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt


class MainWindow(QMainWindow, Ui_MainWindow):

    # 自定义信号
    close_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # qtpandas 设置 tableWidget
        tableWidget = self.tableWidget
        self.model = DataFrameModel()
        tableWidget.setViewModel(self.model)

        self.df = None  # 用于操作的数据
        self.df_temp = None  # 用于撤销的数据
        self.df_original = None  # 用于初始化的数据

        # 数据导入 tab
        self.pushButton_openfile.clicked.connect(self.open_file)

        self.saveDataButton.clicked.connect(self.save_data)
        self.initDataButton.clicked.connect(self.init_data)
        self.pushButton_refresh.clicked.connect(self.data_shape)
        self.pushButton_refresh.clicked.connect(self.data_describe)
        self.plotButton.clicked.connect(self.image_plot)

        # 特征预处理 tab
        self.checkNaNButton.clicked.connect(self.check_NaN)
        self.okButton_nan.clicked.connect(self.deal_with_NaN)
        self.repealButton_nan.clicked.connect(self.repeal_action)
        self.okButton_nondim.clicked.connect(self.nondim_action)
        self.repealButton_nondim.clicked.connect(self.repeal_action)
        self.initButton_nondim.clicked.connect(self.init_data)

        # 绘图设置 tab
        self.radioButton_2D.toggled.connect(self.set_page_show)
        self.radioButton_2Dline.toggled.connect(self.set_page_show)
        self.radioButton_2Dscatter.toggled.connect(self.set_page_show)
        self.radioButton_2Dhist.toggled.connect(self.set_page_show)
        # self.radioButton_2Dpie.toggled.connect(self.set_page_show)
        self.comboBox_fitmethod.activated.connect(self.set_page_show)
        self.radioButton_3D.toggled.connect(self.set_page_show)
        self.radioButton_3Dline.toggled.connect(self.set_page_show)
        self.radioButton_3Dscatter.toggled.connect(self.set_page_show)
        self.radioButton_3Dsurface.toggled.connect(self.set_page_show)

        # 跟随主窗口关闭所有绘图
        self.close_signal.connect(lambda: plt.close('all'))

    def open_file(self):
        """ 打开文件
        """
        fname, filetype = QFileDialog.getOpenFileName(
            self, 'Open File', '',
            "CSV files (*.csv);; Excel files (*.xlsx *.xls);; MAT files (*.mat);; All files (*)"
        )
        if not fname:
            return
        try:
            if fname.endswith('.csv'):
                self.df = pd.read_csv(fname, encoding='utf-8')
            elif fname.endswith('.xlsx' or '.xls'):
                self.df = pd.read_excel(fname, encoding='utf-8')
            elif fname.endswith('.mat'):
                tempdata = sio.loadmat(fname)
                data_dic = dict()
                for key in tempdata.keys():
                    if not key.startswith('__'):
                        if not len(tempdata[key]):
                            data_dic[key] = None
                        elif len(tempdata[key]) == 1:
                            data_dic[key] = tempdata[key][0]
                        else:
                            for i in range(len(tempdata[key])):
                                data_dic['{}_{}'.format(key,
                                                        i)] = tempdata[key][i]

                self.df = pd.DataFrame(data_dic)
            else:
                self.textBrowser.append('文件格式不符')
        except Exception as e:
            self.textBrowser.append('无法打开该文件: {}\n 错误原因: {}'.format(fname, e))
            return
        self.df_original = self.df.copy()
        self.load_data_to_table()
        self.filenameLabel.setText(fname.split('/')[-1].split('.')[0])
        self.fileformatLabel.setText(fname.split('.')[-1])
        self.data_shape()
        self.data_describe()
        self.textBrowser.append('文件已加载: {}'.format(fname))

    def load_data_to_table(self):
        """ 载入数据至 qtpandasTable
        """
        try:
            self.model.setDataFrame(self.df)
        except Exception as e:
            self.textBrowser.append('数据载入失败 \n 错误原因: {}'.format(e))
            return

    def data_shape(self):
        """ 显示数据的维度
        """
        (m, n) = self.df.shape
        self.dataDimLabel.setText('{}x{}'.format(m, n))

    def data_describe(self):
        """ 显示数据特征信息
        """
        # pd.set_option('display.max_columns', 1000)
        # self.dataDescTable.setPlainText('{}'.format(self.df.describe()))
        try:
            desc = self.df.describe()
            self.dataDescTable.setRowCount(desc.shape[0])
            self.dataDescTable.setColumnCount(desc.shape[1])
            self.dataDescTable.setHorizontalHeaderLabels(desc.columns)
            self.dataDescTable.setVerticalHeaderLabels(desc.index)
            for i in range(desc.shape[0]):
                for j in range(desc.shape[1]):
                    newItem = QTableWidgetItem(str(round(desc.iloc[i, j], 2)))
                    self.dataDescTable.setItem(i, j, newItem)
        except:
            pass

    def init_data(self):
        """ 数据初始化
        """
        try:
            self.model.setDataFrame(self.df_original)
            self.df = self.df_original.copy()
            self.textBrowser.append('数据已还原')
        except Exception as e:
            self.textBrowser.append('数据初始化失败 \n 错误原因: {}'.format(e))
            return

    def save_data(self):
        """ 保存文件
        """
        fname, filetype = QFileDialog.getSaveFileName(
            self, 'Save File', '',
            "CSV files (*.csv);; Excel files (*.xlsx *.xls);; MAT files (*.mat);; All files (*)"
        )
        if not fname:
            return
        try:
            if fname.endswith('.csv'):
                self.df.to_csv(fname, encoding='utf-8')
            elif fname.endswith('.xlsx' or '.xls'):
                self.df.to_excel(fname, encoding='utf-8')
            elif fname.endswith('.mat'):
                tempdata = {
                    col_name: self.df[col_name].values
                    for col_name in self.df.columns.values
                }
                sio.savemat(fname, tempdata)
            else:
                self.textBrowser.append('无法保存为该文件类型')
        except Exception as e:
            self.textBrowser.append('文件保存失败: {}\n 错误原因: {}'.format(fname, e))
            return

        self.textBrowser.append('文件已保存于: {}'.format(fname))

    def image_plot(self):
        """ 根据绘图设置进行图像绘制
        """
        try:
            x_label = self.lineEdit_datax.text()
            data_x = self.get_data(x_label)
            y_label = self.lineEdit_datay.text()
            data_y = self.get_data(y_label)
            z_label = self.lineEdit_dataz.text()
            data_z = self.get_data(z_label)
            method = self.comboBox_fitmethod.currentIndex()
            if self.radioButton_2D.isChecked():
                fig = plt.figure()
                ax = fig.add_subplot(111)
                if self.radioButton_2Dline.isChecked():
                    self.plot_2Dline(ax, data_x, data_y)
                elif self.radioButton_2Dscatter.isChecked():
                    self.plot_2Dscatter(ax, data_x, data_y, method)
                elif self.radioButton_2Dhist.isChecked():
                    self.plot_2Dhist(ax, data_x)
                # elif self.radioButton_2Dpie.isChecked():
                #     self.plot_2Dpie(ax, data_x)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                if self.radioButton_3Dline.isChecked():
                    self.plot_3Dline(ax, data_x, data_y, data_z)
                elif self.radioButton_3Dscatter.isChecked():
                    self.plot_3Dscatter(ax, data_x, data_y, data_z)
                elif self.radioButton_3Dsurface.isChecked():
                    self.plot_3Dsurface(ax, data_x, data_y, data_z)
            ax.legend()
            plt.show()
        except Exception as e:
            self.textBrowser.append('无法绘制该图形！\n 错误原因: {}'.format(e))
            # 关闭当前错误 figure
            plt.close(fig)

    def get_data(self, label):
        """ 根据序号或者特征名返回数据
        """
        data = []
        try:
            if not label:
                return
            label_list = label.split(',')
            for index in label_list:
                try:
                    i = int(index)
                    data.append(self.df.iloc[:, i])
                except:
                    data.append(self.df[index])
            return data
        except:
            return

    def plot_2Dline(self, ax, x, y):
        plt.xlabel(x[0].name)
        for y_ in y:
            ax.plot(x[0], y_, label=y_.name)

    def plot_2Dscatter(self, ax, x, y, method):
        plt.xlabel(x[0].name)
        for y_ in y:
            ax.scatter(x[0], y_, label=y_.name)
            if self.checkBox_fit.isChecked():
                yvals, func_exp, r_square = self.scatter_fit(x[0], y_, method)
                ax.plot(x[0], yvals, '--', label='{}-fit'.format(y_.name))
                self.textBrowser.append('{} 拟合表达式为：\n'.format(y_.name) +
                                        func_exp)
                self.textBrowser.append('R-square: {}'.format(r_square))

    def scatter_fit(self, x, y, method):
        """二维散点图曲线拟合

        :param x: input 1-dArray x
        :param y: input 1-dArray y
        :param method: int, 0: 多项式拟合; 1: e 指数拟合; 2: 对数拟合
        :param n: 多项式拟合阶数

        :return yvals: 拟合后 1-dArray yvals
        :return func_exp: string, 拟合函数表达式
        :return r_square: float, 确定系数(R-square), range(0,1)
        """

        def func_log(x, a, b):
            return a * np.log(x) + b

        def func_e(x, a, b, c):
            return a * np.exp(b * x) + c

        def compute_r(y, yvals):
            ybar = np.sum(y) / len(y)
            ssreg = np.sum((yvals - ybar)**2)
            sstot = np.sum((y - ybar)**2)
            return ssreg / sstot

        if method == 0:
            n = int(self.comboBox_2d_n.currentText())
            popt = np.polyfit(x, y, n)
            func_exp = '{}'.format(np.poly1d(popt))
            yvals = np.polyval(popt, x)

        elif method == 1:
            popt, _ = curve_fit(func_e, x, y)
            func_exp = 'y = {} * exp({} * x) + {}'.format(
                popt[0], popt[1], popt[2])
            yvals = func_e(x, popt[0], popt[1], popt[2])

        elif method == 2:
            popt, _ = curve_fit(func_log, x, y)
            func_exp = 'y = {} * log(x) + {}'.format(popt[0], popt[1])
            yvals = func_log(x, popt[0], popt[1])

        r_square = compute_r(y, yvals)
        return (yvals, func_exp, r_square)

    def plot_2Dhist(self, ax, x):
        for x_ in x:
            ax.hist(x_, normed=True, histtype='bar', label=x_.name)
            if self.checkBox_histfit.isChecked():
                x_.plot(
                    kind='kde', style='r--', label='{}-kde'.format(x_.name))

    # def plot_2Dpie(self, ax, x):
    #     ax.pie(x[0], startangle=90, autopct='%1.1f%%')
    #     # Equal aspect ratio ensures that pie is drawn as a circle
    #     ax.axis('equal')

    def plot_3Dline(self, ax, x, y, z):
        plt.xlabel(x[0].name)
        plt.ylabel(y[0].name)
        for z_ in z:
            ax.plot(x[0], y[0], z_, label=z_.name)

    def plot_3Dscatter(self, ax, x, y, z):
        plt.xlabel(x[0].name)
        plt.ylabel(y[0].name)
        for z_ in z:
            ax.scatter(x[0], y[0], z_, label=z_.name)
            # if self.checkBox_3dscatterfit.isChecked():
            #     # ax.plot_trisurf(x[0], y[0], z_, label='{}-fit'.format(z_.name))
            #     data = np.c_[x[0], y[0], z_]
            #     mn = np.min(data, axis=0)
            #     mx = np.max(data, axis=0)
            #     X, Y = np.meshgrid(
            #         np.linspace(mn[0], mx[0], 20),
            #         np.linspace(mn[1], mx[1], 20))
            #     XX = X.flatten()
            #     YY = Y.flatten()
            #     if int(n) == 1:
            #         A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
            #         C, _, _, _ = lstsq(A, data[:, 2])
            #         Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)],
            #                    C).reshape(X.shape)
            #         ax.plot_surface(X, Y, Z, label='fit-surface')
            #     elif int(n) == 2:
            #         A = np.c_[np.ones(data.shape[0]), data[:, :2],
            #                   np.prod(data[:, :2], axis=1), data[:, :2]**2]
            #         C, _, _, _ = lstsq(A, data[:, 2])
            #         Z = np.dot(
            #             np.c_[np.ones(XX.shape), XX, YY, XX *
            #                   YY, XX**2, YY**2], C).reshape(X.shape)
            #         ax.plot_surface(X, Y, Z, label='fit-surface')

    def plot_3Dsurface(self, ax, x, y, z):
        x_, y_ = np.meshgrid(x[0], y[0])
        ax.plot_surface(x_, y_, z[0], cmap='hot')

    def check_NaN(self):
        """ 检查 NaN 值分布情况，并给出建议
        """
        temp = sum(
            [any(self.df.iloc[i].isnull()) for i in range(self.df.shape[0])])
        self.textBrowser.append('含缺失值总样本数如下所示: {}'.format(temp))
        if temp / self.df.shape[0] * 100 < 1:
            self.textBrowser.append('带有缺失值样本占总样本数小于 1%，建议删除这些样本')
            self.radioButton_delnan.setChecked(True)
        else:
            self.textBrowser.append('带有缺失值样本占总样本数大于 1%，建议使用均值替代缺失值')
            self.radioButton_fillnan.setChecked(True)

    def deal_with_NaN(self):
        """ 对 NaN 进行处理
        """
        self.df_temp = self.df.copy()
        if self.radioButton_delnan.isChecked():
            self.df.dropna(how='any', inplace=True)
        elif self.radioButton_fillnan.isChecked():
            try:
                self.df.fillna(self.df.mean(), inplace=True)
            except:
                self.textBrowser.append('无法对该数据集使用均值代替 NaN')
                return
        else:
            return
        self.load_data_to_table()
        self.textBrowser.append('缺失值处理已完成')

    def nondim_action(self):
        """ 无量纲化处理
        """
        self.df_temp = self.df.copy()
        columns = self.df.columns.values
        if self.radioButton_01standard.isChecked():
            try:
                min_max_scaler = preprocessing.MinMaxScaler()
                df_01 = min_max_scaler.fit_transform(self.df)
                self.df = pd.DataFrame(df_01, columns=columns)
            except:
                self.textBrowser.append('无法对该数据集进行归一化处理')
                return
        elif self.radioButton_zstandard.isChecked():
            try:
                df_scale = preprocessing.scale(self.df)
                self.df = pd.DataFrame(df_scale, columns=columns)
            except:
                self.textBrowser.append('无法对该数据集进行标准化处理')
                return
        elif self.radioButton_normalizer.isChecked():
            try:
                df_normalized = preprocessing.normalize(self.df, norm='l2')
                self.df = pd.DataFrame(df_normalized, columns=columns)
            except Exception as e:
                self.textBrowser.append('无法对该数据集进行正则化处理 \n 错误原因: {}'.format(e))
                return
        else:
            return
        self.load_data_to_table()
        self.textBrowser.append('数据无量纲化处理已完成')

    def repeal_action(self):
        """ 预处理撤回操作
        """
        try:
            self.model.setDataFrame(self.df_temp)
            self.df = self.df_temp.copy()
            self.textBrowser.append('已撤销该操作')
        except Exception as e:
            self.textBrowser.append('撤销失败 \n 错误原因: {}'.format(e))
            return

    def set_page_show(self):
        """ 根据图像类型单选按钮状态，调整页面显示
        """
        if self.radioButton_2D.isChecked():
            self.groupBox_2D.setEnabled(True)
            self.groupBox_3D.setEnabled(False)
            self.groupBox_fit.setEnabled(False)
            self.checkBox_histfit.setEnabled(False)
            if self.radioButton_2Dline.isChecked(
            ) or self.radioButton_2Dscatter.isChecked():
                self.lineEdit_datax.setEnabled(True)
                self.lineEdit_datay.setEnabled(True)
                self.lineEdit_dataz.setEnabled(False)
                if self.radioButton_2Dscatter.isChecked():
                    self.groupBox_fit.setEnabled(True)
                    self.checkBox_fit.setEnabled(True)
                    self.comboBox_fitmethod.setEnabled(True)
                    self.comboBox_2d_n.setEnabled(False)
                    if self.comboBox_fitmethod.currentIndex() == 0:
                        self.comboBox_2d_n.setEnabled(True)

            elif self.radioButton_2Dhist.isChecked():
                self.lineEdit_datax.setEnabled(True)
                self.lineEdit_datay.setEnabled(False)
                self.lineEdit_dataz.setEnabled(False)
                if self.radioButton_2Dhist.isChecked():
                    self.checkBox_histfit.setEnabled(True)
        else:
            self.groupBox_2D.setEnabled(False)
            self.groupBox_3D.setEnabled(True)

            self.lineEdit_datax.setEnabled(True)
            self.lineEdit_datay.setEnabled(True)
            self.lineEdit_dataz.setEnabled(True)

    def closeEvent(self, event):
        """ 关闭程序
        """
        result = QMessageBox.question(self, 'Confirm Exit...',
                                      'Are you sure to exit?',
                                      QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            self.close_signal.emit()
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
