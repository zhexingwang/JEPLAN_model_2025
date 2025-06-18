#coding: utf-8 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import Normalize
plt.rcParams['font.family'] = 'Times New Roman'

from .get_eval import get_vio_class
from .get_eval import get_obj_class

class figure_save_class:
    def get_contour_data(self, prob, delta):
        mesh = 100
        x_ul = prob.x_ul
        N = len(x_ul[0])
        
        x = np.zeros((N, mesh, mesh))
        if N == 2:
            xx = np.linspace(x_ul[0, 0], x_ul[1, 0], mesh)
            yy = np.linspace(x_ul[0, 1], x_ul[1, 1], mesh)
            # X: [mesh, mesh] xbox in mesh
            # Y: [mesh, mesh] ybox in mesh
            X, Y = np.meshgrid(xx, yy)
            x[0, :, :] = X
            x[1, :, :] = Y
        else:
            for n in range(0, int(N/2)):
                j = int(2*n)
                xx = np.linspace(x_ul[0, j], x_ul[1, j], mesh)
                yy = np.linspace(x_ul[0, j+1], x_ul[1, j+1], mesh)
                X, Y = np.meshgrid(xx, yy)
                x[j, :, :] = X
                x[j+1, :, :] = Y
    
        sample_x = x[:, 0, 0]
        (g, h) = get_vio_class().get_vio(prob, sample_x)
        num_g = len(g)
        num_h = len(h)
        num_c = num_g + num_h
        obj_box_ = np.zeros((mesh, mesh))
        vio_box_ = np.zeros((num_c, mesh, mesh))
        for i in range(0, mesh):
            for j in range(0, mesh):
                #x_ : (N, mesh, mesh)
                x_ = x[:, i, j]
                obj_box_[i, j] = get_eval_class().get_eval(x_, prob)
                (g, h) = get_vio_class().get_vio(prob, x_)
                vio_box_[:num_g,i,j] = g
                for jj in range(0, num_h):
                    if abs(h[jj]) < delta:
                        vio_box_[num_g+jj,i,j] = 0
                    else:
                        vio_box_[num_g+jj,i,j] = abs(h[jj])
        vio_box_ = np.where(vio_box_<0, 0, vio_box_)
        vio_box_ = np.sum(vio_box_, axis=0)
        fea_box = np.where(vio_box_==0, -1, 1)
        #all_fea_box = np.all(fea_box==-1, axis=0)
        #all_fea_box = np.where(all_fea_box == True, -1, 1)

        return X, Y, obj_box_, vio_box_, fea_box, x_ul

    def trend(self, figure_label, x, y, fig_file_name, scale='linear'):
        def _yscale(ax, scale, y_min, y_max, mask_type):
            if scale == 'linear':
                ax.set_ylim([y_min, y_max])
            elif scale == 'log':
                if mask_type:
                    ax.set_yscale(value=scale, nonpositive='mask')
                else:
                    ax.set_yscale(value=scale)
        x_label_name = figure_label[0]
        y_label_name = figure_label[1]
        y_min = figure_label[2]
        y_max = figure_label[3]

        fig = plt.figure(figsize=(6,4))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, y, "C0", lw=0.5)
        ax.set_xlabel(x_label_name, fontsize=16)
        ax.set_ylabel(y_label_name, fontsize=16)
        ax.tick_params(labelsize=12)
        #ax.set_xlim([x_min, x_max])
        _yscale(ax, scale, y_min, y_max, mask_type=True)

        plt.tick_params(labelsize=12, direction = "in")
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")


    def double_trend(self, figure_label, x, y, fig_file_name, scale1='linear', scale2='linear'):
        def _yscale(ax, scale, y_min, y_max, mask_type):
            if scale == 'linear':
                ax.set_ylim([y_min, y_max])
            elif scale == 'log':
                if mask_type:
                    ax.set_yscale(value=scale, nonpositive='mask')
                else:
                    ax.set_yscale(value=scale)

        x_label_name = figure_label[0]
        y1_label_name = figure_label[1]
        y2_label_name = figure_label[2]
        legend_name1 = figure_label[3]
        legend_name2 = figure_label[4]
        y1_min = figure_label[5]
        y1_max = figure_label[6]
        y2_min = figure_label[7]
        y2_max = figure_label[8]

        fig = plt.figure(figsize=(6,4))
        fig.patch.set_facecolor('white')
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(x, y[:,0], "C0", label=legend_name1, lw=0.5)
        ax1.set_xlabel(x_label_name, fontsize=16)
        ax1.set_ylabel(y1_label_name, fontsize=16)
        ax1.tick_params(labelsize=12)
        #ax.set_xlim([x_min, x_max])
        _yscale(ax1, scale1, y1_min, y1_max, mask_type=True)

        ax2 = ax1.twinx()
        ax2.plot(x, y[:,1], "C1", label=legend_name2, lw=0.5)
        ax2.set_ylabel(y2_label_name, fontsize=16)
        _yscale(ax2, scale2, y2_min, y2_max, mask_type=False)

        
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')

        plt.tick_params(labelsize=12, direction = "in")
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")


    def gbest_trend_uplow_dis(self, figure_label, x, y1, y21, y22, y23, fig_file_name):
        x_label_name = figure_label[0]
        y1_label_name = figure_label[1]
        y2_label_name = figure_label[2]
        legend_name1 = figure_label[3]
        legend_name21 = figure_label[4]
        legend_name22 = figure_label[5]
        legend_name23 = figure_label[6]
        y1_min = figure_label[7]
        y1_max = figure_label[8]
        y2_min = figure_label[9]
        y2_max = figure_label[10]

        fig = plt.figure(figsize=(6,4))
        fig.patch.set_facecolor('white')
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(x, y1, "C0", label=legend_name1, lw=0.5)
        ax1.set_xlabel(x_label_name, fontsize=16)
        ax1.set_ylabel(y1_label_name, fontsize=16)
        ax1.tick_params(labelsize=12)
        #ax.set_xlim([x_min, x_max])
        ax1.set_ylim([y1_min, y1_max])

        ax2 = ax1.twinx()
        ax2.plot(x, y21, "C1", label=legend_name21, lw=0.5)
        ax2.plot(x, y22, "C2", label=legend_name22, lw=0.5)
        ax2.plot(x, y23, "C3", label=legend_name23, lw=0.5)
        ax2.set_ylabel(y2_label_name, fontsize=16)
        ax2.set_ylim([y2_min, y2_max])


        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')

        plt.tick_params(labelsize=12, direction = "in")
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")



    def all_trend(self, figure_label, x, y1, y2, fig_file_name):
        x_label_name = figure_label[0]
        y1_label_name = figure_label[1]
        y2_label_name = figure_label[2]
        legend_name1 = figure_label[3]
        legend_name2 = figure_label[4]
        y1_min = figure_label[5]
        y1_max = figure_label[6]
        y2_min = figure_label[7]
        y2_max = figure_label[8]
        m = len(y1[0])

        handles = []
        labels = ["C0", "C1"]

        fig = plt.figure(figsize=(6,4))
        fig.patch.set_facecolor('white')
        ax1 = fig.add_subplot(1,1,1)
        for i in range(0, m):
            if i == 0:
                ax1.plot(x, y1[:, i], "C0", label=legend_name1, lw=0.5, marker="o", markeredgecolor="C0", markeredgewidth=0.5, markerfacecolor="None", markersize=5, linestyle='None')
            else:
                ax1.plot(x, y1[:, i], "C0", lw=0.5, marker="o", markeredgecolor="C0", markeredgewidth=0.5, markerfacecolor="None", markersize=5, linestyle='None')
        ax1.set_xlabel(x_label_name, fontsize=16)
        ax1.set_ylabel(y1_label_name, fontsize=16)
        ax1.tick_params(labelsize=12)
        #ax.set_xlim([x_min, x_max])
        ax1.set_ylim([y1_min, y1_max])

        ax2 = ax1.twinx()
        for i in range(0, m):
            if i == 0:
                ax2.plot(x, y2[:, i], "C1", label=legend_name2, lw=0.5, marker="o", markeredgecolor="C1", markeredgewidth=0.5, markerfacecolor="None", markersize=5, linestyle='None')
            else:
                ax2.plot(x, y2[:, i], "C1", lw=0.5, marker="o", markeredgecolor="C1", markeredgewidth=0.5, markerfacecolor="None", markersize=5, linestyle='None')
        ax2.set_ylabel(y2_label_name, fontsize=16)
        ax2.set_ylim([y2_min, y2_max])


        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')

        plt.tick_params(labelsize=12, direction = "in")
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")



    def scatter_solution_space_contour(self, figure_label, x1, x2, gbest, X, Y, obj_mesh_box, feas_mesh_box, fig_file_name):
        x_label_name = figure_label[0]
        y_label_name = figure_label[1]
        label_name1 = figure_label[2]
        label_name2 = figure_label[3]
        x_min = figure_label[4]
        x_max = figure_label[5]
        y_min = figure_label[6]
        y_max = figure_label[7]

        cmax = np.max(obj_mesh_box) - 50
        cmin = np.min(obj_mesh_box)
        n_delta = 200
        levels_array = np.linspace(cmin, cmax, n_delta)


        fig = plt.figure(figsize=(5,5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)

        # contour
        ax.contour(X, Y, obj_mesh_box, linestyles='dashdot', alpha=1.0, cmap='coolwarm', levels=levels_array, norm=Normalize(vmin=cmin, vmax=cmax), linewidths = 0.5)
        #for g in range(0, len(feas_mesh_box[:, 0, 0])):
        #    ax.contourf(X, Y, feas_mesh_box[g, :, :], alpha=0.4, cmap='Greys')
        ax.contourf(X, Y, feas_mesh_box, alpha=0.3, cmap='Greys')
 
        # scatter
        ax.scatter(x1, x2, color="C0", s=40, alpha=1.0, marker="o", label=label_name1)

        # gbest scatter
        ax.scatter(gbest[0], gbest[1], s=20, marker="s", edgecolor="C1", facecolor="None", label=label_name2)


        ax.legend(loc='upper right')

        ax.set_xlabel(x_label_name, fontsize=16)
        ax.set_ylabel(y_label_name, fontsize=16)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(direction = "in")

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")



    def scatter_solution_space(self, figure_label, x1, x2, gbest, fig_file_name):
        x_label_name = figure_label[0]
        y_label_name = figure_label[1]
        label_name1 = figure_label[2]
        label_name2 = figure_label[3]
        x_min = figure_label[4]
        x_max = figure_label[5]
        y_min = figure_label[6]
        y_max = figure_label[7]


        fig = plt.figure(figsize=(5,5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)
 
        # scatter
        ax.scatter(x1, x2, color="C0", s=40, alpha=1.0, marker="o", label=label_name1)

        # gbest scatter
        ax.scatter(gbest[0], gbest[1], s=20, marker="s", edgecolor="C1", facecolor="None", label=label_name2)


        ax.legend(loc='upper right')

        ax.set_xlabel(x_label_name, fontsize=16)
        ax.set_ylabel(y_label_name, fontsize=16)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(direction = "in")

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")


    def scatter_obj_vio_space_contour(self, figure_label, x1, x2, gbest, obj_mesh_box, vio_mesh_box, feas_mesh_box, fig_file_name):
        x_label_name = figure_label[0]
        y_label_name = figure_label[1]
        label_name1 = figure_label[2]
        label_name2 = figure_label[3]
        x_min = figure_label[4]
        x_max = figure_label[5]
        y_min = figure_label[6]
        y_max = figure_label[7]


        fig = plt.figure(figsize=(5,5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)

        # contour
        #for g in range(0, len(vio_mesh_box[:, 0, 0])):
        for i in range(0, len(obj_mesh_box[:, 0])):
            ax.scatter(obj_mesh_box[i, :], vio_mesh_box[i, :], color="lightgray", alpha=0.3, s=20, marker="+", edgecolor='face')

        # xi scatter
        ax.scatter(x1, x2, color="C0", s=40, marker="o", label=label_name1)

        # gbest scatter
        ax.scatter(gbest[0], gbest[1], s=20, marker="s", edgecolor="C1", facecolor="None", label=label_name2)
 
        ax.legend(loc='upper right')

        ax.set_xlabel(x_label_name, fontsize=16)
        ax.set_ylabel(y_label_name, fontsize=16)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(direction = "in")

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")



    def scatter_obj_vio_space(self, figure_label, x1, x2, gbest, fig_file_name):
        x_label_name = figure_label[0]
        y_label_name = figure_label[1]
        label_name1 = figure_label[2]
        label_name2 = figure_label[3]
        x_min = figure_label[4]
        x_max = figure_label[5]
        y_min = figure_label[6]
        y_max = figure_label[7]


        fig = plt.figure(figsize=(5,5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)

        # xi scatter
        ax.scatter(x1, x2, color="C0", s=40, marker="o", label=label_name1)

        # gbest scatter
        ax.scatter(gbest[0], gbest[1], s=20, marker="s", edgecolor="C1", facecolor="None", label=label_name2)
 
        ax.legend(loc='upper right')

        ax.set_xlabel(x_label_name, fontsize=16)
        ax.set_ylabel(y_label_name, fontsize=16)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(direction = "in")

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")



    def scatter(self, figure_label, x1, x2, fig_file_name):
        x_label_name = figure_label[0]
        y_label_name = figure_label[1]
        x_min = figure_label[2]
        x_max = figure_label[3]
        y_min = figure_label[4]
        y_max = figure_label[5]


        fig = plt.figure(figsize=(5,5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1,1,1)
 
        ax.scatter(x1, x2, color="C0", s=40, marker="o")

        ax.set_xlabel(x_label_name, fontsize=16)
        ax.set_ylabel(y_label_name, fontsize=16)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(direction = "in")

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(fig_file_name, bbox_inches="tight")
