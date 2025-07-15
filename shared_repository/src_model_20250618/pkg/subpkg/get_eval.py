#coding: utf-8
import numpy as np
import pandas as pd

class scaling_class:
    def scaling_values(self, prob_, values_):
        def min_max(x, axis=None):
            eps = pow(10,-15)
            min_ = x.min(axis=axis, keepdims=True)
            max_ = x.max(axis=axis, keepdims=True)
            return (x - min_)/(max_ - min_ + eps)

        if prob_.scaling_type == 1:
            values = values_
        elif prob_.scaling_type == 2:
            # values: (m, g_num)
            if values_.ndim > 1:
                values = np.array([min_max(values_[:, j]) for j in range(0, values_.shape[1])]).T
            # values: (m, 1)
            else:
                values = min_max(values_)
        return values

class get_obj_class:
    def get_obj(self, prob, x_):
        # x_: (m, N)
        if x_.ndim > 1:
            # obj_: (m, 1)
            obj_ = np.array([prob.object_function(x_[i, :]) for i in range(0, x_.shape[0])])
        # x_: (N)
        else:
            # obj_: (1)
            obj_ = prob.object_function(x_)
        return obj_

    def get_opt_sol(self, prob):
        opt_sol = []
        for i in range(0, prob.num_of_signs):
            opt_sol.append((-prob.sign_entries[i][0], prob.sign_entries[i][3]))
        return opt_sol


    def get_filter_penalty(self, x_, archive, pena_para):
        penalty_coef = pena_para[2]
        penalty = penalty_coef*np.sum(x_) / len(x_)
        return penalty


class get_vio_class:
    def get_rank(self, array_):
        clm = ["value", "rank"]
        df_ = pd.DataFrame(np.zeros((len(array_), 2)), columns=clm)
        df_[clm[0]] = array_
        list_ = list(set(list(array_)))
        list_.sort()
        rank = 0
        for ele_ in list_:
            df_.loc[df_[clm[0]]==ele_, clm[1]] = rank
            rank = rank + len(df_.loc[df_[clm[0]]==ele_, clm[1]])
        rank_ = df_[clm[1]].values
        return rank_

    def get_each_vio(self, prob_, x_):
        # x_: (m, N)
        if x_.ndim > 1:
            # each_g: (m, L)
            each_g = np.array([prob_.constraint_function(x_[i, :]) for i in range(0, x_.shape[0])])
        # x_: (N)
        else:
            # each_g: (L)
            each_g = prob_.constraint_function(x_)
        each_vio = np.where(each_g<0, 0, each_g)
        return each_vio


    # vio = sigma_{j=1, L} orm_j
    # vio = sigma_{j=1, L} (orm_max - orm_j / orm_max - orm_min)
    def get_vio(self, prob_, x_, type_vio = 'sum'):
        # each_vio: (m, L) or (L)
        each_vio = self.get_each_vio(prob_, x_)
        # each_vio (m) and x (m, N)
        if each_vio.ndim == 1 and x_.ndim > 1:
            vio = each_vio
        # each_vio (1) and x (N)
        elif each_vio.ndim == 0 and x_.ndim == 1:
            vio = each_vio
        else:
            if type_vio == 'sum':
                if each_vio.ndim >= 2:
                    vio = np.sum(each_vio, axis=1)
                elif each_vio.ndim == 1:
                    vio = np.sum(each_vio)
            elif type_vio == 'max':
                if each_vio.ndim >= 2:
                    vio = np.max(each_vio, axis=1)
                elif each_vio.ndim == 1:
                    vio = np.max(each_vio)
        return vio, each_vio

    def get_const_ranking(self, prob_, x_, delta):
        # num points
        m_ = len(x_)
            
        # constraints violation
        (v, Nv, M_g, M_h) = self.get_vio_each_const(prob_, x_, delta)

        # v: (m_, M_g+M_h)
        # v_sum: (m_, 1) sum violation
        v_sum = np.sum(v, axis=1)

        # num of constraint violation ranking
        rank_Nv = self.get_rank(Nv) + 1

        # constraints violation ranking
        rank_v = np.zeros((m_, M_g+M_h))
        for j in range(0, M_g+M_h):
            rank_v[:, j] = self.get_rank(v[:, j]) + 1

        # rank_v: (m_, M_g+M_h)
        rank_v_df = pd.DataFrame(np.concatenate([rank_Nv.reshape(m_, 1), rank_v], 1))
        rank_v_df["sum v_rank"] = rank_v_df.sum(axis = 1)

        # total ranking
        v_rank = self.get_rank(rank_v_df["sum v_rank"].values)
        rank_v_df["rank"] = v_rank

        return v_rank, v_sum


class Superior:
    def get_clm(self, eval_type):
        return eval_type-1

    def eval_scaler(self, prob, x_):
        if x_.ndim > 1:
            obj = np.zeros((x_.shape[0], 5))
            obj[:, 0] = get_obj_class().get_obj(prob, x_)
            (obj[:, 1], each_vio) = get_vio_class().get_vio(prob, x_)
            obj[:, 2] = obj[:, 1] + obj[:, 0]
            obj[:, [3,4]] = obj[:, [0,1]].copy()
        else:
            obj = np.zeros(5)
            obj[0] = get_obj_class().get_obj(prob, x_)
            (obj[1], each_vio) = get_vio_class().get_vio(prob, x_)
            obj[2] = obj[1] + obj[0]
            obj[[3,4]] = obj[[0,1]].copy()
        return obj, each_vio
    
    def get_min_idx_array(self, A, eval_type):
        # argmin index
        if eval_type <= 3:
            clm = self.get_clm(eval_type)
            idx_ = np.argmin(A[:, clm])
        elif eval_type == 4:
            if np.any(A[:, 1] == 0):
                idx__ = np.where(A[:, 1] == 0)[0]
                idx_ = idx__[np.argmin(A[idx__, 0])]
            else:
                idx_ = np.argmin(A[:, 1])
        return idx_
    
    def get_rank(self, A, eval_type):
        if eval_type <= 3:
            clm = self.get_clm(eval_type)
            idx_ = np.argsort(A[:, clm])
        elif eval_type == 4:
            if np.any(A[:, 1] == 0):
                idx_feas = np.where(A[:, 1] == 0)[0]
                idx_infeas = np.where(A[:, 1] > 0)[0]
                idx_ = np.concatenate([idx_feas[np.argsort(A[idx_feas, 0])], idx_infeas[np.argsort(A[idx_infeas, 1])]])
            else:
                idx_ = np.argsort(A[:, 1])
        return idx_

    def get_sup_idx_array(self, A, B, eval_type):
        # A>B => index True
        idx_ = []
        if (A.ndim > 1) and (B.ndim > 1):
            if eval_type <= 3:
                clm = self.get_clm(eval_type)
                idx_ = np.where(A[:, clm] > B[:, clm])[0]
            elif eval_type == 4:
                idx_ = [i for i in range(0, A.shape[0]) if self.get_sup_bool_sca(A[i, :], B[i, :], eval_type)]
        elif (A.ndim == 1) and (B.ndim > 1):
            if eval_type <= 3:
                clm = self.get_clm(eval_type)
                idx_ = np.where(A[clm] > B[:, clm])[0]
            elif eval_type == 4:
                if np.any(A[1] == B[:, 1]):
                    idx__ = np.where(A[1] == B[:, 1])[0]
                    if np.any(A[0] > B[idx__, 0]):
                        idx_ = idx__[np.where(A[0] > B[idx__, 0])]
                elif np.any(A[1] > B[:, 1]):
                    idx_ = np.where(A[1] > B[:, 1])[0]
        return idx_

    def get_sup_bool_sca(self, A, B, eval_type):
        # A>B => True
        bool_ = False
        if eval_type <= 3:
            clm = self.get_clm(eval_type)
            bool_ = (A[clm] > B[clm])
        elif eval_type == 4:
            if (A[1] == B[1]) or (A[1] == 0 and B[1] == 0):
                bool_ = (A[0] > B[0])
            else:
                bool_ = (A[1] > B[1])
        return bool_
