import os
from datetime import datetime
import casadi as cs
import time
import numpy as np
import torch
import config
from multiprocessing import Process, Lock, Event
from multiprocessing.shared_memory import SharedMemory
from utils import sm_array


def rbf_ard_kernel(x, x_prime, sigma_f, length_scales):
    if len(x_prime.shape) != 2:
        x_prime = x_prime.unsqueeze(0)
    x_scaled = x / torch.sqrt(length_scales)
    x_prime_scaled = x_prime / torch.sqrt(length_scales)
    xx = (x_scaled ** 2).sum(dim=1).unsqueeze(1)
    yy = (x_prime_scaled ** 2).sum(dim=1).unsqueeze(0)
    xy = x_scaled @ x_prime_scaled.t()
    sqdist = xx + yy - 2 * xy
    return sigma_f * torch.exp(-0.5 * sqdist)


class GaussianProcess:
    MAX_NUM_DATA = 450
    X_SIZE = 5
    Y_SIZE = 3
    def __init__(self, n_inducing, smm, data_queue):
        self.n_inducing = n_inducing
        self.data_queue = data_queue
        self.sigma_fs = torch.tensor([0.0954, 0.1284, 2.1050])
        self.length_scales = torch.tensor([
            [1.7374, 1.5327, 1.6253, 1.4038, 1.6207],
            [2.2494, 1.4749, 1.3088, 1.9541, 2.0376],
            [0.2958, 0.0965, 0.2554, 0.1502, 1.1564]
        ])
        self.sigma_ns = torch.tensor([0.1012, 0.0960, 0.0970])

        self.kill_event = Event()
        self.shared_state_lock = Lock()

        n_bytes = self.Y_SIZE * 8
        self.num_data_shared_memory = smm.SharedMemory(size=n_bytes)
        self.shared_num_data = sm_array(self.num_data_shared_memory, self.Y_SIZE, dtype=np.int64)

        n_bytes = self.Y_SIZE * self.MAX_NUM_DATA * 8
        self.knn_diag_shared_memory = smm.SharedMemory(size=n_bytes)
        self.shared_knn_diag = sm_array(self.knn_diag_shared_memory, (self.Y_SIZE, self.MAX_NUM_DATA), dtype=np.float64)

        n_bytes = self.Y_SIZE * self.MAX_NUM_DATA * 2 * max(self.X_SIZE, self.Y_SIZE) * 8
        self.xy_shared_memory = smm.SharedMemory(size=n_bytes)
        self.shared_xy = sm_array(
            self.xy_shared_memory, (self.Y_SIZE, self.MAX_NUM_DATA, 2, max(self.X_SIZE, self.Y_SIZE)),
            dtype=np.float64
        )

        self.gp_update_process = Process(target=self._update_gp, args=(
            self.sigma_fs, self.length_scales,
            self.num_data_shared_memory.name, self.knn_diag_shared_memory.name, self.xy_shared_memory.name,
            self.shared_state_lock, data_queue))
        self.gp_update_process.start()

    def _update_gp(self, sigma_fs, length_scales,
                   num_data_memory_name, knn_memory_name, xy_memory_name, shared_state_lock, data_queue):

        num_data_shared_memory = SharedMemory(num_data_memory_name)
        shared_num_data = sm_array(num_data_shared_memory, self.Y_SIZE, dtype=np.int64)
        num_data = np.zeros(self.Y_SIZE, dtype=np.int64)

        knn_shared_memory = SharedMemory(knn_memory_name)
        shared_knn_diag = sm_array(knn_shared_memory, (self.Y_SIZE, self.MAX_NUM_DATA), dtype=np.float64)
        knn = torch.zeros((self.Y_SIZE, self.MAX_NUM_DATA, self.MAX_NUM_DATA), dtype=torch.float64)

        xy_shared_memory = SharedMemory(xy_memory_name)
        shared_xy = sm_array(
            xy_shared_memory, (self.Y_SIZE, self.MAX_NUM_DATA, 2, self.X_SIZE),
            dtype=np.float64
        )
        xy = torch.zeros((self.Y_SIZE, self.MAX_NUM_DATA, 2, self.X_SIZE+2), dtype=torch.float64)

        variances = torch.zeros(self.Y_SIZE, self.MAX_NUM_DATA)
        timestamps = torch.zeros(self.Y_SIZE, self.MAX_NUM_DATA)

        errors = []


        Ls = None
        while True:
            # state:
            # 0 vx
            # 1 vy
            # 2 w
            # 3 steer
            # 4 throttle
            # 5 ax
            # 6 ay
            # 7 dw
            pos_info, input_state, output_state = data_queue.get()

            if pos_info is None:
                date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
                result_folder = "session/%s" % date_time
                os.makedirs(result_folder)
                np.save(f"{result_folder}/errors.npy", np.asarray(errors))
                for i in range(self.Y_SIZE):
                    np.save(f"{result_folder}/xy_{i}.npy", xy[i, :num_data[i], :, :].numpy())

                num_data_shared_memory.close()
                knn_shared_memory.close()
                xy_shared_memory.close()
                return

            pos_info = torch.from_numpy(pos_info).to(torch.float64)
            input_state = torch.from_numpy(input_state).to(torch.float64)
            output_state = torch.from_numpy(output_state).to(torch.float64)
            timestamp = time.time_ns()

            mean = torch.zeros(self.Y_SIZE)
            variance = torch.zeros(self.Y_SIZE)
            for output_i in range(self.Y_SIZE):
                kzz = rbf_ard_kernel(input_state[None, :], input_state[None, :], sigma_fs[output_i], length_scales[output_i])
                kzZ = rbf_ard_kernel(input_state[None, :], xy[output_i, :num_data[output_i], 0, :self.X_SIZE], sigma_fs[output_i], length_scales[output_i])
                if Ls is not None:
                    temp = torch.linalg.solve_triangular(Ls[output_i], xy[output_i, :num_data[output_i], 1, 0].unsqueeze(1), upper=False)
                    a = torch.linalg.solve_triangular(Ls[output_i].T, temp, upper=True)
                    mean[output_i] = kzZ @ a
                    v = torch.linalg.solve_triangular(Ls[output_i], kzZ.T, upper=False)
                    variance[output_i] = kzz - v.T @ v

            nominal_error = output_state.numpy()
            corrected_error = output_state.numpy() - mean.numpy()
            errors.append(np.r_[pos_info.numpy()[0], nominal_error, corrected_error, variance.numpy()])

            # In dimensions where the new point has high enough variance, consider replacing an existing point:
            updated_output_dims = []
            for output_i in torch.arange(self.Y_SIZE):
                output_i = output_i.item()
                if num_data[output_i] == self.MAX_NUM_DATA:
                    time_since_s = (timestamp - timestamps[output_i, :num_data[output_i]]) / 1e9
                    time_scalers = torch.exp(-(time_since_s/config.gp_h)**2 / 2)
                    time_adjusted_variances = time_scalers * variances[output_i, :num_data[output_i]]
                    if variance[output_i] > torch.median(time_adjusted_variances):
                        idx = torch.argmin(time_adjusted_variances)
                    else:
                        continue
                else:
                    num_data[output_i] = min(num_data[output_i] + 1, self.MAX_NUM_DATA)
                    idx = num_data[output_i] - 1

                updated_output_dims.append(output_i)
                xy[output_i, idx, 0, :self.X_SIZE] = input_state
                xy[output_i, idx, 0, -2:] = pos_info[1:3]
                xy[output_i, idx, 1, 0] = output_state[output_i]
                timestamps[output_i, idx] = timestamp

            if Ls is None:
                assert len(updated_output_dims) == self.Y_SIZE
                Ls = [None] * self.Y_SIZE

            # For the output dimensions we updated above, recompute the covariance matrix and the posterior variance
            for output_i in updated_output_dims:
                knn[output_i, :num_data[output_i], :num_data[output_i]] = rbf_ard_kernel(
                    xy[output_i, :num_data[output_i], 0, :self.X_SIZE],
                    xy[output_i, :num_data[output_i], 0, :self.X_SIZE],
                    sigma_fs[output_i], length_scales[output_i]
                )
                knn_i = knn[output_i, :num_data[output_i], :num_data[output_i]]
                L = torch.linalg.cholesky(knn_i + self.sigma_ns[output_i] * torch.eye(num_data[output_i]))
                Ls[output_i] = L
                v = torch.linalg.solve_triangular(L, knn_i.T, upper=False)
                variances[output_i, :num_data[output_i]] = torch.diagonal(knn_i - v.T @ v)

            diagonals = {
                output_i: knn[output_i, :num_data[output_i], :num_data[output_i]].diagonal(dim1=0, dim2=1)
                for output_i in updated_output_dims
            }

            with shared_state_lock:
                shared_num_data[:] = num_data
                for output_i in updated_output_dims:
                    shared_knn_diag[output_i, :num_data[output_i]] = diagonals[output_i].numpy()
                    shared_xy[output_i, :num_data[output_i], :, :self.X_SIZE] = xy[output_i, :num_data[output_i], :, :self.X_SIZE].numpy()

            t2 = time.time_ns()
            # print("Took %d ms, with %s entries" % ((t2 - timestamp) / 1e6, num_data.tolist()))

    def _symbolic_rbf_ard_kernel_sx(self):
        x = cs.SX.sym("x", self.n_inducing, self.X_SIZE)
        x_prime = cs.SX.sym("x_prime", 1, self.X_SIZE)
        sigma_f = cs.SX.sym("sigma_f", 1)
        lengthscale = cs.SX.sym("lengthscale", 1, self.X_SIZE)

        diff = ((x.T - x_prime.T)**2 / lengthscale.T).T
        diff = cs.sum2(diff)
        kernel = sigma_f * cs.exp(-0.5 * diff)

        return cs.Function("kernel", [x, x_prime, sigma_f, lengthscale], [kernel])

    def get_mu_symbolic_expression(self, z):
        kernel_creation_fn = self._symbolic_rbf_ard_kernel_sx()
        Zind = cs.MX.sym("Zind", self.n_inducing, self.X_SIZE)

        outputs = cs.vertcat([])
        sym_params_list = [Zind]
        for output_i in range(self.Y_SIZE):
            alpha = cs.MX.sym("alpha_%d" % output_i, self.n_inducing)
            sigma_f = cs.MX.sym("sigma_f_%d" % output_i, 1)
            lengthscale = cs.MX.sym("lengthscale_%d" % output_i, self.X_SIZE)

            sym_params_list.extend([alpha, sigma_f, lengthscale])
            K_Zind_z = kernel_creation_fn(Zind, z, sigma_f, lengthscale)
            outputs = cs.vertcat(outputs, K_Zind_z.T @ alpha)

        params = cs.vcat([cs.reshape(mx, np.prod(mx.shape), 1) for mx in sym_params_list])
        return outputs, params

    def get_mu_jacobian_fun(self):
        z = cs.MX.sym("z", self.X_SIZE)
        outputs, params = self.get_mu_symbolic_expression(z)
        jac_outputs = cs.jacobian(outputs, z)
        return cs.Function("mu_jac", [z, params], [jac_outputs])

    def get_params(self, Zind):
        Zind = torch.from_numpy(Zind).to(torch.float64)

        knn_diags = []
        xys = []
        with self.shared_state_lock:
            num_data = self.shared_num_data.copy()
            for output_i in range(self.Y_SIZE):
                knn_diags.append(torch.from_numpy(self.shared_knn_diag[output_i, :num_data[output_i]].copy()))
                xys.append(torch.from_numpy(self.shared_xy[output_i, :num_data[output_i], :, :].copy()))

        t1 = time.time_ns()

        params_list = [Zind]
        Kmm_list = []
        Qm_list = []
        alpha_list = []
        for output_i in range(self.Y_SIZE):
            if num_data[output_i] > 0:#self.MAX_NUM_DATA/2:
                xy = xys[output_i]
                knn_diag = knn_diags[output_i]
                Kmm = rbf_ard_kernel(Zind, Zind, self.sigma_fs[output_i], self.length_scales[output_i])
                Knm = rbf_ard_kernel(
                    xy[:, 0, :self.X_SIZE], Zind,
                    self.sigma_fs[output_i], self.length_scales[output_i]
                )
                L = torch.linalg.cholesky(Kmm + 0.001 * torch.eye(len(Kmm)))
                terms = torch.linalg.solve_triangular(L, Knm.T, upper=False)
                lamda_diag = knn_diag - torch.sum(terms**2, dim=0)
                lamda_term = 1/(lamda_diag + self.sigma_ns[output_i] * torch.ones(num_data[output_i]))
                Qm_intermediate = Knm.T * lamda_term  # Element-wise multiplication
                Qm = Kmm + Qm_intermediate @ Knm

                L = torch.linalg.cholesky(Qm + 0.001 * torch.eye(len(Kmm)))
                alpha = torch.linalg.solve_triangular(L.T, torch.linalg.solve_triangular(L, Knm.T, upper=False), upper=True)
                alpha = alpha * lamda_term @ xy[:, 1, 0]
                alpha_list.append(alpha)
            else:
                Kmm = None
                Qm = None
                alpha = torch.zeros(len(Zind))
            params_list.extend([alpha, self.sigma_fs[output_i], self.length_scales[output_i]])
            Kmm_list.append(Kmm)
            Qm_list.append(Qm)
            alpha_list.append(alpha)

        t2 = time.time_ns()
        # print("Took total %d ms, with %s entries" % ((t2-t1)/1e6, num_data.tolist()))

        params = np.hstack([p.numpy().flatten(order="F") for p in params_list])
        return params, Kmm_list, Qm_list, alpha_list

    def predict_covar(self, Z, Zind, Kmm, Qm):
        Z = torch.from_numpy(Z).to(torch.float64)
        Zind = torch.from_numpy(Zind).to(torch.float64)

        variances = torch.zeros([self.Y_SIZE, len(Z)])
        for output_i in range(self.Y_SIZE):
            Kzz = rbf_ard_kernel(Z, Z, self.sigma_fs[output_i], self.length_scales[output_i])
            Kmz = rbf_ard_kernel(Zind, Z, self.sigma_fs[output_i], self.length_scales[output_i])

            L1 = torch.linalg.cholesky(Kmm[output_i] + 0.001 * torch.eye(len(Kmm[output_i])))
            v1 = torch.linalg.solve_triangular(L1, Kmz, upper=False)

            L2 = torch.linalg.cholesky(Qm[output_i] + 0.001 * torch.eye(len(Qm[output_i])))
            v2 = torch.linalg.solve_triangular(L2, Kmz, upper=False)

            variances[output_i] = (Kzz - (v1.T @ v1 - v2.T @ v2) + self.sigma_ns[output_i]).diagonal()

        return variances.numpy().T

    def shutdown(self):
        # self.kill_event.set()
        self.data_queue.put((None, None, None))
        self.gp_update_process.join()
        self.num_data_shared_memory.close()
        self.knn_diag_shared_memory.close()
        self.xy_shared_memory.close()

    def __del__(self):
        self.shutdown()