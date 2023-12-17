import time
import grpc
import config
import os
import shutil
import json
import numpy as np
from rpc import mpc_pb2
import traceback
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from rpc.mpc_pb2_grpc import ModelPredictiveControllerServicer, add_ModelPredictiveControllerServicer_to_server
from solver.acados_solver import Solver
from multiprocessing.managers import SharedMemoryManager
from solver.gaussian_process import GaussianProcess


class MPC(ModelPredictiveControllerServicer):
    def __init__(self):
        self.data_queue = None
        self.smm = None
        self.gp = None
        self.solver = None
        self.n_measured_solve_times = 0
        self.measured_solve_times = np.zeros(250)

    def initialize_solver(self, request, context):
        print("INITIALIZING")

        shutil.rmtree('c_generated_code', ignore_errors=True)
        try:
            os.remove("acados_ocp.json")
        except:
            pass

        self.n_measured_solve_times = 0

        try:
            if self.gp is not None:
                self.gp.shutdown()
                self.smm.shutdown()

            bicycle_params = json.loads(request.bicycle_params)

            self.data_queue = Queue()
            self.smm = SharedMemoryManager()
            self.smm.start()

            if request.use_gp:
                self.gp = GaussianProcess(15, self.smm, self.data_queue)

            midpoints = (np.frombuffer(request.midpoints, dtype=np.float32).reshape(request.n_midpoints, 4).copy() /
                         config.pos_scaler)
            refpoints = (np.frombuffer(request.refpoints, dtype=np.float32).reshape(request.n_refpoints, 4).copy() /
                         [config.pos_scaler, config.pos_scaler, config.pos_scaler, config.lin_vel_acc_scaler])

            self.solver = Solver(
                request.mpc_N, request.mpc_sample_time, bicycle_params, midpoints,
                refpoints, self.gp, request.constraint_tightening)
            return mpc_pb2.Response(status=1)

        except Exception as e:
            print(traceback.format_exc())
            return mpc_pb2.Response(status=0)

    def solve(self, request, context):
        try:
            scaler = np.array([
                config.pos_scaler, config.pos_scaler, 1, config.lin_vel_acc_scaler, config.lin_vel_acc_scaler,
                1, 1, 1, config.pos_scaler])
            initial_state = np.frombuffer(request.initial_state, dtype=np.float32).copy() / scaler

            start = time.time_ns()
            if request.delay_compensation and self.n_measured_solve_times > 0:
                initial_state = self.solver.delay_compensation(
                    initial_state, np.zeros(3), np.median(self.measured_solve_times[-self.n_measured_solve_times:]))
            track_tighteners = self.solver.initialize(initial_state, request.max_speed / config.lin_vel_acc_scaler)
            state_horizon, control_horizon, success = self.solver.solve()
            now = time.time_ns()
            solve_time = (now - start) / 1e9

            state_horizon *= scaler

            self.measured_solve_times[:-1] = self.measured_solve_times[1:]
            self.measured_solve_times[-1] = solve_time
            self.n_measured_solve_times = min(len(self.measured_solve_times), self.n_measured_solve_times+1)

            return mpc_pb2.Solution(state_horizon=state_horizon.copy().astype(np.float32).tobytes(),
                                    control_horizon=control_horizon.copy().astype(np.float32).tobytes(),
                                    track_tighteners=track_tighteners.copy().astype(np.float32).tobytes(),
                                    success=success)

        except Exception as e:
            print(traceback.format_exc())
            return mpc_pb2.Solution(state_horizon=np.zeros(1).copy().astype(np.float32).tobytes(),
                                    control_horizon=np.zeros(1).copy().astype(np.float32).tobytes(),
                                    track_tighteners=np.zeros(1).copy().astype(np.float32).tobytes(),
                                    success=False)


    def learn_from_data(self, request, context):
        if self.gp is not None:
            info_scaler = np.array([config.pos_scaler, config.pos_scaler, config.pos_scaler])
            in_scaler = np.array([config.lin_vel_acc_scaler, config.lin_vel_acc_scaler, 1, 1, 1])
            out_scaler = np.array([config.lin_vel_acc_scaler, config.lin_vel_acc_scaler, 1])
            pos_info = np.frombuffer(request.pos_info, dtype=np.float32).copy() / info_scaler
            input_state = np.frombuffer(request.inputs, dtype=np.float32).copy() / in_scaler
            output_state = np.frombuffer(request.outputs, dtype=np.float32).copy() / out_scaler
            self.data_queue.put((pos_info, input_state, output_state))
            return mpc_pb2.Response(status=1)
        return mpc_pb2.Response(status=0)


    def done(self, request, context):
        print("DONE CALLED")
        self.gp.shutdown()
        self.smm.shutdown()
        self.gp = None
        return mpc_pb2.Empty()


if __name__ == "__main__":
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=2), options=options)
    add_ModelPredictiveControllerServicer_to_server(MPC(), server)
    server.add_insecure_port('0.0.0.0:8000')
    server.start()
    print("Solver serving at %s" % 'localhost:8000')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(1)
