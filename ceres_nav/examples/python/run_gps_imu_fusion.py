import subprocess
import os
from dataclasses import dataclass
import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from navlie.lib.datasets import SimulatedInertialGPSDataset
from navlie.lib.imu import IMU, IMUState
from navlie.types import Measurement, StateWithCovariance
from navlie.utils import associate_stamps, GaussianResultList, plot_error, plot_poses
from pymlg import SO3, SE23

cur_dir = os.path.abspath(os.path.dirname(__file__))

# Plot settings
sns.set_theme(style="whitegrid")
plt.rc("lines", linewidth=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--")
# plt.rcParams.update({"font.size": 14})
plt.rc("text", usetex=True)
colors = sns.color_palette("deep")

@dataclass
class InertialNavigationExampleConfig:
    gyro_white_noise: float = 0.01
    accel_white_noise: float = 0.01
    gyro_random_walk: float = 0.0001
    accel_random_walk: float = 0.0001
    gps_white_noise: float = 0.25
    imu_freq: int = 100
    gps_freq: int = 10
    t_end: float = 20.0
    noise_active: bool = True
    gravity_mag = 9.80665
    lie_direction: str = "left"

    # Output direct and output file names
    output_dir: str = "unset_output_dir"
    gt_fname: str = "ground_truth.txt"
    imu_data_fname: str = "imu_data.txt"
    gps_data_fname: str = "gps_data.txt"
    # Filenames for the output IMU states
    init_imu_fname: str = "init_imu_states.txt"
    est_imu_fname: str = "optimized_imu_states.txt"

def load_imu_states_from_asl(file_path: str) -> typing.List[IMUState]:
    """Loads IMU states from a text file in the ASL format."""
    data = np.loadtxt(file_path, delimiter=",")
    imu_states = []
    for row in data:
        stamp = row[0]
        position = row[1:4]
        quat = row[4:8]
        velocity = row[8:11]
        bias_gyro = row[11:14]
        bias_accel = row[14:17]

        attitude = SO3.from_quat(quat, order="wxyz")

        nav_state = SE23.from_components(attitude, velocity, position)
        imu_state = IMUState(
            stamp=stamp,
            nav_state=nav_state,
            bias_gyro=bias_gyro,
            bias_accel=bias_accel,
        )
        imu_states.append(imu_state)
    return imu_states

def load_covariances_from_file(file: str, dof: int) -> typing.List[np.ndarray]:
    """Loads the full covariances from a file, where each row of the file is
    assumed to have a timestmap and the covariance entries in column major order.""" 
    cov_mats_file_np = np.loadtxt(file, delimiter=",")
    cov_mats: typing.List[np.ndarray] = []
    stamps: typing.List[float] = []
    for row in cov_mats_file_np:
        cov_mats.append(row[1:].reshape((dof, dof), order="F"))
        stamps.append(row[0])

    return cov_mats, stamps

def write_imu_states_to_asl(
    imu_state_list: typing.List[IMUState],
    outfile: str,
):
    """Writes a list of IMU states to a text file in the ASL format.

    Note, ASL format is in the form
    t, pos, qw, qx, qy, qz, vel, bg, ba.
    """

    n_timesteps = len(imu_state_list)
    output_mat = np.ndarray((n_timesteps, 17))

    for i in range(n_timesteps):
        cur_state = imu_state_list[i]

        quat = SO3.to_quat(cur_state.attitude, order="wxyz").ravel()
        output_mat[i, 0] = cur_state.stamp
        output_mat[i, 1:4] = cur_state.position
        output_mat[i, 4:8] = quat
        output_mat[i, 8:11] = cur_state.velocity
        output_mat[i, 11:14] = cur_state.bias_gyro
        output_mat[i, 14:17] = cur_state.bias_accel

    np.savetxt(outfile, output_mat, delimiter=",")


def write_gps_data_list(gps_data_list: typing.List[Measurement], outfile: str):
    """Writes a list of GPS measurements to a text file"""
    n_timesteps = len(gps_data_list)
    output_mat = np.ndarray((n_timesteps, 4))
    for i in range(n_timesteps):
        cur_meas = gps_data_list[i]
        output_mat[i, 0] = cur_meas.stamp
        output_mat[i, 1:4] = cur_meas.value
    np.savetxt(outfile, output_mat, delimiter=",")


def write_imu_data_list(imu_data_list: typing.List[IMU], outfile: str):
    """Writes a list of IMU measurements to a text file.

    Format of text file is
    timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
    """

    n_timesteps = len(imu_data_list)
    output_mat = np.ndarray((n_timesteps, 7))

    for i in range(n_timesteps):
        cur_meas = imu_data_list[i]
        output_mat[i, 0] = cur_meas.stamp
        output_mat[i, 1:4] = cur_meas.gyro
        output_mat[i, 4:7] = cur_meas.accel

    np.savetxt(outfile, output_mat, delimiter=",")


def generate_and_save_data(config: InertialNavigationExampleConfig, save_dir: str):
    """Generates a simulated inertial and GPS dataset and saves all data to text files."""
    Q_c = np.identity(12)
    Q_c[0:3, 0:3] *= config.gyro_white_noise**2
    Q_c[3:6, 3:6] *= config.accel_white_noise
    Q_c[6:9, 6:9] *= config.gyro_random_walk**2
    Q_c[9:12, 9:12] *= config.accel_random_walk**2

    R = np.identity(3) * config.gps_white_noise**2

    print("Noise active:", config.noise_active)
    data = SimulatedInertialGPSDataset(
        Q=Q_c,
        R=R,
        t_start=0.0,
        t_end=config.t_end,
        input_freq=config.imu_freq,
        meas_freq=config.gps_freq,
        noise_active=config.noise_active,
    )

    gt_states = data.get_ground_truth()
    imu_data = data.get_input_data()
    gps_data = data.get_measurement_data()

    # # Plot the IMU data
    # fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # stamps = [x.stamp for x in imu_data]
    # gyro_data = np.array([x.gyro for x in imu_data])
    # accel_data = np.array([x.accel for x in imu_data])
    # ax[0].plot(stamps, gyro_data[:, 0], label="Gyro X")
    # ax[0].plot(stamps, gyro_data[:, 1], label="Gyro Y")
    # ax[0].plot(stamps, gyro_data[:, 2], label="Gyro Z")
    # ax[0].set_title("IMU Gyroscope Data")
    # ax[0].set_xlabel("Time (s)")
    # ax[0].set_ylabel("Gyro (rad/s)")

    # ax[1].plot(stamps, accel_data[:, 0], label="Accel X")
    # ax[1].plot(stamps, accel_data[:, 1], label="Accel Y")
    # ax[1].plot(stamps, accel_data[:, 2], label="Accel Z")
    # ax[1].set_title("IMU Accelerometer Data")
    # plt.show()

    # Save the data to text files
    gt_states_file = os.path.join(save_dir, "ground_truth.txt")
    imu_data_file = os.path.join(save_dir, "imu_data.txt")
    gps_data_file = os.path.join(save_dir, "gps_data.txt")

    write_imu_data_list(imu_data, imu_data_file)
    write_gps_data_list(gps_data, gps_data_file)
    write_imu_states_to_asl(gt_states, gt_states_file)

    output_fpaths = {
        "ground_truth": gt_states_file,
        "imu_data": imu_data_file,
        "gps_data": gps_data_file,
    }

    return output_fpaths


def run_gps_imu_fusion(
    config: InertialNavigationExampleConfig,
    executable_path: str,
    data_fpaths: typing.Dict[str, str],
    estimator_type: str = "full_batch"
):
    """Runs the GPS/IMU sliding window filter example."""
    # Run example
    if not os.path.exists(executable_path):
        print(
            f"Executable not found at {executable_path}. Please build the project first."
        )

    cmd = []
    run_with_valgrind = False
    if run_with_valgrind:
        cmd.append("valgrind")
        cmd.append("--tool=memcheck")
        cmd.append("--leak-check=full")

    cmd.append(executable_path)

    args = [
        "--imu_data",
        data_fpaths["imu_data"],
        "--gps_data",
        data_fpaths["gps_data"],
        "--ground_truth",
        data_fpaths["ground_truth"],
        "--sigma_gyro_continuous",
        str(config.gyro_white_noise),
        "--sigma_accel_continuous",
        str(config.accel_white_noise),
        "--sigma_gyro_random_walk",
        str(config.gyro_random_walk),
        "--sigma_accel_random_walk",
        str(config.accel_random_walk),
        "--sigma_gps",
        str(config.gps_white_noise),
        "--lie_direction",
        config.lie_direction,
        "--gravity_mag",
        str(config.gravity_mag),
        "--output_dir",
        config.output_dir,
        "--estimator_type",
        estimator_type,
    ]

    cmd.extend(args)

    try:
        print("Trying to run command")
        result = subprocess.run(cmd, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the executable: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")


def evaluate_imu_states(est_file: str, gt_file: str, cov_file: str):
    """Evaluates the IMU states against the ground truth."""
    gt_states = load_imu_states_from_asl(gt_file)
    est_states = load_imu_states_from_asl(est_file)
    covariances, cov_stamps = load_covariances_from_file(cov_file, 15)

    if len(est_states) != len(covariances):
        print("Error: Estimated states and covariances have different lengths.")
        print("Estimated states length:", len(est_states))
        print("Covariances length:", len(covariances))
        return

    estimate_stamps = [float(x.stamp) for x in est_states]
    gt_stamps = [x.stamp for x in gt_states]

    matches = associate_stamps(estimate_stamps, gt_stamps)

    est_list: typing.List[IMUState] = []
    gt_list: typing.List[IMUState] = []
    init_list: typing.List[IMUState] = []
    cov_list: typing.List[np.ndarray] = []
    for match in matches:
        gt_list.append(gt_states[match[1]])
        est_list.append(est_states[match[0]])
        cov_list.append(covariances[match[0]])

    # Postprocess the results and plot
    state_with_cov: typing.List[StateWithCovariance] = []
    for est, cov in zip(est_list, cov_list):
        state_with_cov.append(StateWithCovariance(est, cov))
    results = GaussianResultList.from_estimates(state_with_cov, gt_list)

    fig, ax = plot_error(results)
    fig.suptitle("IMU State Error")

    # Plot the poses on a graph
    fig, ax = plot_poses(est_list, line_color="tab:blue", label="Estimated", step=None)
    plot_poses(gt_list, ax=ax, line_color="tab:red", label="Ground Truth", step=None)
    ax.set_title("IMU State Trajectories")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    estimator_type = "full_batch"  # full_batch or sliding_window
    save_dir = os.path.join(cur_dir, "gps_imu_fusion_example")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = InertialNavigationExampleConfig(output_dir=save_dir)

    data_fpaths = generate_and_save_data(config, save_dir)

    # Run the sliding window filter example!
    executable_path = os.path.join(cur_dir, "../../build/examples/gps_imu_example")
    run_gps_imu_fusion(config, executable_path, data_fpaths, estimator_type)

    est_file = os.path.join(save_dir, "optimized_imu_states.txt")
    cov_file = os.path.join(save_dir, "covariances.txt")
    evaluate_imu_states(est_file, data_fpaths["ground_truth"], cov_file)
