import os
from pathlib import Path

import numpy as np
import quaternion
import tensorflow as tf
import torch
from scipy.spatial.transform import Rotation
from train_and_eval.configuration import Configuration
from train_and_eval.constants import Constants
from train_and_eval.tf_model_utils import *
from train_and_eval.tf_models import *
from train_and_eval.utils import SMPL_NR_JOINTS, smpl_reduced_to_full

MODEL_DIR = "train_and_eval/models/tf-1528208085-fine_tuning-imu_v9-birnn-fc1_512-lstm2_512-idrop2-relu-norm_" \
            "ori_acc_smpl-auxloss_acc"

DATA_DIR = Path("./live_demo/data") 
motion = "walking"
frame_start = 400
frame_end = 1500

PELVIS_IDX = 5
CAL_FRAMES = 60  # 1

R_body = torch.tensor([
    [ 0, -1, 0],
    [ 1,  0, 0],
    [ 0,  0, 1],
], dtype=torch.float32)

LEN_PAST_FRAMES = 20
LEN_FUTURE_FRAMES = 5

tracked_joints = [
    18, # LELBOW
    19, # RELBOW
    4, # LKNEE
    5, # RKNEE
    15, # HEAD
    0, # PELVIS
]

C = Constants()
SYSTEM = 'local'

def load_model(sess, model_dir):
    """Load the specified model into the given Tensorflow session."""
    config_dict = Configuration.from_json(os.path.abspath(os.path.join(model_dir, "config.json")))

    config_dict["system"] = SYSTEM
    config = Configuration(**config_dict)
    config.set('eval_dir', os.path.join(model_dir, "evaluation"), override=True)
    config.set('model_dir', model_dir, override=True)

    if not os.path.exists(config.get('eval_dir')):
        os.makedirs(config.get('eval_dir'))

    data_placeholders = dict()
    data_placeholders[C.PL_INPUT] = tf.placeholder(tf.float32, shape=[1, None, C.SIZE_ORI+C.SIZE_ACC])
    data_placeholders[C.PL_TARGET] = tf.placeholder(tf.float32, shape=[1, None, C.SIZE_SMPL])
    data_placeholders[C.PL_SEQ_LEN] = tf.placeholder(tf.int32, shape=[1])

    # Load data statistics
    stat_path = os.path.join(config.get('model_dir'), "stats.npz")
    assert os.path.exists(stat_path), "Data statistics file is required."
    stats = dict(np.load(stat_path, allow_pickle=True))

    # Create Model
    model_cls = config.model_cls
    model = model_cls(config=config,
                      session=sess,
                      reuse=False,
                      mode="sampling",
                      placeholders=data_placeholders,
                      input_dims=[C.SIZE_ORI + C.SIZE_ACC],
                      target_dims=[C.SIZE_SMPL],
                      data_stats=stats)
    model.build_graph()

    # Restore weights
    try:
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(config.get('model_dir'))

        print("Loading model " + checkpoint_path)
        saver.restore(sess, checkpoint_path)
    except Exception:
        raise Exception("Could not load model weights.")

    return model


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)

def quaternion_to_rotation_matrix(q: torch.Tensor):
    r"""
    Turn (unnormalized) quaternions wxyz into rotation matrices. (torch, batch)

    :param q: Quaternion tensor that can reshape to [batch_size, 4].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    q = normalize_tensor(q.view(-1, 4))
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r = torch.cat((- 2 * c * c - 2 * d * d + 1, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                   2 * b * c + 2 * a * d, - 2 * b * b - 2 * d * d + 1, 2 * c * d - 2 * a * b,
                   2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, - 2 * b * b - 2 * c * c + 1), dim=1)
    return r.view(-1, 3, 3)

def normalize(orientations, accelerations, root_idx, first_only=False):
    """
    Normalize orientations and accelerations with respect to `root_idx`. If `first_only` is true, then
    it will normalize only with respect to the root orientation occurring at the first frame in the sequence.
    `orientations` is expected in shape (seq_length, n*9) and `acceleration` in shape (seq_length, 6*3).
    The first five sensors are normalized and returned, i.e. two arrays of shape (seq_length, 5*9) and
    (seq_length, 5*3) respectively unless `first_only` is true in which case the first 6 sensors are returned.
    """
    seq_length = orientations.shape[0]
    oris = np.reshape(orientations, [seq_length, -1, 3, 3])
    accs = np.reshape(accelerations, [seq_length, -1, 3, 1])

    if first_only:
        first_root = oris[0:1, root_idx]
        first_root = np.repeat(first_root, seq_length, axis=0)
        root_inv = np.transpose(first_root, [0, 2, 1])
    else:
        root_inv = np.transpose(oris[:, root_idx], [0, 2, 1])

    # normalize orientations - if first_only is true we keep the root orientation, otherwise we discard it
    end_idx = 6 if first_only else 5
    oris_normalized = np.matmul(root_inv[:, np.newaxis, ...], oris)
    oris_normalized = np.reshape(oris_normalized[:, :end_idx], [seq_length, -1])

    # normalize accelerations
    accs_s = accs - accs[:, -1:]
    accs_s = accs_s[:, :end_idx]
    accs_normalized = np.matmul(root_inv[:, np.newaxis, ...], accs_s)
    accs_normalized = np.reshape(accs_normalized, [seq_length, -1])

    return oris_normalized, accs_normalized


def make_calibration_pose(num_frames: int = 1) -> torch.Tensor:
    pose = torch.eye(3).repeat(num_frames, 24, 1, 1)

    left_arm_down = torch.tensor(
        Rotation.from_euler("z", -85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_arm_down = torch.tensor(
        Rotation.from_euler("z", 85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    left_elbow_bend = torch.tensor(
        Rotation.from_euler("xy", [-20,-10], degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_elbow_bend = torch.tensor(
        Rotation.from_euler("xy", [-20,10], degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    pose[:, 16] = left_arm_down
    pose[:, 17] = right_arm_down

    pose[:, 18] = left_elbow_bend
    pose[:, 19] = right_elbow_bend

    # left_wrist_bend = torch.tensor(
    #     Rotation.from_euler("x", 30, degrees=True).as_matrix(),
    #     dtype=torch.float32,
    # )

    # right_wrist_bend = torch.tensor(
    #     Rotation.from_euler("x", -30, degrees=True).as_matrix(),
    #     dtype=torch.float32,
    # )

    # pose[:, SMPLJoint.LWRIST] = left_wrist_bend
    # pose[:, SMPLJoint.RWRIST] = right_wrist_bend

    return pose


## START MAIN PROGRAM
# if __name__ == '__main__':
print("Starting IMU processing!")

# prepare_model
tf.reset_default_graph()
sess = tf.Session()
model = load_model(sess, MODEL_DIR)
buffer_size = LEN_PAST_FRAMES + 1 + LEN_FUTURE_FRAMES
input_buffer_size = LEN_FUTURE_FRAMES + 1

print("Buffer size: ",buffer_size)
buffer = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = TransPoseNet().to(device)
# NO f-strings in Python 3.5
acc_raw = torch.load(str(DATA_DIR / ("{}_accelerations.pt".format(motion))))  # noqa: UP032
print(acc_raw[0])
print(acc_raw.shape)
ori_raw = torch.load(str(DATA_DIR / ("{}_orientations.pt".format(motion))))  # noqa: UP032

acc_raw = acc_raw[frame_start:frame_end]
ori_raw = ori_raw[frame_start:frame_end]

pose_cal = make_calibration_pose(1)[0]  # (24,3,3)
R_bone_cal = pose_cal[tracked_joints]  # (6,3,3)

num_frames = ori_raw.shape[0]

ori_rot = torch.stack(
    [
        quaternion_to_rotation_matrix(ori_raw[i])  # (6, 4) -> (6, 3, 3)
        for i in range(num_frames)
    ],
    dim=0,
)

R_var = R_body
R_fix = torch.stack([
    R_var,   # radius_l_imu
    R_var,   # radius_r_imu
    R_var,  # tibia_l_imu
    R_var,  # tibia_r_imu
    R_var,  # head_imu
    R_var,  # pelvis_imu
]).to(ori_rot.device)

ori_rot = ori_rot @ R_fix
acc_raw = torch.matmul(
    acc_raw.unsqueeze(-2),
    R_fix.unsqueeze(0)
).squeeze(-2)

# Calibration frame
ori_cal = ori_raw[:CAL_FRAMES].mean(dim=0)
acc_cal = acc_raw[:CAL_FRAMES].mean(dim=0)
R_ori_cal = quaternion_to_rotation_matrix(ori_cal.reshape(-1, 4)).view(6, 3, 3)
R_ori_cal = R_ori_cal @ R_fix
smpl2imu = R_ori_cal[PELVIS_IDX].T
print(smpl2imu)

device2bone = smpl2imu.matmul(R_ori_cal).transpose(1, 2).matmul(R_bone_cal)
acc_offsets = smpl2imu.matmul(
    acc_cal.unsqueeze(-1)
)  # [num_imus, 3, 1], already in global inertial frame

print(device2bone.shape)
print(acc_offsets.shape)

acc_in = (smpl2imu.matmul(acc_raw.unsqueeze(-1)) - acc_offsets.unsqueeze(0)).squeeze(-1)
# torch.save(acc_in.cpu(),os.path.join(paths.example_dir, f"{motion}_cal_accelerations.pt"))
ori_in = smpl2imu.matmul(ori_rot).matmul(device2bone)
preds = []
oris = []
accs = []
for i in range(len(ori_rot)):

    # for j in range(buffer_size):
    #     # print("In loop with i=",i)
    #     # imu_m = buffer[i]
    #     # print(ori_rot[i])
    oris.append(np.array(ori_in[i])[np.newaxis, :])
    accs.append(np.array(acc_in[i])[np.newaxis, :])
# ori = ori_rot[np.newaxis, :]
# acc = acc_raw[np.newaxis, :]
ori = np.concatenate(oris, axis=0)
acc = np.concatenate(accs, axis=0)
    # print(ori.shape)
    # print(acc.shape)

    # model_io = {'input_ori': np.copy(ori),
    #             'input_acc': np.copy(acc)}

    # normalize w.r.t. root sensor
root_idx = 5
ori_n, acc_n = normalize(ori, acc, root_idx)

# get the prediction from the model, previous state unused
pred, _, x_input = model.inference_step(ori_n, acc_n, previous_state=None)  # (seq_length, dof)

    # if i % 100 == 0:
    #     print("GOT A PREDICTION FROM THE MODEL i=", i)
    # print(pred)

    # print("pred before slice", pred.shape)
    # print("ori before slice", ori.shape)

    # pred = pred[-(LEN_FUTURE_FRAMES+1):-LEN_FUTURE_FRAMES]
    # ori = ori[-(LEN_FUTURE_FRAMES+1):-LEN_FUTURE_FRAMES]

    # print("pred after slice", pred.shape)
    # print("ori after slice", ori.shape)
    # model_io['output_actual'] = np.copy(pred)
print(pred.shape)

# Flatten ori
# ori = ori.reshape(ori.shape[0], -1)   # (1, 54)

# root_ori = ori[:, 9 * root_idx:9 * (root_idx + 1)]
root_ori = ori[:, root_idx].reshape(-1, 9)

# add unused joints
prediction = smpl_reduced_to_full(pred)
# print(prediction)
# print(prediction.shape)

# replace root with root input sensor value
prediction[:, 0:9] = root_ori

# # end = time.time()

# # convert to angle axis
prediction = prediction.reshape(-1, SMPL_NR_JOINTS, 3, 3)
# preds.append(prediction)
    # print(prediction)
    # print(prediction.shape)
    # prediction_aa = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(prediction))
    # prediction_aa = np.reshape(prediction_aa, [-1])
# preds_out = np.array(preds)
# print(preds_out)
# print(preds_out.shape)
print(prediction.shape)

torch.save(prediction,str(DATA_DIR / "{}_smpl_pose.pt".format(motion)))  # noqa: UP032
