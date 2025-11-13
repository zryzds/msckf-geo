# GEOVINS 滤波器测试报告

## 总览
- 测试用例数: 13
- 失败: 0  错误: 0  跳过: 0
- 总用时: 14.01s

## 详细用例
- 套件 `pytest`: 13 个用例, 失败 0, 错误 0, 跳过 0, 用时 14.01s
  - [passed] tests.test_filter.TestEKF::test_creation (0.000s)
  - [passed] tests.test_filter.TestEKF::test_predict (0.002s)
  - [passed] tests.test_filter.TestEKF::test_update (0.000s)
  - [passed] tests.test_filter.TestEKF::test_marginalize (0.000s)
  - [passed] tests.test_filter.TestEKF::test_augment (0.000s)
  - [passed] tests.test_filter.TestMSCKF::test_creation (0.001s)
  - [passed] tests.test_filter.TestMSCKF::test_imu_propagation (0.007s)
  - [passed] tests.test_filter.TestMSCKF::test_state_augmentation (0.001s)
  - [passed] tests.test_filter.TestMSCKF::test_camera_state_marginalization (0.001s)
  - [passed] tests.test_filter.TestMSCKF::test_get_state (0.000s)
  - [passed] tests.test_filter.TestMSCKF::test_get_covariance (0.000s)
  - [passed] tests.test_filter.TestIMUPropagation::test_constant_velocity (0.001s)
  - [passed] tests.test_filter.TestIMUFromCSV::test_msckf_with_csv (0.117s)

## IMU CSV 指标摘要
- 数据文件: `c:\Users\Admin\Desktop\msckf-geo\TD_07\export\imu.csv`
- 采样条数: 2000
- dt 中位数: 0.005285s (min 0.002100s, max 0.007000s)
- 末端位置: [5.809189421953886, 50.683545388517395, -2.1717830555983104]
- 末端速度: [0.8105910515835635, 9.656957855839453, -0.49403389311204465]
- 位移: 51.062 m
- 协方差最小特征值: 1.987516e-07