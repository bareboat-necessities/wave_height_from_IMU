//! Raspberry Pi demo

extern crate linux_embedded_hal as hal;
extern crate mpu9250;
extern crate simple_moving_average as sma;

#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use hal::Delay;
use hal::I2cdev;
use mpu9250::{Mpu9250, MargMeasurements, AccelDataRate, Dlpf};

use ahrs::{Ahrs, Madgwick};
use nalgebra::{Vector3, Quaternion, UnitQuaternion};
use std::f64;

use rulinalg::vector::Vector;
use linearkalman::{KalmanFilter, KalmanState, update_step, predict_step};

use sma::{SumTreeSMA, SMA};

fn main() -> io::Result<()> {
    let i2c = I2cdev::new("/dev/i2c-1").expect("unable to open /dev/i2c-1");

    let mut mpu9250 =
        Mpu9250::marg_default(i2c, &mut Delay).expect("unable to make MPU9250");

    let who_am_i = mpu9250.who_am_i().expect("could not read WHO_AM_I");
    let mag_who_am_i = mpu9250.ak8963_who_am_i()
        .expect("could not read magnetometer's WHO_AM_I");
    println!("WHO_AM_I: 0x{:x}", who_am_i);
    println!("AK8963 WHO_AM_I: 0x{:x}", mag_who_am_i);
    assert_eq!(who_am_i, 0x71);

    let acc_bias: [f32; 3] = mpu9250.get_accel_bias().expect("Err accel_bias");
    println!("accel_bias {:>7.3} {:>7.3} {:>7.3}", acc_bias[0], acc_bias[1], acc_bias[2]);
    println!("accel_resolution {:>15.10}", mpu9250.accel_resolution());

    mpu9250.accel_data_rate(AccelDataRate::DlpfConf(Dlpf::_0)).expect("Err setting rate");

    let gyro_bias: [f32; 3] = mpu9250.get_gyro_bias().expect("Err gyro_bias");
    println!("gyro_bias {:>7.3} {:>7.3} {:>7.3}", gyro_bias[0], gyro_bias[1], gyro_bias[2]);
    println!("gyro_resolution {:>15.10}", mpu9250.gyro_resolution());

    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    const WAIT_SEC: f64 = 0.1;

    let mut ahrs = Madgwick::new_with_quat(
        WAIT_SEC,
        0.1f64,
        UnitQuaternion::new_unchecked(Quaternion::new(
            nalgebra::one(),
            nalgebra::zero(),
            nalgebra::zero(),
            nalgebra::zero(),
        )));

    let pos_integral_trans_variance: f64 = 1.0;
    let pos_integral_variance: f64 = 1.0;

    let dt = WAIT_SEC;
    let b = Vector::new(vec![((1.0 / 6.0) * dt.powi(3)), (0.5 * dt.powi(2)), dt]);

    let x0 = vector![0.0, 0.0, 0.0];
    let p0 = matrix![pos_integral_variance, 0.0, 0.0;
                                                  0.0, 0.0, 0.0;
                                                  0.0, 0.0, 0.0];
    let kf = KalmanFilter {
        // Process noise covariance
        q: matrix![pos_integral_trans_variance, 0.0, 0.0;
                                           0.0, 0.1, 0.0;
                                           0.0, 0.0, 0.05],
        // Measurement noise matrix
        r: matrix![pos_integral_variance],
        // Observation matrix
        h: matrix![1.0, 0.0, 0.0],
        // State transition matrix
        f: matrix![1.0,  dt,  dt.powi(2)/2.0;
                   0.0, 1.0,              dt;
                   0.0, 0.0,             1.0],
        // Initial guess for state mean at time 0
        x0,
        // Initial guess for state covariance at time 0
        p0,
    };

    let mut predicted: KalmanState = KalmanState { x: (kf.x0).clone(), p: (kf.p0).clone() };
    let mut filtered: KalmanState;
    const SAMPLES: usize = (45.0 / WAIT_SEC) as usize;
    let mut acc_mean = SumTreeSMA::<f64, f64, SAMPLES>::from_zero(0.0);

    writeln!(&mut stdout, "Give process a couple of minutes to self calibrate\n")?;
    writeln!(&mut stdout,
             "   Accel XYZ(m/s^2)  |   Gyro XYZ (rad/s)  |  Mag Field XYZ(uT)  | Temp (C) | Roll   | Pitch  | Yaw    | Vert Acc - g (m/s^2) | VVel(m/s) | VPos(m)")?;

    let mut vert_pos = 0.0;
    let mut vert_vel = 0.0;
    let g = 9.806;
    let mut t: usize = 0;
    //loop {
        thread::sleep(Duration::from_micros((WAIT_SEC * 1000000.0) as u64));

        loop {
            match mpu9250.all::<[f32; 3]>() {
                Ok(all ) => {
                    // Obtain sensor values from a source
                    let gyroscope: Vector3<f64> = Vector3::new(all.gyro[0] as f64, all.gyro[1] as f64, all.gyro[2] as f64);
                    let accelerometer: Vector3<f64> = Vector3::new(all.accel[0] as f64, all.accel[1] as f64, all.accel[2] as f64);
                    let magnetometer: Vector3<f64> = Vector3::new(all.mag[0] as f64, all.mag[1] as f64, all.mag[2] as f64);

                    // Run inputs through AHRS filter (gyroscope must be radians/s)
                    let quat = ahrs
                        .update(
                            &gyroscope,
                            &accelerometer,
                            &magnetometer
                        )
                        .unwrap();
                    let (roll, pitch, yaw) = quat.euler_angles();

                    let rotated_acc= quat.transform_vector(
                        &Vector3::new(accelerometer[0] as f64, accelerometer[1] as f64, accelerometer[2] as f64));

                    let vert_acc_minus_g = rotated_acc[2] - g;

                    t = t + 1;
                    if SAMPLES <= t && t <= (2 * SAMPLES) {
                        acc_mean.add_sample(vert_acc_minus_g);
                    }

                    if t >= (2 * SAMPLES) {
                        filtered = update_step(&kf, &predicted, &Vector::new(vec![0.0]));
                        filtered.x = &filtered.x + &b * (vert_acc_minus_g - acc_mean.get_average());
                        predicted = predict_step(&kf, &filtered);
                        vert_pos = filtered.x[1];
                        vert_vel = filtered.x[2];
                        t = 2 * SAMPLES
                    }

                    write!(&mut stdout,
                           "\r{:>6.2} {:>6.2} {:>6.2} |{:>6.1} {:>6.1} {:>6.1} |{:>6.1} {:>6.1} {:>6.1} | {:>4.1}     | {:>6.1} | {:>6.1} | {:>6.1} | {:>7.3}              | {:>7.2}   | {:>7.3}",
                           accelerometer[0],
                           accelerometer[1],
                           accelerometer[2],
                           gyroscope[0],
                           gyroscope[1],
                           gyroscope[2],
                           magnetometer[0],
                           magnetometer[1],
                           magnetometer[2],
                           all.temp,
                           roll * 180.0 / f64::consts::PI,
                           pitch * 180.0 / f64::consts::PI,
                           yaw * 180.0 / f64::consts::PI,
                           vert_acc_minus_g - acc_mean.get_average(),
                           vert_vel,
                           vert_pos
                    )?;
                    stdout.flush()?;
                }
                Err(err) => {
                    println!("{:>?}", err)
                }
            }
        }
    //}
}
