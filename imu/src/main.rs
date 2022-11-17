//! Raspberry Pi4 demo

// TODO: fix reading IMU (timestamps), etc. Not working example!!!

extern crate linux_embedded_hal as hal;
extern crate mpu9250;
extern crate simple_moving_average as sma;

#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use std::io::{self, Write};
use std::thread;
use std::time::{Duration};

use hal::Delay;
use hal::I2cdev;
use mpu9250::{Mpu9250, AccelDataRate, Dlpf};

use ahrs::{Ahrs, Madgwick};
use nalgebra::{Vector3, Quaternion, UnitQuaternion};

use std::{f64};
use std::time::Instant;

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
    println!("who_am_I              |     0x{:x}", who_am_i);
    println!("ak8963 who_am_I       |     0x{:x}", mag_who_am_i);
    assert_eq!(who_am_i, 0x71);

    mpu9250.accel_data_rate(AccelDataRate::DlpfConf(Dlpf::_0)).expect("Err setting rate");

    println!("accel resolution      | {:>8.5}", mpu9250.accel_resolution());
    println!("gyro resolution       | {:>8.5}", mpu9250.gyro_resolution());
    println!("mag resolution        | {:>8.5}", mpu9250.mag_resolution());
    let acc_bias: [f32; 3] = mpu9250.get_accel_bias().expect("Err accel_bias");
    println!("accel bias            | {:>8.3} {:>8.3} {:>8.3}", acc_bias[0], acc_bias[1], acc_bias[2]);
    let gyro_bias: [f32; 3] = mpu9250.get_gyro_bias().expect("Err gyro_bias");
    println!("gyro bias             | {:>8.3} {:>8.3} {:>8.3}", gyro_bias[0], gyro_bias[1], gyro_bias[2]);
    let mag_sens_adj: [f32; 3] = mpu9250.mag_sensitivity_adjustments();
    println!("mag sense adj         | {:>8.3} {:>8.3} {:>8.3}", mag_sens_adj[0], mag_sens_adj[1], mag_sens_adj[2]);

    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    const IMU_SAMPLE_SEC: f64 = 0.01;
    //const IMU_POLLING_SEC: f64 = 0.004;
    const ACC_SAMPLE_PERIOD_SEC: f64 = IMU_SAMPLE_SEC * 1.0;
    const ACC_AVG_PERIOD_SEC: f64 = 45.0;
    const WARMUP_PERIOD_SEC: f64 = 50.0;
    const AVG_SAMPLES: usize = (ACC_AVG_PERIOD_SEC / IMU_SAMPLE_SEC) as usize;

    let mut ahrs = Madgwick::new_with_quat(
        IMU_SAMPLE_SEC,
        0.1f64,
        UnitQuaternion::new_unchecked(Quaternion::new(
            nalgebra::one(),
            nalgebra::zero(),
            nalgebra::zero(),
            nalgebra::zero(),
        )));

    let pos_integral_trans_variance: f64 = 0.1;
    let pos_integral_variance: f64 = 0.1;

    let dt = ACC_SAMPLE_PERIOD_SEC;
    let b = Vector::new(vec![((1.0 / 6.0) * dt.powi(3)), (0.5 * dt.powi(2)), dt]);

    let x0 = vector![0.0, 0.0, 0.0];
    let p0 = matrix![pos_integral_variance, 0.0, 0.0;
                                                  0.0, 0.0, 0.0;
                                                  0.0, 0.0, 0.0];
    let kf = KalmanFilter {
        // Process noise covariance
        q: matrix![pos_integral_trans_variance, 0.0, 0.0;
                                           0.0, 0.2, 0.0;
                                           0.0, 0.0, 0.1],
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

    let g = 9.81;

    let mut vert_pos = 0.0;
    let mut vert_vel = 0.0;
    let mut k: usize = 0;

    let start = Instant::now();

    let mut acc_mean_filter = SumTreeSMA::<f64, f64, AVG_SAMPLES>::from_zero(0.0);
    let mut acc_mean: f64 = 0.0;

    loop {
        let mut ta = Instant::now();
        let mut loop_time = ta.elapsed();
        loop {
            let mut t = Instant::now();
            match mpu9250.all::<[f32; 3]>() {
                Ok(all ) => {
                    // Obtain sensor values from a source
                    let gyroscope: Vector3<f64> = Vector3::new(all.gyro[0] as f64, all.gyro[1] as f64, all.gyro[2] as f64);
                    let accelerometer: Vector3<f64> = Vector3::new(all.accel[0] as f64, all.accel[1] as f64, all.accel[2] as f64);
                    let magnetometer: Vector3<f64> = Vector3::new(all.mag[0] as f64, all.mag[1] as f64, all.mag[2] as f64);

                    loop {
                        // Use previous measurements if time slot measurement was skipped
                        let sample_time = start.elapsed().as_millis();

                        // Run inputs through AHRS filter (gyroscope must be radians/s)
                        let quat = ahrs
                            .update(
                                &gyroscope,
                                &accelerometer,
                                &magnetometer
                            )
                            .unwrap();
                        let (roll, pitch, yaw) = quat.euler_angles();

                        let acc_abs = Vector3::new(accelerometer[0] as f64, accelerometer[1] as f64, accelerometer[2] as f64).magnitude();

                        let rotated_acc = quat.transform_vector(
                            &Vector3::new(accelerometer[0] as f64, accelerometer[1] as f64, accelerometer[2] as f64));

                        let vert_acc_minus_g = rotated_acc[2] - &g;
                        acc_mean_filter.add_sample(vert_acc_minus_g);
                        if acc_mean_filter.get_num_samples() == AVG_SAMPLES {
                            acc_mean =  acc_mean_filter.get_average();
                        }

                        if period_expired(ta.elapsed(), ACC_SAMPLE_PERIOD_SEC)
                            && period_expired(start.elapsed(), WARMUP_PERIOD_SEC) {
                            k = k + 1;
                            ta = Instant::now();

                            filtered = update_step(&kf, &predicted, &Vector::new(vec![0.0]));
                            filtered.x = &filtered.x + &b * (vert_acc_minus_g - acc_mean);
                            predicted = predict_step(&kf, &filtered);
                            vert_pos = filtered.x[1];
                            vert_vel = filtered.x[2];
                        }

                        write!(&mut stdout, "accel XYZ     (m/s^2) | {:>8.3} {:>8.3} {:>8.3}\n", accelerometer[0], accelerometer[1], accelerometer[2])?;
                        write!(&mut stdout, "gyro XYZ      (rad/s) | {:>8.2} {:>8.2} {:>8.2}\n", gyroscope[0], gyroscope[1], gyroscope[2])?;
                        write!(&mut stdout, "mag field XYZ    (uT) | {:>8.2} {:>8.2} {:>8.2}\n", magnetometer[0], magnetometer[1], magnetometer[2])?;
                        write!(&mut stdout, "roll/pitch/yaw  (deg) | {:>8.1} {:>8.1} {:>8.1}\n", roll * 180.0 / f64::consts::PI, pitch * 180.0 / f64::consts::PI, yaw * 180.0 / f64::consts::PI)?;
                        write!(&mut stdout, "temp              (C) | {:>8.2}\n", all.temp)?;
                        write!(&mut stdout, "accel ref xyz (m/s^2) | {:>8.3} {:>8.3} {:>8.3}\n", rotated_acc[0], rotated_acc[1], rotated_acc[2])?;
                        write!(&mut stdout, "acc_abs       (m/s^2) | {:>8.3} \n", acc_abs)?;
                        write!(&mut stdout, "acc_z/avg     (m/s^2) | {:>8.3} {:>8.3}\n", vert_acc_minus_g - acc_mean, acc_mean)?;
                        write!(&mut stdout, "vert_vel        (m/s) | {:>8.3} \n", vert_vel)?;
                        write!(&mut stdout, "vert_pos          (m) | {:>8.3} \n", vert_pos)?;
                        write!(&mut stdout, "timestamp    (millis) | {:>8?}                 \n", sample_time)?;
                        write!(&mut stdout, "time elapsed (micros) | {:>8?}                 \n", t.elapsed().as_micros())?;
                        write!(&mut stdout, "loop time    (micros) | {:>8?}                 \n", loop_time.as_micros())?;
                        stdout.flush()?;
                        //write!(&mut stdout, "{}", move_up_csi_sequence(13))?;

                        let tt = t.elapsed();
                        match Duration::from_micros((IMU_SAMPLE_SEC * (1000000.0 - 1200.0)) as u64).checked_sub(t.elapsed()) {
                            Some(diff) => {
                                thread::sleep(diff);
                                loop_time = tt;
                                break;
                            },
                            None => {
                                // will use previous measurements
                                println!("skipped measurement");
                                loop_time = tt;
                                t = Instant::now();
                                continue;
                            }
                        }
                    }
                }
                Err(err) => {
                    println!("{:>?}", err)
                }
            }
        }
    }
}

macro_rules! csi {
    ($( $l:expr ),*) => { concat!("\x1B[", $( $l ),*) };
}

fn move_up_csi_sequence(count: u16) -> String {
    format!(csi!("{}A"), count)
}

fn period_expired(time: Duration, period_sec: f64) -> bool {
    time.saturating_sub(Duration::from_micros((period_sec * 1000000.0) as u64)) != Duration::ZERO
}
