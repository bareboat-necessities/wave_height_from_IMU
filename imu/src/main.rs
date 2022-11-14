//! Raspberry Pi demo

extern crate linux_embedded_hal as hal;
extern crate mpu9250;

use std::io::{self, Write};
use std::thread;
use std::time::Duration;

use hal::Delay;
use hal::I2cdev;
use mpu9250::{Mpu9250, MargMeasurements};

use ahrs::{Ahrs, Madgwick};
use nalgebra::Vector3;
use std::f64;

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

    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut ahrs = Madgwick::default();
    writeln!(&mut stdout,
             "   Accel XYZ(m/s^2)  |   Gyro XYZ (rad/s)  |  Mag Field XYZ(uT)  | Temp (C) | Roll | Pitch | Yaw")?;
    loop {
        let all: MargMeasurements<[f32; 3]> = mpu9250.all().expect("unable to read from MPU!");

        // Obtain sensor values from a source
        let gyroscope: Vector3<f64> = Vector3::new(all.gyro[0] as f64, all.gyro[1] as f64, all.gyro[2] as f64);
        let accelerometer: Vector3<f64> = Vector3::new(all.accel[0] as f64, all.accel[1] as f64, all.accel[2] as f64);
        let magnetometer: Vector3<f64> = Vector3::new(all.mag[0] as f64, all.mag[1] as f64, all.mag[2] as f64);

        // Run inputs through AHRS filter (gyroscope must be radians/s)
        let quat = ahrs
            .update(
                &gyroscope,
                &accelerometer,
                &magnetometer,
            )
            .unwrap();
        let (roll, pitch, yaw) = quat.euler_angles();

        write!(&mut stdout,
               "\r{:>6.2} {:>6.2} {:>6.2} |{:>6.1} {:>6.1} {:>6.1} |{:>6.1} {:>6.1} {:>6.1} | {:>4.1}     | {:>4.1} | {:>4.1} | {:>4.1}",
               all.accel[0],
               all.accel[1],
               all.accel[2],
               all.gyro[0],
               all.gyro[1],
               all.gyro[2],
               all.mag[0],
               all.mag[1],
               all.mag[2],
               all.temp,
               roll * 180.0 / f64::consts::PI,
               pitch * 180.0 / f64::consts::PI,
               yaw * 180.0 / f64::consts::PI
        )?;
        stdout.flush()?;
        thread::sleep(Duration::from_micros(100000));
    }
}
