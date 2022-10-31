# wave_height_from_IMU
Calculate sea wave height from IMU

Algorithm description

- Project acceleration to Earth center to get vertical acceleration
- Use Kalman filter to adjust vertical acceleration
- Calculate integral for vertical velocity and second integral for height
- Adjust interpolation to zero height when acceleration is low
- Capture min and max height over periods of time and report difference as wave height


TODO: 
- generate NMEA 0183 sentence for wave height
- run this code as part of pypilot server

MWH - Wave Height
NMEA 0183
Approved by the NMEA 0183 Standard Committee
as of
October 1, 2008

$--MWH,x.x,f,x.x,M*hh
Wave height, meters
Wave height, feet

