# wave_height_from_IMU
Calculate sea wave height from IMU

Algorithm description

https://www.researchgate.net/profile/Mehdi-Hendijanizadeh/publication/264713649_A_Novel_Kalman_Filter_Based_Technique_for_Calculating_the_Time_History_of_Vertical_Displacement_of_a_Boat_from_Measured_Acceleration/links/53ec88db0cf24f241f1584c5/A-Novel-Kalman-Filter-Based-Technique-for-Calculating-the-Time-History-of-Vertical-Displacement-of-a-Boat-from-Measured-Acceleration.pdf

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

