# wave_height_from_IMU

(DEPRECATED REPOSITORY)
Note: For actual implementation see: https://github.com/bareboat-necessities/bbn-wave-period-esp32

Calculate sea wave height from IMU 

Algorithm description:

https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html

Kalman filter based on

https://www.researchgate.net/profile/Mehdi-Hendijanizadeh/publication/264713649_A_Novel_Kalman_Filter_Based_Technique_for_Calculating_the_Time_History_of_Vertical_Displacement_of_a_Boat_from_Measured_Acceleration/links/53ec88db0cf24f241f1584c5/A-Novel-Kalman-Filter-Based-Technique-for-Calculating-the-Time-History-of-Vertical-Displacement-of-a-Boat-from-Measured-Acceleration.pdf

And another model is based on vertical acceleration of trochoidal waves


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
    
    
    
XDR - Transducer Measurement 
Talker ID - 'YD' - Transducer - Displacement, Angular or Linear
    
            1 2   3 4            n
            | |   | |            |
    $YDXDR,a,x.x,a,c--c, ..... *hh<CR><LF>
    
    Field Number:
    
        Transducer Type
    
        Measurement Data
    
        Units of measurement A = Amperes B = Bars B = Binary C = Celsius D = Degrees H = Hertz I = liters/second K = Kelvin K = kg/m3 M = Meters M = cubic Meters N = Newton P = % of full range P = Pascal R = RPM S = Parts per thousand V = Volts
    
        Name of transducer
    
There may be any number of quadruplets like this, each describing a sensor. The last field will be a checksum as usual.
    
Example for (reporting displacement of a vessel from mean sea surface due to a wave of -0.33 m height):

    $YDXDR,W,-0.33,M,VERT_DISPLACEMENT*<checksum>

Or something like this from Autopilot:

    
    $APXDR,A,-0.087,M,VDIS*<checksum>
    $APXDR,A,0.08,,VVEL*<checksum>
    $APXDR,A,-0.087,,VACC*<checksum>
    $APMWH,3.44,f,1.05,M*<checksum>
    
    VDIS - vertical displacement (m)
    VVEL - vertical velocity (m/s)
    VACC - vertical acceleration (m/s^2)
    $--MWH - wave height (feet and meters)
