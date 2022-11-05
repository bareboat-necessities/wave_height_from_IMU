clear;

dt = 0.01;
t=0:dt:70;

accx_var_best = 0.0005; % (m/s^2)^2
accx_var_good = 0.0007; % (m/s^2)^2
accx_var_worst = 0.001; % (m/s^2)^2

accx_ref_noise = randn(size(t))*sqrt(accx_var_best);
accx_good_noise = randn(size(t))*sqrt(accx_var_good);
accx_worst_noise = randn(size(t))*sqrt(accx_var_worst);

accx_basesignal = (-225*sin(0.3*t)-2*sin(0.04*t))/2500;

accx_ref = accx_basesignal + accx_ref_noise;
velx_ref = (15*cos(0.3*t) + cos(0.04*t))/50;
distx_ref = sin(0.3*t) + 0.5*sin(0.04*t);


accx_good_offset =  0.001 + 0.0004*sin(0.05*t);

accx_good = accx_basesignal + accx_good_noise + accx_good_offset;
velx_good = cumsum(accx_good)*dt;
distx_good = cumsum(velx_good)*dt;


accx_worst_offset =  -0.08 + 0.004*sin(0.07*t);

accx_worst = accx_basesignal + accx_worst_noise + accx_worst_offset;
velx_worst = cumsum(accx_worst)*dt;
distx_worst = cumsum(velx_worst)*dt;

fid = fopen('wavedata.txt','wt');
fprintf('%6d\n', numel(accx_worst));
for ii = 1:numel(accx_worst)
    fprintf(fid,'%6.4f, %12.8f, %12.8f, %12.8f\n', (ii-1)*dt, accx_worst(ii), distx_ref(ii), velx_ref(ii));
end
fclose(fid);

subplot(3,1,1);
plot(t, accx_ref);
hold on;
plot(t, accx_good);
plot(t, accx_basesignal);
hold off;
grid minor;
legend('ref', 'good', 'worst');
title('accx');

subplot(3,1,2);
plot(t, velx_ref);
hold on;
%plot(t, velx_good);
%plot(t, velx_worst);
hold off;
grid minor;
legend('ref');
title('velx');

subplot(3,1,3);
plot(t, distx_ref);
hold on;
%plot(t, distx_good);
%plot(t, distx_worst);
hold off;
grid minor;
legend('ref');
title('distx');
