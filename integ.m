fun_1 = @(x_0, x_1) 1/(2*pi) * exp(-1/2 * (x_0.^2 + x_1.^2));

fun_2 = @(x_0, x_1) 1/(2*pi) * exp(-1/2 * (1/2 * x_0.^2 + 2 * x_1.^2));

x_1_min = -inf;
x_1_max = @(x) -1/sqrt(2) * x;
x_2_min = @(x) 1/sqrt(2) * x;
x_2_max = inf;
x_3_min = @(x) -1/sqrt(2) * x;
x_3_max = @(x) 1/sqrt(2) * x;

integ_1 = integral2(fun_1, -inf, inf, x_1_min, x_1_max);
integ_2 = integral2(fun_1, -inf, inf, x_2_min, x_2_max);
integ_3 = integral2(fun_2, -inf, inf, x_3_min, x_3_max);

result = 0.5 * (integ_1 + integ_2 + integ_3)