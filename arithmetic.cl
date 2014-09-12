__kernel void mult(__global float *a, __global float *b, __global
		float *result)
{
	int i;

	i = get_global_id(0);

	result[i] = a[i] * b[i];
}

__kernel void complex_mult(__global float *a_r, __global float *a_i,
		__global float *b_r, __global float *b_i, __global float
		*result_r, __global float *result_i)
{
	int i;
	
	i = get_global_id(0);

	result_r[i] = a_r[i] * b_r[i] - a_i[i] * b_i[i];
	result_i[i] = a_r[i] * b_i[i] + a_i[i] * b_r[i];
}

/* conj(a) * b */
__kernel void complex_conj_mult(__global float *a_r, __global float
		*a_i, __global float *b_r, __global float *b_i, __global
		float *result_r, __global float *result_i)
{
	int i;
	
	i = get_global_id(0);

	result_r[i] = a_r[i] * b_r[i] + a_i[i] * b_i[i];
	result_i[i] = a_r[i] * b_i[i] - a_i[i] * b_r[i];
}

__kernel void divide(__global float *a, __global float *b, __global
		float *result)
{
	int i;

	i = get_global_id(0);

	if (b[i] != 0) {
		result[i] = a[i] / b[i];
	} else {
		result[i] = 0;
	}
}
