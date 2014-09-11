__kernel void multiply(__global float *a, __global float *b, __global
		float *result)
{
	int i;

	i = get_global_id(0);

	result[i] = a[i] * b[i];
}

__kernel void divide(__global float *a, __global float *b, __global
		float *result)
{
	int i;

	i = get_global_id(0);

	result[i] = a[i] / b[i];
}
