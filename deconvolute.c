#include "deconvolute.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fftw3.h>
#include <CL/opencl.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"

/********************************/
/* STATIC VARS, PROTOTYPES, ETC */
/********************************/

static int width, height;
static uint16_t *input_image;
static uint8_t *psf_image;

/* real images */
static float *norm_input_image[3];
static float *norm_current_image[3];
static float *norm_output_image[3];
static float *norm_psf_image[3];

/* complex images */


/* fftw vars */
static fftwf_plan fft_forward_plan;
static fftwf_plan fft_backward_plan;
static float *fft_real;
static fftwf_complex *fft_complex;

/* opencl vars */
static size_t global_work_size[1];
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel mult_k[3];
static cl_kernel complex_mult_k[3];
static cl_kernel complex_conj_mult_k[3];
static cl_kernel divide_k[3];

/* functions */
static int init_images(char *input_image_filename, char
		*psf_image_filename);
static void cleanup_init_images();

static int init_fftw();
static void cleanup_init_fftw();

static int init_opencl();
static void cleanup_init_opencl();


/******************/
/* IMPLEMENTATION */
/******************/

/* returns 0 on success, anything else on failure */
int deconvolute_image(char *input_image_filename, char
		*psf_image_filename, char *output_image_filename, int
		n_iterations)
{
	int ret;

	ret = init_images(input_image_filename, psf_image_filename);
	if (ret != 0)
		goto out_no_init_images;

	ret = init_fftw();
	if (ret != 0)
		goto out_no_init_fftw;

	ret = init_opencl();
	if (ret != 0)
		goto out_no_init_opencl;

	cleanup_init_opencl();
out_no_init_opencl:
	cleanup_init_fftw();
out_no_init_fftw:
	cleanup_init_images();
out_no_init_images:
	return ret;
}

/*
 * read in images, alloc memory for real images, pad and normalize
 * psf
 *
 * returns 0 on success, anything else otherwise
 */
int init_images(char *input_image_filename, char *psf_image_filename)
{
	int i, j, c;
	int psf_width, psf_height;
	int x, y, index, psf_index;

	/* read in images */
	input_image = read_tiff16(input_image_filename, &width,
			&height);
	if (input_image == NULL)
		goto out_no_input_image;

	psf_image = read_tiff8(psf_image_filename, &psf_width,
			&psf_height);
	if (psf_image == NULL)
		goto out_no_psf_image;

	/* alloc memory for images */
	for (c = 0; c < 3; c++) {
		norm_input_image[c] = malloc(width * height *
				sizeof(*norm_input_image[c]));
		norm_current_image[c] = malloc(width * height *
				sizeof(*norm_current_image[c]));
		norm_output_image[c] = malloc(width * height *
				sizeof(*norm_output_image[c]));
		norm_psf_image[c] = calloc(width * height,
				sizeof(*norm_psf_image[c]));

		if (norm_input_image[c] == NULL)
			goto out_nomem;
		if (norm_current_image[c] == NULL)
			goto out_nomem;
		if (norm_output_image[c] == NULL)
			goto out_nomem;
		if (norm_psf_image[c] == NULL)
			goto out_nomem;
	}

	/* convert input image over to float */
	for (i = 0; i < 3 * width * height; i++) {
		norm_input_image[i%3][i/3] = (float)input_image[i]/UINT16_MAX;
		norm_current_image[i%3][i/3] = (float)input_image[i]/UINT16_MAX;
	}

	float total[3] = {0, 0, 0};
	for (i = 0; i < 3 * psf_width * psf_height; i++) {
		total[i%3] += (float)psf_image[i];
	}

	/* copy psf over to padded float psf image */
	for (c = 0; c < 3; c++) {
		for (i = 0; i < psf_width; i++) {
			for (j = 0; j < psf_height; j++) {
				x = (width - psf_width)/2 + i;
				y = (height - psf_height)/2 + j;
				index = y * width + x;
				psf_index = 3 * (j * psf_width + i) + c;

				norm_psf_image[c][index] = (float)
					psf_image[psf_index]/total[c];
			}
		}
	}

	return 0;

out_nomem:
	for (c = 0; c < 3; c++) {
		if (norm_input_image[c] != NULL)
			free(norm_input_image[c]);
		if (norm_current_image[c] != NULL)
			free(norm_current_image[c]);
		if (norm_output_image[c] != NULL)
			free(norm_output_image[c]);
		if (norm_psf_image[c] != NULL)
			free(norm_psf_image[c]);
	}
	free(psf_image);
out_no_psf_image:
	free(input_image);
out_no_input_image:
	fprintf(stderr, "init_images: failed\n");
	fflush(stderr);
	return -1;
}

void cleanup_init_images()
{
	int c;

	for (c = 0; c < 3; c++) {
		free(norm_psf_image[c]);
		free(norm_output_image[c]);
		free(norm_current_image[c]);
		free(norm_input_image[c]);
	}

	free(psf_image);
	free(input_image);
}

/*
 * allocate fftw input/output arrays and create plans
 *
 * returns 0 on success, anything else otherwise
 */
int init_fftw()
{
	/* allocate memory for doing fft computations */
	fft_real = fftwf_malloc(width * height * sizeof(*fft_real));
	if (fft_real == NULL)
		goto out_no_fft_real;

	fft_complex = fftwf_malloc(width * (height/2 + 1) *
			sizeof(*fft_complex));
	if (fft_complex == NULL)
		goto out_no_fft_complex;

	/* create fftw plans for both forward and backward ffts */
	fft_forward_plan = fftwf_plan_dft_r2c_2d(width, height,
			fft_real, fft_complex, FFTW_MEASURE);
	if (fft_forward_plan == NULL)
		goto out_no_forward_plan;

	fft_backward_plan = fftwf_plan_dft_c2r_2d(width, height,
			fft_complex, fft_real, FFTW_MEASURE);
	if (fft_backward_plan == NULL)
		goto out_no_backward_plan;

	return 0;

out_no_backward_plan:
	fftwf_destroy_plan(fft_forward_plan);
out_no_forward_plan:
	fftwf_free(fft_complex);
out_no_fft_complex:
	fftwf_free(fft_real);
out_no_fft_real:
	fprintf(stderr, "init_fftw: failed\n");
	fflush(stderr);
	return -1;
}

void cleanup_init_fftw()
{
	fftwf_destroy_plan(fft_backward_plan);
	fftwf_destroy_plan(fft_forward_plan);
	fftwf_free(fft_complex);
	fftwf_free(fft_real);
}

/*
 * create opencl context, queue, program, and kernels and alloc opencl
 * buffers
 *
 * returns 0 on success, anything else otherwise
 */
int init_opencl()
{
	int ret;
	int c;

	/* set global work size */
	global_work_size[0] = width * height;

	/* setup context, queue, program, and kernels */
	ret = cl_utils_setup_gpu(&context, &queue, &device);
	if (ret != 0)
		goto out_gpu_fail;

	ret = cl_utils_create_program(&program, "arithmetic.cl", context, device);
	if (ret != 0)
		goto out_no_program;

	for (c = 0; c < 3; c++) {
		mult_k[c] = clCreateKernel(program, "mult", NULL);
		complex_mult_k[c] = clCreateKernel(program,
				"complex_mult", NULL);
		complex_conj_mult_k[c] = clCreateKernel(program,
				"complex_conj_mult", NULL);
		divide_k[c] = clCreateKernel(program, "divide", NULL);

		if (mult_k[c] == NULL)
			goto out_no_kernel;
		if (complex_mult_k[c] == NULL)
			goto out_no_kernel;
		if (complex_conj_mult_k[c] == NULL)
			goto out_no_kernel;
		if (divide_k[c] == NULL)
			goto out_no_kernel;
	}

	/* allocate opencl buffers */


	return 0;

out_no_kernel:
	for (c = 0; c < 3; c++) {
		if (mult_k[c] != NULL)
			clReleaseKernel(mult_k[c]);
		if (complex_mult_k[c] != NULL)
			clReleaseKernel(complex_mult_k[c]);
		if (complex_conj_mult_k[c] != NULL)
			clReleaseKernel(complex_conj_mult_k[c]);
		if (divide_k[c] != NULL)
			clReleaseKernel(divide_k[c]);
	}
	clReleaseProgram(program);
out_no_program:
	cl_utils_cleanup_gpu(context, queue);
out_gpu_fail:
	fprintf(stderr, "init_opencl: failed\n");
	fflush(stderr);
	return -1;
}

void cleanup_init_opencl()
{
	int c;

	for (c = 0; c < 3; c++) {
		clReleaseKernel(mult_k[c]);
		clReleaseKernel(complex_mult_k[c]);
		clReleaseKernel(complex_conj_mult_k[c]);
		clReleaseKernel(divide_k[c]);
	}

	clReleaseProgram(program);
	cl_utils_cleanup_gpu(context, queue);
}
