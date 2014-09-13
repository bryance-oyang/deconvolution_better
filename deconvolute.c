#include "deconvolute.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fftw3.h>
#include <CL/opencl.h>
#include "tiff_goodness.h"
#include "opencl_utils.h"

#define say_function_failed() \
	fprintf(stderr, "%s: %s: failed\n", __FILE__, __func__); \
	fflush(stderr);

/********************************/
/* STATIC VARS, PROTOTYPES, ETC */
/********************************/

static int width, height;
static uint16_t *original_input_image;
static uint8_t *original_psf_image;

/* real images */
static float *input_image[3];
static float *current_image[3];
static float *psf_image[3];
static float *image_a[3];
static float *image_b[3];

/* complex images */
static float *cimage_a[3][2];
static float *cimage_b[3][2];
static float *cimage_psf[3][2];

/* fftw vars */
static fftwf_plan fft_forward_plan;
static fftwf_plan fft_backward_plan;
static float *fft_real;
static fftwf_complex *fft_complex;

/* opencl vars */
static size_t global_work_size[2];
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel mult_k[3];
static cl_kernel complex_mult_k[3];
static cl_kernel complex_conj_mult_k[3];
static cl_kernel divide_k[3];
/* wait (sync) events */
static cl_event copy_events[3][3];
static cl_event kernel_events[3];
/* opencl memory buffers */
static cl_mem k_input_image[3];
static cl_mem k_image_a[3];
static cl_mem k_image_b[3];
static cl_mem k_image_c[3];
static cl_mem k_cimage_a[3][2];
static cl_mem k_cimage_b[3][2];
static cl_mem k_cimage_psf[3][2];

/* functions */
static int init_images(char *input_image_filename, char
		*psf_image_filename);
static void cleanup_init_images();

static int init_fftw(int n_threads);
static void cleanup_init_fftw();

static int init_opencl();
static void cleanup_init_opencl();

static int copy_reusables_to_opencl();

static int do_iteration();

static int output(char *output_image_filename);

static int cpsf_multiply(float *in[3][2], float *out[3][2]);
static int image_input_divide(float *in[3], float *out[3]);
static int cpsf_conj_multiply(float *in[3][2], float *out[3][2]);
static int image_multiply(float *a[3], float *b[3], float *out[3]);

static void fft(float *in, float *out[2]);
static void ifft(float *in[2], float *out);

/******************/
/* IMPLEMENTATION */
/******************/

/*
 * global function to deconvolute an image via Richardson–Lucy
 *
 * the input image must be a 16-bit RGB TIFF
 * the psf image must be a 8-bit RGB TIFF (due to GIMP limitations)
 * the outputted image will be a 16-bit RGB TIFF
 *
 * if any part of it fails, it will undo itself (goto styled stack-esque
 * wind and unwind)
 *
 * returns 0 on success, anything else on failure
 */
int deconvolute_image(char *input_image_filename, char
		*psf_image_filename, char *output_image_filename, int
		n_iterations, int n_threads)
{
	int ret;
	int i;

	/* setup */
	ret = init_images(input_image_filename, psf_image_filename);
	if (ret != 0)
		goto out_no_init_images;

	ret = init_fftw(n_threads);
	if (ret != 0)
		goto out_no_init_fftw;

	ret = init_opencl();
	if (ret != 0)
		goto out_no_init_opencl;

	ret = copy_reusables_to_opencl();
	if (ret != 0)
		goto out_no_copy_reusables;

	/* run deconvolution */
	for (i = 0; i < n_iterations; i++) {
		ret = do_iteration();
		if (ret != 0)
			goto out_iteration_failed;
	}

	/* output result */
	ret = output(output_image_filename);
	if (ret != 0)
		goto out_no_output;

out_no_output:
out_iteration_failed:
out_no_copy_reusables:
	cleanup_init_opencl();
out_no_init_opencl:
	cleanup_init_fftw();
out_no_init_fftw:
	cleanup_init_images();
out_no_init_images:
	return ret;
}

/********************/
/* STATIC FUNCTIONS */
/********************/

/*
 * read in images, alloc memory for real images, pad and normalize
 * psf
 *
 * returns 0 on success, anything else otherwise
 */
static int init_images(char *input_image_filename, char *psf_image_filename)
{
	int i, j, c;
	int psf_width, psf_height;
	int x, y, index, psf_index;

	/* read in images */
	original_input_image = read_tiff16(input_image_filename, &width,
			&height);
	if (original_input_image == NULL)
		goto out_err;

	original_psf_image = read_tiff8(psf_image_filename, &psf_width,
			&psf_height);
	if (original_psf_image == NULL)
		goto out_err;

	/* alloc memory for images */
	for (c = 0; c < 3; c++) {
		input_image[c] = malloc(width * height *
				sizeof(*input_image[c]));
		current_image[c] = malloc(width * height *
				sizeof(*current_image[c]));
		psf_image[c] = calloc(width * height,
				sizeof(*psf_image[c]));
		image_a[c] = malloc(width * height *
				sizeof(*image_a[c]));
		image_b[c] = malloc(width * height *
				sizeof(*image_b[c]));

		if (input_image[c] == NULL)
			goto out_err;
		if (current_image[c] == NULL)
			goto out_err;
		if (psf_image[c] == NULL)
			goto out_err;
		if (image_a[c] == NULL)
			goto out_err;
		if (image_b[c] == NULL)
			goto out_err;

		/* alloc memory for complex images */
		for (i = 0; i < 2; i++) {
			cimage_a[c][i] = malloc((width/2 + 1) * height *
					sizeof(*cimage_a[c][i]));
			cimage_b[c][i] = malloc((width/2 + 1) * height *
					sizeof(*cimage_b[c][i]));
			cimage_psf[c][i] = malloc((width/2 + 1) * height
					* sizeof(*cimage_psf[c][i]));

			if (cimage_a[c][i] == NULL)
				goto out_err;
			if (cimage_b[c][i] == NULL)
				goto out_err;
			if (cimage_psf[c][i] == NULL)
				goto out_err;
		}
	}

	/* convert input image over to float */
	for (i = 0; i < 3 * width * height; i++) {
		input_image[i%3][i/3] = (float)original_input_image[i]/UINT16_MAX;
		current_image[i%3][i/3] = (float)original_input_image[i]/UINT16_MAX;
	}

	float total[3] = {0, 0, 0};
	for (i = 0; i < 3 * psf_width * psf_height; i++) {
		total[i%3] += (float)original_psf_image[i];
	}

	/* copy psf over to padded float psf image */
	for (c = 0; c < 3; c++) {
		for (i = 0; i < psf_width; i++) {
			for (j = 0; j < psf_height; j++) {
				x = (width - psf_width/2 + i) % width;
				y = (height - psf_height/2 + j) % height;
				index = y * width + x;
				psf_index = 3 * (j * psf_width + i) + c;

				psf_image[c][index] = (float)
					original_psf_image[psf_index]/total[c];
			}
		}
	}

	return 0;

out_err:
	say_function_failed();
	cleanup_init_images();
	return -1;
}

/* will only be called once */
static void cleanup_init_images()
{
	int c, i;

	for (c = 0; c < 3; c++) {
		if (input_image[c] != NULL)
			free(input_image[c]);
		if (current_image[c] != NULL)
			free(current_image[c]);
		if (psf_image[c] != NULL)
			free(psf_image[c]);
		if (image_a[c] != NULL)
			free(image_a[c]);
		if (image_b[c] != NULL)
			free(image_b[c]);

		for (i = 0; i < 2; i++) {
			if (cimage_a[c][i] != NULL)
				free(cimage_a[c][i]);
			if (cimage_b[c][i] != NULL)
				free(cimage_b[c][i]);
			if (cimage_psf[c][i] != NULL)
				free(cimage_psf[c][i]);
		}
	}

	if (original_psf_image != NULL)
		free(original_psf_image);
	if (original_input_image != NULL)
		free(original_input_image);
}

/*
 * allocate fftw input/output arrays and create plans
 *
 * returns 0 on success, anything else otherwise
 */
static int init_fftw(int n_threads)
{
	/* make fftw multithreaded */

	if (fftwf_init_threads() == 0)
		goto out_err;

	fftwf_plan_with_nthreads(n_threads);

	/* allocate memory for doing fft computations */
	fft_real = fftwf_malloc(width * height * sizeof(*fft_real));
	if (fft_real == NULL)
		goto out_err;

	fft_complex = fftwf_malloc((width/2 + 1) * height *
			sizeof(*fft_complex));
	if (fft_complex == NULL)
		goto out_err;

	/* create fftw plans for both forward and backward ffts */
	fft_forward_plan = fftwf_plan_dft_r2c_2d(height, width,
			fft_real, fft_complex, FFTW_MEASURE);
	if (fft_forward_plan == NULL)
		goto out_err;

	fft_backward_plan = fftwf_plan_dft_c2r_2d(height, width,
			fft_complex, fft_real, FFTW_MEASURE);
	if (fft_backward_plan == NULL)
		goto out_err;

	return 0;

out_err:
	say_function_failed();
	cleanup_init_fftw();
	return -1;
}

/* will only be called once */
static void cleanup_init_fftw()
{
	if (fft_backward_plan != NULL)
		fftwf_destroy_plan(fft_backward_plan);
	if (fft_forward_plan != NULL)
		fftwf_destroy_plan(fft_forward_plan);
	if (fft_complex != NULL)
		fftwf_free(fft_complex);
	if (fft_real != NULL)
		fftwf_free(fft_real);
}

/*
 * create opencl context, queue, program, and kernels and alloc opencl
 * buffers
 *
 * returns 0 on success, anything else otherwise
 */
static int init_opencl()
{
	int ret;
	int c, i;

	/* set work sizes */
	global_work_size[0] = width * height;
	global_work_size[1] = (width/2 + 1) * height;

	/* setup context, queue, program, and kernels */
	ret = cl_utils_setup_gpu(&context, &queue, &device);
	if (ret != 0)
		goto out_err;

	ret = cl_utils_create_program(&program, "arithmetic.cl",
			context, device);
	if (ret != 0)
		goto out_err;

	for (c = 0; c < 3; c++) {
		mult_k[c] = clCreateKernel(program, "mult", NULL);
		complex_mult_k[c] = clCreateKernel(program,
				"complex_mult", NULL);
		complex_conj_mult_k[c] = clCreateKernel(program,
				"complex_conj_mult", NULL);
		divide_k[c] = clCreateKernel(program, "divide", NULL);

		if (mult_k[c] == NULL)
			goto out_err;
		if (complex_mult_k[c] == NULL)
			goto out_err;
		if (complex_conj_mult_k[c] == NULL)
			goto out_err;
		if (divide_k[c] == NULL)
			goto out_err;
	}

	/* allocate opencl buffers */
	for (c = 0; c < 3; c++) {
		k_input_image[c] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, width * height *
				sizeof(cl_float), NULL, NULL);
		k_image_a[c] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, width * height *
				sizeof(cl_float), NULL, NULL);
		k_image_b[c] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, width * height *
				sizeof(cl_float), NULL, NULL);
		k_image_c[c] = clCreateBuffer(context,
				CL_MEM_READ_WRITE, width * height *
				sizeof(cl_float), NULL, NULL);

		if (k_input_image[c] == NULL)
			goto out_err;
		if (k_image_a[c] == NULL)
			goto out_err;
		if (k_image_b[c] == NULL)
			goto out_err;
		if (k_image_c[c] == NULL)
			goto out_err;

		/* allocate complex buffers */
		for (i = 0; i < 2; i++) {
			k_cimage_a[c][i] = clCreateBuffer(context,
					CL_MEM_READ_WRITE, (width/2 + 1)
					* height * sizeof(cl_float),
					NULL, NULL);
			k_cimage_b[c][i] = clCreateBuffer(context,
					CL_MEM_READ_WRITE, (width/2 + 1)
					* height * sizeof(cl_float),
					NULL, NULL);
			k_cimage_psf[c][i] = clCreateBuffer(context,
					CL_MEM_READ_ONLY, (width/2 + 1)
					* height * sizeof(cl_float),
					NULL, NULL);

			if (k_cimage_a[c][i] == NULL)
				goto out_err;
			if (k_cimage_b[c][i] == NULL)
				goto out_err;
			if (k_cimage_psf[c][i] == NULL)
				goto out_err;
		}
	}

	return 0;

out_err:
	say_function_failed();
	cleanup_init_opencl();
	return -1;
}

/* will only be called once */
static void cleanup_init_opencl()
{
	int c, i;

	for (c = 0; c < 3; c++) {
		if (k_input_image[c] != NULL)
			clReleaseMemObject(k_input_image[c]);
		if (k_image_a[c] != NULL)
			clReleaseMemObject(k_image_a[c]);
		if (k_image_b[c] != NULL)
			clReleaseMemObject(k_image_b[c]);
		if (k_image_c[c] != NULL)
			clReleaseMemObject(k_image_c[c]);

		for (i = 0; i < 2; i++) {
			if (k_cimage_a[c][i] != NULL)
				clReleaseMemObject(k_cimage_a[c][i]);
			if (k_cimage_b[c][i] != NULL)
				clReleaseMemObject(k_cimage_b[c][i]);
			if (k_cimage_psf[c][i] != NULL)
				clReleaseMemObject(k_cimage_psf[c][i]);
		}
	}

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

	if (program != NULL)
		clReleaseProgram(program);

	cl_utils_cleanup_gpu(&context, &queue);
}

/*
 * copy reusable images to opencl buffers (also computes fft of psf
 * before copying that)
 *
 * returns 0 on success, anything else on failure
 */
static int copy_reusables_to_opencl()
{
	cl_int ret;
	int c, i;

	/* compute fft of psf */
	for (c = 0; c < 3; c++) {
		fft(psf_image[c], cimage_psf[c]);
	}

	for (c = 0; c < 3; c++) {
		ret = clEnqueueWriteBuffer(queue, k_input_image[c],
				CL_TRUE, 0, width * height *
				sizeof(cl_float), input_image[c], 0,
				NULL, &copy_events[c][2]);
		if (ret != CL_SUCCESS)
			goto out_err;

		for (i = 0; i < 2; i++) {
			ret = clEnqueueWriteBuffer(queue,
					k_cimage_psf[c][i], CL_TRUE, 0,
					(width/2 + 1) * height *
					sizeof(cl_float),
					cimage_psf[c][i], 0, NULL,
					&copy_events[c][i]);
			if (ret != CL_SUCCESS)
				goto out_err;
		}

		ret = clWaitForEvents(3, copy_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	return 0;

out_err:
	say_function_failed();
	return ret;
}

/* do one iteration of Richardson–Lucy deconvolution */
static int do_iteration()
{
	int ret;
	int c;

	/* compute convolution of psf and current image */
	for (c = 0; c < 3; c++) {
		fft(current_image[c], cimage_a[c]);
	}

	ret = cpsf_multiply(cimage_a, cimage_b);
	if (ret != 0)
		goto out_err;

	for (c = 0; c < 3; c++) {
		ifft(cimage_b[c], image_a[c]);
	}

	/* compute original image/(convolution of psf and current image) */
	ret = image_input_divide(image_a, image_b);
	if (ret != 0)
		goto out_err;

	/* compute convolution of psf(-x) and previous result */
	for (c = 0; c < 3; c++) {
		fft(image_b[c], cimage_b[c]);
	}

	ret = cpsf_conj_multiply(cimage_b, cimage_a);
	if (ret != 0)
		goto out_err;

	for (c = 0; c < 3; c++) {
		ifft(cimage_a[c], image_a[c]);
	}

	/* multiply current image by previous result to get new current
	 * image */
	ret = image_multiply(current_image, image_a, current_image);
	if (ret != 0)
		goto out_err;

	return 0;

out_err:
	say_function_failed();
	return -1;
}

static int output(char *output_image_filename)
{
	int i;
	int ret;
	uint16_t *original_output_image;

	ret = -1;

	/* allocate output buffer */
	original_output_image = malloc(3 * width * height *
			sizeof(*original_output_image));
	if (original_output_image == NULL)
		goto out_nomem;

	/* copy the current image to output buffer */
	for (i = 0; i < 3 * width * height; i++) {
		if (current_image[i%3][i/3] >= 1) {
			original_output_image[i] = UINT16_MAX;
		} else {
			original_output_image[i] =
				current_image[i%3][i/3] * UINT16_MAX;
		}
	}

	/* write TIFF image */
	ret = write_tiff16(output_image_filename, original_output_image,
			width, height);
	if (ret != 0)
		goto out_no_write;

	free(original_output_image);
	return 0;

out_no_write:
	free(original_output_image);
out_nomem:
	say_function_failed();
	return ret;
}

/*
 * multiply complex psf with complex image
 *
 * returns 0 on success, anything else otherwise
 */
static int cpsf_multiply(float *in[3][2], float *out[3][2])
{
	cl_int ret;
	int c, i;

	/* copy in to opencl buffers */
	for (c = 0; c < 3; c++) {
		for (i = 0; i < 2; i++) {
			ret = clEnqueueWriteBuffer(queue,
					k_cimage_a[c][i], CL_TRUE, 0,
					(width/2 + 1) * height *
					sizeof(cl_float), in[c][i], 0,
					NULL, &copy_events[c][i]);
			if (ret != CL_SUCCESS)
				goto out_err;
		}

		ret = clWaitForEvents(2, copy_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	/* run kernels */
	for (c = 0; c < 3; c++) {
		ret = clSetKernelArg(complex_mult_k[c], 0, sizeof(cl_mem),
				&k_cimage_psf[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_mult_k[c], 1, sizeof(cl_mem),
				&k_cimage_psf[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_mult_k[c], 2, sizeof(cl_mem),
				&k_cimage_a[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_mult_k[c], 3, sizeof(cl_mem),
				&k_cimage_a[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_mult_k[c], 4, sizeof(cl_mem),
				&k_cimage_b[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_mult_k[c], 5, sizeof(cl_mem),
				&k_cimage_b[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clEnqueueNDRangeKernel(queue, complex_mult_k[c],
				1, NULL, &global_work_size[1], NULL, 0,
				NULL, &kernel_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	ret = clWaitForEvents(3, kernel_events);
	if (ret != CL_SUCCESS)
		goto out_err;

	/* copy opencl buffers to out */
	for (c = 0; c < 3; c++) {
		for (i = 0; i < 2; i++) {
			ret = clEnqueueReadBuffer(queue,
					k_cimage_b[c][i], CL_TRUE, 0,
					(width/2 + 1) * height *
					sizeof(cl_float), out[c][i], 0,
					NULL, NULL);
			if (ret != CL_SUCCESS)
				goto out_err;
		}
	}

	return 0;

out_err:
	say_function_failed();
	return ret;
}

/*
 * divide input image with a real image
 *
 * returns 0 on success, anything else otherwise
 */
static int image_input_divide(float *in[3], float *out[3])
{
	cl_int ret;
	int c;

	/* copy in to opencl buffers */
	for (c = 0; c < 3; c++) {
		ret = clEnqueueWriteBuffer(queue, k_image_a[c], CL_TRUE,
				0, width * height * sizeof(cl_float),
				in[c], 0, NULL, &copy_events[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clWaitForEvents(1, copy_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	/* run kernels */
	for (c = 0; c < 3; c++) {
		ret = clSetKernelArg(divide_k[c], 0, sizeof(cl_mem),
				&k_input_image[c]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(divide_k[c], 1, sizeof(cl_mem),
				&k_image_a[c]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(divide_k[c], 2, sizeof(cl_mem),
				&k_image_b[c]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clEnqueueNDRangeKernel(queue, divide_k[c], 1,
				NULL, &global_work_size[0], NULL, 0,
				NULL, &kernel_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	ret = clWaitForEvents(3, kernel_events);
	if (ret != CL_SUCCESS)
		goto out_err;

	/* copy opencl buffers to out */
	for (c = 0; c < 3; c++) {
		ret = clEnqueueReadBuffer(queue, k_image_b[c], CL_TRUE,
				0, width * height * sizeof(cl_float),
				out[c], 0, NULL, NULL);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	return 0;

out_err:
	say_function_failed();
	return ret;
}

/*
 * multiply conj(psf) with complex image
 *
 * returns 0 on success, anything else otherwise
 */
static int cpsf_conj_multiply(float *in[3][2], float *out[3][2])
{
	cl_int ret;
	int c, i;

	/* copy in to opencl buffers */
	for (c = 0; c < 3; c++) {
		for (i = 0; i < 2; i++) {
			ret = clEnqueueWriteBuffer(queue,
					k_cimage_a[c][i], CL_TRUE, 0,
					(width/2 + 1) * height *
					sizeof(cl_float), in[c][i], 0,
					NULL, &copy_events[c][i]);
			if (ret != CL_SUCCESS)
				goto out_err;
		}

		ret = clWaitForEvents(2, copy_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	/* run kernels */
	for (c = 0; c < 3; c++) {
		ret = clSetKernelArg(complex_conj_mult_k[c], 0,
				sizeof(cl_mem), &k_cimage_psf[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_conj_mult_k[c], 1,
				sizeof(cl_mem), &k_cimage_psf[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_conj_mult_k[c], 2,
				sizeof(cl_mem), &k_cimage_a[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_conj_mult_k[c], 3,
				sizeof(cl_mem), &k_cimage_a[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_conj_mult_k[c], 4,
				sizeof(cl_mem), &k_cimage_b[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(complex_conj_mult_k[c], 5,
				sizeof(cl_mem), &k_cimage_b[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clEnqueueNDRangeKernel(queue,
				complex_conj_mult_k[c], 1, NULL,
				&global_work_size[1], NULL, 0, NULL,
				&kernel_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	ret = clWaitForEvents(3, kernel_events);
	if (ret != CL_SUCCESS)
		goto out_err;

	/* copy opencl buffers to out */
	for (c = 0; c < 3; c++) {
		for (i = 0; i < 2; i++) {
			ret = clEnqueueReadBuffer(queue,
					k_cimage_b[c][i], CL_TRUE, 0,
					(width/2 + 1) * height *
					sizeof(cl_float), out[c][i], 0,
					NULL, NULL);
			if (ret != CL_SUCCESS)
				goto out_err;
		}
	}

	return 0;

out_err:
	say_function_failed();
	return ret;
}

/*
 * multiply two real images
 *
 * returns 0 on success, anything else otherwise
 */
static int image_multiply(float *a[3], float *b[3], float *out[3])
{
	cl_int ret;
	int c;

	/* copy images to opencl buffers */
	for (c = 0; c < 3; c++) {
		ret = clEnqueueWriteBuffer(queue, k_image_a[c], CL_TRUE,
				0, width * height * sizeof(cl_float),
				a[c], 0, NULL, &copy_events[c][0]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clEnqueueWriteBuffer(queue, k_image_b[c], CL_TRUE,
				0, width * height * sizeof(cl_float),
				b[c], 0, NULL, &copy_events[c][1]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clWaitForEvents(2, copy_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	/* run kernels */
	for (c = 0; c < 3; c++) {
		ret = clSetKernelArg(mult_k[c], 0, sizeof(cl_mem),
				&k_image_a[c]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(mult_k[c], 1, sizeof(cl_mem),
				&k_image_b[c]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clSetKernelArg(mult_k[c], 2, sizeof(cl_mem),
				&k_image_c[c]);
		if (ret != CL_SUCCESS)
			goto out_err;

		ret = clEnqueueNDRangeKernel(queue, mult_k[c], 1, NULL,
				&global_work_size[0], NULL, 0, NULL,
				&kernel_events[c]);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	ret = clWaitForEvents(3, kernel_events);
	if (ret != CL_SUCCESS)
		goto out_err;

	/* copy opencl buffers to out */
	for (c = 0; c < 3; c++) {
		ret = clEnqueueReadBuffer(queue, k_image_c[c], CL_TRUE,
				0, width * height * sizeof(cl_float),
				out[c], 0, NULL, NULL);
		if (ret != CL_SUCCESS)
			goto out_err;
	}

	return 0;

out_err:
	say_function_failed();
	return ret;
}

/*
 * helper function to compute forward fft of real image data
 *
 * image must be width x height (static var)
 */
static void fft(float *in, float *out[2])
{
	int i;

	for (i = 0; i < width * height; i++) {
		fft_real[i] = in[i];
	}

	fftwf_execute(fft_forward_plan);

	for (i = 0; i < (width/2 + 1) * height; i++) {
		out[0][i] = fft_complex[i][0];
		out[1][i] = fft_complex[i][1];
	}
}

/*
 * helper function to compute inverse fft of complex image data
 * to real image data
 *
 * image must be width x height (static var)
 */
static void ifft(float *in[2], float *out)
{
	int i;

	for (i = 0; i < (width/2 + 1) * height; i++) {
		fft_complex[i][0] = in[0][i];
		fft_complex[i][1] = in[1][i];
	}

	fftwf_execute(fft_backward_plan);

	for (i = 0; i < width * height; i++) {
		out[i] = fft_real[i] / (width * height);
	}
}
