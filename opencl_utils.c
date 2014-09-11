#include "opencl_utils.h"

/*
 * reads a file and returns a malloced char* of its contents
 * that needs to freed
 */
char *cl_utils_read_file(char *filename)
{
	FILE *file;
	int size;
	char *contents;

	if ((file = fopen(filename, "r")) == NULL)
		goto out_no_open;
	if (fseek(file, 0, SEEK_END) == -1)
		goto out_info_err;
	if ((size = ftell(file)) == -1)
		goto out_info_err;
	if (fseek(file, 0, SEEK_SET) == -1)
		goto out_info_err;

	contents = malloc(size + 1);
	if (contents == NULL)
		goto out_nomem;

	fread(contents, 1, size, file);
	if (ferror(file)) {
		free(contents);
		goto out_read_err;
	}

	fclose(file);
	contents[size] = '\0';
	goto out;

out_read_err:
	free(contents);
out_nomem:
out_info_err:
	fclose(file);
out_no_open:
	contents = NULL;
out:
	return contents;
}

/*
 * creates opencl context and command queue using gpu device
 * 
 * returns 0 on success, anything else on failure
 */
int cl_utils_setup_gpu(cl_context *context, cl_command_queue
		*command_queue, cl_device_id *device)
{
	int ret;
	cl_int err;
	cl_platform_id platform;

	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS)
		goto out_err;

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device,
			NULL);
	if (err != CL_SUCCESS)
		goto out_err;

	*context = clCreateContext(0, 1, device, NULL, NULL, &err);
	if (err != CL_SUCCESS)
		goto out_err;

	*command_queue = clCreateCommandQueue(*context, *device, 0,
			&err);
	if (err != CL_SUCCESS)
		goto out_no_queue;

	ret = 0;
	goto out;

out_no_queue:
	clReleaseContext(*context);
out_err:
	ret = -1;
out:
	return ret;
}

/*
 * create a opencl program from source code in filename
 * 
 * returns 0 on success, anything else otherwise
 */
int cl_utils_create_program(cl_program *program, char *filename,
		cl_context context, cl_device_id device)
{
	int ret;
	cl_int err;
	char *source_code;
	size_t build_log_size;
	char *build_log;

	source_code = cl_utils_read_file(filename);
	if (source_code == NULL) {
		fprintf(stderr, "cl_utils_create_program: could not read %s\n",
				filename);
		fflush(stderr);
		goto out_no_source;
	}

	*program = clCreateProgramWithSource(context, 1,
			(const char **) &source_code, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed\n");
		fflush(stderr);
		goto out_no_program;
	}

	err = clBuildProgram(*program, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram failed\n");

		clGetProgramBuildInfo(*program, device,
				CL_PROGRAM_BUILD_LOG, 0, NULL,
				&build_log_size);
		build_log = malloc(build_log_size);
		if (build_log == NULL) {
			fprintf(stderr, "No memory for build log\n");
			fflush(stderr);
			goto out_build_fail;
		}

		clGetProgramBuildInfo(*program, device,
				CL_PROGRAM_BUILD_LOG, build_log_size,
				build_log, NULL);
		fprintf(stderr, "%s\n", build_log);
		fflush(stderr);
		free(build_log);
		goto out_build_fail;
	}

	free(source_code);
	ret = 0;
	goto out;

out_build_fail:
	clReleaseProgram(*program);
out_no_program:
	free(source_code);
out_no_source:
	ret = -1;
out:
	return ret;
}
