#ifndef _OPEN_CL_UTILS_H_
#define _OPEN_CL_UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>

int cl_utils_setup_gpu(cl_context *context, cl_command_queue
		*command_queue, cl_device_id *device);
int cl_utils_create_program(cl_program *program, char *filename,
		cl_context context, cl_device_id device);

#endif /* !_OPEN_CL_UTILS_H_ */
