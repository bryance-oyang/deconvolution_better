#ifndef _OPEN_CL_UTILS_H_
#define _OPEN_CL_UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>
#include "emalloc.h"

char *cl_utils_read_file(char *filename);
void cl_utils_setup_gpu(cl_context *context, cl_command_queue
		*command_queue, cl_device_id *device);
cl_program cl_utils_create_program(char *filename, cl_context context,
		cl_device_id device);

#endif /* !_OPEN_CL_UTILS_H_ */
