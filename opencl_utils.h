/*
 * Utilities for dealing with OpenCL
 *
 * Copyright (C) 2014 Bryance Oyang
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#ifndef _OPEN_CL_UTILS_H_
#define _OPEN_CL_UTILS_H_

#include <CL/opencl.h>

int cl_utils_setup_gpu(cl_context *context, cl_command_queue
		*command_queue, cl_device_id *device);
void cl_utils_cleanup_gpu(cl_context *context, cl_command_queue
		*command_queue);
int cl_utils_create_program(cl_program *program, char *filename,
		cl_context context, cl_device_id device);

#endif /* !_OPEN_CL_UTILS_H_ */
