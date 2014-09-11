#ifndef _TIFF_GOODNESS_H_
#define _TIFF_GOODNESS_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <tiffio.h>

uint16_t *read_tiff16(char *filename, int *width, int *height);
uint8_t *read_tiff8(char *filename, int *width, int *height);
int write_tiff16(char *filename, uint16_t *image_data, int width, int height);

#endif /* !_TIFF_GOODNESS_H_ */
