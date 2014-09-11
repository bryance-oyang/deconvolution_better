#include "tiff_goodness.h"

/*
 * read tiff, assumes tiff file has 3 channels per pixel, RGB,
 * with 16-bit channels
 *
 * returns a malloced uint16_t* that needs to be freed
 */
uint16_t *read_tiff(char *filename, int *width, int *height)
{
	int i;
	TIFF *tif;
	uint16_t *result;
	int scanline_size;

	if ((tif = TIFFOpen(filename, "r")) == NULL) {
		fprintf(stderr, "read_tiff: could not open %s\n",
				filename);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);
	scanline_size = TIFFScanlineSize(tif);
	if (scanline_size != 3 * (*width) * sizeof(*result)) {
		fprintf(stderr, "read_tiff: %s is not in correct format.  TIFF file should have 16-bit channels in RGBRGB format.\n",
				filename);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	result = emalloc((*height) * scanline_size);

	for (i = 0; i < (*height); i++) {
		if (TIFFReadScanline(tif, result + 3 * (*width) * i,
					i, 0)
				== -1) {
			fprintf(stderr, "read_tiff: error in reading %s row %d\n",
					filename, i);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
	}

	TIFFClose(tif);
	return result;
}

/*
 * read tiff, assumes tiff file has 3 channels per pixel, RGB,
 * with 8-bit channels
 *
 * returns a malloced uint8_t* that needs to be freed
 */
uint8_t *read_tiff8(char *filename, int *width, int *height)
{
	int i;
	TIFF *tif;
	uint8_t *result;
	int scanline_size;

	if ((tif = TIFFOpen(filename, "r")) == NULL) {
		fprintf(stderr, "read_tiff: could not open %s\n",
				filename);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);
	scanline_size = TIFFScanlineSize(tif);
	if (scanline_size != 3 * (*width) * sizeof(*result)) {
		fprintf(stderr, "read_tiff: %s is not in correct format.  TIFF file should have 8-bit channels in RGBRGB format.\n",
				filename);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	result = emalloc((*height) * scanline_size);

	for (i = 0; i < (*height); i++) {
		if (TIFFReadScanline(tif, result + 3 * (*width) * i,
					i, 0)
				== -1) {
			fprintf(stderr, "read_tiff: error in reading %s row %d\n",
					filename, i);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
	}

	TIFFClose(tif);
	return result;
}

/* write tiff with 3 channels per pixel, RGB, 16-bit per channel */
int write_tiff(char *filename, uint16_t *image_data, int width, int height)
{
	int i;
	TIFF *out;

	out = TIFFOpen(filename, "w");

	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 3);
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);
	TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

	for (i = 0; i < height; i++) {
		if (TIFFWriteScanline(out, image_data + 3 * i
					* width, i, 0) == -1) {
			fprintf(stderr, "write_tiff: error in writing %s on row %d",
					filename, i);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
	}

	TIFFClose(out);
	return 0;
}
