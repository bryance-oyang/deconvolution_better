#ifndef _DECONVOLUTE_H_
#define _DECONVOLUTE_H_

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
		n_iterations);

#endif /* _DECONVOLUTE_H_ */
