#ifndef _UTIL_IMAGE_H_
#define _UTIL_IMAGE_H_

#ifdef __cplusplus
	extern "C" {
#endif

#define UTIM_INT32 int

typedef UTIM_INT32 utim_int32_t;
typedef unsigned UTIM_INT32 utim_uint32_t;

#ifndef byte
	#define byte unsigned char
#endif

/*
 * Defined as little-endian
 */
#define UTIM_BMP 0x00706d62 /* b m p \0 */
#define UTIM_PNG 0x00676e70 /* p n g \0 */
#define UTIM_JPG 0x0067706a /* j p g \0 */
#define UTIM_TGA 0x00616774 /* t g a \0 */

#define UTIM_MAX_CHANNELS 4

enum {
	UTIM_ERR_ALLOC,
	UTIM_ERR_FILE_OPS,
	UTIM_ERR_BAD_ARG
};

typedef struct {
	byte *pixels;
	int   xsize;
	int   ysize;
	int   channels;
} utim_image_t; /* For 8bit color-depth images */

typedef int  utim_point_t[2]; /* X, Y */
typedef byte utim_color_t[4]; /* R, G, B, A */

/*
 * PSF2 fonts
 * View https://www.win.tue.nl/~aeb/linux/kbd/font-formats-1.html
 */
struct psf2_header {
	unsigned char magic[4];
	unsigned int version;
	unsigned int headersize;    /* offset of bitmaps in file */
	unsigned int flags;
	unsigned int length;        /* number of glyphs */
	unsigned int charsize;      /* number of bytes for each character */
	unsigned int height, width; /* max dimensions of glyphs */
	/* charsize = height * ((width + 7) / 8) */
};

typedef struct {
	struct psf2_header header;
	byte              *glyphs;
	unsigned int      *utf8index;
} utim_font_t;

#define UTIM_POINT_X 0
#define UTIM_POINT_Y 1

#define UTIM_COLOR_R 0
#define UTIM_COLOR_G 1
#define UTIM_COLOR_B 2
#define UTIM_COLOR_A 3

#define UTIM_BACK_COLOR 255

#define UTIM_RESIZE_NEAREST 0
#define UTIM_RESIZE_LINEAR  1

utim_image_t *utim_read(const char *filename);

int utim_write(const char *filename, utim_image_t *img);
int utim_write_ctrl(const char *filename,
	utim_image_t *img, int comp, int quality);

utim_image_t *utim_clone(utim_image_t *img);
utim_image_t *utim_create(int x, int y, int nch, int c);

void utim_free_image(utim_image_t *img);

utim_image_t *utim_resize(utim_image_t *img, int x, int y, int mode);

/* Swap 2 channels */
int utim_swap_chl(utim_image_t *img, int a, int b);

int utim_img2rgb (utim_image_t *img);
int utim_img2gray(utim_image_t *img);
int utim_img2rgba(utim_image_t *img);

/* gray2rgb will copy gray channel 2 times */
int utim_gray2rgb (utim_image_t *gray);
int utim_gray2rgba(utim_image_t *gray);
int utim_rgb2bgr  (utim_image_t *rgb);
int utim_rgb2gray (utim_image_t *rgb);
int utim_rgb2rgba (utim_image_t *rgb);
int utim_rgba2rgb (utim_image_t *rgba);
int utim_rgba2gray(utim_image_t *rgba);

utim_image_t *utim_rgb_by_rgba(utim_image_t *rgba);
utim_image_t *utim_bgr_by_rgb (utim_image_t *rgb);
utim_image_t *utim_gray_by_rgb(utim_image_t *rgb);
utim_image_t *utim_rgb_by_gray(utim_image_t *gray);

utim_image_t *utim_stack   (utim_image_t **chx, int nch);
utim_image_t *utim_pick_chl(utim_image_t *img, int ich);
utim_image_t *utim_set_chl (utim_image_t *img, int ich, int color);

void utim_negative_color(utim_image_t *img);

#define utim_set_point(point, x, y) \
	point[UTIM_POINT_X] = x;    \
	point[UTIM_POINT_Y] = y;

#define utim_set_color(color, r, g, b, a) \
	color[UTIM_COLOR_R] = r;          \
	color[UTIM_COLOR_G] = g;          \
	color[UTIM_COLOR_B] = b;          \
	color[UTIM_COLOR_A] = a;

/*
 * Support Gray, RGB, RGBA images. 
 * If not RGBA image, will convert to RGBA image
 */
int utim_set_opacity(utim_image_t *img, int opacity);

int utim_superpose(utim_image_t *bg, utim_image_t *img, utim_point_t p);

/*
 * Simple Drawing functions
 */
int utim_set_pixel(utim_image_t *img, utim_point_t p, utim_color_t c);

int utim_draw_point(utim_image_t *img, utim_point_t p, utim_color_t c);

void utim_set_draw_point_fn(int (*fn)
	(utim_image_t*, utim_point_t, utim_color_t));

void utim_draw_line(utim_image_t *img,
	utim_point_t a, utim_point_t b, utim_color_t c, int width);

void utim_draw_rect(utim_image_t *img,
	utim_point_t a, int w, int h, utim_color_t c, int width);

void utim_draw_circle(utim_image_t *img,
	utim_point_t center, int radius, utim_color_t color, int width);

void utim_draw_filled_circle(utim_image_t *img,
	utim_point_t center, int radius, utim_color_t color);

/*
 * Simple text
 */
utim_font_t *utim_load_font(const char *filename);

void utim_free_font(utim_font_t *font);

utim_image_t *utim_text(utim_font_t *font, char *text, utim_color_t color);

#undef byte

#ifdef __cplusplus
	}
#endif

#endif /* _UTIL_IMAGE_H_ */
