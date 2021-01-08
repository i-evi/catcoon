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
} UTIM_IMG; /* For 8bit color-depth images */

typedef int  UTIM_POINT[2]; /* X, Y */
typedef byte UTIM_COLOR[4]; /* R, G, B, A */

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
} UTIM_FONT;

#define UTIM_POINT_X 0
#define UTIM_POINT_Y 1

#define UTIM_COLOR_R 0
#define UTIM_COLOR_G 1
#define UTIM_COLOR_B 2
#define UTIM_COLOR_A 3

#define UTIM_BACK_COLOR 255

#define UTIM_RESIZE_NEAREST 0
#define UTIM_RESIZE_LINEAR  1

UTIM_IMG *utim_read(const char *filename);

int utim_write(const char *filename, UTIM_IMG *img);
int utim_write_ctrl(const char *filename,
	UTIM_IMG *img, int comp, int quality);

UTIM_IMG *utim_clone(UTIM_IMG *img);
UTIM_IMG *utim_create(int x, int y, int nch, int init);

void utim_free_image(UTIM_IMG *img);

UTIM_IMG *utim_resize(UTIM_IMG *img, int x, int y, int mode);

/* Swap 2 channels */
int utim_swap_chl(UTIM_IMG *img, int a, int b);

int utim_img2rgb (UTIM_IMG *img);
int utim_img2gray(UTIM_IMG *img);
int utim_img2rgba(UTIM_IMG *img);

/* gray2rgb will copy gray channel 2 times */
int utim_gray2rgb (UTIM_IMG *gray);
int utim_gray2rgba(UTIM_IMG *gray);
int utim_rgb2bgr  (UTIM_IMG *rgb);
int utim_rgb2gray (UTIM_IMG *rgb);
int utim_rgb2rgba (UTIM_IMG *rgb);
int utim_rgba2rgb (UTIM_IMG *rgba);
int utim_rgba2gray(UTIM_IMG *rgba);

UTIM_IMG *utim_rgb_by_rgba(UTIM_IMG *rgba);
UTIM_IMG *utim_bgr_by_rgb (UTIM_IMG *rgb);
UTIM_IMG *utim_gray_by_rgb(UTIM_IMG *rgb);
UTIM_IMG *utim_rgb_by_gray(UTIM_IMG *gray);

UTIM_IMG *utim_stack   (UTIM_IMG **chx, int nch);
UTIM_IMG *utim_pick_chl(UTIM_IMG *img, int ich);
UTIM_IMG *utim_set_chl (UTIM_IMG *img, int ich, int color);

void utim_negative_color(UTIM_IMG *img);

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
int utim_set_opacity(UTIM_IMG *img, int opacity);

int utim_superpose(UTIM_IMG *bg, UTIM_IMG *img, UTIM_POINT p);

/*
 * Simple Drawing functions
 */
int utim_set_pixel(UTIM_IMG *img, UTIM_POINT p, UTIM_COLOR c);

int utim_draw_point(UTIM_IMG *img, UTIM_POINT p, UTIM_COLOR c);

void utim_set_draw_point_fn(int (*fn)
	(UTIM_IMG*, UTIM_POINT, UTIM_COLOR));

void utim_draw_line(UTIM_IMG *img,
	UTIM_POINT a, UTIM_POINT b, UTIM_COLOR c, int width);

void utim_draw_rect(UTIM_IMG *img,
	UTIM_POINT a, int w, int h, UTIM_COLOR c, int width);

void utim_draw_circle(UTIM_IMG *img,
	UTIM_POINT center, int radius, UTIM_COLOR color, int width);

void utim_draw_filled_circle(UTIM_IMG *img,
	UTIM_POINT center, int radius, UTIM_COLOR color);

/*
 * Simple text
 */
UTIM_FONT *utim_load_font(const char *filename);

void utim_free_font(UTIM_FONT *font);

UTIM_IMG *utim_text(UTIM_FONT *font, char *text, UTIM_COLOR color);

#undef byte

#ifdef __cplusplus
	}
#endif

#endif /* _UTIL_IMAGE_H_ */
