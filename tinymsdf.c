#include "tinymsdf.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

typedef float vec2[2];

typedef struct {
	double distance, dot;
} signed_distance_t;

typedef enum {
	EDGE_COLOR_BLACK = 0,
	EDGE_COLOR_RED = 1,
	EDGE_COLOR_GREEN = 2,
	EDGE_COLOR_YELLOW = 3,
	EDGE_COLOR_BLUE = 4,
	EDGE_COLOR_MAGENTA = 5,
	EDGE_COLOR_CYAN = 6,
	EDGE_COLOR_WHITE = 7
} edge_color_t;

typedef enum {
	EDGE_TYPE_LINEAR = 0,
	EDGE_TYPE_QUADRATIC = 1,
	EDGE_TYPE_CUBIC = 2
} edge_type_t;

typedef struct {
	edge_color_t color;
	vec2 p[4];
	edge_type_t type;
} edge_segment_t;

typedef struct {
	void *fake_data;
} context_t;

int tinymsdf_move_to(const FT_Vector *to, void *user)
{
	context_t *context = user;
	printf("%s\n", __func__);
	return 0;
}

int tinymsdf_line_to(const FT_Vector *to, void *user)
{
	context_t *context = user;
	printf("%s\n", __func__);
	return 0;
}

int tinymsdf_conic_to(const FT_Vector *control, const FT_Vector *to, void *user)
{
	context_t *context = user;
	printf("%s\n", __func__);
	return 0;
}

int tinymsdf_cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user)
{
	context_t *context = user;
	printf("%s\n", __func__);
	return 0;
}

tinymsdf_error_t tinymsdf_load_glyph(FT_Face face, unicode_t unicode)
{
	FT_Error error = FT_Load_Char(face, unicode, FT_LOAD_NO_SCALE);
	if (error)
		return TINYMSDF_GLYPH_LOAD;

	context_t context;
	
	FT_Outline_Funcs outlineFuncs;
	outlineFuncs.move_to = &tinymsdf_move_to;
	outlineFuncs.line_to = &tinymsdf_line_to;
	outlineFuncs.conic_to = &tinymsdf_conic_to;
	outlineFuncs.cubic_to = &tinymsdf_cubic_to;
	outlineFuncs.shift = 0;
	outlineFuncs.delta = 0;

	error = FT_Outline_Decompose(&face->glyph->outline, &outlineFuncs, &context);
	if (error)
		return TINYMSDF_OUTLINE_DECOMPOSE;

	return TINYMSDF_SUCCESS;
}

tinymsdf_error_t tinymsdf_create_bitmap(float **pixels, int width, int height, int components) {
	*pixels = malloc(sizeof(float) * components * width * height);
	if (*pixels == NULL)
		return TINYMSDF_NOMEM;
	memset(*pixels, 0.0f, sizeof(float) * components * width * height);
	return TINYMSDF_SUCCESS;
}

tinymsdf_error_t tinymsdf_generate_mtsdf(float *pixels, int width, int height, FT_Face face, unicode_t unicode)
{
	tinymsdf_error_t error = tinymsdf_load_glyph(face, unicode);
	if (error)
		return error;

	error = tinymsdf_create_bitmap(&pixels, width, height, 4);
	if (error)
		return error;
	
	return TINYMSDF_SUCCESS;
}
