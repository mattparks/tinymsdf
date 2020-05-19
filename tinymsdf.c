#include "tinymsdf.h"

#include <math.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
	double x, y;
} vec2;

vec2 vec2_add(vec2 a, vec2 b) {
	return (vec2){a.x+b.x, a.y+b.y};
}

vec2 vec2_sub(vec2 a, vec2 b) {
	return (vec2){a.x-b.x, a.y-b.y};
}

vec2 vec2_mul(double a, vec2 b) {
	return (vec2) {a*b.x, a*b.y};
}

vec2 vec2_mix(vec2 a, vec2 b, double weight) {
	vec2 r;
	r.x = (1-weight)*a.x+weight*b.x;
	r.y = (1-weight)*a.y+weight*b.y;
	return r;
}

int vec2_eql(vec2 a, vec2 b) {
	return a.x == b.x && a.y == b.y;
}

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
	edge_type_t type;
	vec2 p[4];
} edge_segment_t;

edge_segment_t edge_segment_linear(vec2 p0, vec2 p1, edge_color_t color /* = EDGE_COLOR_WHITE*/) {
	edge_segment_t edge_segment = {color, EDGE_TYPE_LINEAR};
	edge_segment.p[0] = p0;
	edge_segment.p[1] = p1;
	return edge_segment;
}

edge_segment_t edge_segment_quadratic(vec2 p0, vec2 p1, vec2 p2, edge_color_t color /* = EDGE_COLOR_WHITE*/) {
	if (vec2_eql(p1, p0) || vec2_eql(p1, p2))
		p1 = vec2_mul(0.5f, vec2_add(p0, p2));
	
	edge_segment_t edge_segment = {color, EDGE_TYPE_QUADRATIC};
	edge_segment.p[0] = p0;
	edge_segment.p[1] = p1;
	edge_segment.p[2] = p2;
	return edge_segment;
}

edge_segment_t edge_segment_cubic(vec2 p0, vec2 p1, vec2 p2, vec2 p3, edge_color_t color /* = EDGE_COLOR_WHITE*/) {
	edge_segment_t edge_segment = {color, EDGE_TYPE_CUBIC};
	edge_segment.p[0] = p0;
	edge_segment.p[1] = p1;
	edge_segment.p[2] = p2;
	edge_segment.p[3] = p3;
	return edge_segment;
}

vec2 edge_segment_point(const edge_segment_t *edge, double param) {
	switch (edge->type) {
	case EDGE_TYPE_LINEAR:
		return vec2_mix(edge->p[0], edge->p[1], param);
	case EDGE_TYPE_QUADRATIC:
		return vec2_mix(vec2_mix(edge->p[0], edge->p[1], param), vec2_mix(edge->p[1], edge->p[2], param), param);
	case EDGE_TYPE_CUBIC:
	{
		vec2 p12 = vec2_mix(edge->p[1], edge->p[2], param);
		return vec2_mix(vec2_mix(vec2_mix(edge->p[0], edge->p[1], param), p12, param), vec2_mix(p12, vec2_mix(edge->p[2], edge->p[3], param), param), param);
	}
	}
}

/// Splits the edge segments into thirds which together represent the original edge.
void edge_segment_split_in_thirds(const edge_segment_t *src, edge_segment_t *part1, edge_segment_t *part2, edge_segment_t *part3) {
	switch (src->type) {
	case EDGE_TYPE_LINEAR:
	{
		*part1 = edge_segment_linear(src->p[0], edge_segment_point(src, 1/3.0), src->color);
		*part2 = edge_segment_linear(edge_segment_point(src, 1/3.0), edge_segment_point(src, 2/3.0), src->color);
		*part3 = edge_segment_linear(edge_segment_point(src, 2/3.0), src->p[1], src->color);
		break;
	}
	case EDGE_TYPE_QUADRATIC:
	{
		*part1 = edge_segment_quadratic(src->p[0], vec2_mix(src->p[0], src->p[1], 1/3.0), edge_segment_point(src, 1/3.0), src->color);
		*part2 = edge_segment_quadratic(edge_segment_point(src, 1/3.0), vec2_mix(vec2_mix(src->p[0], src->p[1], 5/9.0), vec2_mix(src->p[1], src->p[2], 4/9.0), 0.5), edge_segment_point(src, 2/3.0), src->color);
		*part3 = edge_segment_quadratic(edge_segment_point(src, 2/3.0), vec2_mix(src->p[1], src->p[2], 2/3.0), src->p[2], src->color);
		break;
	}
	case EDGE_TYPE_CUBIC:
	{
		*part1 = edge_segment_cubic(src->p[0], vec2_eql(src->p[0], src->p[1]) ? src->p[0] : vec2_mix(src->p[0], src->p[1], 1/3.0), vec2_mix(vec2_mix(src->p[0], src->p[1], 1/3.0), vec2_mix(src->p[1], src->p[2], 1/3.0), 1/3.0), edge_segment_point(src, 1/3.0), src->color);
		*part2 = edge_segment_cubic(edge_segment_point(src, 1/3.0),
			vec2_mix(vec2_mix(vec2_mix(src->p[0], src->p[1], 1/3.0), vec2_mix(src->p[1], src->p[2], 1/3.0), 1/3.0), vec2_mix(vec2_mix(src->p[1], src->p[2], 1/3.0), vec2_mix(src->p[2], src->p[3], 1/3.0), 1/3.0), 2/3.0),
			vec2_mix(vec2_mix(vec2_mix(src->p[0], src->p[1], 2/3.0), vec2_mix(src->p[1], src->p[2], 2/3.0), 2/3.0), vec2_mix(vec2_mix(src->p[1], src->p[2], 2/3.0), vec2_mix(src->p[2], src->p[3], 2/3.0), 2/3.0), 1/3.0),
			edge_segment_point(src, 2/3.0), src->color);
		*part3 = edge_segment_cubic(edge_segment_point(src, 2/3.0), vec2_mix(vec2_mix(src->p[1], src->p[2], 2/3.0), vec2_mix(src->p[2], src->p[3], 2/3.0), 2/3.0), vec2_eql(src->p[2], src->p[3]) ? src->p[3] : vec2_mix(src->p[2], src->p[3], 2/3.0), src->p[3], src->color);
		break;
	}
	}
}

typedef struct {
	edge_segment_t *edges;
	int edge_count;
} contour_t;

int contour_add_edge(contour_t *contour, edge_segment_t edge) {
	int count = ++contour->edge_count;
	contour->edges = realloc(contour->edges, sizeof(edge_segment_t)*count);
	contour->edges[count-1] = edge;
	return count-1; // index
}

void free_contour(contour_t *contour) {
	free(contour->edges), contour->edges = NULL;
	contour->edge_count = 0;
}

typedef struct {
	contour_t *contours;
	int contour_count;
} shape_t;

int shape_add_contour(shape_t *shape, contour_t contour) {
	int count = ++shape->contour_count;
	shape->contours = realloc(shape->contours, sizeof(contour_t)*count);
	shape->contours[count-1] = contour;
	return count-1; // index
}

void free_shape(shape_t *shape) {
	for (int i = 0; i < shape->contour_count; i++)
		free_contour(&shape->contours[i]);
	free(shape->contours), shape->contours = NULL;
	shape->contour_count = 0;
}

typedef struct {
	vec2 position;
	shape_t *shape;
	contour_t *contour;
} context_t;

#define F26DOT6_TO_DOUBLE(x) (1/64.0*(double)x)

inline vec2 ftPoint2(const FT_Vector *vector) {
	return (vec2){ F26DOT6_TO_DOUBLE(vector->x), F26DOT6_TO_DOUBLE(vector->y)};
}

int tinymsdf_move_to(const FT_Vector *to, void *user)
{
	context_t *context = user;
	if (!(context->contour != NULL && context->contour->edge_count == 0)) {
		int index = context->shape->contour_count;
		shape_add_contour(context->shape, (contour_t){NULL, 0});
		context->contour = &context->shape->contours[index];
	}
	context->position = ftPoint2(to);
	return 0;
}

int tinymsdf_line_to(const FT_Vector *to, void *user)
{
	context_t *context = user;
	vec2 endpoint = ftPoint2(to);
	if (!vec2_eql(endpoint, context->position)) {
		contour_add_edge(context->contour, edge_segment_linear(context->position, endpoint, EDGE_COLOR_WHITE));
		context->position = endpoint;
	}
	return 0;
}

int tinymsdf_conic_to(const FT_Vector *control, const FT_Vector *to, void *user)
{
	context_t *context = user;
	contour_add_edge(context->contour, edge_segment_quadratic(context->position, ftPoint2(control), ftPoint2(to), EDGE_COLOR_WHITE));
	context->position = ftPoint2(to);
	return 0;
}

int tinymsdf_cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user)
{
	context_t *context = user;
	contour_add_edge(context->contour, edge_segment_cubic(context->position, ftPoint2(control1), ftPoint2(control2), ftPoint2(to), EDGE_COLOR_WHITE));
	context->position = ftPoint2(to);
	return 0;
}

tinymsdf_error_t tinymsdf_load_glyph(shape_t *shape, FT_Face face, unicode_t unicode)
{
	FT_Error error = FT_Load_Char(face, unicode, FT_LOAD_NO_SCALE);
	if (error)
		return TINYMSDF_GLYPH_LOAD;

	FT_Outline_Funcs outlineFuncs;
	outlineFuncs.move_to = &tinymsdf_move_to;
	outlineFuncs.line_to = &tinymsdf_line_to;
	outlineFuncs.conic_to = &tinymsdf_conic_to;
	outlineFuncs.cubic_to = &tinymsdf_cubic_to;
	outlineFuncs.shift = 0;
	outlineFuncs.delta = 0;

	context_t context;
	context.position = (vec2){0, 0};
	context.shape = shape;
	context.contour = NULL;
	
	error = FT_Outline_Decompose(&face->glyph->outline, &outlineFuncs, &context);
	if (error)
		return TINYMSDF_OUTLINE_DECOMPOSE;

	/// Performs basic checks to determine if the object represents a valid shape.
	for (int i = 0; i < shape->contour_count; i++) {
		contour_t *contour = &shape->contours[i];
		if (contour->edge_count != 0) {
			vec2 corner = edge_segment_point(&contour->edges[contour->edge_count-1], 1);
			for (int j = 0; j < contour->edge_count; j++) {
				edge_segment_t *edge = &contour->edges[j];
				if (!vec2_eql(edge_segment_point(edge, 0), corner))
					return TINYMSDF_INVALID_GEOMETRY;
				corner = edge_segment_point(edge, 1);
			}
		}
	}
	
	/// Normalizes the shape geometry for distance field generation.
	for (int i = 0; i < shape->contour_count; i++) {
		contour_t *contour = &shape->contours[i];
		if (contour->edge_count == 1) {
			edge_segment_t parts[3];
			edge_segment_split_in_thirds(&contour->edges[0], &parts[0], &parts[1], &parts[2]);
			free_contour(contour);
			contour_add_edge(contour, parts[0]);
			contour_add_edge(contour, parts[1]);
			contour_add_edge(contour, parts[2]);
		}
	}

	return TINYMSDF_SUCCESS;
}

tinymsdf_error_t tinymsdf_create_bitmap(float **pixels, int width, int height, int components) {
	*pixels = malloc(sizeof(float)*components*width*height);
	memset(*pixels, 0.0f, sizeof(float)*components*width*height);
	return TINYMSDF_SUCCESS;
}

tinymsdf_error_t tinymsdf_generate_mtsdf(float *pixels, int width, int height, FT_Face face, unicode_t unicode)
{
	shape_t shape = {NULL, 0};
	tinymsdf_error_t error = tinymsdf_load_glyph(&shape, face, unicode);
	if (error)
		goto error_exit;

	error = tinymsdf_create_bitmap(&pixels, width, height, 4);
	if (error)
		goto error_exit;

error_exit:
	free_shape(&shape);
	return error;
}
