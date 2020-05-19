#include "tinymsdf.h"

#include <math.h>
#include <stdbool.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef float vec3[3];
#define P(x, y, w, arr) ((vec3){arr[(3*(((y)*w)+x))], arr[(3*(((y)*w)+x))+1], arr[(3*(((y)*w)+x))+2]})

typedef struct {
	double x, y;
} vec2;

double vec2_len(vec2 a) {
	return sqrt(a.x*a.x + a.y*a.y);
}

vec2 vec2_norm(vec2 a, bool allowZero /*= false */) {
	double len = vec2_len(a);
	if (len == 0)
		return (vec2){0, !allowZero};
	return (vec2) {a.x/len, a.y/len};
}

/// Dot product of two vectors.
double vec2_dot(vec2 a, vec2 b) {
	return a.x*b.x + a.y*b.y;
}

/// A special version of the cross product for 2D vectors (returns scalar value).
double vec2_cross(vec2 a, vec2 b) {
	return a.x*b.y - a.y*b.x;
}

vec2 vec2_neg(vec2 a) {
	return (vec2){-a.x, -a.y};
}

vec2 vec2_add(vec2 a, vec2 b) {
	return (vec2){a.x+b.x, a.y+b.y};
}

vec2 vec2_sub(vec2 a, vec2 b) {
	return (vec2){a.x-b.x, a.y-b.y};
}

vec2 vec2_mul(vec2 a, vec2 b) {
	return (vec2){a.x*b.x, a.y*b.y};
}

vec2 vec2_div(vec2 a, vec2 b) {
	return (vec2){a.x/b.x, a.y/b.y};
}

vec2 vec2_scale(double a, vec2 b) {
	return (vec2){a*b.x, a*b.y};
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

#define F26DOT6_TO_DOUBLE(x) (1/64.0*(double)x)

vec2 ftPoint2(const FT_Vector *vector) {
	return (vec2){F26DOT6_TO_DOUBLE(vector->x), F26DOT6_TO_DOUBLE(vector->y)};
}

#define TOO_LARGE_RATIO 1e12

int solve_quadratic(double x[2], double a, double b, double c) {
	// a = 0 -> linear equation
	if (a == 0 || fabs(b) + fabs(c) > TOO_LARGE_RATIO * fabs(a)) {
		// a, b = 0 -> no solution
		if (b == 0 || fabs(c) > TOO_LARGE_RATIO * fabs(b)) {
			if (c == 0)
				return -1; // 0 = 0
			return 0;
		}
		x[0] = -c / b;
		return 1;
	}
	double dscr = b * b - 4 * a * c;
	if (dscr > 0) {
		dscr = sqrt(dscr);
		x[0] = (-b + dscr) / (2 * a);
		x[1] = (-b - dscr) / (2 * a);
		return 2;
	} else if (dscr == 0) {
		x[0] = -b / (2 * a);
		return 1;
	} else
		return 0;
}

int solve_cubic_normed(double x[3], double a, double b, double c) {
	double a2 = a * a;
	double q = (a2 - 3 * b) / 9;
	double r = (a * (2 * a2 - 9 * b) + 27 * c) / 54;
	double r2 = r * r;
	double q3 = q * q * q;
	if (r2 < q3) {
		double t = r / sqrt(q3);
		if (t < -1) t = -1;
		if (t > 1) t = 1;
		t = acos(t);
		a /= 3; q = -2 * sqrt(q);
		x[0] = q * cos(t / 3) - a;
		x[1] = q * cos((t + 2 * M_PI) / 3) - a;
		x[2] = q * cos((t - 2 * M_PI) / 3) - a;
		return 3;
	} else {
		double A = -pow(fabs(r) + sqrt(r2 - q3), 1 / 3.0);
		if (r < 0) A = -A;
		double B = A == 0 ? 0 : q / A;
		a /= 3;
		x[0] = (A + B) - a;
		x[1] = -0.5 * (A + B) - a;
		x[2] = 0.5 * sqrt(3.0) * (A - B);
		if (fabs(x[2]) < 1e-14)
			return 2;
		return 1;
	}
}

inline int solve_cubic(double x[3], double a, double b, double c, double d) {
	if (a != 0) {
		double bn = b / a, cn = c / a, dn = d / a;
		// Check that a isn't "almost zero"
		if (fabs(bn) < TOO_LARGE_RATIO && fabs(cn) < TOO_LARGE_RATIO && fabs(dn) < TOO_LARGE_RATIO)
			return solve_cubic_normed(x, bn, cn, dn);
	}
	return solve_quadratic(x, b, c, d);
}

typedef struct {
	double distance, dot;
} signed_distance_t;

typedef struct {
	double l, b, r, t;
} bounds_t;

void bound_point(vec2 p, bounds_t *bound) {
	if (p.x < bound->l) bound->l = p.x;
	if (p.y < bound->b) bound->b = p.y;
	if (p.x > bound->r) bound->r = p.x;
	if (p.y > bound->t) bound->t = p.y;
}

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

edge_segment_t edge_segment_linear(vec2 p0, vec2 p1, edge_color_t color /*= EDGE_COLOR_WHITE*/) {
	edge_segment_t edge_segment = {color, EDGE_TYPE_LINEAR};
	edge_segment.p[0] = p0;
	edge_segment.p[1] = p1;
	return edge_segment;
}

edge_segment_t edge_segment_quadratic(vec2 p0, vec2 p1, vec2 p2, edge_color_t color /*= EDGE_COLOR_WHITE*/) {
	if (vec2_eql(p1, p0) || vec2_eql(p1, p2))
		p1 = vec2_scale(0.5, vec2_add(p0, p2));
	
	edge_segment_t edge_segment = {color, EDGE_TYPE_QUADRATIC};
	edge_segment.p[0] = p0;
	edge_segment.p[1] = p1;
	edge_segment.p[2] = p2;
	return edge_segment;
}

edge_segment_t edge_segment_cubic(vec2 p0, vec2 p1, vec2 p2, vec2 p3, edge_color_t color /*= EDGE_COLOR_WHITE*/) {
	edge_segment_t edge_segment = {color, EDGE_TYPE_CUBIC};
	edge_segment.p[0] = p0;
	edge_segment.p[1] = p1;
	edge_segment.p[2] = p2;
	edge_segment.p[3] = p3;
	return edge_segment;
}

/// Returns the point on the edge specified by the parameter (between 0 and 1).
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

/// Returns the direction the edge has at the point specified by the parameter.
vec2 edge_segment_direction(const edge_segment_t *edge, double param) {
	switch (edge->type) {
	case EDGE_TYPE_LINEAR:
		return vec2_sub(edge->p[1], edge->p[0]);
	case EDGE_TYPE_QUADRATIC:
	{
		vec2 tangent = vec2_mix(vec2_sub(edge->p[1], edge->p[0]), vec2_sub(edge->p[2], edge->p[1]), param);
		if (!(!tangent.x && !tangent.y))
			return vec2_sub(edge->p[2], edge->p[0]);
		return tangent;
	}
	case EDGE_TYPE_CUBIC:
	{
		vec2 tangent = vec2_mix(vec2_mix(vec2_sub(edge->p[1], edge->p[0]), vec2_sub(edge->p[2], edge->p[1]), param), vec2_mix(vec2_sub(edge->p[2], edge->p[1]), vec2_sub(edge->p[3], edge->p[2]), param), param);
		if (!(!tangent.x && !tangent.y)) {
			if (param == 0) return vec2_sub(edge->p[2], edge->p[0]);
			if (param == 1) return vec2_sub(edge->p[3], edge->p[1]);
		}
		return tangent;
	}
	}
}

/// Splits the edge segments into thirds which together represent the original edge.
void edge_segment_split_in_thirds(const edge_segment_t *edge, edge_segment_t *part1, edge_segment_t *part2, edge_segment_t *part3) {
	switch (edge->type) {
	case EDGE_TYPE_LINEAR:
	{
		*part1 = edge_segment_linear(edge->p[0], edge_segment_point(edge, 1/3.0), edge->color);
		*part2 = edge_segment_linear(edge_segment_point(edge, 1/3.0), edge_segment_point(edge, 2/3.0), edge->color);
		*part3 = edge_segment_linear(edge_segment_point(edge, 2/3.0), edge->p[1], edge->color);
		break;
	}
	case EDGE_TYPE_QUADRATIC:
	{
		*part1 = edge_segment_quadratic(edge->p[0], vec2_mix(edge->p[0], edge->p[1], 1/3.0), edge_segment_point(edge, 1/3.0), edge->color);
		*part2 = edge_segment_quadratic(edge_segment_point(edge, 1/3.0), vec2_mix(vec2_mix(edge->p[0], edge->p[1], 5/9.0), vec2_mix(edge->p[1], edge->p[2], 4/9.0), 0.5), edge_segment_point(edge, 2/3.0), edge->color);
		*part3 = edge_segment_quadratic(edge_segment_point(edge, 2/3.0), vec2_mix(edge->p[1], edge->p[2], 2/3.0), edge->p[2], edge->color);
		break;
	}
	case EDGE_TYPE_CUBIC:
	{
		*part1 = edge_segment_cubic(edge->p[0], vec2_eql(edge->p[0], edge->p[1]) ? edge->p[0] : vec2_mix(edge->p[0], edge->p[1], 1/3.0), vec2_mix(vec2_mix(edge->p[0], edge->p[1], 1/3.0), vec2_mix(edge->p[1], edge->p[2], 1/3.0), 1/3.0), edge_segment_point(edge, 1/3.0), edge->color);
		*part2 = edge_segment_cubic(edge_segment_point(edge, 1/3.0),
			vec2_mix(vec2_mix(vec2_mix(edge->p[0], edge->p[1], 1/3.0), vec2_mix(edge->p[1], edge->p[2], 1/3.0), 1/3.0), vec2_mix(vec2_mix(edge->p[1], edge->p[2], 1/3.0), vec2_mix(edge->p[2], edge->p[3], 1/3.0), 1/3.0), 2/3.0),
			vec2_mix(vec2_mix(vec2_mix(edge->p[0], edge->p[1], 2/3.0), vec2_mix(edge->p[1], edge->p[2], 2/3.0), 2/3.0), vec2_mix(vec2_mix(edge->p[1], edge->p[2], 2/3.0), vec2_mix(edge->p[2], edge->p[3], 2/3.0), 2/3.0), 1/3.0),
			edge_segment_point(edge, 2/3.0), edge->color);
		*part3 = edge_segment_cubic(edge_segment_point(edge, 2/3.0), vec2_mix(vec2_mix(edge->p[1], edge->p[2], 2/3.0), vec2_mix(edge->p[2], edge->p[3], 2/3.0), 2/3.0), vec2_eql(edge->p[2], edge->p[3]) ? edge->p[3] : vec2_mix(edge->p[2], edge->p[3], 2/3.0), edge->p[3], edge->color);
		break;
	}
	}
}

/// Adjusts the bounding box to fit the edge segment.
void edge_segment_bound(const edge_segment_t *edge, bounds_t *bound) {
	switch (edge->type) {
	case EDGE_TYPE_LINEAR:
	{
		bound_point(edge->p[0], bound);
		bound_point(edge->p[1], bound);
		break;
	}
	case EDGE_TYPE_QUADRATIC:
	{
		bound_point(edge->p[0], bound);
		bound_point(edge->p[2], bound);
		vec2 bot = vec2_sub(vec2_sub(edge->p[1], edge->p[0]), vec2_sub(edge->p[2], edge->p[1]));
		if (bot.x) {
			double param = (edge->p[1].x - edge->p[0].x) / bot.x;
			if (param > 0 && param < 1)
				bound_point(edge_segment_point(edge, param), bound);
		}
		if (bot.y) {
			double param = (edge->p[1].y - edge->p[0].y) / bot.y;
			if (param > 0 && param < 1)
				bound_point(edge_segment_point(edge, param), bound);
		}
		break;
	}
	case EDGE_TYPE_CUBIC:
	{
		bound_point(edge->p[0], bound);
		bound_point(edge->p[3], bound);
		vec2 a0 = vec2_sub(edge->p[1], edge->p[0]);
		vec2 a1 = vec2_scale(2.0, vec2_sub(vec2_sub(edge->p[2], edge->p[1]), a0));
		vec2 a2 = vec2_add(vec2_sub(edge->p[3], vec2_scale(3.0, edge->p[2])), vec2_sub(vec2_scale(3.0, edge->p[1]), edge->p[0]));
		double params[2];
		int solutions = solve_quadratic(params, a2.x, a1.x, a0.x);
		for (int i = 0; i < solutions; ++i)
			if (params[i] > 0 && params[i] < 1)
				bound_point(edge_segment_point(edge, params[i]), bound);
		solutions = solve_quadratic(params, a2.y, a1.y, a0.y);
		for (int i = 0; i < solutions; ++i)
			if (params[i] > 0 && params[i] < 1)
				bound_point(edge_segment_point(edge, params[i]), bound);
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

/// Adjusts the bounding box to fit the shape.
void shape_bound(shape_t *shape, bounds_t *bound) {
	for (int i = 0; i < shape->contour_count; i++) {
		contour_t *contour = &shape->contours[i];
		for (int j = 0; j < contour->edge_count; j++) {
			edge_segment_bound(&contour->edges[j], bound);
		}
	}
}

/// Adjusts the bounding box to fit the shape border's mitered corners.
void shape_bound_miters(shape_t *shape, bounds_t *bound, double border, double miterLimit, int polarity) {
	for (int i = 0; i < shape->contour_count; i++) {
		contour_t *contour = &shape->contours[i];
		if (contour->edge_count == 0)
			return;
		vec2 prevDir = vec2_norm(edge_segment_direction(&contour->edges[contour->edge_count - 1], 1.0), true);
		for (int j = 0; j < contour->edge_count; j++) {
			edge_segment_t *edge = &contour->edges[j];
			
			vec2 dir = vec2_neg(vec2_norm(edge_segment_direction(edge, 0.0), true));
			if (polarity * vec2_cross(prevDir, dir) >= 0.0) {
				double miterLength = miterLimit;
				double q = 0.5 * (1.0 - vec2_dot(prevDir, dir));
				if (q > 0.0)
					miterLength = min(1.0 / sqrt(q), miterLimit);
				vec2 miter = vec2_add(edge_segment_point(edge, 0.0), vec2_scale(border * miterLength, vec2_norm(vec2_add(prevDir, dir), true)));
				bound_point(miter, bound);
			}
			prevDir = vec2_norm(edge_segment_direction(edge, 1.0), true);
		}
	}
}

/// Computes the minimum bounding box that fits the shape, optionally with a (mitered) border.
bounds_t shape_get_bounds(shape_t *shape, double border /*= 0*/, double miterLimit /*= 0*/, int polarity /*= 0*/) {
#define LARGE_VALUE 1e240
	bounds_t bounds = {+LARGE_VALUE, +LARGE_VALUE, -LARGE_VALUE, -LARGE_VALUE};
	shape_bound(shape, &bounds);
	if (border > 0) {
		bounds.l -= border, bounds.b -= border;
		bounds.r += border, bounds.t += border;
		if (miterLimit > 0)
			shape_bound_miters(shape, &bounds, border, miterLimit, polarity);
	}
	return bounds;
}

typedef struct {
	vec2 position;
	shape_t *shape;
	contour_t *contour;
} context_t;

inline int move_to(const FT_Vector *to, void *user)
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

inline int line_to(const FT_Vector *to, void *user)
{
	context_t *context = user;
	vec2 endpoint = ftPoint2(to);
	if (!vec2_eql(endpoint, context->position)) {
		contour_add_edge(context->contour, edge_segment_linear(context->position, endpoint, EDGE_COLOR_WHITE));
		context->position = endpoint;
	}
	return 0;
}

inline int conic_to(const FT_Vector *control, const FT_Vector *to, void *user)
{
	context_t *context = user;
	contour_add_edge(context->contour, edge_segment_quadratic(context->position, ftPoint2(control), ftPoint2(to), EDGE_COLOR_WHITE));
	context->position = ftPoint2(to);
	return 0;
}

inline int cubic_to(const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user)
{
	context_t *context = user;
	contour_add_edge(context->contour, edge_segment_cubic(context->position, ftPoint2(control1), ftPoint2(control2), ftPoint2(to), EDGE_COLOR_WHITE));
	context->position = ftPoint2(to);
	return 0;
}

tinymsdf_error_t load_glyph(shape_t *shape, FT_Face face, unicode_t unicode)
{
	FT_Error error = FT_Load_Char(face, unicode, FT_LOAD_NO_SCALE);
	if (error)
		return TINYMSDF_GLYPH_LOAD;

	FT_Outline_Funcs outlineFuncs;
	outlineFuncs.move_to = &move_to;
	outlineFuncs.line_to = &line_to;
	outlineFuncs.conic_to = &conic_to;
	outlineFuncs.cubic_to = &cubic_to;
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

inline tinymsdf_error_t create_bitmap(float **pixels, int width, int height, int components) {
	*pixels = malloc(sizeof(float)*components*width*height);
	memset(*pixels, 0.0f, sizeof(float)*components*width*height);
	return TINYMSDF_SUCCESS;
}

inline bool is_corner(vec2 aDir, vec2 bDir, double crossThreshold) {
	return vec2_dot(aDir, bDir) <= 0 || fabs(vec2_cross(aDir, bDir)) > crossThreshold;
}

inline void switch_color(edge_color_t *color, unsigned long long *seed, edge_color_t banned /*= EDGE_COLOR_BLACK*/) {
	edge_color_t combined = *color & banned;
	if (combined == EDGE_COLOR_RED || combined == EDGE_COLOR_GREEN || combined == EDGE_COLOR_BLUE) {
		*color = combined ^ EDGE_COLOR_WHITE;
		return;
	}
	if (*color == EDGE_COLOR_BLACK || *color == EDGE_COLOR_WHITE) {
		static const edge_color_t start[3] = {EDGE_COLOR_CYAN, EDGE_COLOR_MAGENTA, EDGE_COLOR_YELLOW};
		*color = start[*seed % 3];
		*seed /= 3;
		return;
	}
	int shifted = *color << (1 + (*seed & 1));
	*color = (shifted | shifted >> 3) & EDGE_COLOR_WHITE;
	*seed >>= 1;
}

/** Assigns colors to edges of the shape in accordance to the multi-channel distance field technique.
 *  May split some edges if necessary.
 *  angleThreshold specifies the maximum angle (in radians) to be considered a corner, for example 3 (~172 degrees).
 *  Values below 1/2 PI will be treated as the external angle.
 */
void edge_coloring_simple(shape_t *shape, double angleThreshold, unsigned long long seed) {
	size_t corner_count = 0;
	for (int i = 0; i < shape->contour_count; ++i) {
		for (int j = 0; j < shape->contours[i].edge_count; ++j)
			corner_count++;
	}

	int *corners = malloc(sizeof(int) * corner_count);
	int corner_index = 0;
	
	for (int i = 0; i < shape->contour_count; i++) {
		contour_t *contour = &shape->contours[i];
	}
	
	free(corners);
}

inline bool detect_clash(const float *a, const float *b, double threshold) {
	// Sort channels so that pairs (a0, b0), (a1, b1), (a2, b2) go from biggest to smallest absolute difference
	float a0 = a[0], a1 = a[1], a2 = a[2];
	float b0 = b[0], b1 = b[1], b2 = b[2];
	float tmp;
	if (fabsf(b0 - a0) < fabsf(b1 - a1)) {
		tmp = a0, a0 = a1, a1 = tmp;
		tmp = b0, b0 = b1, b1 = tmp;
	}
	if (fabsf(b1 - a1) < fabsf(b2 - a2)) {
		tmp = a1, a1 = a2, a2 = tmp;
		tmp = b1, b1 = b2, b2 = tmp;
		if (fabsf(b0 - a0) < fabsf(b1 - a1)) {
			tmp = a0, a0 = a1, a1 = tmp;
			tmp = b0, b0 = b1, b1 = tmp;
		}
	}
	return (fabsf(b1 - a1) >= threshold) &&
		!(b0 == b1 && b0 == b2) && // Ignore if other pixel has been equalized
		fabsf(a2 - 0.5f) >= fabsf(b2 - 0.5f); // Out of the pair, only flag the pixel farther from a shape edge
}

void generate_distance_field(float *pixels, int width, int height, shape_t *shape, double range, vec2 scale, vec2 translate) {
	
}

/// Resolves multi-channel signed distance field values that may cause interpolation artifacts.
void msdf_error_correction(float *pixels, int width, int height, vec2 threshold) {
	typedef struct {
		int x, y;
	} clashes_t;
	clashes_t *clashes = malloc(sizeof(clashes_t) * width * height);

	free(clashes);
}

tinymsdf_error_t tinymsdf_generate_mtsdf(float **pixels, int width, int height, FT_Face face, unicode_t unicode)
{
	// Load Freetype glyph into a shape.
	shape_t shape = {NULL, 0};
	tinymsdf_error_t error = load_glyph(&shape, face, unicode);
	if (error)
		goto error_exit;

	// Create the bitmap.
	error = create_bitmap(pixels, width, height, 4);
	if (error)
		goto error_exit;

	// Auto-frame
	vec2 scale = {1.0, 1.0};
	double avgScale = 0.5 * (scale.x + scale.y);
	bounds_t bounds = shape_get_bounds(&shape, 0, 0, 0);

	double pxRange = 2;
	vec2 translate;
	bool autoFrame = true;
	
	if (autoFrame) {
		double l = bounds.l, b = bounds.b, r = bounds.r, t = bounds.t;
		vec2 frame = {width - pxRange, height - pxRange};
		if (l >= r || b >= t)
			l = 0, b = 0, r = 1, t = 1;
		if (frame.x <= 0 || frame.y <= 0) {
			error = TINYMSDF_CANNOT_FIT_RANGE;
			goto error_exit;
		}
		vec2 dims = {r - l, t - b};
		if (dims.x * frame.y < dims.y * frame.x) {
			translate.x = 0.5 * (frame.x / frame.y * dims.y - dims.x) - l, translate.y = -b;
			avgScale = frame.y / dims.y;
			scale = (vec2){avgScale, avgScale};
		} else {
			translate.x = -l, translate.y = 0.5 * (frame.y / frame.x * dims.x - dims.y) - b;
			avgScale = frame.x / dims.x;
			scale = (vec2){avgScale, avgScale};
		}
		translate = vec2_add(translate, vec2_scale(2.0 / pxRange, scale));
	}

	double range = pxRange / min(scale.x, scale.y);
	double advance = F26DOT6_TO_DOUBLE(face->glyph->advance.x);

	// Print metrics
	if (bounds.r >= bounds.l && bounds.t >= bounds.b)
		printf("bounds = %.12g, %.12g, %.12g, %.12g\n", bounds.l, bounds.b, bounds.r, bounds.t);
	if (advance != 0)
		printf("advance = %.12g\n", advance);
	if (autoFrame) {
		printf("scale = %.12g\n", avgScale);
		printf("translate = %.12g, %.12g\n", translate.x, translate.y);
	}
	printf("range = %.12g\n", range);


	// Compute output
	double angleThreshold = 3;
	unsigned long long coloringSeed = 0;
	double edgeThreshold = TINYMSDF_DEFAULT_ERROR_CORRECTION_THRESHOLD;
	edge_coloring_simple(&shape, angleThreshold, coloringSeed);
	// TODO: No overlapping contour support.
	generate_distance_field(*pixels, width, height, &shape, range, scale, translate);
	if (edgeThreshold > 0)
		msdf_error_correction(*pixels, width, height, vec2_scale(1.0 / edgeThreshold, vec2_scale(range, scale)));
	
error_exit:
	free_shape(&shape);
	return error;
}
