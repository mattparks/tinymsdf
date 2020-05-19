/**
 * Based on the C++ implementation by Viktor Chlumský.
 * https://github.com/Chlumsky/msdfgen
 */

#ifndef TINYMSDF_H
#define TINYMSDF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FT_LibraryRec_ *FT_Library;
typedef struct FT_FaceRec_ *FT_Face;

typedef unsigned unicode_t;

/// Error code type, where 0 = success.
typedef int tinymsdf_error_t;

enum {
	TINYMSDF_SUCCESS,
	TINYMSDF_GLYPH_LOAD,
	TINYMSDF_OUTLINE_DECOMPOSE,
	TINYMSDF_INVALID_GEOMETRY,
	TINYMSDF_CANNOT_FIT_RANGE
};

#define TINYMSDF_DEFAULT_ERROR_CORRECTION_THRESHOLD 1.001

/// Generates a conventional single-channel signed distance field.
tinymsdf_error_t tinymsdf_generate_sdf(float **pixels, int width, int height, FT_Face face, unicode_t unicode);
/// Generates a single-channel signed pseudo-distance field.
tinymsdf_error_t tinymsdf_generate_pseudo_sdf(float **pixels, int width, int height, FT_Face face, unicode_t unicode);
/// Generates a multi-channel signed distance field.
tinymsdf_error_t tinymsdf_generate_msdf(float **pixels, int width, int height, FT_Face face, unicode_t unicode);
/// Generates a multi-channel signed distance field with true distance in the alpha channel.
tinymsdf_error_t tinymsdf_generate_mtsdf(float **pixels, int width, int height, FT_Face face, unicode_t unicode);

#ifdef __cplusplus
}
#endif

#endif // TINYMSDF_H