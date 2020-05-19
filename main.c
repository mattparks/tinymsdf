#include <ft2build.h>
#include FT_FREETYPE_H
#include <tinymsdf.h>

int main(int argc, const char **argv) {
#define ABORT(msg) { printf(msg); return 1; }
	
	const char *input = "D:\\Workspaces\\Mattparks\\tinymsdf\\fonts\\Roboto-Regular.ttf";
	const char *output = "output.png";
	const char *testRender = "test.png";
	unicode_t unicode = 'p';
	int width = 26, height = 26;
	int testWidth = 512, testHeight = 512;

	FT_Library library;
	if (FT_Init_FreeType(&library)) {
		ABORT("Failed to initalize Freetype");
	}
	FT_Face face;
	if (FT_New_Face(library, input, 0, &face)) {
		FT_Done_FreeType(library);
		ABORT("Failed to load font file");
	}

	float *pixels = NULL;
	if (tinymsdf_generate_mtsdf(pixels, width, height, face, unicode)) {
		FT_Done_Face(face);
		FT_Done_FreeType(library);
		ABORT("Failed to load glyph from font file");
	}

	FT_Done_Face(face);
	FT_Done_FreeType(library);
	
	return 0;
}
