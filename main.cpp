#include <algorithm>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <tinymsdf.hpp>
#include <lodepng.h>
#include <vector>

typedef unsigned char byte;
inline byte PixelFloatToByte(float x) {
	return byte(std::clamp(256.0f * x, 0.0f, 255.0f));
}
bool SavePng(const tinymsdf::Bitmap<float, 4> &bitmap, const char *filename) {
	std::vector<byte> pixels(4 * bitmap.width * bitmap.height);
	std::vector<byte>::iterator it = pixels.begin();
	for (int y = bitmap.height - 1; y >= 0; --y)
		for (int x = 0; x < bitmap.width; ++x) {
			*it++ = PixelFloatToByte(bitmap(x, y)[0]);
			*it++ = PixelFloatToByte(bitmap(x, y)[1]);
			*it++ = PixelFloatToByte(bitmap(x, y)[2]);
			*it++ = PixelFloatToByte(bitmap(x, y)[3]);
		}
	return !lodepng::encode(filename, pixels, bitmap.width, bitmap.height, LCT_RGBA);
}

int main(int argc, const char **argv) {
#define ABORT(msg) { printf(msg); return 1; }
	const char *input = argv[0];
	const char *output = "output.png";
	const char *testRender = "test.png";
	tinymsdf::unicode_t unicode = 'p';
	int width = 32, height = 32;
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

	tinymsdf::Bitmap<float, 4> mtsdf(width, height);
	if (tinymsdf::GenerateMTSDF(mtsdf, face, unicode)) {
		FT_Done_Face(face);
		FT_Done_FreeType(library);
		ABORT("Failed to load glyph from font file");
	}
	SavePng(mtsdf, output);

	FT_Done_Face(face);
	FT_Done_FreeType(library);

	return 0;
}
