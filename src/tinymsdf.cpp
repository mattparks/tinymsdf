#include "tinymsdf.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tinymsdf {
class ArithmeticException : std::runtime_error {
public:
	explicit ArithmeticException(const std::string &err) : runtime_error(err) {}
	~ArithmeticException() override = default;
};
class FtException : std::runtime_error {
public:
	explicit FtException(const std::string &err) : runtime_error(err) {}
	~FtException() override = default;
};
class MsdfException : std::runtime_error {
public:
	explicit MsdfException(const std::string &err) : runtime_error(err) {}
	~MsdfException() override = default;
};

/// Returns the middle out of three values
template <typename T>
T Median(T a, T b, T c) {
	return std::max(std::min(a, b), std::min(std::max(a, b), c));
}
/// Returns the weighted average of a and b.
template <typename T, typename S>
T Mix(T a, T b, S weight) {
	return T((S(1) - weight) * a + weight * b);
}
/// Returns 1 for positive values, -1 for negative values, and 0 for zero.
template <typename T>
int Sign(T n) {
	return (T(0) < n) - (n < T(0));
}
/// Returns 1 for non-negative values and -1 for negative values.
template <typename T>
int NonZeroSign(T n) {
	return 2 * (n > T(0)) - 1;
}

#define F26DOT6_TO_DOUBLE(x) (1/64.0*double(x))

class Vector2 {
public:
	Vector2() = default;
	Vector2(double val) :
		x(val),
		y(val) {
	}
	Vector2(double x, double y) :
		x(x),
		y(y) {
	}
	Vector2(const FT_Vector &vector) :
		x(F26DOT6_TO_DOUBLE(vector.x)),
		y(F26DOT6_TO_DOUBLE(vector.y)) {
	}

	/// Returns the vector's length.
	double Length() const {
		return std::sqrt(x * x + y * y);
	}
	/// Returns the angle of the vector in radians (atan2).
	double Direction() const {
		return std::atan2(y, x);
	}
	/// Returns the normalized vector - one that has the same direction but unit length.
	Vector2 Normalize() const {
		double len = Length();
		if (len == 0)
			throw ArithmeticException("Vector length is zero");
		return Vector2(x / len, y / len);
	}
	/// Dot product of two vectors.
	double Dot(const Vector2 &right) const {
		return x * right.x + y * right.y;
	}
	/// A special version of the cross product for 2D vectors (returns scalar value).
	double Cross(const Vector2 &right) const {
		return x * right.y - y * right.x;
	}
	/// Returns a vector with the same length that is orthogonal to this one.
	Vector2 Orthonormal() const {
		double len = Length();
		if (len == 0)
			throw ArithmeticException("Vector length is zero");
		return Vector2(y / len, -x / len);
	}

	bool operator==(const Vector2 &right) const {
		return x == right.x && y == right.y;
	}
	bool operator!=(const Vector2 &right) const {
		return x != right.x || y != right.y;
	}
	Vector2 operator-() const {
		return Vector2(-x, -y);
	}
	Vector2 operator+(const Vector2 &right) const {
		return Vector2(x + right.x, y + right.y);
	}
	Vector2 operator-(const Vector2 &right) const {
		return Vector2(x - right.x, y - right.y);
	}
	Vector2 operator*(const Vector2 &right) const {
		return Vector2(x * right.x, y * right.y);
	}
	Vector2 operator/(const Vector2 &right) const {
		return Vector2(x / right.x, y / right.y);
	}
	Vector2 operator*(double right) const {
		return Vector2(x * right, y * right);
	}
	Vector2 operator/(double right) const {
		return Vector2(x / right, y / right);
	}
	friend Vector2 operator*(double left, const Vector2 &right) {
		return Vector2(left * right.x, left * right.y);
	}
	friend Vector2 operator/(double left, const Vector2 &right) {
		return Vector2(left / right.x, left / right.y);
	}
	
	Vector2 &operator+=(const Vector2 &right) {
		x += right.x, y += right.y;
		return *this;
	}
	Vector2 &operator-=(const Vector2 &right) {
		x -= right.x, y -= right.y;
		return *this;
	}
	Vector2 &operator*=(const Vector2 &right) {
		x *= right.x, y *= right.y;
		return *this;
	}
	Vector2 &operator/=(const Vector2 &right) {
		x /= right.x, y /= right.y;
		return *this;
	}
	Vector2 &operator*=(double right) {
		x *= right, y *= right;
		return *this;
	}
	Vector2 &operator/=(double right) {
		x /= right, y /= right;
		return *this;
	}

	double x = 0.0, y = 0.0;
};

constexpr double InfinateDistance = -1e240;
class SignedDistance {
public:
	SignedDistance() = default;
	SignedDistance(double dist, double d) : distance(dist), dot(d) {}

	friend bool operator<(const SignedDistance &left, const SignedDistance &right) {
		return std::fabs(left.distance) < std::fabs(right.distance) || (std::fabs(left.distance) == std::fabs(right.distance) && left.dot < right.dot);
	}

	double distance = InfinateDistance;
	double dot = 1;
};

class Bounds {
public:
	void PointBounds(const Vector2 &p) {
		if (p.x < l) l = p.x;
		if (p.y < b) b = p.y;
		if (p.x > r) r = p.x;
		if (p.y > t) t = p.y;
	}
	
	double l, b, r, t;
};

/// Edge color specifies which color channels an edge belongs to.
enum EdgeColor {
	BLACK = 0,
	RED = 1,
	GREEN = 2,
	YELLOW = 3,
	BLUE = 4,
	MAGENTA = 5,
	CYAN = 6,
	WHITE = 7
};

/// An abstract edge segment.
class EdgeSegment {
public:
	explicit EdgeSegment(EdgeColor edgeColor = WHITE) :
		color(edgeColor) {
	}
	virtual ~EdgeSegment() = default;

	/// Returns the point on the edge specified by the parameter (between 0 and 1).
	virtual Vector2 Point(double param) const = 0;
	/// Returns the direction the edge has at the point specified by the parameter.
	virtual Vector2 Direction(double param) const = 0;
	/// Returns the minimum signed distance between origin and the edge.
	virtual SignedDistance MinSignedDistance(Vector2 origin, double &param) const = 0;
	/// Converts a previously retrieved signed distance from origin to pseudo-distance.
	virtual void DistanceToPseudoDistance(SignedDistance &distance, Vector2 origin, double param) const {
		if (param < 0) {
			Vector2 dir = Direction(0).Normalize();
			Vector2 aq = origin - Point(0);
			double ts = aq.Dot(dir);
			if (ts < 0) {
				double pseudoDistance = aq.Cross(dir);
				if (std::fabs(pseudoDistance) <= std::fabs(distance.distance)) {
					distance.distance = pseudoDistance;
					distance.dot = 0;
				}
			}
		} else if (param > 1) {
			Vector2 dir = Direction(1).Normalize();
			Vector2 bq = origin - Point(1);
			double ts = bq.Dot(dir);
			if (ts > 0) {
				double pseudoDistance = bq.Cross(dir);
				if (std::fabs(pseudoDistance) <= std::fabs(distance.distance)) {
					distance.distance = pseudoDistance;
					distance.dot = 0;
				}
			}
		}
	}
	/// Outputs a list of (at most three) intersections (their X coordinates) with an infinite horizontal scanline at y and returns how many there are.
	virtual int ScanlineIntersections(double x[3], int dy[3], double y) const = 0;
	/// Adjusts the bounding box to fit the edge segment.
	virtual void Bound(Bounds &bounds) const = 0;

	/// Moves the start point of the edge segment.
	virtual void MoveStartPoint(Vector2 to) = 0;
	/// Moves the end point of the edge segment.
	virtual void MoveEndPoint(Vector2 to) = 0;
	/// Splits the edge segments into thirds which together represent the original edge.
	virtual void SplitInThirds(std::unique_ptr<EdgeSegment> &part1, std::unique_ptr<EdgeSegment> &part2, std::unique_ptr<EdgeSegment> &part3) const = 0;

	EdgeColor color;
};

constexpr double TOO_LARGE_RATIO = 1e12;
// Parameters for iterative search of closest point on a cubic Bezier curve. Increase for higher precision.
constexpr int MSDFGEN_CUBIC_SEARCH_STARTS = 4;
constexpr int MSDFGEN_CUBIC_SEARCH_STEPS = 4;

int SolveQuadratic(double x[2], double a, double b, double c) {
	// a = 0 -> linear equation
	if (a == 0 || std::fabs(b) + std::fabs(c) > TOO_LARGE_RATIO * std::fabs(a)) {
		// a, b = 0 -> no solution
		if (b == 0 || std::fabs(c) > TOO_LARGE_RATIO * std::fabs(b)) {
			if (c == 0)
				return -1; // 0 = 0
			return 0;
		}
		x[0] = -c / b;
		return 1;
	}
	double dscr = b * b - 4 * a * c;
	if (dscr > 0) {
		dscr = std::sqrt(dscr);
		x[0] = (-b + dscr) / (2 * a);
		x[1] = (-b - dscr) / (2 * a);
		return 2;
	} else if (dscr == 0) {
		x[0] = -b / (2 * a);
		return 1;
	} else
		return 0;
}

int SolveCubicNormed(double x[3], double a, double b, double c) {
	double a2 = a * a;
	double q = (a2 - 3 * b) / 9;
	double r = (a * (2 * a2 - 9 * b) + 27 * c) / 54;
	double r2 = r * r;
	double q3 = q * q * q;
	double A, B;
	if (r2 < q3) {
		double t = r / std::sqrt(q3);
		if (t < -1) t = -1;
		if (t > 1) t = 1;
		t = std::acos(t);
		a /= 3; q = -2 * std::sqrt(q);
		x[0] = q * std::cos(t / 3) - a;
		x[1] = q * std::cos((t + 2 * M_PI) / 3) - a;
		x[2] = q * std::cos((t - 2 * M_PI) / 3) - a;
		return 3;
	} else {
		A = -std::pow(std::fabs(r) + std::sqrt(r2 - q3), 1 / 3.);
		if (r < 0) A = -A;
		B = A == 0 ? 0 : q / A;
		a /= 3;
		x[0] = (A + B) - a;
		x[1] = -0.5 * (A + B) - a;
		x[2] = 0.5 * std::sqrt(3.) * (A - B);
		if (std::fabs(x[2]) < 1e-14)
			return 2;
		return 1;
	}
}

int SolveCubic(double x[3], double a, double b, double c, double d) {
	if (a != 0) {
		double bn = b / a, cn = c / a, dn = d / a;
		// Check that a isn't "almost zero"
		if (std::fabs(bn) < TOO_LARGE_RATIO && std::fabs(cn) < TOO_LARGE_RATIO && std::fabs(dn) < TOO_LARGE_RATIO)
			return SolveCubicNormed(x, bn, cn, dn);
	}
	return SolveQuadratic(x, b, c, d);
}

/// A line segment.
class LinearSegment : public EdgeSegment {
public:
	LinearSegment(Vector2 p0, Vector2 p1, EdgeColor edgeColor = WHITE) :
		EdgeSegment(edgeColor) {
		p[0] = p0;
		p[1] = p1;
	}

	Vector2 Point(double param) const override {
		return Mix(p[0], p[1], param);
	}
	Vector2 Direction(double param) const override {
		return p[1] - p[0];
	}
	SignedDistance MinSignedDistance(Vector2 origin, double &param) const override {
		Vector2 aq = origin - p[0];
		Vector2 ab = p[1] - p[0];
		param = aq.Dot(ab) / ab.Dot(ab);
		Vector2 eq = p[param > .5] - origin;
		double endpointDistance = eq.Length();
		if (param > 0 && param < 1) {
			double orthoDistance = ab.Orthonormal().Dot(aq);
			if (std::fabs(orthoDistance) < endpointDistance)
				return SignedDistance(orthoDistance, 0);
		}
		return SignedDistance(NonZeroSign(aq.Cross(ab)) * endpointDistance, std::fabs(ab.Normalize().Dot(eq.Normalize())));
	}
	int ScanlineIntersections(double x[3], int dy[3], double y) const override {
		if ((y >= p[0].y && y < p[1].y) || (y >= p[1].y && y < p[0].y)) {
			double param = (y - p[0].y) / (p[1].y - p[0].y);
			x[0] = Mix(p[0].x, p[1].x, param);
			dy[0] = Sign(p[1].y - p[0].y);
			return 1;
		}
		return 0;
	}
	void Bound(Bounds &bounds) const override {
		bounds.PointBounds(p[0]);
		bounds.PointBounds(p[1]);
	}

	void MoveStartPoint(Vector2 to) override {
	}
	void MoveEndPoint(Vector2 to) override {
	}
	void SplitInThirds(std::unique_ptr<EdgeSegment> &part1, std::unique_ptr<EdgeSegment> &part2, std::unique_ptr<EdgeSegment> &part3) const override {
		part1 = std::make_unique<LinearSegment>(p[0], Point(1 / 3.0), color);
		part2 = std::make_unique<LinearSegment>(Point(1 / 3.0), Point(2 / 3.0), color);
		part3 = std::make_unique<LinearSegment>(Point(2 / 3.0), p[1], color);
	}

	Vector2 p[2];
};

/// A quadratic Bezier curve.
class QuadraticSegment : public EdgeSegment {
public:
	QuadraticSegment(Vector2 p0, Vector2 p1, Vector2 p2, EdgeColor edgeColor = WHITE) :
		EdgeSegment(edgeColor) {
		if (p1 == p0 || p1 == p2)
			p1 = 0.5 * (p0 + p2);
		p[0] = p0;
		p[1] = p1;
		p[2] = p2;
	}

	Vector2 Point(double param) const override {
		return Mix(Mix(p[0], p[1], param), Mix(p[1], p[2], param), param);
	}
	Vector2 Direction(double param) const override {
		Vector2 tangent = Mix(p[1] - p[0], p[2] - p[1], param);
		if (!tangent.x && !tangent.y)
			return p[2] - p[0];
		return tangent;
	}
	SignedDistance MinSignedDistance(Vector2 origin, double &param) const override {
		Vector2 qa = p[0] - origin;
		Vector2 ab = p[1] - p[0];
		Vector2 br = p[2] - p[1] - ab;
		double a = br.Dot(br);
		double b = 3 * ab.Dot(br);
		double c = 2 * ab.Dot(ab) + qa.Dot(br);
		double d = qa.Dot(ab);
		double t[3];
		int solutions = SolveCubic(t, a, b, c, d);

		Vector2 epDir = Direction(0);
		double minDistance = NonZeroSign(epDir.Cross(qa)) * qa.Length(); // distance from A
		param = -qa.Dot(epDir) / epDir.Dot(epDir);
		{
			epDir = Direction(1);
			double distance = NonZeroSign(epDir.Cross(p[2] - origin)) * (p[2] - origin).Length(); // distance from B
			if (std::fabs(distance) < std::fabs(minDistance)) {
				minDistance = distance;
				param = (origin - p[1]).Dot(epDir) / epDir.Dot(epDir);
			}
		}
		for (int i = 0; i < solutions; ++i) {
			if (t[i] > 0 && t[i] < 1) {
				Vector2 qe = p[0] + 2 * t[i] * ab + t[i] * t[i] * br - origin;
				double distance = NonZeroSign((p[2] - p[0]).Cross(qe)) * qe.Length();
				if (std::fabs(distance) <= std::fabs(minDistance)) {
					minDistance = distance;
					param = t[i];
				}
			}
		}

		if (param >= 0 && param <= 1)
			return SignedDistance(minDistance, 0);
		if (param < .5)
			return SignedDistance(minDistance, std::fabs(Direction(0).Normalize().Dot(qa.Normalize())));
		else
			return SignedDistance(minDistance, std::fabs(Direction(1).Normalize().Dot((p[2] - origin).Normalize())));
	}
	int ScanlineIntersections(double x[3], int dy[3], double y) const override {
		int total = 0;
		int nextDY = y > p[0].y ? 1 : -1;
		x[total] = p[0].x;
		if (p[0].y == y) {
			if (p[0].y < p[1].y || (p[0].y == p[1].y && p[0].y < p[2].y))
				dy[total++] = 1;
			else
				nextDY = 1;
		}
		{
			Vector2 ab = p[1] - p[0];
			Vector2 br = p[2] - p[1] - ab;
			double t[2];
			int solutions = SolveQuadratic(t, br.y, 2 * ab.y, p[0].y - y);
			// Sort solutions
			double tmp;
			if (solutions >= 2 && t[0] > t[1])
				tmp = t[0], t[0] = t[1], t[1] = tmp;
			for (int i = 0; i < solutions && total < 2; ++i) {
				if (t[i] >= 0 && t[i] <= 1) {
					x[total] = p[0].x + 2 * t[i] * ab.x + t[i] * t[i] * br.x;
					if (nextDY * (ab.y + t[i] * br.y) >= 0) {
						dy[total++] = nextDY;
						nextDY = -nextDY;
					}
				}
			}
		}
		if (p[2].y == y) {
			if (nextDY > 0 && total > 0) {
				--total;
				nextDY = -1;
			}
			if ((p[2].y < p[1].y || (p[2].y == p[1].y && p[2].y < p[0].y)) && total < 2) {
				x[total] = p[2].x;
				if (nextDY < 0) {
					dy[total++] = -1;
					nextDY = 1;
				}
			}
		}
		if (nextDY != (y >= p[2].y ? 1 : -1)) {
			if (total > 0)
				--total;
			else {
				if (fabs(p[2].y - y) < fabs(p[0].y - y))
					x[total] = p[2].x;
				dy[total++] = nextDY;
			}
		}
		return total;
	}
	void Bound(Bounds &bounds) const override {
		bounds.PointBounds(p[0]);
		bounds.PointBounds(p[2]);
		Vector2 bot = (p[1] - p[0]) - (p[2] - p[1]);
		if (bot.x) {
			double param = (p[1].x - p[0].x) / bot.x;
			if (param > 0 && param < 1)
				bounds.PointBounds(Point(param));
		}
		if (bot.y) {
			double param = (p[1].y - p[0].y) / bot.y;
			if (param > 0 && param < 1)
				bounds.PointBounds(Point(param));
		}
	}

	void MoveStartPoint(Vector2 to) override {
	}
	void MoveEndPoint(Vector2 to) override {
	}
	void SplitInThirds(std::unique_ptr<EdgeSegment> &part1, std::unique_ptr<EdgeSegment> &part2, std::unique_ptr<EdgeSegment> &part3) const override {
		part1 = std::make_unique<QuadraticSegment>(p[0], Mix(p[0], p[1], 1 / 3.0), Point(1 / 3.0), color);
		part2 = std::make_unique<QuadraticSegment>(Point(1 / 3.0), Mix(Mix(p[0], p[1], 5 / 9.0), Mix(p[1], p[2], 4 / 9.0), 0.5), Point(2 / 3.0), color);
		part3 = std::make_unique<QuadraticSegment>(Point(2 / 3.0), Mix(p[1], p[2], 2 / 3.0), p[2], color);
	}

	Vector2 p[3];
};

/// A cubic Bezier curve.
class CubicSegment : public EdgeSegment {
public:
	CubicSegment(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, EdgeColor edgeColor = WHITE) :
		EdgeSegment(edgeColor) {
		p[0] = p0;
		p[1] = p1;
		p[2] = p2;
		p[3] = p3;
	}

	Vector2 Point(double param) const override {
		Vector2 p12 = Mix(p[1], p[2], param);
		return Mix(Mix(Mix(p[0], p[1], param), p12, param), Mix(p12, Mix(p[2], p[3], param), param), param);
	}
	Vector2 Direction(double param) const override {
		Vector2 tangent = Mix(Mix(p[1] - p[0], p[2] - p[1], param), Mix(p[2] - p[1], p[3] - p[2], param), param);
		if (!tangent.x && !tangent.y) {
			if (param == 0) return p[2] - p[0];
			if (param == 1) return p[3] - p[1];
		}
		return tangent;
	}
	SignedDistance MinSignedDistance(Vector2 origin, double &param) const override {
		Vector2 qa = p[0] - origin;
		Vector2 ab = p[1] - p[0];
		Vector2 br = p[2] - p[1] - ab;
		Vector2 as = (p[3] - p[2]) - (p[2] - p[1]) - br;

		Vector2 epDir = Direction(0);
		double minDistance = NonZeroSign(epDir.Cross(qa)) * qa.Length(); // distance from A
		param = -qa.Dot(epDir) / epDir.Dot(epDir);
		{
			epDir = Direction(1);
			double distance = NonZeroSign(epDir.Cross(p[3] - origin)) * (p[3] - origin).Length(); // distance from B
			if (fabs(distance) < fabs(minDistance)) {
				minDistance = distance;
				param = (epDir - (p[3] - origin)).Dot(epDir) / epDir.Dot(epDir);
			}
		}
		// Iterative minimum distance search
		for (int i = 0; i <= MSDFGEN_CUBIC_SEARCH_STARTS; ++i) {
			double t = (double)i / MSDFGEN_CUBIC_SEARCH_STARTS;
			for (int step = 0;; ++step) {
				Vector2 qe = p[0] + 3 * t * ab + 3 * t * t * br + t * t * t * as - origin; // do not simplify with qa !!!
				double distance = NonZeroSign(Direction(t).Cross(qe)) * qe.Length();
				if (fabs(distance) < fabs(minDistance)) {
					minDistance = distance;
					param = t;
				}
				if (step == MSDFGEN_CUBIC_SEARCH_STEPS)
					break;
				// Improve t
				Vector2 d1 = 3 * as * t * t + 6 * br * t + 3 * ab;
				Vector2 d2 = 6 * as * t + 6 * br;
				t -= qe.Dot(d1) / (d1.Dot(d1) + qe.Dot(d2));
				if (t < 0 || t > 1)
					break;
			}
		}

		if (param >= 0 && param <= 1)
			return SignedDistance(minDistance, 0);
		if (param < .5)
			return SignedDistance(minDistance, std::fabs(Direction(0).Normalize().Dot(qa.Normalize())));
		else
			return SignedDistance(minDistance, std::fabs(Direction(1).Normalize().Dot((p[3] - origin).Normalize())));
	}
	int ScanlineIntersections(double x[3], int dy[3], double y) const override {
		int total = 0;
		int nextDY = y > p[0].y ? 1 : -1;
		x[total] = p[0].x;
		if (p[0].y == y) {
			if (p[0].y < p[1].y || (p[0].y == p[1].y && (p[0].y < p[2].y || (p[0].y == p[2].y && p[0].y < p[3].y))))
				dy[total++] = 1;
			else
				nextDY = 1;
		}
		{
			Vector2 ab = p[1] - p[0];
			Vector2 br = p[2] - p[1] - ab;
			Vector2 as = (p[3] - p[2]) - (p[2] - p[1]) - br;
			double t[3];
			int solutions = SolveCubic(t, as.y, 3 * br.y, 3 * ab.y, p[0].y - y);
			// Sort solutions
			double tmp;
			if (solutions >= 2) {
				if (t[0] > t[1])
					tmp = t[0], t[0] = t[1], t[1] = tmp;
				if (solutions >= 3 && t[1] > t[2]) {
					tmp = t[1], t[1] = t[2], t[2] = tmp;
					if (t[0] > t[1])
						tmp = t[0], t[0] = t[1], t[1] = tmp;
				}
			}
			for (int i = 0; i < solutions && total < 3; ++i) {
				if (t[i] >= 0 && t[i] <= 1) {
					x[total] = p[0].x + 3 * t[i] * ab.x + 3 * t[i] * t[i] * br.x + t[i] * t[i] * t[i] * as.x;
					if (nextDY * (ab.y + 2 * t[i] * br.y + t[i] * t[i] * as.y) >= 0) {
						dy[total++] = nextDY;
						nextDY = -nextDY;
					}
				}
			}
		}
		if (p[3].y == y) {
			if (nextDY > 0 && total > 0) {
				--total;
				nextDY = -1;
			}
			if ((p[3].y < p[2].y || (p[3].y == p[2].y && (p[3].y < p[1].y || (p[3].y == p[1].y && p[3].y < p[0].y)))) && total < 3) {
				x[total] = p[3].x;
				if (nextDY < 0) {
					dy[total++] = -1;
					nextDY = 1;
				}
			}
		}
		if (nextDY != (y >= p[3].y ? 1 : -1)) {
			if (total > 0)
				--total;
			else {
				if (std::fabs(p[3].y - y) < std::fabs(p[0].y - y))
					x[total] = p[3].x;
				dy[total++] = nextDY;
			}
		}
		return total;
	}
	void Bound(Bounds &bounds) const override {
		bounds.PointBounds(p[0]);
		bounds.PointBounds(p[3]);
		Vector2 a0 = p[1] - p[0];
		Vector2 a1 = 2 * (p[2] - p[1] - a0);
		Vector2 a2 = p[3] - 3 * p[2] + 3 * p[1] - p[0];
		double params[2];
		int solutions;
		solutions = SolveQuadratic(params, a2.x, a1.x, a0.x);
		for (int i = 0; i < solutions; ++i)
			if (params[i] > 0 && params[i] < 1)
				bounds.PointBounds(Point(params[i]));
		solutions = SolveQuadratic(params, a2.y, a1.y, a0.y);
		for (int i = 0; i < solutions; ++i)
			if (params[i] > 0 && params[i] < 1)
				bounds.PointBounds(Point(params[i]));
	}

	void MoveStartPoint(Vector2 to) override {
	}
	void MoveEndPoint(Vector2 to) override {
	}
	void SplitInThirds(std::unique_ptr<EdgeSegment> &part1, std::unique_ptr<EdgeSegment> &part2, std::unique_ptr<EdgeSegment> &part3) const override {
		part1 = std::make_unique<CubicSegment>(p[0], p[0] == p[1] ? p[0] : Mix(p[0], p[1], 1 / 3.0), Mix(Mix(p[0], p[1], 1 / 3.0), Mix(p[1], p[2], 1 / 3.0), 1 / 3.0), Point(1 / 3.0), color);
		part2 = std::make_unique<CubicSegment>(Point(1 / 3.0),
			Mix(Mix(Mix(p[0], p[1], 1 / 3.0), Mix(p[1], p[2], 1 / 3.0), 1 / 3.0), Mix(Mix(p[1], p[2], 1 / 3.0), Mix(p[2], p[3], 1 / 3.0), 1 / 3.0), 2 / 3.0),
			Mix(Mix(Mix(p[0], p[1], 2 / 3.0), Mix(p[1], p[2], 2 / 3.0), 2 / 3.0), Mix(Mix(p[1], p[2], 2 / 3.0), Mix(p[2], p[3], 2 / 3.0), 2 / 3.0), 1 / 3.0),
			Point(2 / 3.0), color);
		part3 = std::make_unique<CubicSegment>(Point(2 / 3.0), Mix(Mix(p[1], p[2], 2 / 3.0), Mix(p[2], p[3], 2 / 3.0), 2 / 3.0), p[2] == p[3] ? p[3] : Mix(p[2], p[3], 2 / 3.0), p[3], color);
	}

	Vector2 p[4];
};

double Shoelace(const Vector2 &a, const Vector2 &b) {
	return (b.x - a.x) * (a.y + b.y);
}

/// A single closed contour of a shape.
class Contour {
public:
	/// Adds an edge to the contour.
	template<typename T, typename... Args,
		typename = std::enable_if_t<std::is_convertible_v<T *, EdgeSegment *>>>
		EdgeSegment &AddEdge(Args ... args) {
		return *edges.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
	}
	/// Adjusts the bounding box to fit the contour.
	void Bound(Bounds &bounds) const {
		for (const auto &edge : edges)
			edge->Bound(bounds);
	}
	/// Adjusts the bounding box to fit the contour border's mitered corners.
	void BoundMiters(Bounds &bounds, double border, double miterLimit, int polarity) const {
		if (edges.empty())
			return;
		Vector2 prevDir = edges.back()->Direction(1).Normalize();
		for (const auto &edge : edges) {
			Vector2 dir = -edge->Direction(0).Normalize();
			if (polarity * prevDir.Cross(dir) >= 0) {
				double miterLength = miterLimit;
				double q = 0.5 * (1.0 - prevDir.Dot(dir));
				if (q > 0)
					miterLength = std::min(1 / sqrt(q), miterLimit);
				auto miter = edge->Point(0) + border * miterLength * (prevDir + dir).Normalize();
				bounds.PointBounds(miter);
			}
			prevDir = edge->Direction(1).Normalize();
		}
	}
	/// Computes the winding of the contour. Returns 1 if positive, -1 if negative.
	int Winding() const {
		if (edges.empty())
			return 0;
		double total = 0;
		if (edges.size() == 1) {
			Vector2 a = edges[0]->Point(0), b = edges[0]->Point(1 / 3.), c = edges[0]->Point(2 / 3.);
			total += Shoelace(a, b);
			total += Shoelace(b, c);
			total += Shoelace(c, a);
		} else if (edges.size() == 2) {
			Vector2 a = edges[0]->Point(0), b = edges[0]->Point(.5), c = edges[1]->Point(0), d = edges[1]->Point(.5);
			total += Shoelace(a, b);
			total += Shoelace(b, c);
			total += Shoelace(c, d);
			total += Shoelace(d, a);
		} else {
			Vector2 prev = edges.back()->Point(0);
			for (const auto &edge : edges) {
				Vector2 cur = edge->Point(0);
				total += Shoelace(prev, cur);
				prev = cur;
			}
		}
		return Sign(total);
	}

	/// The sequence of edges that make up the contour.
	std::vector<std::unique_ptr<EdgeSegment>> edges;
};

/// Vector shape representation.
class Shape {
public:
	/// Adds a contour.
	Contour &AddContour() {
		return contours.emplace_back();
	}
	/// Performs basic checks to determine if the object represents a valid shape.
	bool Validate() const {
		for (const auto &contour : contours) {
			if (!contour.edges.empty()) {
				auto corner = contour.edges.back()->Point(1);
				for (const auto &edge : contour.edges) {
					if (!edge)
						return false;
					if (edge->Point(0) != corner)
						return false;
					corner = edge->Point(1);
				}
			}
		}
		return true;
	}
	/// Normalizes the shape geometry for distance field generation.
	void Normalize() {
		for (auto &contour : contours) {
			if (contour.edges.size() == 1) {
				std::unique_ptr<EdgeSegment> parts[3];
				contour.edges[0]->SplitInThirds(parts[0], parts[1], parts[2]);
				contour.edges.clear();
				contour.edges.emplace_back(std::move(parts[0]));
				contour.edges.emplace_back(std::move(parts[1]));
				contour.edges.emplace_back(std::move(parts[2]));
			}
		}
	}
	/// Adjusts the bounding box to fit the shape.
	void Bound(Bounds &bounds) const {
		for (auto &contour : contours)
			contour.Bound(bounds);
	}
	/// Adjusts the bounding box to fit the shape border's mitered corners.
	void BoundMiters(Bounds &bounds, double border, double miterLimit, int polarity) const {
		for (auto &contour : contours)
			contour.BoundMiters(bounds, border, miterLimit, polarity);
	}
	/// Computes the minimum bounding box that fits the shape, optionally with a (mitered) border.
	Bounds GetBounds(double border = 0, double miterLimit = 0, int polarity = 0) const {
		static constexpr double LARGE_VALUE = 1e240;
		Bounds bounds = {+LARGE_VALUE, +LARGE_VALUE, -LARGE_VALUE, -LARGE_VALUE};
		Bound(bounds);
		if (border > 0) {
			bounds.l -= border, bounds.b -= border;
			bounds.r += border, bounds.t += border;
			if (miterLimit > 0)
				BoundMiters(bounds, border, miterLimit, polarity);
		}
		return bounds;
	}
	/// Returns the total number of edge segments
	int EdgeCount() const {
		int total = 0;
		for (auto &contour : contours)
			total += contour.edges.size();
		return total;
	}

	/// The list of contours the shape consists of.
	std::vector<Contour> contours;
	/// Specifies whether the shape uses bottom-to-top (false) or top-to-bottom (true) Y coordinates.
	bool inverseYAxis = false;
};

Shape LoadGlyph(FT_Face face, unicode_t unicode) {
	FT_Error error = FT_Load_Char(face, unicode, FT_LOAD_NO_SCALE);
	if (error)
		throw FtException("Unable to load character");

	Shape shape;

	struct FtContext {
		Shape *shape;
		Contour *contour = nullptr;
		Vector2 position;
	} context = {&shape};

	FT_Outline_Funcs ftFunctions = {};
	ftFunctions.move_to = [](const FT_Vector *to, void *user) {
		auto context = reinterpret_cast<FtContext *>(user);
		if (!(context->contour && context->contour->edges.empty()))
			context->contour = &context->shape->AddContour();
		context->position = *to;
		return 0;
	};
	ftFunctions.line_to = [](const FT_Vector *to, void *user) {
		auto context = reinterpret_cast<FtContext *>(user);
		Vector2 endpoint = *to;
		if (endpoint != context->position) {
			context->contour->AddEdge<LinearSegment>(context->position, endpoint);
			context->position = endpoint;
		}
		return 0;
	};
	ftFunctions.conic_to = [](const FT_Vector *control, const FT_Vector *to, void *user) {
		auto context = reinterpret_cast<FtContext *>(user);
		context->contour->AddEdge<QuadraticSegment>(context->position, *control, *to);
		context->position = *to;
		return 0;
	};
	ftFunctions.cubic_to = [](const FT_Vector *control1, const FT_Vector *control2, const FT_Vector *to, void *user) {
		auto context = reinterpret_cast<FtContext *>(user);
		context->contour->AddEdge<CubicSegment>(context->position, *control1, *control2, *to);
		context->position = *to;
		return 0;
	};

	error = FT_Outline_Decompose(&face->glyph->outline, &ftFunctions, &context);
	if (error)
		throw FtException("Unable to decompose glyph");

	if (!shape.contours.empty() && shape.contours.back().edges.empty())
		shape.contours.pop_back();
	return shape;
}

void AutoFrame(const Shape &shape, int width, int height, double pxRange, Vector2 &translate, Vector2 &scale) {
	auto bounds = shape.GetBounds();

	double l = bounds.l, b = bounds.b, r = bounds.r, t = bounds.t;
	Vector2 frame(width - pxRange, height - pxRange);
	if (l >= r || b >= t)
		l = 0, b = 0, r = 1, t = 1;
	if (frame.x <= 0 || frame.y <= 0)
		throw MsdfException("Cannot fit the specified pixel range");
	Vector2 dims(r - l, t - b);
	if (dims.x *frame.y < dims.y *frame.x) {
		translate = Vector2(0.5 * (frame.x / frame.y * dims.y - dims.x) - l, -b);
		scale = frame.y / dims.y;
	} else {
		translate = Vector2(-l, 0.5 * (frame.y / frame.x * dims.x - dims.y) - b);
		scale = frame.x / dims.x;
	}
	translate += 0.5 * pxRange / scale;
}

bool IsCorner(const Vector2 &aDir, const Vector2 &bDir, double crossThreshold) {
	return aDir.Dot(bDir) <= 0 || std::fabs(aDir.Cross(bDir)) > crossThreshold;
}
void SwitchColor(EdgeColor &color, unsigned long long &seed, EdgeColor banned = BLACK) {
	EdgeColor combined = EdgeColor(color & banned);
	if (combined == RED || combined == GREEN || combined == BLUE) {
		color = EdgeColor(combined ^ WHITE);
		return;
	}
	if (color == BLACK || color == WHITE) {
		static const EdgeColor start[3] = {CYAN, MAGENTA, YELLOW};
		color = start[seed % 3];
		seed /= 3;
		return;
	}
	int shifted = color << (1 + (seed & 1));
	color = EdgeColor((shifted | shifted >> 3) & WHITE);
	seed >>= 1;
}
void EdgeColoringSimple(Shape &shape, double angleThreshold, unsigned long long seed) {
	double crossThreshold = sin(angleThreshold);
	std::vector<int> corners;
	for (auto &contour : shape.contours) {
		// Identify corners
		corners.clear();
		if (!contour.edges.empty()) {
			Vector2 prevDirection = contour.edges.back()->Direction(1);
			int index = 0;
			for (const auto &edge : contour.edges) {
				if (IsCorner(prevDirection.Normalize(), edge->Direction(0).Normalize(), crossThreshold))
					corners.emplace_back(index);
				prevDirection = edge->Direction(1);
				++index;
			}
		}

		// Smooth contour
		if (corners.empty()) {
			for (const auto &edge : contour.edges)
				edge->color = WHITE;
		} else if (corners.size() == 1) { // "Teardrop" case
			EdgeColor colors[3] = {WHITE, WHITE};
			SwitchColor(colors[0], seed);
			SwitchColor(colors[2] = colors[0], seed);
			int corner = corners[0];
			if (contour.edges.size() >= 3) {
				int m = (int)contour.edges.size();
				for (int i = 0; i < m; ++i)
					contour.edges[(corner + i) % m]->color = (colors + 1)[int(3 + 2.875 * i / (m - 1) - 1.4375 + .5) - 3];
			} else if (contour.edges.size() >= 1) {
				// Less than three edge segments for three colors => edges must be split
				std::unique_ptr<EdgeSegment> parts[7];
				contour.edges[0]->SplitInThirds(parts[0 + 3 * corner], parts[1 + 3 * corner], parts[2 + 3 * corner]);
				if (contour.edges.size() >= 2) {
					contour.edges[1]->SplitInThirds(parts[3 - 3 * corner], parts[4 - 3 * corner], parts[5 - 3 * corner]);
					parts[0]->color = parts[1]->color = colors[0];
					parts[2]->color = parts[3]->color = colors[1];
					parts[4]->color = parts[5]->color = colors[2];
				} else {
					parts[0]->color = colors[0];
					parts[1]->color = colors[1];
					parts[2]->color = colors[2];
				}
				contour.edges.clear();
				for (auto &&part : parts)
					contour.edges.emplace_back(std::move(part));
			}
		} else { // Multiple corners
			int cornerCount = (int)corners.size();
			int spline = 0;
			int start = corners[0];
			int m = (int)contour.edges.size();
			EdgeColor color = WHITE;
			SwitchColor(color, seed);
			EdgeColor initialColor = color;
			for (int i = 0; i < m; ++i) {
				int index = (start + i) % m;
				if (spline + 1 < cornerCount && corners[spline + 1] == index) {
					++spline;
					SwitchColor(color, seed, EdgeColor((spline == cornerCount - 1) * (int)initialColor));
				}
				contour.edges[index]->color = color;
			}
		}
	}
}

constexpr double DISTANCE_DELTA_FACTOR = 1.001;

struct MultiDistance {
	double r, g, b;
};
struct MultiAndTrueDistance : MultiDistance {
	double a;
};

static void InitDistance(double &distance) {
	distance = InfinateDistance;
}

static void InitDistance(MultiDistance &distance) {
	distance.r = InfinateDistance;
	distance.g = InfinateDistance;
	distance.b = InfinateDistance;
}

static double ResolveDistance(double distance) {
	return distance;
}

static double ResolveDistance(const MultiDistance &distance) {
	return Median(distance.r, distance.g, distance.b);
}

/// Selects the nearest edge by its true distance.
class TrueDistanceSelector {
public:
	using DistanceType = double;

	struct EdgeCache {
		Vector2 point;
		double absDistance = 0.0;
	};

	void Reset(const Vector2 &p) {
		double delta = DISTANCE_DELTA_FACTOR * (p - this->p).Length();
		minDistance.distance += NonZeroSign(minDistance.distance) * delta;
		this->p = p;
	}
	void AddEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge) {
		double delta = DISTANCE_DELTA_FACTOR * (p - cache.point).Length();
		if (cache.absDistance - delta <= std::fabs(minDistance.distance)) {
			double dummy;
			SignedDistance distance = edge->MinSignedDistance(p, dummy);
			if (distance < minDistance)
				minDistance = distance;
			cache.point = p;
			cache.absDistance = std::fabs(distance.distance);
		}
	}
	void Merge(const TrueDistanceSelector &other) {
		if (other.minDistance < minDistance)
			minDistance = other.minDistance;
	}
	DistanceType Distance() const {
		return minDistance.distance;
	}

private:
	Vector2 p;
	SignedDistance minDistance;
};

class PseudoDistanceSelectorBase {
public:
	struct EdgeCache {
		Vector2 point;
		double absDistance = 0.0;
		double edgeDomainDistance = 0.0;
		double pseudoDistance = 0.0;
	};

	static double EdgeDomainDistance(const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge, const Vector2 &p, double param) {
		if (param < 0) {
			Vector2 prevEdgeDir = -prevEdge->Direction(1).Normalize();
			Vector2 edgeDir = edge->Direction(0).Normalize();
			Vector2 pointDir = p - edge->Point(0);
			return pointDir.Dot((prevEdgeDir - edgeDir).Normalize());
		}
		if (param > 1) {
			Vector2 edgeDir = -edge->Direction(1).Normalize();
			Vector2 nextEdgeDir = nextEdge->Direction(0).Normalize();
			Vector2 pointDir = p - edge->Point(1);
			return pointDir.Dot((nextEdgeDir - edgeDir).Normalize());
		}
		return 0;
	}

	void Reset(double delta) {
		minTrueDistance.distance += NonZeroSign(minTrueDistance.distance) * delta;
		minNegativePseudoDistance.distance = -std::fabs(minTrueDistance.distance);
		minPositivePseudoDistance.distance = std::fabs(minTrueDistance.distance);
		nearEdge = NULL;
		nearEdgeParam = 0;
	}
	bool IsEdgeRelevant(const EdgeCache &cache, const EdgeSegment *edge, const Vector2 &p) const {
		double delta = DISTANCE_DELTA_FACTOR * (p - cache.point).Length();
		return (
			cache.absDistance - delta <= std::fabs(minTrueDistance.distance) ||
			(cache.edgeDomainDistance > 0 ?
				cache.edgeDomainDistance - delta <= 0 :
				(cache.pseudoDistance < 0 ?
					cache.pseudoDistance + delta >= minNegativePseudoDistance.distance :
					cache.pseudoDistance - delta <= minPositivePseudoDistance.distance
					)
				)
			);
	}
	void AddEdgeTrueDistance(const EdgeSegment *edge, const SignedDistance &distance, double param) {
		if (distance < minTrueDistance) {
			minTrueDistance = distance;
			nearEdge = edge;
			nearEdgeParam = param;
		}
	}
	void AddEdgePseudoDistance(const SignedDistance &distance) {
		SignedDistance &minPseudoDistance = distance.distance < 0 ? minNegativePseudoDistance : minPositivePseudoDistance;
		if (distance < minPseudoDistance)
			minPseudoDistance = distance;
	}
	void Merge(const PseudoDistanceSelectorBase &other) {
		if (other.minTrueDistance < minTrueDistance) {
			minTrueDistance = other.minTrueDistance;
			nearEdge = other.nearEdge;
			nearEdgeParam = other.nearEdgeParam;
		}
		if (other.minNegativePseudoDistance < minNegativePseudoDistance)
			minNegativePseudoDistance = other.minNegativePseudoDistance;
		if (other.minPositivePseudoDistance < minPositivePseudoDistance)
			minPositivePseudoDistance = other.minPositivePseudoDistance;
	}
	double ComputeDistance(const Vector2 &p) const {
		double minDistance = minTrueDistance.distance < 0 ? minNegativePseudoDistance.distance : minPositivePseudoDistance.distance;
		if (nearEdge) {
			SignedDistance distance = minTrueDistance;
			nearEdge->DistanceToPseudoDistance(distance, p, nearEdgeParam);
			if (std::fabs(distance.distance) < std::fabs(minDistance))
				minDistance = distance.distance;
		}
		return minDistance;
	}
	SignedDistance TrueDistance() const {
		return minTrueDistance;
	}

private:
	SignedDistance minTrueDistance;
	SignedDistance minNegativePseudoDistance;
	SignedDistance minPositivePseudoDistance;
	const EdgeSegment *nearEdge = nullptr;
	double nearEdgeParam = 0.0;
};

/// Selects the nearest edge by its pseudo-distance.
class PseudoDistanceSelector : public PseudoDistanceSelectorBase {
public:
	using DistanceType = double;

	void Reset(const Vector2 &p) {
		double delta = DISTANCE_DELTA_FACTOR * (p - this->p).Length();
		PseudoDistanceSelectorBase::Reset(delta);
		this->p = p;
	}
	void AddEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge) {
		if (IsEdgeRelevant(cache, edge, p)) {
			double param;
			SignedDistance distance = edge->MinSignedDistance(p, param);
			double edd = EdgeDomainDistance(prevEdge, edge, nextEdge, p, param);
			AddEdgeTrueDistance(edge, distance, param);
			cache.point = p;
			cache.absDistance = std::fabs(distance.distance);
			cache.edgeDomainDistance = edd;
			if (edd <= 0) {
				edge->DistanceToPseudoDistance(distance, p, param);
				AddEdgePseudoDistance(distance);
				cache.pseudoDistance = distance.distance;
			}
		}
	}
	DistanceType Distance() const {
		return ComputeDistance(p);
	}

private:
	Vector2 p;
};

/// Selects the nearest edge for each of the three channels by its pseudo-distance.
class MultiDistanceSelector {
public:
	using DistanceType = MultiDistance;
	using EdgeCache = PseudoDistanceSelectorBase::EdgeCache;

	void Reset(const Vector2 &p) {
		double delta = DISTANCE_DELTA_FACTOR * (p - this->p).Length();
		r.Reset(delta);
		g.Reset(delta);
		b.Reset(delta);
		this->p = p;
	}
	void AddEdge(EdgeCache &cache, const EdgeSegment *prevEdge, const EdgeSegment *edge, const EdgeSegment *nextEdge) {
		if (
			(edge->color & RED && r.IsEdgeRelevant(cache, edge, p)) ||
			(edge->color & GREEN && g.IsEdgeRelevant(cache, edge, p)) ||
			(edge->color & BLUE && b.IsEdgeRelevant(cache, edge, p))
			) {
			double param;
			SignedDistance distance = edge->MinSignedDistance(p, param);
			double edd = PseudoDistanceSelectorBase::EdgeDomainDistance(prevEdge, edge, nextEdge, p, param);
			if (edge->color & RED)
				r.AddEdgeTrueDistance(edge, distance, param);
			if (edge->color & GREEN)
				g.AddEdgeTrueDistance(edge, distance, param);
			if (edge->color & BLUE)
				b.AddEdgeTrueDistance(edge, distance, param);
			cache.point = p;
			cache.absDistance = fabs(distance.distance);
			cache.edgeDomainDistance = edd;
			if (edd <= 0) {
				edge->DistanceToPseudoDistance(distance, p, param);
				if (edge->color & RED)
					r.AddEdgePseudoDistance(distance);
				if (edge->color & GREEN)
					g.AddEdgePseudoDistance(distance);
				if (edge->color & BLUE)
					b.AddEdgePseudoDistance(distance);
				cache.pseudoDistance = distance.distance;
			}
		}
	}
	void Merge(const MultiDistanceSelector &other) {
		r.Merge(other.r);
		g.Merge(other.g);
		b.Merge(other.b);
	}
	DistanceType Distance() const {
		MultiDistance multiDistance;
		multiDistance.r = r.ComputeDistance(p);
		multiDistance.g = g.ComputeDistance(p);
		multiDistance.b = b.ComputeDistance(p);
		return multiDistance;
	}
	SignedDistance TrueDistance() const {
		SignedDistance distance = r.TrueDistance();
		if (g.TrueDistance() < distance)
			distance = g.TrueDistance();
		if (b.TrueDistance() < distance)
			distance = b.TrueDistance();
		return distance;
	}

private:
	Vector2 p;
	PseudoDistanceSelectorBase r, g, b;
};

/// Selects the nearest edge for each of the three color channels by its pseudo-distance and by true distance for the alpha channel.
class MultiAndTrueDistanceSelector : public MultiDistanceSelector {
public:
	using DistanceType = MultiAndTrueDistance;

	DistanceType Distance() const {
		MultiDistance multiDistance = MultiDistanceSelector::Distance();
		MultiAndTrueDistance mtd;
		mtd.r = multiDistance.r;
		mtd.g = multiDistance.g;
		mtd.b = multiDistance.b;
		mtd.a = TrueDistance().distance;
		return mtd;
	}
};

/// Simply selects the nearest contour.
template<class EdgeSelector>
class SimpleContourCombiner {
public:
	using EdgeSelectorType = EdgeSelector;
	using DistanceType = typename EdgeSelector::DistanceType;

	explicit SimpleContourCombiner(const Shape &shape) {}
	void Reset(const Vector2 &p) {
		shapeEdgeSelector.Reset(p);
	}
	DistanceType Distance() const {
		return shapeEdgeSelector.Distance();
	}
	EdgeSelector &GetEdgeSelector(int i) { return shapeEdgeSelector; }

private:
	EdgeSelector shapeEdgeSelector;
};
template class SimpleContourCombiner<TrueDistanceSelector>;
template class SimpleContourCombiner<PseudoDistanceSelector>;
template class SimpleContourCombiner<MultiDistanceSelector>;
template class SimpleContourCombiner<MultiAndTrueDistanceSelector>;

/// Selects the nearest contour that actually forms a border between filled and unfilled area.
template <class EdgeSelector>
class OverlappingContourCombiner {

public:
	using EdgeSelectorType = EdgeSelector;
	using DistanceType = typename EdgeSelector::DistanceType;

	explicit OverlappingContourCombiner(const Shape &shape) {
		windings.reserve(shape.contours.size());
		for (const auto &contour : shape.contours)
			windings.emplace_back(contour.Winding());
		edgeSelectors.resize(shape.contours.size());
	}
	void Reset(const Vector2 &p) {
		this->p = p;
		for (auto &contourEdgeSelector : edgeSelectors)
			contourEdgeSelector.Reset(p);
	}
	DistanceType Distance() const {
		int contourCount = (int)edgeSelectors.size();
		EdgeSelector shapeEdgeSelector;
		EdgeSelector innerEdgeSelector;
		EdgeSelector outerEdgeSelector;
		shapeEdgeSelector.Reset(p);
		innerEdgeSelector.Reset(p);
		outerEdgeSelector.Reset(p);
		for (int i = 0; i < contourCount; ++i) {
			DistanceType edgeDistance = edgeSelectors[i].Distance();
			shapeEdgeSelector.Merge(edgeSelectors[i]);
			if (windings[i] > 0 && ResolveDistance(edgeDistance) >= 0)
				innerEdgeSelector.Merge(edgeSelectors[i]);
			if (windings[i] < 0 && ResolveDistance(edgeDistance) <= 0)
				outerEdgeSelector.Merge(edgeSelectors[i]);
		}

		DistanceType shapeDistance = shapeEdgeSelector.Distance();
		DistanceType innerDistance = innerEdgeSelector.Distance();
		DistanceType outerDistance = outerEdgeSelector.Distance();
		double innerScalarDistance = ResolveDistance(innerDistance);
		double outerScalarDistance = ResolveDistance(outerDistance);
		DistanceType distance;
		InitDistance(distance);

		int winding = 0;
		if (innerScalarDistance >= 0 && std::fabs(innerScalarDistance) <= std::fabs(outerScalarDistance)) {
			distance = innerDistance;
			winding = 1;
			for (int i = 0; i < contourCount; ++i)
				if (windings[i] > 0) {
					DistanceType contourDistance = edgeSelectors[i].Distance();
					if (std::fabs(ResolveDistance(contourDistance)) < std::fabs(outerScalarDistance) && ResolveDistance(contourDistance) > ResolveDistance(distance))
						distance = contourDistance;
				}
		} else if (outerScalarDistance <= 0 && std::fabs(outerScalarDistance) < std::fabs(innerScalarDistance)) {
			distance = outerDistance;
			winding = -1;
			for (int i = 0; i < contourCount; ++i)
				if (windings[i] < 0) {
					DistanceType contourDistance = edgeSelectors[i].Distance();
					if (std::fabs(ResolveDistance(contourDistance)) < std::fabs(innerScalarDistance) && ResolveDistance(contourDistance) < ResolveDistance(distance))
						distance = contourDistance;
				}
		} else {
			return shapeDistance;
		}

		for (int i = 0; i < contourCount; ++i)
			if (windings[i] != winding) {
				DistanceType contourDistance = edgeSelectors[i].Distance();
				if (ResolveDistance(contourDistance) * ResolveDistance(distance) >= 0 && std::fabs(ResolveDistance(contourDistance)) < std::fabs(ResolveDistance(distance)))
					distance = contourDistance;
			}
		if (ResolveDistance(distance) == ResolveDistance(shapeDistance))
			distance = shapeDistance;
		return distance;
	}
	EdgeSelector &GetEdgeSelector(int i) { return edgeSelectors[i]; }

private:
	Vector2 p;
	std::vector<int> windings;
	std::vector<EdgeSelector> edgeSelectors;
};
template class OverlappingContourCombiner<TrueDistanceSelector>;
template class OverlappingContourCombiner<PseudoDistanceSelector>;
template class OverlappingContourCombiner<MultiDistanceSelector>;
template class OverlappingContourCombiner<MultiAndTrueDistanceSelector>;

template <typename DistanceType>
class DistancePixelConversion;

template <>
class DistancePixelConversion<double> {
public:
	using BitmapType = Bitmap<float, 1>;
	static void Convert(float *pixels, double distance, double range) {
		*pixels = float(distance / range + .5);
	}
};

template <>
class DistancePixelConversion<MultiDistance> {
public:
	using BitmapType = Bitmap<float, 3>;
	static void Convert(float *pixels, const MultiDistance &distance, double range) {
		pixels[0] = float(distance.r / range + .5);
		pixels[1] = float(distance.g / range + .5);
		pixels[2] = float(distance.b / range + .5);
	}
};

template <>
class DistancePixelConversion<MultiAndTrueDistance> {
public:
	using BitmapType = Bitmap<float, 4>;
	static void Convert(float *pixels, const MultiAndTrueDistance &distance, double range) {
		pixels[0] = float(distance.r / range + .5);
		pixels[1] = float(distance.g / range + .5);
		pixels[2] = float(distance.b / range + .5);
		pixels[3] = float(distance.a / range + .5);
	}
};

template <class ContourCombiner>
void GenerateDistanceField(typename DistancePixelConversion<typename ContourCombiner::DistanceType>::BitmapType &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
	int edgeCount = shape.EdgeCount();
	ContourCombiner contourCombiner(shape);
	std::vector<typename ContourCombiner::EdgeSelectorType::EdgeCache> shapeEdgeCache(edgeCount);
	bool rightToLeft = false;
	Vector2 p;
	for (int y = 0; y < output.height; ++y) {
		int row = shape.inverseYAxis ? output.height - y - 1 : y;
		p.y = (y + 0.5) / scale.y - translate.y;
		for (int col = 0; col < output.width; ++col) {
			int x = rightToLeft ? output.width - col - 1 : col;
			p.x = (x + 0.5) / scale.x - translate.x;

			contourCombiner.Reset(p);
			typename ContourCombiner::EdgeSelectorType::EdgeCache *edgeCache = &shapeEdgeCache[0];

			for (auto contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
				if (!contour->edges.empty()) {
					typename ContourCombiner::EdgeSelectorType &edgeSelector = contourCombiner.GetEdgeSelector(int(contour - shape.contours.begin()));

					const EdgeSegment *prevEdge = contour->edges.size() >= 2 ? (contour->edges.end() - 2)->get() : contour->edges.begin()->get();
					const EdgeSegment *curEdge = contour->edges.back().get();
					for (auto edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
						const EdgeSegment *nextEdge = edge->get();
						edgeSelector.AddEdge(*edgeCache++, prevEdge, curEdge, nextEdge);
						prevEdge = curEdge;
						curEdge = nextEdge;
					}
				}
			}

			typename ContourCombiner::DistanceType distance = contourCombiner.Distance();
			DistancePixelConversion<typename ContourCombiner::DistanceType>::Convert(output(x, row), distance, range);
		}
		rightToLeft = !rightToLeft;
	}
}

bool DetectClash(const float *a, const float *b, double threshold) {
	// Sort channels so that pairs (a0, b0), (a1, b1), (a2, b2) go from biggest to smallest absolute difference
	float a0 = a[0], a1 = a[1], a2 = a[2];
	float b0 = b[0], b1 = b[1], b2 = b[2];
	float tmp;
	if (std::fabsf(b0 - a0) < std::fabsf(b1 - a1)) {
		tmp = a0, a0 = a1, a1 = tmp;
		tmp = b0, b0 = b1, b1 = tmp;
	}
	if (std::fabsf(b1 - a1) < std::fabsf(b2 - a2)) {
		tmp = a1, a1 = a2, a2 = tmp;
		tmp = b1, b1 = b2, b2 = tmp;
		if (std::fabsf(b0 - a0) < std::fabsf(b1 - a1)) {
			tmp = a0, a0 = a1, a1 = tmp;
			tmp = b0, b0 = b1, b1 = tmp;
		}
	}
	return (std::fabsf(b1 - a1) >= threshold) &&
		!(b0 == b1 && b0 == b2) && // Ignore if other pixel has been equalized
		std::fabsf(a2 - 0.5f) >= std::fabsf(b2 - 0.5f); // Out of the pair, only flag the pixel farther from a shape edge
}

/// Resolves multi-channel signed distance field values that may cause interpolation artifacts.
template <int N>
void MsdfErrorCorrection(Bitmap<float, N> &output, const Vector2 &threshold) {
	std::vector<std::pair<int, int> > clashes;
	int w = output.width, h = output.height;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			if ((x > 0 && DetectClash(output(x, y), output(x - 1, y), threshold.x)) ||
				(x < w - 1 && DetectClash(output(x, y), output(x + 1, y), threshold.x)) ||
				(y > 0 && DetectClash(output(x, y), output(x, y - 1), threshold.y)) ||
				(y < h - 1 && DetectClash(output(x, y), output(x, y + 1), threshold.y))) {
				clashes.emplace_back(x, y);
			}
		}
	}
	for (const auto &clash : clashes) {
		float *pixel = output(clash.first, clash.second);
		float med = Median(pixel[0], pixel[1], pixel[2]);
		pixel[0] = med, pixel[1] = med, pixel[2] = med;
	}
	clashes.clear();
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x) {
			if ((x > 0 && y > 0 && DetectClash(output(x, y), output(x - 1, y - 1), threshold.x + threshold.y)) ||
				(x < w - 1 && y > 0 && DetectClash(output(x, y), output(x + 1, y - 1), threshold.x + threshold.y)) ||
				(x > 0 && y < h - 1 && DetectClash(output(x, y), output(x - 1, y + 1), threshold.x + threshold.y)) ||
				(x < w - 1 && y < h - 1 && DetectClash(output(x, y), output(x + 1, y + 1), threshold.x + threshold.y))) {
				clashes.emplace_back(x, y);
			}
		}
	for (const auto &clash : clashes) {
		float *pixel = output(clash.first, clash.second);
		float med = Median(pixel[0], pixel[1], pixel[2]);
		pixel[0] = med, pixel[1] = med, pixel[2] = med;
	}
}

bool GenerateSdf(Bitmap<float, 1> &output, FT_Face face, unicode_t unicode) {
	auto shape = LoadGlyph(face, unicode);

	// Validate and normalize shape
	if (!shape.Validate())
		throw MsdfException("The geometry of the loaded shape is invalid");
	shape.Normalize();

	double pxRange = 2.0;
	Vector2 translate, scale;
	AutoFrame(shape, output.width, output.height, pxRange, translate, scale);
	double range = pxRange / std::min(scale.x, scale.y);

	double angleThreshold = 3;
	unsigned long long coloringSeed = 0;
	EdgeColoringSimple(shape, angleThreshold, coloringSeed);

	double edgeThreshold = DEFAULT_ERROR_CORRECTION_THRESHOLD;
	GenerateDistanceField<OverlappingContourCombiner<TrueDistanceSelector>>(output, shape, range, scale, translate);
	if (edgeThreshold > 0)
		MsdfErrorCorrection(output, edgeThreshold / (scale * range));

	return false;
}

bool GeneratePseudoSdf(Bitmap<float, 1> &output, FT_Face face, unicode_t unicode) {
	auto shape = LoadGlyph(face, unicode);

	// Validate and normalize shape
	if (!shape.Validate())
		throw MsdfException("The geometry of the loaded shape is invalid");
	shape.Normalize();

	double pxRange = 2.0;
	Vector2 translate, scale;
	AutoFrame(shape, output.width, output.height, pxRange, translate, scale);
	double range = pxRange / std::min(scale.x, scale.y);

	double angleThreshold = 3;
	unsigned long long coloringSeed = 0;
	EdgeColoringSimple(shape, angleThreshold, coloringSeed);

	double edgeThreshold = DEFAULT_ERROR_CORRECTION_THRESHOLD;
	GenerateDistanceField<OverlappingContourCombiner<PseudoDistanceSelector>>(output, shape, range, scale, translate);
	if (edgeThreshold > 0)
		MsdfErrorCorrection(output, edgeThreshold / (scale * range));

	return false;
}

bool GenerateMsdf(Bitmap<float, 3> &output, FT_Face face, unicode_t unicode) {
	auto shape = LoadGlyph(face, unicode);

	// Validate and normalize shape
	if (!shape.Validate())
		throw MsdfException("The geometry of the loaded shape is invalid");
	shape.Normalize();

	double pxRange = 2.0;
	Vector2 translate, scale;
	AutoFrame(shape, output.width, output.height, pxRange, translate, scale);
	double range = pxRange / std::min(scale.x, scale.y);

	double angleThreshold = 3;
	unsigned long long coloringSeed = 0;
	EdgeColoringSimple(shape, angleThreshold, coloringSeed);

	double edgeThreshold = DEFAULT_ERROR_CORRECTION_THRESHOLD;
	GenerateDistanceField<OverlappingContourCombiner<MultiDistanceSelector>>(output, shape, range, scale, translate);
	if (edgeThreshold > 0)
		MsdfErrorCorrection(output, edgeThreshold / (scale * range));

	return false;
}

bool GenerateMtsdf(Bitmap<float, 4> &output, FT_Face face, unicode_t unicode) {
	auto shape = LoadGlyph(face, unicode);

	// Validate and normalize shape
	if (!shape.Validate())
		throw MsdfException("The geometry of the loaded shape is invalid");
	shape.Normalize();

	double pxRange = 2.0;
	Vector2 translate, scale;
	AutoFrame(shape, output.width, output.height, pxRange, translate, scale);
	double range = pxRange / std::min(scale.x, scale.y);

	double angleThreshold = 3;
	unsigned long long coloringSeed = 0;
	EdgeColoringSimple(shape, angleThreshold, coloringSeed);

	double edgeThreshold = DEFAULT_ERROR_CORRECTION_THRESHOLD;
	GenerateDistanceField<OverlappingContourCombiner<MultiAndTrueDistanceSelector>>(output, shape, range, scale, translate);
	if (edgeThreshold > 0)
		MsdfErrorCorrection(output, edgeThreshold / (scale * range));

	return false;
}
}
