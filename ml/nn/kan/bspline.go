package kan

import "math"

// BSplineGrid holds a uniform knot vector and evaluates cubic B-spline basis functions.
// All computation is done in pure Go on CPU since the coefficient tensors are tiny.
//
// The grid can be dynamically expanded via Expand() when logits fall outside the
// current [GridMin, GridMax] range. Expansion preserves the step size and existing
// knot positions, so previously-evaluated basis functions are unchanged.
type BSplineGrid struct {
	Order    int
	NumBasis int
	Knots    []float32 // length = NumBasis + Order + 1
	GridMin  float32   // lower bound of the active region
	GridMax  float32   // upper bound of the active region
	Step     float32   // uniform spacing between interior knots
}

// NewBSplineGrid creates a uniform B-spline grid with the given parameters.
// For cubic B-splines (order=3) with numBasis=8, this creates 12 knots
// uniformly spaced in [gridMin, gridMax] with extended boundary knots.
func NewBSplineGrid(order, numBasis int, gridMin, gridMax float32) *BSplineGrid {
	numInterior := numBasis - order
	if numInterior < 1 {
		numInterior = 1
	}

	step := (gridMax - gridMin) / float32(numInterior)
	numKnots := numBasis + order + 1
	knots := make([]float32, numKnots)

	for i := range knots {
		knots[i] = gridMin + float32(i-order)*step
	}

	return &BSplineGrid{
		Order:    order,
		NumBasis: numBasis,
		Knots:    knots,
		GridMin:  gridMin,
		GridMax:  gridMax,
		Step:     step,
	}
}

// Expand returns a new BSplineGrid covering [newMin, newMax] with the same step
// size and order. Boundaries are snapped outward to step multiples so the knot
// vector stays uniform.
//
// Returns the new grid and a leftOffset: old basis function i maps to new basis
// function (i + leftOffset). This offset is needed to reposition existing
// coefficients in the expanded weight vector.
//
// If the current grid already covers [newMin, newMax], returns (self, 0).
func (g *BSplineGrid) Expand(newMin, newMax float32) (*BSplineGrid, int) {
	// How many steps to extend in each direction?
	leftSteps := 0
	if newMin < g.GridMin {
		leftSteps = int(math.Ceil(float64(g.GridMin-newMin) / float64(g.Step)))
	}
	rightSteps := 0
	if newMax > g.GridMax {
		rightSteps = int(math.Ceil(float64(newMax-g.GridMax) / float64(g.Step)))
	}

	if leftSteps == 0 && rightSteps == 0 {
		return g, 0
	}

	expandedMin := g.GridMin - float32(leftSteps)*g.Step
	expandedMax := g.GridMax + float32(rightSteps)*g.Step
	newNumBasis := g.NumBasis + leftSteps + rightSteps

	newGrid := NewBSplineGrid(g.Order, newNumBasis, expandedMin, expandedMax)
	// Force the same step to avoid floating-point drift
	newGrid.Step = g.Step

	return newGrid, leftSteps
}

// Evaluate computes the values of all basis functions at a single point x.
// Uses the Cox-de Boor recursion algorithm.
// Returns a slice of length NumBasis.
func (g *BSplineGrid) Evaluate(x float32) []float32 {
	k := g.Order + 1 // B-spline order (degree + 1), e.g., 4 for cubic
	n := g.NumBasis
	t := g.Knots

	// Start with degree-0 basis functions (piecewise constant)
	numIntervals := len(t) - 1
	basis := make([]float32, numIntervals)
	for i := 0; i < numIntervals; i++ {
		if (x >= t[i] && x < t[i+1]) || (i == numIntervals-1 && x == t[i+1]) {
			basis[i] = 1.0
		}
	}

	// Cox-de Boor recursion for degrees 1..order
	for d := 1; d < k; d++ {
		newBasis := make([]float32, numIntervals-d)
		for i := range newBasis {
			var left, right float32

			denom1 := t[i+d] - t[i]
			if denom1 > 0 {
				left = (x - t[i]) / denom1 * basis[i]
			}

			denom2 := t[i+d+1] - t[i+1]
			if denom2 > 0 {
				right = (t[i+d+1] - x) / denom2 * basis[i+1]
			}

			newBasis[i] = left + right
		}
		basis = newBasis
	}

	// Ensure we return exactly NumBasis values
	if len(basis) > n {
		basis = basis[:n]
	}
	for len(basis) < n {
		basis = append(basis, 0)
	}

	return basis
}

// EvaluateBatch computes basis function values for a batch of input points.
// Returns a [len(xs)][NumBasis] matrix (row-major).
func (g *BSplineGrid) EvaluateBatch(xs []float32) [][]float32 {
	result := make([][]float32, len(xs))
	for i, x := range xs {
		result[i] = g.Evaluate(x)
	}
	return result
}

// InitSoftmaxApprox returns initial B-spline coefficients that make the KAN
// approximate the identity function: kan(x) ≈ x.
//
// Since the forward pass applies exp(kan(x) - rowMax) / sum(exp(...)),
// initializing to identity means the full pipeline computes:
//
//	exp(x - max) / sum(exp(...)) ≈ softmax(x)
//
// This gives the KAN a near-perfect starting point, and training refines
// it from there.
//
// The coefficients are the Greville abscissae of the knot vector, which is
// the standard way to make a B-spline curve interpolate the identity. No
// further normalization is applied — shifting or rescaling the coefficients
// would change the effective slope away from 1.0 and break the identity
// approximation.
func InitSoftmaxApprox(grid *BSplineGrid) []float32 {
	n := grid.NumBasis
	coeffs := make([]float32, n)

	// To approximate identity with B-splines, the coefficients should be
	// the x-coordinates at each basis function's center (Greville abscissae).
	// For a uniform knot vector, these are evenly spaced.
	k := grid.Order + 1 // degree + 1
	for i := 0; i < n; i++ {
		// Greville abscissa: average of k-1 consecutive knots starting at i+1
		sum := float32(0)
		count := 0
		for j := 1; j < k; j++ {
			idx := i + j
			if idx < len(grid.Knots) {
				sum += grid.Knots[idx]
				count++
			}
		}
		if count > 0 {
			coeffs[i] = sum / float32(count)
		}
	}

	return coeffs
}
